// Python bindings: a thin nanobind layer over the same seams the CLI uses
// (config::apply_overrides, cli::train_with_progress, io::save/load_booster).
// No training or prediction logic lives here.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/cli/common.hpp"
#include "bonsai/cli/pipeline.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/toml.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/registry/objective_dispatch.hpp"

namespace nb = nanobind;

namespace
{

using array_2d = nb::ndarray<float const, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using array_1d = nb::ndarray<float const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

bonsai::cli::LabeledData make_labeled(array_2d const &X, array_1d const &y,
                                      bonsai::BinMappers const &mappers,
                                      bonsai::Config const     &cfg)
{
    size_t const n = X.shape(0);
    size_t const f = X.shape(1);

    bonsai::detail::ColumnBatch batch;
    batch.features.assign(f, std::vector<float>(n));
    for (size_t c = 0; c < f; ++c)
    {
        for (size_t r = 0; r < n; ++r)
        {
            batch.features[c][r] = X(r, c);
        }
    }
    batch.labels.assign(y.data(), y.data() + n);

    bonsai::cli::FeatureBuffer buf;
    buf.n_rows     = n;
    buf.n_features = f;
    buf.data.assign(X.data(), X.data() + (n * f));

    return bonsai::cli::LabeledData{.dataset =
                                        bonsai::Dataset::bin(batch, mappers, cfg.data),
                                    .features = std::move(buf),
                                    .labels   = std::move(batch.labels)};
}

// A trained model: booster + the bin mappers and config it was fit with.
class Model
{
  public:
    Model(std::unique_ptr<bonsai::IBooster> booster, bonsai::BinMappers mappers,
          bonsai::Config cfg)
        : booster_(std::move(booster)), mappers_(std::move(mappers)),
          cfg_(std::move(cfg))
    {
    }

    nb::ndarray<nb::numpy, float> predict(array_2d const &X,
                                          size_t          num_iteration = 0) const
    {
        size_t const n   = X.shape(0);
        auto        *out = new std::vector<float>(n, 0.0F);
        {
            nb::gil_scoped_release release;
            booster_->predict_at(bonsai::features_view{X.data(), n, X.shape(1)}, *out,
                                 num_iteration);
            bonsai::apply_link_inverse_by_name(cfg_.dispatch.objective_name, *out);
        }
        nb::capsule owner(out, [](void *p) noexcept
                          { delete static_cast<std::vector<float> *>(p); });
        return {out->data(), {n}, owner};
    }

    // (n_iters, n_rows): prediction after each boosting iteration.
    nb::ndarray<nb::numpy, float> staged_predict(array_2d const &X) const
    {
        size_t const n   = X.shape(0);
        size_t const k   = booster_->n_iters();
        auto        *out = new std::vector<float>(k * n, 0.0F);
        {
            nb::gil_scoped_release release;
            booster_->predict_staged(bonsai::features_view{X.data(), n, X.shape(1)},
                                     *out);
            for (size_t t = 0; t < k; ++t)
            {
                bonsai::apply_link_inverse_by_name(
                    cfg_.dispatch.objective_name,
                    bonsai::floats_out{out->data() + (t * n), n});
            }
        }
        nb::capsule owner(out, [](void *p) noexcept
                          { delete static_cast<std::vector<float> *>(p); });
        return {out->data(), {k, n}, owner};
    }

    // (n_rows, n_iters): the leaf each row lands in, per tree.
    nb::ndarray<nb::numpy, uint32_t> predict_leaf(array_2d const &X) const
    {
        size_t const n   = X.shape(0);
        size_t const k   = booster_->n_iters();
        auto        *out = new std::vector<bonsai::node_id_t>(n * k, 0);
        {
            nb::gil_scoped_release release;
            booster_->predict_leaf(bonsai::features_view{X.data(), n, X.shape(1)},
                                   std::span<bonsai::node_id_t>{*out});
        }
        nb::capsule owner(out, [](void *p) noexcept
                          { delete static_cast<std::vector<bonsai::node_id_t> *>(p); });
        return {out->data(), {n, k}, owner};
    }

    std::string dump() const
    {
        return booster_->dump(mappers_.feature_names());
    }

    // (n_rows, n_features + 1): TreeSHAP contributions, last column = bias.
    // Rows sum to the raw (pre-link) prediction exactly.
    nb::ndarray<nb::numpy, double> pred_contribs(array_2d const &X) const
    {
        size_t const n    = X.shape(0);
        size_t const nf   = X.shape(1);
        size_t const cols = nf + 1;
        auto        *out  = new std::vector<double>(n * cols, 0.0);
        {
            nb::gil_scoped_release release;
            booster_->pred_contribs(bonsai::features_view{X.data(), n, nf},
                                    std::span<double>{*out}, nf);
        }
        nb::capsule owner(out, [](void *p) noexcept
                          { delete static_cast<std::vector<double> *>(p); });
        return {out->data(), {n, cols}, owner};
    }

    void save(std::string const &path) const
    {
        bonsai::io::save_booster(*booster_, path, mappers_, cfg_);
    }

    // type: "gain" (total loss reduction) or "split" (split count),
    // padded to the trained feature count.
    nb::ndarray<nb::numpy, double> feature_importance(std::string const &type) const
    {
        bonsai::ImportanceType const t = [&]
        {
            if (type == "gain")
            {
                return bonsai::ImportanceType::gain;
            }
            if (type == "split")
            {
                return bonsai::ImportanceType::split;
            }
            throw std::invalid_argument("importance type must be 'gain' or 'split'");
        }();
        auto *out = new std::vector<double>(booster_->feature_importance(t));
        out->resize(std::max(out->size(), mappers_.size()), 0.0);
        nb::capsule owner(out, [](void *p) noexcept
                          { delete static_cast<std::vector<double> *>(p); });
        return {out->data(), {out->size()}, owner};
    }

    size_t n_iters() const
    {
        return booster_->n_iters();
    }

    std::string config_toml() const
    {
        return bonsai::config::dump_toml(cfg_);
    }

  private:
    std::unique_ptr<bonsai::IBooster> booster_;
    bonsai::BinMappers                mappers_;
    bonsai::Config                    cfg_;
};

bonsai::Config
config_from_params(std::vector<std::pair<std::string, std::string>> const &params)
{
    bonsai::Config                        cfg;
    std::vector<bonsai::config::Override> overrides;
    overrides.reserve(params.size());
    for (auto const &[key, value] : params)
    {
        overrides.push_back({.key = key, .value = value});
    }
    bonsai::config::apply_overrides(cfg, overrides);
    return cfg;
}

Model train(std::vector<std::pair<std::string, std::string>> const &params,
            array_2d const &X, array_1d const &y,
            std::optional<std::pair<array_2d, array_1d>> const &eval_set,
            std::optional<std::string> const                   &init_model)
{
    bonsai::Config const cfg = config_from_params(params);
    bonsai::parallel::set_n_threads(cfg.parallel.n_threads);

    std::optional<bonsai::io::LoadedBooster> init;
    if (init_model)
    {
        init.emplace(bonsai::io::load_booster(*init_model));
    }

    bonsai::detail::ColumnBatch fit_batch;
    size_t const                n = X.shape(0);
    size_t const                f = X.shape(1);
    fit_batch.features.assign(f, std::vector<float>(n));
    for (size_t c = 0; c < f; ++c)
    {
        for (size_t r = 0; r < n; ++r)
        {
            fit_batch.features[c][r] = X(r, c);
        }
    }
    fit_batch.feature_names.reserve(f);
    for (size_t c = 0; c < f; ++c)
    {
        fit_batch.feature_names.push_back("f" + std::to_string(c));
    }

    nb::gil_scoped_release release;

    bonsai::cli::LoadedTrainValid loaded;
    loaded.mappers = init ? std::move(init->mappers)
                          : bonsai::BinMappers::fit(fit_batch, cfg.bin_mapper);
    loaded.train   = make_labeled(X, y, loaded.mappers, cfg);
    if (eval_set)
    {
        loaded.valid =
            make_labeled(eval_set->first, eval_set->second, loaded.mappers, cfg);
    }

    auto booster = bonsai::cli::train_with_progress(
        cfg, loaded, {}, init ? std::move(init->booster) : nullptr);
    return Model{std::move(booster), std::move(loaded.mappers), cfg};
}

Model load(std::string const &path)
{
    auto loaded = bonsai::io::load_booster(path);
    return Model{std::move(loaded.booster), std::move(loaded.mappers),
                 std::move(loaded.cfg)};
}

} // namespace

NB_MODULE(_bonsai, m)
{
    m.doc() = "bonsai gradient-boosted trees (native module)";

    nb::class_<Model>(m, "Model")
        .def("predict", &Model::predict, nb::arg("X"), nb::arg("num_iteration") = 0)
        .def("staged_predict", &Model::staged_predict, nb::arg("X"))
        .def("predict_leaf", &Model::predict_leaf, nb::arg("X"))
        .def("dump", &Model::dump)
        .def("pred_contribs", &Model::pred_contribs, nb::arg("X"))
        .def("feature_importance", &Model::feature_importance, nb::arg("type") = "gain")
        .def("save", &Model::save, nb::arg("path"))
        .def_prop_ro("n_iters", &Model::n_iters)
        .def_prop_ro("config_toml", &Model::config_toml);

    m.def("train", &train, nb::arg("params"), nb::arg("X"), nb::arg("y"),
          nb::arg("eval_set") = nb::none(), nb::arg("init_model") = nb::none(),
          "Train a booster on row-major float32 features. `params` is a list "
          "of (dotted-key, value-string) config overrides, e.g. "
          "('tree.max_depth', '8').");
    m.def("load", &load, nb::arg("path"), "Load a model saved by Model.save.");

    m.def("default_config_toml", [] { return bonsai::config::dump_toml({}); });
}

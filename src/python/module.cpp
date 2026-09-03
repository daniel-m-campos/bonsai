
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/cli/common.hpp"
#include "bonsai/cli/pipeline.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/sections/all.hpp"
#include "bonsai/config/toml.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/predict.hpp"
#include "bonsai/cuda/shap.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/registry/objective_dispatch.hpp"

namespace nb = nanobind;

namespace
{

using array_2d = nb::ndarray<float const, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using array_1d = nb::ndarray<float const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using cuda_2d  = nb::ndarray<float const, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using cuda_1d  = nb::ndarray<float const, nb::ndim<1>, nb::c_contig, nb::device::cuda>;

bonsai::features_view as_view(array_2d const &X)
{
    return bonsai::features_view{X.data(), X.shape(0), X.shape(1)};
}

template <typename T>
nb::ndarray<nb::numpy, T> to_numpy(std::unique_ptr<std::vector<T>> out,
                                   std::initializer_list<size_t>   shape)
{
    auto       *raw = out.release();
    nb::capsule owner(raw, [](void *p) noexcept
                      { delete static_cast<std::vector<T> *>(p); });
    return {raw->data(), shape, owner};
}

template <typename Array>
void place_device_array(Array const &arr, uint32_t device_id, char const *what)
{
    if (arr.data() == nullptr || arr.size() == 0)
    {
        throw std::invalid_argument(std::string{what} + " is an empty device array");
    }
    if (!bonsai::cuda_available())
    {
        throw std::invalid_argument(std::string{what} +
                                    " is device-resident (DLPack), which needs a CUDA "
                                    "build and a visible device; cuda_available() is "
                                    "False");
    }
    bonsai::cuda_select_device(device_id);
    auto const on = static_cast<uint32_t>(arr.device_id());
    if (on != device_id)
    {
        throw std::invalid_argument(
            std::string{what} + " is resident on CUDA device " + std::to_string(on) +
            "; parallel.device_id=" + std::to_string(device_id) +
            " would train on another device. Train with device_id=" +
            std::to_string(on) + " or move the array to device " +
            std::to_string(device_id) + ".");
    }
}

struct MatrixArg
{
    std::optional<array_2d> host;
    std::optional<cuda_2d>  device;
    bonsai::DeviceMatrix    dev;
    size_t                  n_rows     = 0;
    size_t                  n_features = 0;

    bool on_device() const
    {
        return dev.data != nullptr;
    }
    bonsai::features_view view() const
    {
        return as_view(*host);
    }
};

MatrixArg resolve_matrix(nb::handle X, uint32_t device_id)
{
    MatrixArg out;
    if (cuda_2d device; nb::try_cast(X, device))
    {
        place_device_array(device, device_id, "X");
        out.dev        = {.data    = device.data(),
                          .n_rows  = device.shape(0),
                          .n_feats = device.shape(1)};
        out.n_rows     = device.shape(0);
        out.n_features = device.shape(1);
        out.device     = std::move(device);
        return out;
    }
    array_2d host;
    if (!nb::try_cast(X, host))
    {
        throw std::invalid_argument(
            "X must be a row-major float32 numpy array, or a CUDA array supporting "
            "DLPack (cupy, torch, jax)");
    }
    out.host       = host;
    out.n_rows     = host.shape(0);
    out.n_features = host.shape(1);
    return out;
}

struct VectorArg
{
    std::optional<array_1d> host;
    std::vector<float>      owned;

    bonsai::floats_view view() const
    {
        return host ? bonsai::floats_view{host->data(), host->shape(0)}
                    : bonsai::floats_view{owned};
    }
    size_t size() const
    {
        return host ? host->shape(0) : owned.size();
    }
};

VectorArg resolve_vector(nb::handle v, uint32_t device_id, char const *what)
{
    if (cuda_1d device; nb::try_cast(v, device))
    {
        place_device_array(device, device_id, what);
        VectorArg out;
        out.owned.resize(device.shape(0));
        bonsai::cuda_download(device.data(), device.shape(0), out.owned.data());
        return out;
    }
    array_1d host;
    if (!nb::try_cast(v, host))
    {
        throw std::invalid_argument(std::string{what} +
                                    " must be a float32 numpy array, or a CUDA array "
                                    "supporting DLPack (cupy, torch, jax)");
    }
    VectorArg out;
    out.host = host;
    return out;
}

struct Inputs
{
    MatrixArg                xarg;
    VectorArg                yarg;
    std::optional<VectorArg> warg;
};

Inputs resolve_inputs(nb::handle X, nb::handle y, nb::handle weight, uint32_t device_id,
                      char const *weight_name, std::string_view where)
{
    Inputs in{.xarg = resolve_matrix(X, device_id),
              .yarg = resolve_vector(y, device_id, "y"),
              .warg = std::nullopt};
    if (!weight.is_none())
    {
        in.warg = resolve_vector(weight, device_id, weight_name);
    }
    if (in.yarg.size() != in.xarg.n_rows)
    {
        throw std::invalid_argument(std::string{where} +
                                    "len(y) must equal the row count");
    }
    if (in.warg && in.warg->size() != in.xarg.n_rows)
    {
        throw std::invalid_argument(std::string{where} + "len(" + weight_name +
                                    ") must equal the row count");
    }
    return in;
}

void check_feature_names(std::vector<std::string> const &names, size_t n_features)
{
    if (names.size() != n_features)
    {
        throw std::invalid_argument("feature_names has " +
                                    std::to_string(names.size()) +
                                    " entries and X has " + std::to_string(n_features) +
                                    " columns; one name per column");
    }
    std::unordered_set<std::string_view> seen;
    for (auto const &name : names)
    {
        if (!seen.insert(name).second)
        {
            throw std::invalid_argument("feature_names must be unique; '" + name +
                                        "' appears more than once");
        }
    }
}

std::vector<std::string>
resolve_feature_names(size_t                                         n_features,
                      std::optional<std::vector<std::string>> const &supplied)
{
    if (supplied)
    {
        check_feature_names(*supplied, n_features);
        return *supplied;
    }
    std::vector<std::string> names;
    names.reserve(n_features);
    for (size_t c = 0; c < n_features; ++c)
    {
        names.push_back("f" + std::to_string(c));
    }
    return names;
}

bonsai::BinMappers fit_mappers(MatrixArg const &X, std::vector<std::string> names,
                               bonsai::Config const   &cfg,
                               bonsai::BinEdges const &edges = {})
{
    if (!X.on_device())
    {
        return bonsai::BinMappers::fit(X.view(), std::move(names), cfg.bin_mapper,
                                       edges);
    }
    auto const         rows = bonsai::bin_sample_rows(X.n_rows, cfg.bin_mapper);
    size_t const       k    = rows.empty() ? X.n_rows : rows.size();
    std::vector<float> sample(k * X.n_features);
    bonsai::cuda_gather_rows(X.dev, rows, sample);
    return bonsai::BinMappers::fit(
        bonsai::features_view{sample.data(), k, X.n_features}, std::move(names),
        cfg.bin_mapper, edges);
}

bonsai::cli::LabeledData make_labeled(MatrixArg const &X, bonsai::floats_view y,
                                      bonsai::BinMappers const &mappers,
                                      bonsai::Config const &cfg, bool on_device,
                                      bonsai::floats_view weights = {})
{
    bonsai::cli::FeatureBuffer buf;
    buf.n_rows     = X.n_rows;
    buf.n_features = X.n_features;

    if (X.on_device())
    {
        return bonsai::cli::LabeledData{
            .dataset = bonsai::Dataset::bin(
                X.n_rows, X.n_features, y, mappers, cfg.data,
                bonsai::cuda_ingest_device(X.dev, mappers), weights),
            .features = std::move(buf),
            .labels   = std::vector<float>(y.begin(), y.end())};
    }
    buf.borrowed = std::span{X.host->data(), X.n_rows * X.n_features};

    if (on_device)
    {
        bonsai::cuda_select_device(cfg.parallel.device_id);
    }
    auto plane = on_device ? bonsai::cuda_ingest(X.view(), mappers) : nullptr;
    return bonsai::cli::LabeledData{
        .dataset  = bonsai::Dataset::bin(X.view(), y, mappers, cfg.data,
                                         std::move(plane), weights),
        .features = std::move(buf),
        .labels   = std::vector<float>(y.begin(), y.end())};
}

bonsai::cli::LabeledData make_validation_labeled(array_2d const &X, array_1d const &y)
{
    size_t const n = X.shape(0);

    bonsai::cli::FeatureBuffer buf;
    buf.n_rows     = n;
    buf.n_features = X.shape(1);
    buf.borrowed   = std::span{X.data(), n * X.shape(1)};
    return bonsai::cli::LabeledData{.dataset  = {},
                                    .features = std::move(buf),
                                    .labels =
                                        std::vector<float>(y.data(), y.data() + n)};
}

std::string view_shape_phrase(bonsai::RowView const &view)
{
    switch (view.form())
    {
    case bonsai::RowView::Form::Range:
        return "1 range";
    case bonsai::RowView::Form::Segments:
        return std::to_string(view.n_runs()) + " segments";
    case bonsai::RowView::Form::Gather:
        break;
    }
    return std::to_string(view.size()) + " gathered rows";
}

std::string two_decimals(double value)
{
    std::string const text = std::to_string(value);
    size_t const      dot  = text.find('.');
    return dot == std::string::npos ? text : text.substr(0, dot + 3);
}

struct IndexAxis
{
    std::string_view                            keyword;
    std::string_view                            one;
    std::string_view                            many;
    std::string_view                            empty_consequence;
    std::optional<std::span<std::string const>> names;
};

IndexAxis row_axis()
{
    return {.keyword           = "rows",
            .one               = "row",
            .many              = "rows",
            .empty_consequence = "an empty training set has no gradients to boost on",
            .names             = std::nullopt};
}

IndexAxis column_axis(std::span<std::string const> names)
{
    return {.keyword           = "columns",
            .one               = "feature",
            .many              = "features",
            .empty_consequence = "a dataset with no columns has nothing to split on",
            .names             = names};
}

nb::object name_positions(nb::object const &np, nb::object const &arr,
                          IndexAxis const &axis)
{
    std::vector<int64_t> ids;
    for (nb::handle const item :
         nb::cast<nb::sequence>(np.attr("atleast_1d")(arr).attr("tolist")()))
    {
        auto const want = nb::cast<std::string>(nb::str(item));
        auto const it   = std::ranges::find(*axis.names, want);
        if (it == axis.names->end())
        {
            throw nb::key_error(std::format("Dataset.subset({}=...) names the {} '{}', "
                                            "which this dataset does not have",
                                            axis.keyword, axis.one, want)
                                    .c_str());
        }
        ids.push_back(std::distance(axis.names->begin(), it));
    }
    return np.attr("asarray")(nb::cast(ids));
}

template <typename IdT>
std::vector<IdT> parse_index_selection(nb::handle selection, size_t n,
                                       IndexAxis const &axis)
{
    nb::object const np  = nb::module_::import_("numpy");
    nb::object       arr = nb::isinstance<nb::slice>(selection)
                               ? [&]
    {
        auto const [start, stop, step, len] = nb::cast<nb::slice>(selection).compute(n);
        return np.attr("arange")(start, stop, step);
    }()
                               : np.attr("asarray")(selection);
    if (nb::cast<size_t>(arr.attr("size")) == 0)
    {
        throw nb::value_error(std::format("Dataset.subset({}=...) selected no {}; {}",
                                          axis.keyword, axis.many,
                                          axis.empty_consequence)
                                  .c_str());
    }
    auto const kind = nb::cast<std::string>(arr.attr("dtype").attr("kind"));
    if (kind == "b")
    {
        if (size_t const given = nb::cast<size_t>(arr.attr("size")); given != n)
        {
            throw nb::value_error(
                std::format(
                    "Dataset.subset({}=<bool mask>): the mask has {} entries and "
                    "the dataset has {} {}; a mask names one {} per entry",
                    axis.keyword, given, n, axis.many, axis.one)
                    .c_str());
        }
        arr = np.attr("flatnonzero")(arr);
    }
    else if (axis.names && (kind == "U" || kind == "S" || kind == "O"))
    {
        arr = name_positions(np, arr, axis);
    }
    else if (kind != "i" && kind != "u")
    {
        throw nb::type_error(
            std::format("Dataset.subset({}=...) takes a slice, a boolean mask, {}an "
                        "integer array{}; got dtype kind '{}'",
                        axis.keyword, axis.names ? "" : "or ",
                        axis.names ? std::format(", or {} names", axis.one) : "", kind)
                .c_str());
    }
    arr = np.attr("ascontiguousarray")(arr, np.attr("int64"));
    if (nb::cast<size_t>(arr.attr("ndim")) != 1)
    {
        throw nb::value_error(
            std::format("Dataset.subset({}=...) takes one dimension of {} ids",
                        axis.keyword, axis.one)
                .c_str());
    }
    auto const ids = nb::cast<
        nb::ndarray<int64_t const, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(arr);
    std::vector<IdT> out;
    out.reserve(ids.shape(0));
    for (int64_t const id : std::span<int64_t const>{ids.data(), ids.shape(0)})
    {
        if (id < 0)
        {
            throw nb::index_error(
                std::format("Dataset.subset({}=...) got the negative index "
                            "{}; {} ids index the binned plane and do not wrap",
                            axis.keyword, id, axis.one)
                    .c_str());
        }
        if (static_cast<size_t>(id) >= n)
        {
            throw nb::index_error(
                std::format("Dataset.subset({}=...) got {} {}, out of range "
                            "for a dataset of {} {}",
                            axis.keyword, axis.one, id, n, axis.many)
                    .c_str());
        }
        out.push_back(static_cast<IdT>(id));
    }
    return out;
}

class Dataset
{
  public:
    Dataset(nb::handle X, nb::handle y, nb::handle weight, std::optional<int> max_bin,
            std::optional<size_t> n_samples, std::optional<uint64_t> seed,
            std::optional<int>                               min_data_in_bin,
            std::optional<std::map<size_t, array_1d>> const &bin_edges,
            std::optional<std::string> const &device, std::optional<uint32_t> device_id,
            uint32_t n_threads, Dataset const *reference,
            std::optional<std::vector<std::string>> const &feature_names)
    {
        std::optional<uint32_t> const ref_device =
            reference != nullptr ? reference->device_id_ : std::nullopt;
        uint32_t const dev_id =
            device_id.value_or(ref_device.value_or(bonsai::ParallelConfig{}.device_id));
        auto const [xarg, yarg, warg] =
            resolve_inputs(X, y, weight, dev_id, "weight", "Dataset: ");
        std::string const hint = device.value_or(
            xarg.on_device() || ref_device.has_value() ? "cuda" : "cpu");
        bool const on_device = hint == "cuda";
        if (!on_device && hint != "cpu")
        {
            throw std::invalid_argument(R"(Dataset: device must be "cpu" or "cuda")");
        }
        if (on_device && !bonsai::cuda_available())
        {
            throw std::invalid_argument(
                "Dataset(device=\"cuda\") needs a CUDA build and a visible device; "
                "cuda_available() is False");
        }
        if (xarg.on_device() && !on_device)
        {
            throw std::invalid_argument(
                "Dataset: X is device-resident (DLPack), so device=\"cpu\" would "
                "copy it back to the host. Drop the device argument to bin it "
                "where it already lives, or pass a host array.");
        }
        bonsai::BinMapperConfig const base =
            reference != nullptr ? reference->bin_cfg_ : bonsai::BinMapperConfig{};
        bonsai::Config cfg;
        cfg.bin_mapper.max_bin         = max_bin.value_or(base.max_bin);
        cfg.bin_mapper.n_samples       = n_samples.value_or(base.n_samples);
        cfg.bin_mapper.seed            = seed.value_or(base.seed);
        cfg.bin_mapper.min_data_in_bin = min_data_in_bin.value_or(base.min_data_in_bin);
        cfg.parallel.n_threads         = n_threads;
        cfg.parallel.device_id         = dev_id;
        if (reference != nullptr)
        {
            check_reference(*reference, cfg.bin_mapper, bin_edges.has_value(),
                            feature_names.has_value(), xarg.n_features);
        }

        std::vector<std::string> names =
            resolve_feature_names(xarg.n_features, feature_names);
        bonsai::BinEdges edges;
        if (bin_edges)
        {
            edges.reserve(bin_edges->size());
            for (auto const &[col, arr] : *bin_edges)
            {
                edges.emplace_back(
                    col, std::vector<float>(arr.data(), arr.data() + arr.shape(0)));
            }
        }
        bonsai::floats_view const w = warg ? warg->view() : bonsai::floats_view{};
        nb::gil_scoped_release    release;
        bonsai::parallel::set_n_threads(cfg.parallel.n_threads);
        bin_cfg_         = cfg.bin_mapper;
        loaded_->mappers = reference != nullptr
                               ? reference->loaded_->mappers
                               : fit_mappers(xarg, std::move(names), cfg, edges);
        loaded_->train =
            make_labeled(xarg, yarg.view(), loaded_->mappers, cfg, on_device, w);
        if (loaded_->train.dataset.ingest_plane())
        {
            device_id_ = dev_id;
        }
        x_          = xarg.host;
        n_features_ = xarg.n_features;
    }

    Dataset(Dataset const &parent, bonsai::RowView rows, nb::object base)
        : x_(parent.x_), n_features_(parent.n_features_), loaded_(parent.loaded_),
          view_train_(
              std::make_shared<bonsai::cli::LabeledData>(bonsai::cli::LabeledData{
                  .dataset  = parent.bins().with_rows(std::move(rows)),
                  .features = parent.train_data().features,
                  .labels   = parent.train_data().labels})),
          base_(std::move(base)), device_id_(parent.device_id_),
          bin_cfg_(parent.bin_cfg_)
    {
    }

    Dataset(bonsai::Dataset gathered, bonsai::BinMapperConfig cfg,
            std::optional<uint32_t> device_id)
        : n_features_(gathered.n_features()),
          device_id_(gathered.ingest_plane() != nullptr ? device_id : std::nullopt),
          bin_cfg_(cfg)
    {
        loaded_->mappers = gathered.mappers();
        loaded_->train   = bonsai::cli::LabeledData{
              .dataset  = std::move(gathered),
              .features = {},
              .labels   = {},
        };
        loaded_->train.labels.assign(loaded_->train.dataset.labels().begin(),
                                     loaded_->train.dataset.labels().end());
    }

    Dataset subset(nb::handle self, nb::handle rows, nb::handle columns) const
    {
        if (rows.is_none() && columns.is_none())
        {
            throw std::invalid_argument(
                "Dataset.subset() needs rows= or columns=: a numpy integer array, a "
                "slice, a boolean mask, or (for columns) feature names");
        }
        if (!columns.is_none())
        {
            Dataset const narrowed =
                rows.is_none() ? *this : subset(self, rows, nb::none());
            auto const names = narrowed.loaded_->mappers.feature_names();
            return {narrowed.bins().select_features(
                        parse_index_selection<bonsai::feature_id_t>(
                            columns, names.size(), column_axis(names))),
                    narrowed.bin_cfg_, narrowed.device_id_};
        }
        std::vector<bonsai::row_id_t> ids =
            parse_index_selection<bonsai::row_id_t>(rows, n_rows(), row_axis());
        if (is_view())
        {
            std::vector<bonsai::row_id_t> const mine = bins().row_view().materialize();
            for (bonsai::row_id_t &id : ids)
            {
                id = mine[id];
            }
        }
        nb::object base = is_view() ? base_ : nb::borrow(self);
        return {root(), bonsai::RowView::encode(ids, bins().plane_n_rows()),
                std::move(base)};
    }

    Dataset reorder(nb::handle self, nb::handle rows) const
    {
        if (rows.is_none())
        {
            throw std::invalid_argument(
                "Dataset.reorder() needs rows=: a permutation of this Dataset's "
                "rows, as an integer array or a slice");
        }
        std::vector<bonsai::row_id_t> const ids =
            parse_index_selection<bonsai::row_id_t>(rows, n_rows(), row_axis());
        std::vector<bool> seen(n_rows(), false);
        bool const        whole = ids.size() == n_rows();
        for (bonsai::row_id_t const id : ids)
        {
            if (!whole || seen[id])
            {
                std::string const msg =
                    "Dataset.reorder(rows=...) takes a permutation of this "
                    "Dataset's " +
                    std::to_string(n_rows()) +
                    " rows: every row exactly once. Use subset(rows=...) "
                    "to keep only some of them.";
                throw nb::value_error(msg.c_str());
            }
            seen[id] = true;
        }
        Dataset const laid = subset(self, rows, nb::none());
        return {laid.bins().materialize(), bin_cfg_, laid.device_id_};
    }

    bool is_view() const
    {
        return view_train_ != nullptr;
    }

    nb::object base() const
    {
        return base_;
    }

    std::string device() const
    {
        return device_id_ ? "cuda" : "cpu";
    }

    std::optional<uint32_t> device_id() const
    {
        return device_id_;
    }

    size_t n_rows() const
    {
        return bins().view_n_rows();
    }
    size_t n_features() const
    {
        return n_features_;
    }

    std::vector<std::string> feature_names() const
    {
        auto const names = loaded_->mappers.feature_names();
        return {names.begin(), names.end()};
    }

    bool has_host_matrix() const
    {
        return x_.has_value();
    }

    bonsai::cli::LoadedTrainValidation const &loaded() const
    {
        return *loaded_;
    }

    bonsai::cli::LabeledData const &train_data() const
    {
        return view_train_ ? *view_train_ : loaded_->train;
    }

    bonsai::Dataset const &bins() const
    {
        return train_data().dataset;
    }

    Dataset const &root() const
    {
        return is_view() ? nb::cast<Dataset const &>(base_) : *this;
    }

    [[noreturn]] void refuse_view(char const *method, char const *because) const
    {
        throw std::invalid_argument(
            std::string{"this Dataset is a row view (its .base names the plane it "
                        "selects from), and "} +
            method + " " + because +
            ". Materialize the rows with Dataset(X[idx], y[idx], reference=parent).");
    }

    bonsai::RowView const &row_view() const
    {
        return bins().row_view();
    }

    array_2d const &host_matrix(char const *method) const
    {
        if (!x_)
        {
            throw std::invalid_argument(
                std::string{"this Dataset was built from device-resident (DLPack) "
                            "input and kept no host matrix, but "} +
                method +
                " reads raw rows. Pass X as a host array, or build the Dataset "
                "from one.");
        }
        return *x_;
    }

  private:
    static void check_reference(Dataset const                 &reference,
                                bonsai::BinMapperConfig const &bin_cfg,
                                bool has_bin_edges, bool has_feature_names,
                                size_t n_features)
    {
        if (bin_cfg != reference.bin_cfg_)
        {
            throw std::invalid_argument(
                "Dataset(reference=...) bins with the reference's cut points, so "
                "max_bin/n_samples/seed/min_data_in_bin cannot differ from the "
                "reference's; leave them out to inherit, or set them there");
        }
        if (has_bin_edges)
        {
            throw std::invalid_argument(
                "Dataset(reference=...) bins with the reference's cut points, so "
                "bin_edges belongs on the reference instead");
        }
        if (has_feature_names)
        {
            throw std::invalid_argument(
                "feature_names cannot be given with reference=: the reference "
                "already names these columns, and its mappers carry the names "
                "through. Name the columns on the reference instead.");
        }
        if (n_features != reference.n_features_)
        {
            throw std::invalid_argument(
                "Dataset(reference=...): X has " + std::to_string(n_features) +
                " columns and the reference has " +
                std::to_string(reference.n_features_) +
                "; one set of cuts describes one set of columns");
        }
    }

    std::optional<array_2d>                             x_;
    size_t                                              n_features_ = 0;
    std::shared_ptr<bonsai::cli::LoadedTrainValidation> loaded_ =
        std::make_shared<bonsai::cli::LoadedTrainValidation>();
    std::shared_ptr<bonsai::cli::LabeledData> view_train_;
    nb::object                                base_ = nb::none();
    std::optional<uint32_t>                   device_id_;
    bonsai::BinMapperConfig                   bin_cfg_;
};

// sync: readers are const and may run concurrently because the bindings
// release the gil, so the cache takes a lock. A refused pack caches too, so a
// host with no device pays one refusal per epoch rather than one per call.
class DevicePlanCache
{
  public:
    std::shared_ptr<bonsai::CudaPredictPlan const>
    predict(bonsai::DevicePlanInput const &in, bonsai::BinMappers const &mappers) const
    {
        std::scoped_lock const lock(mutex_);
        return predict_.get(in.epoch,
                            [&]
                            {
                                return bonsai::cuda_predict_plan(
                                    in.trees, mappers, in.learning_rate, in.init_score);
                            });
    }

    std::shared_ptr<bonsai::CudaShapPlan const>
    shap(bonsai::DevicePlanInput const &in, bonsai::BinMappers const &mappers) const
    {
        std::scoped_lock const lock(mutex_);
        return shap_.get(in.epoch,
                         [&]
                         {
                             return bonsai::cuda_shap_plan(
                                 in.trees, mappers, in.learning_rate, in.init_score);
                         });
    }

  private:
    template <typename Plan> struct Slot
    {
        std::shared_ptr<Plan const> plan;
        uint64_t                    epoch = 0;

        template <typename Pack> std::shared_ptr<Plan const> get(uint64_t at, Pack pack)
        {
            if (epoch != at)
            {
                plan  = pack();
                epoch = at;
            }
            return plan;
        }
    };

    mutable std::mutex                    mutex_;
    mutable Slot<bonsai::CudaPredictPlan> predict_;
    mutable Slot<bonsai::CudaShapPlan>    shap_;
};

class Model
{
  public:
    Model(std::unique_ptr<bonsai::IBooster> booster, bonsai::BinMappers mappers,
          bonsai::Config cfg, std::vector<float> eval_history = {})
        : booster_(std::move(booster)), mappers_(std::move(mappers)),
          cfg_(std::move(cfg)), eval_history_(std::move(eval_history))
    {
    }

    std::vector<float> const &eval_history() const
    {
        return eval_history_;
    }

    void check_width(array_2d const &X, char const *method) const
    {
        bonsai::require_n_features(X.shape(1), mappers_.size(),
                                   std::format("the matrix passed to {}", method));
    }

    nb::ndarray<nb::numpy, float> predict(array_2d const &X,
                                          size_t          num_iteration = 0) const
    {
        check_width(X, "predict");
        size_t const n   = X.shape(0);
        auto         out = std::make_unique<std::vector<float>>(n, 0.0F);
        {
            nb::gil_scoped_release release;
            booster_->predict_at(bonsai::features_view{X.data(), n, X.shape(1)}, *out,
                                 num_iteration);
            bonsai::apply_link_inverse_by_name(cfg_.dispatch.objective_name, *out);
        }
        return to_numpy(std::move(out), {n});
    }

    bool routes_binned(Dataset const &ds, char const *method) const
    {
        if (mappers_.same_cuts(ds.loaded().mappers))
        {
            return true;
        }
        if (ds.is_view())
        {
            ds.refuse_view(method,
                           "was binned with different cut points than this model's, "
                           "so the only route left reads the raw rows its parent "
                           "retained, which are the parent's rows");
        }
        if (!ds.has_host_matrix())
        {
            throw std::invalid_argument(
                std::string{"this Dataset was binned with different cut points "
                            "than this model's and was built from device-resident "
                            "(DLPack) input, so "} +
                method +
                " has neither bins it can route nor raw rows to read. Build the "
                "Dataset with reference= the training dataset, or pass X as a "
                "host array.");
        }
        return false;
    }

    nb::ndarray<nb::numpy, float> predict(Dataset const &ds,
                                          size_t         num_iteration = 0) const
    {
        if (!routes_binned(ds, "predict"))
        {
            return predict(ds.host_matrix("predict"), num_iteration);
        }
        size_t const n   = ds.n_rows();
        auto         out = std::make_unique<std::vector<float>>(n, 0.0F);
        {
            nb::gil_scoped_release release;
            if (!predict_on_device(ds, *out, num_iteration))
            {
                booster_->predict_at_binned(ds.bins(), *out, num_iteration);
            }
            bonsai::apply_link_inverse_by_name(cfg_.dispatch.objective_name, *out);
        }
        return to_numpy(std::move(out), {n});
    }

    nb::ndarray<nb::numpy, double> predict_proba(array_2d const &X) const
    {
        check_width(X, "predict_proba");
        require_proba_objective();
        size_t const n   = X.shape(0);
        size_t const w   = booster_->score_width();
        auto         out = std::make_unique<std::vector<double>>(n * w, 0.0);
        {
            nb::gil_scoped_release release;
            if (w > 1)
            {
                booster_->predict_proba(bonsai::features_view{X.data(), n, X.shape(1)},
                                        std::span<double>{*out});
            }
            else
            {
                std::vector<float> margins(n, 0.0F);
                booster_->predict_at(bonsai::features_view{X.data(), n, X.shape(1)},
                                     bonsai::floats_out{margins.data(), n}, 0);
                bonsai::apply_link_inverse_by_name(
                    cfg_.dispatch.objective_name,
                    bonsai::floats_out{margins.data(), n});
                for (size_t i = 0; i < n; ++i)
                {
                    (*out)[i] = margins[i];
                }
            }
        }
        if (w > 1)
        {
            return to_numpy(std::move(out), {n, w});
        }
        return to_numpy(std::move(out), {n});
    }

    nb::ndarray<nb::numpy, double> predict_proba(Dataset const &ds) const
    {
        require_proba_objective();
        if (!routes_binned(ds, "predict_proba"))
        {
            return predict_proba(ds.host_matrix("predict_proba"));
        }
        size_t const n   = ds.n_rows();
        size_t const w   = booster_->score_width();
        auto         out = std::make_unique<std::vector<double>>(n * w, 0.0);
        {
            nb::gil_scoped_release release;
            if (w > 1)
            {
                booster_->predict_proba_binned(ds.bins(), std::span<double>{*out});
            }
            else
            {
                std::vector<float> margins(n, 0.0F);
                booster_->predict_at_binned(ds.bins(),
                                            bonsai::floats_out{margins.data(), n}, 0);
                bonsai::apply_link_inverse_by_name(
                    cfg_.dispatch.objective_name,
                    bonsai::floats_out{margins.data(), n});
                for (size_t i = 0; i < n; ++i)
                {
                    (*out)[i] = margins[i];
                }
            }
        }
        if (w > 1)
        {
            return to_numpy(std::move(out), {n, w});
        }
        return to_numpy(std::move(out), {n});
    }

    nb::ndarray<nb::numpy, float> staged_predict(array_2d const &X) const
    {
        check_width(X, "staged_predict");
        size_t const n   = X.shape(0);
        size_t const k   = booster_->n_iters();
        auto         out = std::make_unique<std::vector<float>>(k * n, 0.0F);
        {
            nb::gil_scoped_release release;
            booster_->predict_staged(bonsai::features_view{X.data(), n, X.shape(1)},
                                     *out);
            for (size_t t = 0; t < k; ++t)
            {
                bonsai::apply_link_inverse_by_name(
                    cfg_.dispatch.objective_name,
                    bonsai::floats_out{std::span{*out}.subspan(t * n, n)});
            }
        }
        return to_numpy(std::move(out), {k, n});
    }

    nb::ndarray<nb::numpy, float> staged_predict(Dataset const &ds) const
    {
        if (!routes_binned(ds, "staged_predict"))
        {
            return staged_predict(ds.host_matrix("staged_predict"));
        }
        size_t const n   = ds.n_rows();
        size_t const k   = booster_->n_iters();
        auto         out = std::make_unique<std::vector<float>>(k * n, 0.0F);
        {
            nb::gil_scoped_release release;
            booster_->predict_staged_binned(ds.bins(), *out);
            for (size_t t = 0; t < k; ++t)
            {
                bonsai::apply_link_inverse_by_name(
                    cfg_.dispatch.objective_name,
                    bonsai::floats_out{std::span{*out}.subspan(t * n, n)});
            }
        }
        return to_numpy(std::move(out), {k, n});
    }

    nb::ndarray<nb::numpy, uint32_t> predict_leaf(array_2d const &X) const
    {
        check_width(X, "predict_leaf");
        size_t const n   = X.shape(0);
        size_t const k   = booster_->n_trees();
        auto         out = std::make_unique<std::vector<bonsai::node_id_t>>(n * k, 0);
        {
            nb::gil_scoped_release release;
            booster_->predict_leaf(bonsai::features_view{X.data(), n, X.shape(1)},
                                   std::span<bonsai::node_id_t>{*out});
        }
        return to_numpy(std::move(out), {n, k});
    }

    nb::ndarray<nb::numpy, uint32_t> predict_leaf(Dataset const &ds) const
    {
        if (!routes_binned(ds, "predict_leaf"))
        {
            return predict_leaf(ds.host_matrix("predict_leaf"));
        }
        size_t const n   = ds.n_rows();
        size_t const k   = booster_->n_trees();
        auto         out = std::make_unique<std::vector<bonsai::node_id_t>>(n * k, 0);
        {
            nb::gil_scoped_release release;
            booster_->predict_leaf_binned(ds.bins(),
                                          std::span<bonsai::node_id_t>{*out});
        }
        return to_numpy(std::move(out), {n, k});
    }

    std::string dump() const
    {
        return booster_->dump(mappers_.feature_names());
    }

    nb::ndarray<nb::numpy, double> pred_contribs(array_2d const &X) const
    {
        check_width(X, "pred_contribs");
        size_t const n     = X.shape(0);
        size_t const nf    = X.shape(1);
        size_t const cols  = nf + 1;
        size_t const width = booster_->score_width();
        auto         out = std::make_unique<std::vector<double>>(n * width * cols, 0.0);
        {
            nb::gil_scoped_release release;
            booster_->pred_contribs(bonsai::features_view{X.data(), n, nf},
                                    std::span<double>{*out}, nf);
        }
        if (width > 1)
        {
            return to_numpy(std::move(out), {n, width, cols});
        }
        return to_numpy(std::move(out), {n, cols});
    }

    nb::ndarray<nb::numpy, double> pred_contribs(Dataset const &ds) const
    {
        if (!routes_binned(ds, "pred_contribs"))
        {
            return pred_contribs(ds.host_matrix("pred_contribs"));
        }
        size_t const n     = ds.n_rows();
        size_t const nf    = ds.n_features();
        size_t const cols  = nf + 1;
        size_t const width = booster_->score_width();
        auto         out = std::make_unique<std::vector<double>>(n * width * cols, 0.0);
        {
            nb::gil_scoped_release release;
            if (width > 1 || !contribs_on_device(ds, std::span<double>{*out}))
            {
                booster_->pred_contribs_binned(ds.bins(), std::span<double>{*out}, nf);
            }
        }
        if (width > 1)
        {
            return to_numpy(std::move(out), {n, width, cols});
        }
        return to_numpy(std::move(out), {n, cols});
    }

    void save(std::string const &path) const
    {
        bonsai::io::save_booster(*booster_, path, mappers_, cfg_);
    }

    std::vector<uint8_t> save_bytes() const
    {
        return bonsai::io::save_booster_bytes(*booster_, mappers_, cfg_);
    }

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
        auto out =
            std::make_unique<std::vector<double>>(booster_->feature_importance(t));
        out->resize(std::max(out->size(), mappers_.size()), 0.0);
        size_t const n = out->size();
        return to_numpy(std::move(out), {n});
    }

    size_t n_iters() const
    {
        return booster_->n_iters();
    }

    std::string config_toml() const
    {
        return bonsai::config::dump_toml(cfg_);
    }

    std::string objective_name() const
    {
        return cfg_.dispatch.objective_name;
    }

    size_t n_classes() const
    {
        return cfg_.dispatch.objective_name == "softmax" ? cfg_.objective.n_classes : 0;
    }

    std::vector<std::string> feature_names() const
    {
        auto const names = mappers_.feature_names();
        return {names.begin(), names.end()};
    }

  private:
    void require_proba_objective() const
    {
        if (booster_->score_width() == 1 && cfg_.dispatch.objective_name != "logloss")
        {
            throw std::invalid_argument(
                "predict_proba is only defined for classification objectives "
                "(logloss/softmax); this model was trained with '" +
                cfg_.dispatch.objective_name + "'");
        }
    }

    bool predict_on_device(Dataset const &ds, std::vector<float> &out,
                           size_t num_iteration) const
    {
        auto const &bins  = ds.bins();
        auto const  plane = bins.ingest_plane();
        if (!plane || !bins.row_view().is_identity())
        {
            return false;
        }
        auto const in = booster_->device_plan_input();
        if (in.trees.empty())
        {
            return false;
        }
        auto const plan = plan_cache_->predict(in, mappers_);
        return plan && bonsai::cuda_predict(*plan, *plane, bins.plane_n_rows(),
                                            bins.n_features(), num_iteration, out);
    }

    bool contribs_on_device(Dataset const &ds, std::span<double> out) const
    {
        auto const &bins  = ds.bins();
        auto const  plane = bins.ingest_plane();
        if (!plane || !bins.row_view().is_identity())
        {
            return false;
        }
        auto const in = booster_->device_plan_input();
        if (in.trees.empty())
        {
            return false;
        }
        auto const plan = plan_cache_->shap(in, mappers_);
        return plan && bonsai::cuda_pred_contribs(*plan, *plane, bins.plane_n_rows(),
                                                  bins.n_features(), out);
    }

    std::unique_ptr<bonsai::IBooster> booster_;
    bonsai::BinMappers                mappers_;
    bonsai::Config                    cfg_;
    std::vector<float>                eval_history_;
    std::shared_ptr<DevicePlanCache>  plan_cache_ = std::make_shared<DevicePlanCache>();
};

using EvalSet = std::variant<std::pair<array_2d, array_1d>, Dataset const *>;

void check_eval_labels(std::vector<float> const &labels, bonsai::Config const &cfg)
{
    auto const kind = bonsai::task_kind_by_name(cfg.dispatch.objective_name);
    if (kind == bonsai::TaskKind::regression)
    {
        return;
    }
    auto const n_classes =
        kind == bonsai::TaskKind::binary_classification ? 2U : cfg.objective.n_classes;
    float const top = static_cast<float>(n_classes - 1);
    for (float const label : labels)
    {
        if (label >= 0.0F && label <= top && label == std::floor(label))
        {
            continue;
        }
        throw std::invalid_argument(
            "eval_set label " + std::to_string(label) +
            " is not one of the encoded class ids 0.." + std::to_string(n_classes - 1) +
            " this objective scores against. A Dataset is built before fit has "
            "seen the classes, so encode its labels first "
            "(np.searchsorted(clf.classes_, y_valid)) or pass "
            "eval_set=(X_valid, y_valid).");
    }
}

bonsai::cli::LabeledData const *
resolve_eval_set(std::optional<EvalSet> const &eval_set, bonsai::Config const &cfg,
                 bonsai::BinMappers const &mappers, bool warm_start,
                 std::optional<bonsai::cli::LabeledData> &owned)
{
    if (!eval_set)
    {
        return nullptr;
    }
    if (auto const *const arrays =
            std::get_if<std::pair<array_2d, array_1d>>(&*eval_set))
    {
        bonsai::require_n_features(arrays->first.shape(1), mappers.size(),
                                   "the matrix passed as eval_set");
        owned = make_validation_labeled(arrays->first, arrays->second);
        check_eval_labels(owned->labels, cfg);
        return &*owned;
    }
    auto const *const dataset = std::get<Dataset const *>(*eval_set);
    if (!mappers.same_cuts(dataset->loaded().mappers))
    {
        throw std::invalid_argument(
            "the eval_set Dataset was binned with different cut points than this "
            "fit's training data, so its bins name different splits. Build it "
            "against the training data: bonsai.Dataset(X_valid, y_valid, "
            "reference=train_dataset).");
    }
    if (!dataset->bins().weights().empty())
    {
        throw std::invalid_argument(
            "the eval_set Dataset carries sample weights, which the validation "
            "loss does not apply: it is the unweighted metric. Drop the weight "
            "argument from the eval-set Dataset rather than have it ignored.");
    }
    if (!dataset->has_host_matrix() && (warm_start || !dataset->bins().bins_are_u8()))
    {
        throw std::invalid_argument(
            "this eval_set Dataset was built from device-resident (DLPack) input "
            "and kept no host matrix, but this fit scores it from raw rows (an "
            "init_model warm start, or bins above 255 that have no row-major "
            "mirror). Pass eval_set=(X_valid, y_valid), or build the Dataset "
            "from a host array.");
    }
    if (!dataset->is_view())
    {
        check_eval_labels(dataset->train_data().labels, cfg);
        return &dataset->train_data();
    }
    if (warm_start)
    {
        dataset->refuse_view("eval_set", "seeds an init_model warm start from raw "
                                         "rows, which are its parent's");
    }
    if (!dataset->bins().bins_are_u8())
    {
        dataset->refuse_view("eval_set",
                             "carries more than 255 bins, which have no row-major "
                             "mirror for the binned walk to route, leaving only the "
                             "raw rows its parent retained");
    }
    check_eval_labels(dataset->row_view().gather(dataset->train_data().labels), cfg);
    return &dataset->train_data();
}

using ConfigPairs = std::vector<std::pair<std::string, std::string>>;

std::string config_str(nb::handle value)
{
    if (nb::isinstance<nb::bool_>(value))
    {
        return nb::cast<bool>(value) ? "true" : "false";
    }
    if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value))
    {
        std::string joined;
        bool        first = true;
        for (nb::handle item : nb::borrow<nb::object>(value))
        {
            if (!first)
            {
                joined += ',';
            }
            joined += nb::cast<std::string>(nb::str(item));
            first = false;
        }
        return joined;
    }
    return nb::cast<std::string>(nb::str(value));
}

using ParamItems = std::vector<std::pair<std::string, nb::object>>;

ParamItems items_from_params(nb::object params)
{
    if (params.is_none())
    {
        return {};
    }
    if (nb::hasattr(params, "to_dict"))
    {
        if (nb::object const to_dict = params.attr("to_dict");
            PyCallable_Check(to_dict.ptr()) != 0)
        {
            params = to_dict();
        }
    }
    if (!nb::hasattr(params, "items"))
    {
        std::string const msg =
            "params must be a bonsai.Params, a mapping of dotted keys, or None; "
            "got " +
            nb::cast<std::string>(nb::str(params.type().attr("__name__"))) +
            ". For legacy (key, value) pairs, pass dict(pairs).";
        throw nb::type_error(msg.c_str());
    }
    ParamItems       out;
    nb::object const items = params.attr("items")();
    for (nb::handle item : items)
    {
        nb::object const entry = nb::borrow<nb::object>(item);
        nb::object const key   = entry[0];
        if (!nb::isinstance<nb::str>(key))
        {
            std::string const msg =
                "params keys must be dotted config keys (str); got " +
                nb::cast<std::string>(nb::str(key.type().attr("__name__")));
            throw nb::type_error(msg.c_str());
        }
        out.emplace_back(nb::cast<std::string>(key), nb::object(entry[1]));
    }
    return out;
}

std::vector<std::string> stated_keys(ParamItems const &items)
{
    std::vector<std::string> keys;
    for (auto const &[key, value] : items)
    {
        if (key.starts_with("dispatch.") || key.starts_with("objective."))
        {
            keys.push_back(key);
        }
    }
    return keys;
}

ConfigPairs render_params(ParamItems const &items)
{
    ConfigPairs pairs;
    pairs.reserve(items.size());
    for (auto const &[key, value] : items)
    {
        pairs.emplace_back(key, config_str(value));
    }
    return pairs;
}

constexpr std::string_view k_monotone_key = "tree.monotone_constraints";

nb::object take_named_monotone(ParamItems &items)
{
    for (auto it = items.begin(); it != items.end(); ++it)
    {
        if (it->first != k_monotone_key || !nb::hasattr(it->second, "items"))
        {
            continue;
        }
        nb::object const mapping = it->second;
        items.erase(it);
        return mapping;
    }
    return nb::none();
}

std::vector<int> monotone_from_mapping(nb::handle                   mapping,
                                       std::span<std::string const> names)
{
    std::vector<int>         out(names.size(), 0);
    std::vector<std::string> unknown;
    for (nb::handle item : mapping.attr("items")())
    {
        nb::object const entry = nb::borrow<nb::object>(item);
        auto const       name  = nb::cast<std::string>(nb::str(entry[0]));
        auto const       at    = std::ranges::find(names, name);
        if (at == names.end())
        {
            unknown.push_back(name);
            continue;
        }
        nb::object const value = entry[1];
        if (!nb::isinstance<nb::int_>(value) || std::abs(nb::cast<int>(value)) > 1)
        {
            throw std::invalid_argument(std::string{k_monotone_key} + "['" + name +
                                        "'] must be the int -1, 0, or 1; got " +
                                        nb::cast<std::string>(nb::repr(value)));
        }
        out[static_cast<size_t>(at - names.begin())] = nb::cast<int>(value);
    }
    if (unknown.empty())
    {
        return out;
    }
    std::string listed;
    for (auto const &name : unknown)
    {
        listed += (listed.empty() ? "'" : ", '") + name + "'";
    }
    throw std::invalid_argument(
        std::string{k_monotone_key} +
        " names features the training data does not have: " + listed + ". It carries " +
        std::to_string(names.size()) + " feature names.");
}

bool put_monotone(ParamItems &items, nb::handle named,
                  std::span<std::string const> names)
{
    if (named.is_none())
    {
        return false;
    }
    items.emplace_back(std::string{k_monotone_key},
                       nb::cast(monotone_from_mapping(named, names)));
    return true;
}

bonsai::Config config_from_params(ConfigPairs const &params)
{
    std::vector<bonsai::config::Override> overrides;
    overrides.reserve(params.size());
    for (auto const &[key, value] : params)
    {
        overrides.push_back({.key = key, .value = value});
    }
    return bonsai::config::resolve("", overrides);
}

template <typename T> char const *py_type_token()
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return "bool";
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        return "str";
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return "float";
    }
    else if constexpr (std::is_same_v<T, std::vector<std::string>>)
    {
        return "list[str]";
    }
    else if constexpr (std::is_same_v<T, std::vector<int>>)
    {
        return "list[int]";
    }
    else
    {
        static_assert(std::is_integral_v<T>, "unmapped config field type");
        return "int";
    }
}

template <typename Sub, typename T>
void append_field(nb::list &fields, Sub const &sub,
                  bonsai::config::internal::Field<Sub, T> const &f)
{
    nb::dict d;
    d["name"]    = std::string{f.leaf};
    d["type"]    = py_type_token<T>();
    d["default"] = nb::cast(sub.*(f.member));
    fields.append(d);
}

template <typename Sec>
void append_section(nb::list &sections, bonsai::Config const &defaults, Sec const &sec)
{
    nb::list fields;
    std::apply([&](auto const &...fs)
               { (..., append_field(fields, defaults.*(sec.sub), fs)); }, sec.fields);
    nb::dict s;
    s["section"] = std::string{sec.name};
    s["fields"]  = fields;
    sections.append(s);
}

nb::list params_schema()
{
    bonsai::Config const defaults{};
    nb::list             sections;
    std::apply([&](auto const &...secs)
               { (..., append_section(sections, defaults, secs)); },
               bonsai::config::internal::all_sections);
    return sections;
}

Model train(nb::object const &params, nb::handle X, nb::handle y,
            std::optional<EvalSet> const     &eval_set,
            std::optional<std::string> const &init_model, nb::handle sample_weight,
            std::optional<std::vector<std::string>> const &feature_names)
{
    if (feature_names && init_model)
    {
        throw std::invalid_argument(
            "feature_names cannot be given with init_model: a warm start keeps "
            "the loaded model's feature names");
    }
    ParamItems       items          = items_from_params(params);
    nb::object const named_monotone = take_named_monotone(items);
    bonsai::Config   cfg            = config_from_params(render_params(items));
    bonsai::parallel::set_n_threads(cfg.parallel.n_threads);

    auto const [xarg, yarg, warg] = resolve_inputs(
        X, y, sample_weight, cfg.parallel.device_id, "sample_weight", "");

    std::optional<bonsai::io::LoadedBooster> init;
    if (init_model)
    {
        init.emplace(bonsai::io::load_booster(*init_model));
    }

    std::vector<std::string> names =
        resolve_feature_names(xarg.n_features, feature_names);
    if (put_monotone(items, named_monotone,
                     init ? init->mappers.feature_names()
                          : std::span<std::string const>{names}))
    {
        cfg = config_from_params(render_params(items));
    }
    if (init)
    {
        cfg = bonsai::cli::reconcile_warm_start(std::move(cfg), init->cfg,
                                                stated_keys(items));
    }

    nb::gil_scoped_release release;

    bonsai::cli::LoadedTrainValidation loaded;
    loaded.mappers =
        init ? std::move(init->mappers) : fit_mappers(xarg, std::move(names), cfg);
    if (init)
    {
        bonsai::require_n_features(xarg.n_features, loaded.mappers.size(),
                                   "the matrix passed to fit(init_model=...)");
    }
    bonsai::floats_view const wview = warg ? warg->view() : bonsai::floats_view{};
    loaded.train =
        make_labeled(xarg, yarg.view(), loaded.mappers, cfg,
                     bonsai::grower_runs_on_device(cfg.dispatch.grower_name), wview);
    std::optional<bonsai::cli::LabeledData> owned;
    auto const *const                       validation =
        resolve_eval_set(eval_set, cfg, loaded.mappers, init.has_value(), owned);

    std::vector<float> history;
    auto               initial = init ? std::move(init->booster) : nullptr;
    auto               booster =
        validation != nullptr
                          ? bonsai::cli::train_with_progress(cfg, loaded.train, *validation, {},
                                                             std::move(initial), std::ref(history))
                          : bonsai::cli::train_with_progress(cfg, loaded.train, {},
                                                             std::move(initial), std::ref(history));
    return Model{std::move(booster), std::move(loaded.mappers), cfg,
                 std::move(history)};
}

Model train_dataset(nb::object const &params, Dataset const &dataset,
                    std::optional<EvalSet> const     &eval_set,
                    std::optional<std::string> const &init_model)
{
    ParamItems       items = items_from_params(params);
    nb::object const named = take_named_monotone(items);
    put_monotone(items, named, dataset.loaded().mappers.feature_names());
    ConfigPairs const pairs = render_params(items);
    for (auto const &[key, value] : pairs)
    {
        if (key.starts_with("bin_mapper."))
        {
            throw std::invalid_argument(
                "bin_mapper.* is fixed when training from a prebuilt Dataset; set "
                "max_bin/n_samples/seed/min_data_in_bin at Dataset construction "
                "instead");
        }
    }
    bonsai::Config cfg = config_from_params(pairs);
    if (auto const resident = dataset.device_id();
        resident && *resident != cfg.parallel.device_id)
    {
        throw std::invalid_argument(
            "this Dataset is binned on CUDA device " + std::to_string(*resident) +
            "; parallel.device_id=" + std::to_string(cfg.parallel.device_id) +
            " would train on another device. Train with device_id=" +
            std::to_string(*resident) + " or rebuild the Dataset on that device.");
    }
    bonsai::parallel::set_n_threads(cfg.parallel.n_threads);

    std::optional<bonsai::io::LoadedBooster> init;
    if (init_model)
    {
        init.emplace(bonsai::io::load_booster(*init_model));
        if (!init->mappers.same_cuts(dataset.loaded().mappers))
        {
            throw std::invalid_argument(
                "init_model's cuts disagree with the dataset's; one set of "
                "cuts describes one set of columns, so a warm start needs the "
                "dataset binned with the model's own cuts");
        }
        cfg = bonsai::cli::reconcile_warm_start(std::move(cfg), init->cfg,
                                                stated_keys(items));
    }
    nb::gil_scoped_release                  release;
    std::optional<bonsai::cli::LabeledData> owned;
    auto const *const                       validation = resolve_eval_set(
        eval_set, cfg, dataset.loaded().mappers, init.has_value(), owned);
    std::vector<float> history;
    auto               initial = init ? std::move(init->booster) : nullptr;
    auto const        &train   = dataset.train_data();

    auto booster =
        validation != nullptr
            ? bonsai::cli::train_with_progress(cfg, train, *validation, {},
                                               std::move(initial), std::ref(history))
            : bonsai::cli::train_with_progress(cfg, train, {}, std::move(initial),
                                               std::ref(history));
    return Model{std::move(booster), dataset.loaded().mappers, cfg, std::move(history)};
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
        .def("predict",
             nb::overload_cast<array_2d const &, size_t>(&Model::predict, nb::const_),
             nb::arg("X"), nb::arg("num_iteration") = 0,
             "Predict on the response scale (the objective's link inverse "
             "applied).\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "X : float32 array, shape (n_rows, n_features), or Dataset\n"
             "    Row-major features. A Dataset carrying this model's cuts "
             "routes in bin space (device-resident builds included), "
             "bit-identical to the raw walk; other Datasets read the matrix "
             "they retained.\n"
             "num_iteration : int, default 0\n"
             "    Predict with only the first ``num_iteration`` boosting "
             "rounds; 0 uses all.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "float32 array, shape (n_rows,)\n"
             "    One prediction per row.")
        .def("predict",
             nb::overload_cast<Dataset const &, size_t>(&Model::predict, nb::const_),
             nb::arg("X"), nb::arg("num_iteration") = 0)
        .def("predict_proba",
             nb::overload_cast<array_2d const &>(&Model::predict_proba, nb::const_),
             nb::arg("X"),
             "Per-class probabilities for classification models.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "X : float32 array, shape (n_rows, n_features), or Dataset\n"
             "    Row-major features; a Dataset with this model's cuts "
             "routes in bin space, others read the matrix they retained.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "float64 array\n"
             "    Softmax models return ``(n_rows, n_classes)``, a row-wise "
             "softmax of the class logits; logloss returns ``(n_rows,)`` with "
             "P(class 1) via the link inverse.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    For non-classification objectives: the mse link inverse is "
             "the identity, so raw margins would be silently mislabeled as "
             "probabilities.")
        .def("predict_proba",
             nb::overload_cast<Dataset const &>(&Model::predict_proba, nb::const_),
             nb::arg("X"))
        .def("staged_predict",
             nb::overload_cast<array_2d const &>(&Model::staged_predict, nb::const_),
             nb::arg("X"),
             "Predictions after each boosting iteration.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "X : float32 array, shape (n_rows, n_features), or Dataset\n"
             "    Row-major features; a Dataset with this model's cuts "
             "routes in bin space, others read the matrix they retained.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "float32 array, shape (n_iters, n_rows)\n"
             "    Row t is the response-scale prediction using rounds 0..t.")
        .def("staged_predict",
             nb::overload_cast<Dataset const &>(&Model::staged_predict, nb::const_),
             nb::arg("X"))
        .def("predict_leaf",
             nb::overload_cast<array_2d const &>(&Model::predict_leaf, nb::const_),
             nb::arg("X"),
             "The leaf each row lands in, one column per tree.\n"
             "\n"
             "Width-1 objectives have one tree per round, so the columns are "
             "the boosting rounds in order. Softmax models grow one tree per "
             "class per round and the columns stay in that order, so column t "
             "is round ``t // n_classes``, class ``t % n_classes``.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "X : float32 array, shape (n_rows, n_features), or Dataset\n"
             "    Row-major features; a Dataset with this model's cuts "
             "routes in bin space, others read the matrix they retained.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "uint32 array, shape (n_rows, n_trees)\n"
             "    Leaf node id per (row, tree).")
        .def("predict_leaf",
             nb::overload_cast<Dataset const &>(&Model::predict_leaf, nb::const_),
             nb::arg("X"))
        .def("dump", &Model::dump,
             "The trees as human-readable text, with feature names.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "str\n"
             "    One block per tree.")
        .def("pred_contribs",
             nb::overload_cast<array_2d const &>(&Model::pred_contribs, nb::const_),
             nb::arg("X"),
             "TreeSHAP feature contributions, last column the bias.\n"
             "\n"
             "Each row sums to the raw (pre-link) prediction exactly.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "X : float32 array, shape (n_rows, n_features), or Dataset\n"
             "    Row-major features; a Dataset with this model's cuts "
             "routes in bin space, others read the matrix they retained.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "float64 array\n"
             "    ``(n_rows, n_features + 1)``; multiclass models return "
             "``(n_rows, n_classes, n_features + 1)``.")
        .def("pred_contribs",
             nb::overload_cast<Dataset const &>(&Model::pred_contribs, nb::const_),
             nb::arg("X"))
        .def("feature_importance", &Model::feature_importance, nb::arg("type") = "gain",
             "Per-feature importance, padded to the trained feature count.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "type : {'gain', 'split'}, default 'gain'\n"
             "    'gain' is total loss reduction; 'split' is split count.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "float64 array, shape (n_features,)\n"
             "    Importance per feature; features never split score 0.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    For any other ``type`` string.")
        .def("save", &Model::save, nb::arg("path"),
             "Save the model (MessagePack) with its bin mappers and config.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "path : str\n"
             "    Output file; ``bonsai.load(path)`` restores the model.")
        .def(
            "__getstate__",
            [](Model const &m)
            {
                auto const bytes = m.save_bytes();
                return nb::bytes(reinterpret_cast<char const *>(bytes.data()),
                                 bytes.size());
            },
            "Pickle support: the same bytes ``save`` writes, so a pickled\n"
            "Model and a saved file restore identically (neither carries\n"
            "the eval history).")
        .def("__setstate__",
             [](Model &m, nb::bytes const &state)
             {
                 auto const *const first =
                     reinterpret_cast<uint8_t const *>(state.c_str());
                 std::vector<uint8_t> const bytes(first, first + state.size());
                 auto loaded = bonsai::io::load_booster_bytes(bytes);
                 new (&m) Model{std::move(loaded.booster), std::move(loaded.mappers),
                                std::move(loaded.cfg)};
             })
        .def_prop_ro("n_iters", &Model::n_iters, "Boosting rounds this model carries.")
        .def_prop_ro("eval_history", &Model::eval_history,
                     "Per-round valid loss from fit (objective's own eval "
                     "metric); empty without an eval set or after load(). "
                     "Indexed by absolute model round: after an init_model "
                     "warm start the pre-existing rounds appear as NaN "
                     "placeholders.")
        .def_prop_ro("config_toml", &Model::config_toml,
                     "The resolved training config, rendered as TOML.")
        .def_prop_ro("objective_name", &Model::objective_name,
                     "The objective this model was trained with (e.g. mse, "
                     "logloss, softmax).")
        .def_prop_ro("n_classes", &Model::n_classes,
                     "Class count for softmax models; 0 for every other "
                     "objective (including binary logloss).")
        .def_prop_ro("feature_names", &Model::feature_names,
                     "One name per feature, in column order: the names the fit "
                     "was given, or the synthesized ``f0``..``fN``. Survives "
                     "save/load, and its length is the model's feature count.")
        .def("__repr__",
             [](Model const &mo)
             {
                 return "Model(objective='" + mo.objective_name() +
                        "', n_iters=" + std::to_string(mo.n_iters()) + ")";
             });

    constexpr bonsai::ParallelConfig k_parallel_defaults{};
    nb::class_<Dataset>(m, "Dataset")
        .def(
            nb::init<nb::handle, nb::handle, nb::handle, std::optional<int>,
                     std::optional<size_t>, std::optional<uint64_t>, std::optional<int>,
                     std::optional<std::map<size_t, array_1d>> const &,
                     std::optional<std::string> const &, std::optional<uint32_t>,
                     uint32_t, Dataset const *,
                     std::optional<std::vector<std::string>> const &>(),
            nb::arg("X"), nb::arg("y"), nb::arg("weight") = nb::none(),
            nb::arg("max_bin") = nb::none(), nb::arg("n_samples") = nb::none(),
            nb::arg("seed") = nb::none(), nb::arg("min_data_in_bin") = nb::none(),
            nb::arg("bin_edges") = nb::none(), nb::arg("device") = nb::none(),
            nb::arg("device_id")        = nb::none(),
            nb::arg("n_threads")        = k_parallel_defaults.n_threads,
            nb::arg("reference").none() = nb::none(),
            nb::arg("feature_names")    = nb::none(),
            "A pre-binned dataset: bins X once at construction.\n"
            "\n"
            "Reused across ``train(params, dataset)`` calls (hyperparameter "
            "search / CV), skipping the per-fit bin pass. All bin_mapper "
            "settings are fixed here.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "X : float32 array, shape (n_rows, n_features)\n"
            "    Row-major features; any DLPack producer. A device-resident X "
            "(cupy, torch, jax) is binned on the GPU in place, with no host "
            "round trip.\n"
            "y : float32 array, shape (n_rows,)\n"
            "    Labels. May be device-resident; downloaded once, because "
            "bonsai keeps labels on the host.\n"
            "weight : float32 array, shape (n_rows,), optional\n"
            "    Per-row weights; the same device rules as ``y``.\n"
            "max_bin, n_samples, seed, min_data_in_bin : optional\n"
            "    Bin-mapper settings, fixed at construction. Unset means the "
            "library default, or the reference's value in the ``reference=`` "
            "form.\n"
            "bin_edges : dict of {int: float32 array}, optional\n"
            "    Column index to its explicit interior cut points (strictly "
            "increasing; k edges give k+1 bins). Listed columns skip quantile "
            "fitting, and the edges travel inside the model artifact, so "
            "predict/save/load work on raw values with no external "
            "transform.\n"
            "device : {'cpu', 'cuda'}, optional\n"
            "    Defaults to where X already is, or to the reference's device "
            "in the ``reference=`` form (an explicit ``'cpu'`` overrides it). "
            "``'cuda'`` bins on the GPU "
            "and keeps the matrix resident there, so every cuda_* fit adopts "
            "it with no upload (a sweep uploads once, not once per fit); it "
            "raises without a CUDA build and a visible device, and a later "
            "``parallel.device_id`` that disagrees with ``device_id`` raises "
            "rather than migrating. A device-binned Dataset handed to a CPU "
            "grower materializes host bins once, on first use.\n"
            "device_id : int, optional\n"
            "    CUDA device the binned matrix lives on. Unset means the "
            "reference's device in the ``reference=`` form, else 0.\n"
            "n_threads : int, default 0\n"
            "    Sizes the binning pass (0 = auto), the way "
            "``parallel.n_threads`` sizes a fit.\n"
            "reference : Dataset, optional\n"
            "    Bin with this dataset's cut points instead of fitting our "
            "own, which is what a validation set needs: the result can be "
            "handed to ``train(..., eval_set=valid_dataset)`` and every fit "
            "routes it in bin space with no per-fit bin pass. Binning "
            "settings are inherited from the reference when left unset; "
            "setting one to a different value raises, as ``bin_edges`` and "
            "``feature_names`` do at all. ``device`` and ``device_id`` are "
            "inherited the same "
            "way, so a validation set lands beside its training set unless "
            "placed explicitly.\n"
            "feature_names : sequence of str, optional\n"
            "    One name per column, carried into the model: ``dump`` prints "
            "them and ``tree.monotone_constraints`` may be keyed by them. "
            "Unset, the columns get ``f0``..``fN``. Names must number the "
            "columns exactly and be unique, and cannot be given with "
            "``reference=``, which already names them. From pandas, "
            "``Dataset(df.values, y, feature_names=df.columns)``.")
        .def("__reduce__",
             [](Dataset const &) -> nb::object
             {
                 throw std::runtime_error(
                     "Dataset is not picklable: it holds binned columns (device "
                     "memory under device=\"cuda\") that no artifact carries. "
                     "Rebuild it from X and y in the target process, or pickle "
                     "the trained Model instead.");
             })
        .def_prop_ro("device", &Dataset::device,
                     "Where the binned columns live: \"cuda\" for a "
                     "device-binned dataset, \"cpu\" otherwise.")
        .def_prop_ro("n_rows", &Dataset::n_rows,
                     "Rows a fit visits: the binned matrix's, or the "
                     "selection's on a view.")
        .def_prop_ro("n_features", &Dataset::n_features,
                     "Feature columns in the binned matrix.")
        .def_prop_ro("feature_names", &Dataset::feature_names,
                     "One name per column, in column order: the "
                     "``feature_names=`` given here, or the synthesized "
                     "``f0``..``fN``.")
        .def_prop_ro(
            "shape",
            [](Dataset const &d) { return nb::make_tuple(d.n_rows(), d.n_features()); },
            "(n_rows, n_features), the way an array reports it. IDE variable "
            "explorers read it for their size column.")
        .def("__len__", [](Dataset const &d) { return d.n_rows(); })
        .def_prop_ro("base", &Dataset::base,
                     "The Dataset whose binned plane this one selects rows out "
                     "of, or None when it owns its own, the way "
                     "``numpy.ndarray.base`` reads. A chained view names the "
                     "root, not the view it was taken from.")
        .def(
            "subset", [](nb::object self, nb::handle rows, nb::handle columns)
            { return nb::cast<Dataset const &>(self).subset(self, rows, columns); },
            nb::arg("rows") = nb::none(), nb::arg("columns") = nb::none(),
            nb::sig("def subset(self, "
                    "rows: slice | collections.abc.Sequence[int] | "
                    "NDArray[numpy.integer] | NDArray[numpy.bool] | None = None, "
                    "columns: slice | collections.abc.Sequence[int] | "
                    "collections.abc.Sequence[str] | NDArray[numpy.integer] | "
                    "NDArray[numpy.bool] | None = None) -> Dataset"),
            "Select rows and/or columns out of this Dataset.\n"
            "\n"
            "Rows share the parent's binned plane and cost nothing. Columns "
            "rewrite it: the kept columns are gathered into a plane the result "
            "owns, renumbered from zero, which is what packs them into full "
            "device tiles. Ask for both and the rows are applied first, so the "
            "rewrite gathers only the rows that survive.\n"
            "\n"
            "The result is a Dataset that ``train`` fits on the selected rows, "
            "that ``eval_set`` scores over exactly those rows, and that "
            "``predict`` and its family answer one row per selected row in the "
            "selection's order. Nothing is copied: the bins, the cut points "
            "and the row-major mirror stay with the parent, so k folds cost "
            "one plane rather than k. Selections compose, and positions are "
            "always into the Dataset they are given to.\n"
            "\n"
            "A fit over a view is about the selected rows and no others. It "
            "does not carry the rows left out through each finished tree the "
            "way a subsampled fit carries the rows its draw dropped, since "
            "those rejoin the next tree and a view's never do. So the rows "
            "outside the selection keep stale training scores. Nothing the "
            "view's fit produces reads them, and ``predict`` recomputes from "
            "the trees regardless.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "rows : integer array, slice, or boolean mask\n"
            "    Which rows to keep. The order is kept as given and duplicates "
            "count once each, so a bootstrap draw weighs the way the "
            "materialized copy of it would. Out-of-range and negative indices "
            "raise: row ids index the binned plane and do not wrap.\n"
            "columns : integer array, slice, boolean mask, or feature names\n"
            "    Which features to keep, in the order given. Unlike rows this "
            "copies: the bins are gathered into a new plane, so the result "
            "owns its columns and its ``base`` is None. The gather stays "
            "wherever the parent's bins were: a device-resident parent is "
            "rewritten on the device and the result reports that device, so a "
            "feature-selection loop never round-trips through the host. "
            "Out-of-range and negative indices raise, and an unknown name "
            "raises KeyError.\n"
            "\n"
            "Returns\n"
            "-------\n"
            "Dataset\n"
            "    With ``rows=`` alone, a view whose ``base`` is the Dataset "
            "that owns the plane. With ``columns=``, a Dataset that owns its "
            "own plane and whose ``base`` is None.")
        .def(
            "reorder", [](nb::object self, nb::handle rows)
            { return nb::cast<Dataset const &>(self).reorder(self, rows); },
            nb::arg("rows") = nb::none(),
            nb::sig("def reorder(self, "
                    "rows: slice | collections.abc.Sequence[int] | "
                    "NDArray[numpy.integer] | None = None) -> Dataset"),
            "The same rows in a different order, laid out that way.\n"
            "\n"
            "Unlike ``subset(rows=)``, which describes an order and leaves the "
            "bins where they are, this rewrites the plane so the order is the "
            "storage. That is worth doing when the same selections will be fit "
            "repeatedly: a scattered fold costs a gather on every histogram "
            "fill of every tree, while a fold laid out contiguously is a range "
            "the fill reads as a subspan and the device reads fully "
            "coalesced.\n"
            "\n"
            "The usual arrangement is to sort rows into group order once, "
            "after which each fold is a slice::\n"
            "\n"
            "    order = np.argsort(groups, kind=\"stable\")\n"
            "    ds = bonsai.Dataset(X, y).reorder(rows=order)\n"
            "    fold = ds.subset(rows=slice(0, n_first_group))\n"
            "\n"
            "The result is an ordinary Dataset in the new order: row i is the "
            "row ``rows[i]`` named, and labels, predictions and contributions "
            "all follow that order, the way ``X[order]`` would. Nothing is "
            "un-permuted behind the caller's back, because the contiguity is "
            "the thing being bought and hiding it would put it out of "
            "reach.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "rows : integer array or slice\n"
            "    A permutation of this Dataset's rows: every row exactly once. "
            "To keep only some of them, use ``subset(rows=...)``.\n"
            "\n"
            "Returns\n"
            "-------\n"
            "Dataset\n"
            "    A Dataset that owns its own plane; its ``base`` is None.")
        .def("__repr__",
             [](Dataset const &d)
             {
                 std::string out = "Dataset(" + std::to_string(d.n_rows()) + " x " +
                                   std::to_string(d.n_features()) + ", " + d.device();
                 if (d.is_view())
                 {
                     out += ", view: " + view_shape_phrase(d.row_view()) +
                            ", density " + two_decimals(d.row_view().density()) +
                            ", shares parent plane";
                 }
                 return out + ")";
             });

    m.def("train", &train_dataset, nb::arg("params").none(), nb::arg("dataset"),
          nb::arg("eval_set") = nb::none(), nb::arg("init_model") = nb::none(),
          nb::sig("def train(params: bonsai.params.Params | "
                  "collections.abc.Mapping[str, object] | None, dataset: Dataset, "
                  "eval_set: tuple[Annotated[NDArray[numpy.float32], "
                  "dict(shape=(None, None), order='C', device='cpu', "
                  "writable=False)], Annotated[NDArray[numpy.float32], "
                  "dict(shape=(None,), order='C', device='cpu', writable=False)]] "
                  "| Dataset | None = None, init_model: str | None = None) -> "
                  "Model"),
          "Train on a prebuilt Dataset, reusing its binning across calls.\n"
          "\n"
          "Parameters\n"
          "----------\n"
          "params : Params, Mapping, or None\n"
          "    Overrides over the library defaults: a ``bonsai.Params``, a "
          "``{'tree.max_depth': 8}`` dotted-key mapping, or ``None`` for no "
          "overrides. A TOML base composes here too, "
          "``Params.from_toml(path) | overrides``. ``bin_mapper.*`` overrides "
          "are rejected: binning is fixed by the Dataset.\n"
          "\n"
          "    ``tree.monotone_constraints`` also takes a mapping keyed by "
          "feature name, ``{'age': 1, 'debt': -1}``, resolved against the "
          "Dataset's ``feature_names``. Features the mapping leaves out are "
          "free (0); a name the data does not carry raises.\n"
          "dataset : Dataset\n"
          "    The pre-binned training data.\n"
          "eval_set : tuple of (Xv, yv), Dataset, or None\n"
          "    Enables per-iteration eval and early stopping. Arrays are "
          "binned by the fit itself once the rounds it has run have paid for "
          "the pass; a Dataset built with ``reference=`` this one is binned "
          "once and routed in bin space by every fit.\n"
          "init_model : str, optional\n"
          "    Continue training from this saved model (warm start).\n"
          "\n"
          "Returns\n"
          "-------\n"
          "Model\n"
          "    The trained booster.");
    m.def("train", &train, nb::arg("params").none(), nb::arg("X"), nb::arg("y"),
          nb::arg("eval_set") = nb::none(), nb::arg("init_model") = nb::none(),
          nb::arg("sample_weight") = nb::none(), nb::arg("feature_names") = nb::none(),
          nb::sig("def train(params: bonsai.params.Params | "
                  "collections.abc.Mapping[str, object] | None, X: object, "
                  "y: object, eval_set: tuple[Annotated[NDArray[numpy.float32], "
                  "dict(shape=(None, None), order='C', device='cpu', "
                  "writable=False)], Annotated[NDArray[numpy.float32], "
                  "dict(shape=(None,), order='C', device='cpu', writable=False)]] "
                  "| Dataset | None = None, init_model: str | None = None, "
                  "sample_weight: object | None = None, feature_names: "
                  "collections.abc.Sequence[str] | None = None) -> Model"),
          "Train a booster on row-major float32 features.\n"
          "\n"
          "Parameters\n"
          "----------\n"
          "params : Params, Mapping, or None\n"
          "    Overrides over the library defaults: a ``bonsai.Params``, a "
          "``{'tree.max_depth': 8}`` dotted-key mapping, or ``None`` for no "
          "overrides. A TOML base composes here too, "
          "``Params.from_toml(path) | overrides``.\n"
          "\n"
          "    ``tree.monotone_constraints`` also takes a mapping keyed by "
          "feature name, ``{'age': 1, 'debt': -1}``, resolved against this "
          "data's feature names. Features the mapping leaves out are free "
          "(0); a name the data does not carry raises.\n"
          "X : float32 array, shape (n_rows, n_features)\n"
          "    Row-major features. May be a CUDA array supporting DLPack "
          "(cupy, torch, jax): X is then binned on the GPU in place, with no "
          "host round trip. The columns are named by ``feature_names``, or "
          "``f0``..``fN`` when it is unset.\n"
          "y : float32 array, shape (n_rows,)\n"
          "    Labels; a device-resident y is downloaded once.\n"
          "eval_set : tuple of (Xv, yv), Dataset, or None\n"
          "    Host arrays, or a Dataset binned with this fit's cut points. "
          "Stays host-side either way, because the per-iteration eval "
          "predicts on the host.\n"
          "init_model : str, optional\n"
          "    Continue training from this saved model (warm start).\n"
          "sample_weight : float32 array, shape (n_rows,), optional\n"
          "    Per-row weights; scales each row's gradient and hessian. A "
          "device-resident vector is downloaded once.\n"
          "feature_names : sequence of str, optional\n"
          "    One name per column, carried into the model: ``dump`` prints "
          "them and ``tree.monotone_constraints`` may be keyed by them. Unset, "
          "the columns get ``f0``..``fN``. Names must number the columns "
          "exactly and be unique. Giving them with ``init_model`` raises, "
          "because a warm start keeps the loaded model's names.\n"
          "\n"
          "Returns\n"
          "-------\n"
          "Model\n"
          "    The trained booster.");
    m.def("load", &load, nb::arg("path"),
          "Load a model saved by ``Model.save``.\n"
          "\n"
          "Parameters\n"
          "----------\n"
          "path : str\n"
          "    A ``.msgpack`` model file.\n"
          "\n"
          "Returns\n"
          "-------\n"
          "Model\n"
          "    The restored booster, with its bin mappers and config.");

    m.def(
        "default_config_toml", [] { return bonsai::config::dump_toml({}); },
        "The library's default config, rendered as TOML.");
    m.def("cuda_available", &bonsai::cuda_available,
          "True when the binary carries the CUDA backend and a usable device "
          "is present (cuda_* growers can train).");
    m.def(
        "_n_threads", [] { return bonsai::parallel::n_threads(); },
        "Worker count in effect for the process, as the last train or "
        "Dataset call left it (diagnostics).");
    m.def(
        "_params_from_toml",
        [](std::string const &text)
        {
            nb::dict out;
            for (auto const &[key, value] : bonsai::config::typed_overrides(text))
            {
                out[nb::str(key.c_str())] =
                    std::visit([](auto const &v) { return nb::cast(v); }, value);
            }
            return out;
        },
        nb::arg("text"),
        "The dotted keys a TOML config text explicitly sets, with typed "
        "values: only the stated keys, never the resolved whole. Feeds "
        "Params.from_toml/from_model; not public API.");
    m.def("_params_schema", &params_schema,
          "The config section registry as data: [{'section', 'fields': "
          "[{'name', 'type', 'default'}]}] in registry order. Feeds the "
          "build-time bonsai/_params.py generator; not public API.");
} // namespace nb

// Python bindings: a thin nanobind layer over the same seams the CLI uses
// (config::apply_overrides, cli::train_with_progress, io::save/load_booster).
// No training or prediction logic lives here.

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
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
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

// Hand an owning vector to numpy: the capsule takes the vector, the array
// takes the capsule, and the buffer is never copied. The unique_ptr is
// consumed, so a throw before this call still frees.
template <typename T>
nb::ndarray<nb::numpy, T> to_numpy(std::unique_ptr<std::vector<T>> out,
                                   std::initializer_list<size_t>   shape)
{
    auto       *raw = out.release();
    nb::capsule owner(raw, [](void *p) noexcept
                      { delete static_cast<std::vector<T> *>(p); });
    return {raw->data(), shape, owner};
}

// --- device-resident input (DLPack)

// Placement for device-resident input: the buffer decides where the work
// happens, so an array whose device disagrees with parallel.device_id is
// refused rather than migrated behind the caller's back. Stream ordering is
// the producer's job under DLPack, which synchronizes at export; ingest reads
// on the default stream.
template <typename Array>
void place_device_array(Array const &arr, uint32_t device_id, char const *what)
{
    // The import validates dtype, rank, and contiguity; a null pointer or a
    // zero-size axis is all it leaves for bonsai to refuse.
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

// The feature matrix as bonsai received it: a host numpy array, or a device
// buffer bonsai bins where it already lies. The device pointer is borrowed for
// the duration of the call that consumes it and never retained: ingest mints a
// plane that owns its own device bins, so nothing outlives this struct.
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

// Labels and weights stay on the host whichever device the features are on:
// the objective, the eval loop, and the device-resident uploader all read them
// from host memory, so a device vector is downloaded once here rather than
// plumbed through as a second device pointer.
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

// X, y and an optional weight vector resolved together with the two length
// checks every entry point owes its caller. `weight_name` is the keyword the
// caller spells the weights as, so the message names the user's own argument;
// `where` prefixes the entry point when it is not the free train function.
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

// The bin mappers for X. The device arm gathers exactly the rows the host
// sampler would have drawn and fits on those, so the cuts, and therefore the
// bins, are bit-identical to the host path's for the same seed.
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

// Bins straight from the row-major numpy matrix (no column-major float
// materialization); the FeatureBuffer borrows the same buffer, which is
// alive for the duration of the train call.
bonsai::cli::LabeledData make_labeled(MatrixArg const &X, bonsai::floats_view y,
                                      bonsai::BinMappers const &mappers,
                                      bonsai::Config const &cfg, bool on_device,
                                      bonsai::floats_view weights = {})
{
    bonsai::cli::FeatureBuffer buf;
    buf.n_rows     = X.n_rows;
    buf.n_features = X.n_features;

    // Device-resident input: the plane is the only copy of the columns, so
    // ingest never declines here and the FeatureBuffer stays empty. Nothing
    // reads it — the raw matrix is a progress-tick predict input, and these
    // bindings pass no tick callback.
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

    // The ingest transaction (decision 54): the device arm bins on the GPU;
    // cuda_ingest declines (nullptr) when the dataset's bins exceed the
    // resident ceiling, keeping the host fill. Device placement first
    // (issue #158): cudaSetDevice is thread-local and this thread is about
    // to mint the device plane.
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

// Validation-only LabeledData: train_with_progress takes the features and
// labels and bins them itself, once the rounds it has run have paid for the
// pass, so binning here would charge every fit for it and, under cuda
// growers, add a wasted GPU upload per call.
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

// A reusable pre-binned dataset (decision 65): binning runs once at
// construction, and the SAME bonsai::Dataset is fed to every train() call, so a
// hyperparameter sweep or CV loop skips the per-fit bin pass. On GPU the
// resident-matrix upload-skip cache (ensure_dataset) fires because the object
// address is stable across fits. Holds the numpy X alive because the
// FeatureBuffer borrows the row-major matrix; y and weight are copied out by
// Dataset::bin during construction and are not retained.
// Binning follows the device hint: the host by default, the GPU under
// device="cuda", where the resident matrix is then adopted by every fit.
// Device-resident input (DLPack) carries its own answer: the bytes are
// already there, so it bins there whatever the hint's default. A reference
// answers for an unset hint the same way it answers for unset binning
// settings: a validation set follows its training set's device unless the
// caller places it elsewhere explicitly.
// `reference` binds the new dataset to another one's cuts instead of fitting
// its own, which is what makes a validation set reusable: a fit can only
// route rows binned under the cuts its own trees were grown on.
class Dataset
{
  public:
    Dataset(nb::handle X, nb::handle y, nb::handle weight, std::optional<int> max_bin,
            std::optional<size_t> n_samples, std::optional<uint64_t> seed,
            std::optional<int>                               min_data_in_bin,
            std::optional<std::map<size_t, array_1d>> const &bin_edges,
            std::optional<std::string> const &device, std::optional<uint32_t> device_id,
            uint32_t n_threads, Dataset const *reference)
    {
        // The reference's residency answers for both unset device arguments,
        // so a validation set lands beside its training set by default.
        std::optional<uint32_t> const ref_device =
            reference != nullptr ? reference->device_id_ : std::nullopt;
        uint32_t const dev_id =
            device_id.value_or(ref_device.value_or(bonsai::ParallelConfig{}.device_id));
        auto const [xarg, yarg, warg] =
            resolve_inputs(X, y, weight, dev_id, "weight", "Dataset: ");
        // A device hint is an explicit user request, so an absent backend or
        // device is an error here, unlike the engine's own inference from a
        // grower name (which degrades to the host silently). Device-resident
        // input needs no hint: it defaults to the device it is already on.
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
        // A reference already decided the binning, so an unset setting takes
        // its value rather than the library default; only an explicit one
        // that disagrees is an error (check_reference).
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
                            xarg.n_features);
        }

        size_t const             f = xarg.n_features;
        std::vector<std::string> names;
        names.reserve(f);
        for (size_t c = 0; c < f; ++c)
        {
            names.push_back("f" + std::to_string(c));
        }
        // Copy the edge arrays out while the GIL is held; validation
        // (finite, strictly increasing, in-range column) happens inside
        // BinMappers::fit and surfaces as bonsai::ConfigError.
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
        bin_cfg_        = cfg.bin_mapper;
        loaded_.mappers = reference != nullptr
                              ? reference->loaded_.mappers
                              : fit_mappers(xarg, std::move(names), cfg, edges);
        loaded_.train =
            make_labeled(xarg, yarg.view(), loaded_.mappers, cfg, on_device, w);
        // Device state is recorded only when a plane was actually minted:
        // an ingest decline leaves an ordinary host dataset, which no later
        // fit needs to be placed against.
        if (loaded_.train.dataset.ingest_plane())
        {
            device_id_ = dev_id;
        }
        // The host matrix is kept alive because the FeatureBuffer borrows it;
        // a device matrix is not kept at all, since its bins were copied into
        // the plane and nothing else refers to the caller's buffer.
        x_          = xarg.host;
        n_features_ = xarg.n_features;
    }

    // The device the binned columns live on, "cpu" or "cuda". A device
    // request that ingest declined reports "cpu": the columns are on the
    // host and nothing about the fit is constrained.
    std::string device() const
    {
        return device_id_ ? "cuda" : "cpu";
    }

    // The device a device-binned dataset is resident on; empty for host
    // datasets. A fit placed on a different device would have to migrate the
    // matrix, so train() rejects the mismatch instead.
    std::optional<uint32_t> device_id() const
    {
        return device_id_;
    }

    size_t n_rows() const
    {
        return loaded_.train.labels.size();
    }
    size_t n_features() const
    {
        return n_features_;
    }

    // Whether the caller's host matrix is still reachable. Device-resident
    // (DLPack) input leaves no host copy, and the raw-row readers on the
    // eval path (the warm-start seam, the walk before it switches to bin
    // space) have nothing to read without one.
    bool has_host_matrix() const
    {
        return x_.has_value();
    }

    bonsai::cli::LoadedTrainValidation const &loaded() const
    {
        return loaded_;
    }

    // The binned columns, which is all most readers want out of loaded().
    bonsai::Dataset const &bins() const
    {
        return loaded_.train.dataset;
    }

    // The raw rows a host-built Dataset retained; the Model raw-row readers
    // (predict and friends) read them directly, no rebinning. Asking a
    // device-resident (DLPack) build is an error with the remedy.
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
    // A reference supplies the cuts, so every setting that would have shaped
    // them is inert here: unset inherits, and a disagreeing one is said out
    // loud rather than ignored.
    static void check_reference(Dataset const                 &reference,
                                bonsai::BinMapperConfig const &bin_cfg,
                                bool has_bin_edges, size_t n_features)
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
        if (n_features != reference.n_features_)
        {
            throw std::invalid_argument(
                "Dataset(reference=...): X has " + std::to_string(n_features) +
                " columns and the reference has " +
                std::to_string(reference.n_features_) +
                "; one set of cuts describes one set of columns");
        }
    }

    std::optional<array_2d>            x_;
    size_t                             n_features_ = 0;
    bonsai::cli::LoadedTrainValidation loaded_;
    std::optional<uint32_t>            device_id_;
    bonsai::BinMapperConfig            bin_cfg_;
};

// The device plans cached per mutation epoch, so a sweep of predicts or
// explains between fits packs the ensemble once. Readers are const and may be
// concurrent (the bindings release the GIL), hence the lock; a refused pack
// caches too, so a host with no device pays one refusal per epoch, not one
// per call. The two kinds pack independently: leaf paths cost more to pack
// than the node tables, and a caller that never explains never pays for them.
class DevicePlanCache
{
  public:
    std::shared_ptr<bonsai::CudaPredictPlan const>
    predict(bonsai::PredictPlanInput const &in, bonsai::BinMappers const &mappers) const
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
    shap(bonsai::PredictPlanInput const &in, bonsai::BinMappers const &mappers) const
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
    // Booster epochs start at 1, so epoch = 0 is the never-packed state and
    // the epoch compare alone is the miss test.
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

// A trained model: booster + the bin mappers and config it was fit with.
class Model
{
  public:
    Model(std::unique_ptr<bonsai::IBooster> booster, bonsai::BinMappers mappers,
          bonsai::Config cfg, std::vector<float> eval_history = {})
        : booster_(std::move(booster)), mappers_(std::move(mappers)),
          cfg_(std::move(cfg)), eval_history_(std::move(eval_history))
    {
    }

    // Per-round valid loss from fit (objective's own eval metric); empty when
    // no eval set was given or the model was loaded from a file. In-memory
    // only: the model format does not carry it.
    std::vector<float> const &eval_history() const
    {
        return eval_history_;
    }

    nb::ndarray<nb::numpy, float> predict(array_2d const &X,
                                          size_t          num_iteration = 0) const
    {
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

    // Every X-taking method also accepts a Dataset. A Dataset carrying the
    // model's own cuts routes in bin space (bit-identical to the raw walk,
    // no raw rows needed, so DLPack builds work); foreign cuts fall back to
    // the retained matrix; a device-resident build under foreign cuts has
    // nothing either route can read, so it raises with both remedies.
    bool routes_binned(Dataset const &ds, char const *method) const
    {
        if (mappers_.same_cuts(ds.loaded().mappers))
        {
            return true;
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

    // Per-class probabilities. Softmax models return (n_rows, n_classes) — a
    // row-wise softmax of the class logits; width-1 objectives (logloss)
    // return (n_rows,) with P(class 1) via the link inverse.
    nb::ndarray<nb::numpy, double> predict_proba(array_2d const &X) const
    {
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

    // (n_iters, n_rows): prediction after each boosting iteration.
    nb::ndarray<nb::numpy, float> staged_predict(array_2d const &X) const
    {
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
                    bonsai::floats_out{out->data() + (t * n), n});
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
                    bonsai::floats_out{out->data() + (t * n), n});
            }
        }
        return to_numpy(std::move(out), {k, n});
    }

    // (n_rows, n_trees): the leaf each row lands in, per tree. The width is
    // the tree count, not the round count: multiclass grows one tree per
    // class per round and the booster fills a column for each.
    nb::ndarray<nb::numpy, uint32_t> predict_leaf(array_2d const &X) const
    {
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

    // (n_rows, n_features + 1): TreeSHAP contributions, last column = bias.
    // Rows sum to the raw (pre-link) prediction exactly.
    // (n, n_features + 1); multiclass models return (n, K, n_features + 1).
    nb::ndarray<nb::numpy, double> pred_contribs(array_2d const &X) const
    {
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
        // The config struct default is 3; surfacing it for non-softmax models
        // would hand callers a plausible-but-meaningless class count.
        return cfg_.dispatch.objective_name == "softmax" ? cfg_.objective.n_classes : 0;
    }

  private:
    // Only classification objectives define probabilities; the mse link
    // inverse is the identity, so without this guard a regression model would
    // return raw margins silently mislabeled as P(class 1).
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

    // The device walk over a Dataset whose bins are already resident on a
    // device. false means the host bin walk runs instead: the Dataset was
    // binned on the host, the ensemble is not dense (oblivious, multiclass),
    // there is no usable device, or the plane declined this call. Only the
    // width-1 predict rides this; the rest of the prediction family walks on
    // the host.
    bool predict_on_device(Dataset const &ds, std::vector<float> &out,
                           size_t num_iteration) const
    {
        auto const &bins  = ds.bins();
        auto const  plane = bins.ingest_plane();
        auto const  in    = booster_->predict_plan_input();
        if (!plane || in.trees.empty())
        {
            return false;
        }
        auto const plan = plan_cache_->predict(in, mappers_);
        return plan && bonsai::cuda_predict(*plan, *plane, bins.n_rows(),
                                            bins.n_features(), num_iteration, out);
    }

    // predict_on_device's twin for TreeSHAP, declining on the same conditions.
    // One difference of substance: the device walk builds its path polynomials
    // in fp32 where the host walk is fp64 throughout, so this route agrees with
    // the host to a tolerance, not bit for bit.
    bool contribs_on_device(Dataset const &ds, std::span<double> out) const
    {
        auto const &bins  = ds.bins();
        auto const  plane = bins.ingest_plane();
        auto const  in    = booster_->predict_plan_input();
        if (!plane || in.trees.empty())
        {
            return false;
        }
        auto const plan = plan_cache_->shap(in, mappers_);
        return plan && bonsai::cuda_pred_contribs(*plan, *plane, bins.n_rows(),
                                                  bins.n_features(), out);
    }

    std::unique_ptr<bonsai::IBooster> booster_;
    bonsai::BinMappers                mappers_;
    bonsai::Config                    cfg_;
    std::vector<float>                eval_history_;
    // Held indirectly so the Model stays movable past the cache's mutex.
    std::shared_ptr<DevicePlanCache> plan_cache_ = std::make_shared<DevicePlanCache>();
};

// The validation set a fit was handed: raw arrays, which the fit bins itself
// once the rounds it has run have paid for the pass, or a Dataset the caller
// binned once, which routes in bin space from the first round and charges the
// fit nothing. The Dataset arm is borrowed, so one validation Dataset serves
// a whole sweep.
using EvalSet = std::variant<std::pair<array_2d, array_1d>, Dataset const *>;

// A classification fit scores against the encoded ids 0..K-1, so labels
// carrying anything else would be measured as the wrong class and steer early
// stopping from there. Regression has no domain to check.
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

// A prebinned eval set must carry the fit's own cuts: the walk inverts each
// stored threshold into a bin id, which names the same cut only under the
// mappers the trees were grown on. The C++ side asserts that contract; this
// is where a Python caller is told how to satisfy it. The raw-array arm lands
// in `owned`; the Dataset arm is borrowed and outlives the call.
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
    // The raw walk and the warm-start seed read the caller's rows; a
    // device-resident Dataset kept none, so it can only serve the fits that
    // never look at them (bin space from round 1, no rounds to seed).
    if (!dataset->has_host_matrix() && (warm_start || !dataset->bins().bins_are_u8()))
    {
        throw std::invalid_argument(
            "this eval_set Dataset was built from device-resident (DLPack) input "
            "and kept no host matrix, but this fit scores it from raw rows (an "
            "init_model warm start, or bins above 255 that have no row-major "
            "mirror). Pass eval_set=(X_valid, y_valid), or build the Dataset "
            "from a host array.");
    }
    check_eval_labels(dataset->loaded().train.labels, cfg);
    return &dataset->loaded().train;
}

// The internal wire format the config layer reads: dotted key, string value.
using ConfigPairs = std::vector<std::pair<std::string, std::string>>;

// Render one override value the way the dotted-key parser reads it back; bool
// is tested first because a Python bool is also an int.
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

// The public params contract rendered to the wire format: a Params (duck-typed
// on to_dict), a dotted-key mapping, or None.
ConfigPairs pairs_from_params(nb::object params)
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
        throw nb::type_error(
            ("params must be a bonsai.Params, a mapping of dotted keys, or None; "
             "got " +
             nb::cast<std::string>(nb::str(params.type().attr("__name__"))) +
             ". For legacy (key, value) pairs, pass dict(pairs).")
                .c_str());
    }
    ConfigPairs      pairs;
    nb::object const items = params.attr("items")();
    for (nb::handle item : items)
    {
        nb::object const entry = nb::borrow<nb::object>(item);
        nb::object const key   = entry[0];
        if (!nb::isinstance<nb::str>(key))
        {
            throw nb::type_error(
                ("params keys must be dotted config keys (str); got " +
                 nb::cast<std::string>(nb::str(key.type().attr("__name__"))))
                    .c_str());
        }
        pairs.emplace_back(nb::cast<std::string>(key), config_str(entry[1]));
    }
    return pairs;
}

// Overrides over the library defaults; a TOML base composes upstream via
// Params.from_toml, so this layer knows one input.
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

// The annotation the generated dataclass field gets, so it must name the C++
// type, not the TOML shape (a float whose default is 0 still annotates float).
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

// The section registry as data, folding the same all_sections tuple dump_toml
// folds: [{'section', 'fields': [{'name', 'type', 'default'}]}] in registry
// order. Feeds the build-time bonsai/_params.py generator.
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
            std::optional<std::string> const &init_model, nb::handle sample_weight)
{
    bonsai::Config const cfg = config_from_params(pairs_from_params(params));
    bonsai::parallel::set_n_threads(cfg.parallel.n_threads);

    // Device-resident input places the fit itself: the matrix is already on a
    // device, so it bins there whatever grower was named, and a CPU grower
    // materializes host bins from the plane on first use.
    auto const [xarg, yarg, warg] = resolve_inputs(
        X, y, sample_weight, cfg.parallel.device_id, "sample_weight", "");

    std::optional<bonsai::io::LoadedBooster> init;
    if (init_model)
    {
        init.emplace(bonsai::io::load_booster(*init_model));
    }

    nb::gil_scoped_release release;

    size_t const             f = xarg.n_features;
    std::vector<std::string> names;
    names.reserve(f);
    for (size_t c = 0; c < f; ++c)
    {
        names.push_back("f" + std::to_string(c));
    }

    bonsai::cli::LoadedTrainValidation loaded;
    loaded.mappers =
        init ? std::move(init->mappers) : fit_mappers(xarg, std::move(names), cfg);
    bonsai::floats_view const wview = warg ? warg->view() : bonsai::floats_view{};
    loaded.train = make_labeled(xarg, yarg.view(), loaded.mappers, cfg,
                                cfg.dispatch.grower_name.starts_with("cuda"), wview);
    std::optional<bonsai::cli::LabeledData> owned;
    auto const *const                       validation =
        resolve_eval_set(eval_set, cfg, loaded.mappers, init.has_value(), owned);

    std::vector<float> history;
    auto               initial = init ? std::move(init->booster) : nullptr;
    // The eval set is optional at this boundary; the fit takes one or none.
    auto booster =
        validation != nullptr
            ? bonsai::cli::train_with_progress(cfg, loaded.train, *validation, {},
                                               std::move(initial), std::ref(history))
            : bonsai::cli::train_with_progress(cfg, loaded.train, {},
                                               std::move(initial), std::ref(history));
    return Model{std::move(booster), std::move(loaded.mappers), cfg,
                 std::move(history)};
}

// Train on a prebuilt Dataset: reuses its binning (skips BinMappers::fit +
// Dataset::bin) and, on GPU, its resident matrix across calls. Only training
// hyperparameters vary per call; binning is fixed by the Dataset, so
// bin_mapper.* overrides are rejected rather than silently ignored; a TOML
// base arrives through Params.from_toml as ordinary pairs, so this one check
// covers every route.
Model train_dataset(nb::object const &params, Dataset const &dataset,
                    std::optional<EvalSet> const     &eval_set,
                    std::optional<std::string> const &init_model)
{
    ConfigPairs const pairs = pairs_from_params(params);
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
    bonsai::Config const cfg = config_from_params(pairs);
    // A device-binned Dataset is resident on one device and the fit adopts
    // that matrix in place; placing the fit elsewhere would mean migrating
    // it behind the user's back.
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
    }
    nb::gil_scoped_release release;
    // The validation set is per-call state; the train side stays the Dataset's
    // own LabeledData (no copy: a copy would also change the address that keys
    // the GPU upload-skip cache).
    std::optional<bonsai::cli::LabeledData> owned;
    auto const *const                       validation = resolve_eval_set(
        eval_set, cfg, dataset.loaded().mappers, init.has_value(), owned);
    std::vector<float> history;
    auto               initial = init ? std::move(init->booster) : nullptr;
    auto const        &loaded  = dataset.loaded();

    auto booster =
        validation != nullptr
            ? bonsai::cli::train_with_progress(cfg, loaded.train, *validation, {},
                                               std::move(initial), std::ref(history))
            : bonsai::cli::train_with_progress(cfg, loaded.train, {},
                                               std::move(initial), std::ref(history));
    return Model{std::move(booster), loaded.mappers, cfg, std::move(history)};
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
        .def("__repr__",
             [](Model const &mo)
             {
                 return "Model(objective='" + mo.objective_name() +
                        "', n_iters=" + std::to_string(mo.n_iters()) + ")";
             });

    // The binning settings default to none, not to literals: unset means the
    // library default, or the reference's value in the `reference=` form.
    constexpr bonsai::ParallelConfig k_parallel_defaults{};
    nb::class_<Dataset>(m, "Dataset")
        .def(
            nb::init<nb::handle, nb::handle, nb::handle, std::optional<int>,
                     std::optional<size_t>, std::optional<uint64_t>, std::optional<int>,
                     std::optional<std::map<size_t, array_1d>> const &,
                     std::optional<std::string> const &, std::optional<uint32_t>,
                     uint32_t, Dataset const *>(),
            nb::arg("X"), nb::arg("y"), nb::arg("weight") = nb::none(),
            nb::arg("max_bin") = nb::none(), nb::arg("n_samples") = nb::none(),
            nb::arg("seed") = nb::none(), nb::arg("min_data_in_bin") = nb::none(),
            nb::arg("bin_edges") = nb::none(), nb::arg("device") = nb::none(),
            nb::arg("device_id")        = nb::none(),
            nb::arg("n_threads")        = k_parallel_defaults.n_threads,
            nb::arg("reference").none() = nb::none(),
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
            "setting one to a different value raises, as ``bin_edges`` does "
            "at all. ``device`` and ``device_id`` are inherited the same "
            "way, so a validation set lands beside its training set unless "
            "placed explicitly.")
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
        .def_prop_ro("n_rows", &Dataset::n_rows, "Rows in the binned matrix.")
        .def_prop_ro("n_features", &Dataset::n_features,
                     "Feature columns in the binned matrix.")
        .def_prop_ro(
            "shape",
            [](Dataset const &d) { return nb::make_tuple(d.n_rows(), d.n_features()); },
            "(n_rows, n_features), the way an array reports it. IDE variable "
            "explorers read it for their size column.")
        .def("__len__", [](Dataset const &d) { return d.n_rows(); })
        .def("__repr__",
             [](Dataset const &d)
             {
                 return "Dataset(n_rows=" + std::to_string(d.n_rows()) +
                        ", n_features=" + std::to_string(d.n_features()) +
                        ", device='" + d.device() + "')";
             });

    // Dataset overload first: the array overload takes X and y untyped (a numpy
    // array or any DLPack producer), so it would otherwise shadow a Dataset
    // call that also passes an eval set.
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
          nb::arg("sample_weight") = nb::none(),
          nb::sig("def train(params: bonsai.params.Params | "
                  "collections.abc.Mapping[str, object] | None, X: object, "
                  "y: object, eval_set: tuple[Annotated[NDArray[numpy.float32], "
                  "dict(shape=(None, None), order='C', device='cpu', "
                  "writable=False)], Annotated[NDArray[numpy.float32], "
                  "dict(shape=(None,), order='C', device='cpu', writable=False)]] "
                  "| Dataset | None = None, init_model: str | None = None, "
                  "sample_weight: object | None = None) -> Model"),
          "Train a booster on row-major float32 features.\n"
          "\n"
          "Parameters\n"
          "----------\n"
          "params : Params, Mapping, or None\n"
          "    Overrides over the library defaults: a ``bonsai.Params``, a "
          "``{'tree.max_depth': 8}`` dotted-key mapping, or ``None`` for no "
          "overrides. A TOML base composes here too, "
          "``Params.from_toml(path) | overrides``.\n"
          "X : float32 array, shape (n_rows, n_features)\n"
          "    Row-major features. May be a CUDA array supporting DLPack "
          "(cupy, torch, jax): X is then binned on the GPU in place, with no "
          "host round trip.\n"
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
        "values — only the stated keys, never the resolved whole. Feeds "
        "Params.from_toml/from_model; not public API.");
    m.def("_params_schema", &params_schema,
          "The config section registry as data: [{'section', 'fields': "
          "[{'name', 'type', 'default'}]}] in registry order. Feeds the "
          "build-time bonsai/_params.py generator; not public API.");
}

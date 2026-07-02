#include "bonsai/io/model.hpp"
#include "nlohmann/adl_serializer.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

// nlohmann/json v3.11 doesn't ship std::optional<T> conversion; add a tiny
// adl_serializer so the NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE macros below can
// see optional fields (e.g. DataConfig::missing_sentinel).
template <typename T> struct nlohmann::adl_serializer<std::optional<T>>
{
    static void to_json(json &j, std::optional<T> const &opt)
    {
        if (opt.has_value())
        {
            j = *opt;
        }
        else
        {
            j = nullptr;
        }
    }
    static void from_json(json const &j, std::optional<T> &opt)
    {
        if (j.is_null())
        {
            opt = std::nullopt;
        }
        else
        {
            opt = j.get<T>();
        }
    }
};

#include "bonsai/bin_mapper.hpp"
#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/dispatch_config.hpp"
#include "bonsai/config/sampler_config.hpp"
#include "bonsai/registry/configurations.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/tree.hpp"

namespace bonsai
{
// Free-function (to|from)_json for the POD records. Macros expand in
// namespace bonsai so ADL finds them on the nested types. Member order
// here = JSON key order; renaming a member changes the on-disk schema.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenseTree::Node, feature_id, threshold_or_value,
                                   left, right, default_left)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenseTree::Params, depth, n_leaves)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ObliviousTree::LevelSplit, feature_id, threshold,
                                   default_left)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ObliviousTree::Params, depth, n_leaves)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DataConfig, train, valid, test, format, header,
                                   label_column, weight_column, ignore_columns,
                                   missing_nan, missing_sentinel)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BinMapperConfig, max_bin, n_samples, seed,
                                   min_data_in_bin)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TreeConfig, min_child_hess, min_gain_to_split,
                                   lambda_l2, lambda_l1, max_depth, min_data_in_leaf,
                                   max_leaves, feature_fraction, feature_seed,
                                   monotone_constraints, interaction_constraints)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BoosterConfig, n_iters, learning_rate, random_seed,
                                   log_intervals, early_stopping_rounds,
                                   dart_drop_rate)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DispatchConfig, objective_name, grower_name,
                                   sampler_name)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SamplerConfig, subsample, top_rate, other_rate)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MetricsConfig, fit, eval)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ParallelConfig, n_threads)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ObjectiveConfig, huber_delta, quantile_alpha)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config, data, bin_mapper, tree_config, sampler,
                                   booster_config, dispatch, metrics, parallel,
                                   objective)
} // namespace bonsai

namespace bonsai::io
{

namespace
{

using json = nlohmann::json;

constexpr std::string_view k_magic          = "bonsai01";
constexpr uint32_t         k_format_version = 4;

// ---- Tree <-> JSON --------------------------------------------------------

json tree_to_json(DenseTree const &t)
{
    json out     = t.params();
    out["nodes"] = t.nodes();
    return out;
}

template <typename TreeT> TreeT tree_from_json(json const &j);

template <> DenseTree tree_from_json<DenseTree>(json const &j)
{
    return DenseTree{j.at("nodes").get<DenseTree::Nodes>(), j.get<DenseTree::Params>()};
}

json tree_to_json(ObliviousTree const &t)
{
    json out      = t.params();
    out["splits"] = t.splits();
    out["leaves"] = t.leaf_table();
    return out;
}

template <> ObliviousTree tree_from_json<ObliviousTree>(json const &j)
{
    return ObliviousTree{
        j.at("splits").get<ObliviousTree::LevelSplits>(),
        j.at("leaves").get<ObliviousTree::LeafTable>(),
    };
}

// ---- BinMappers <-> JSON --------------------------------------------------

json mappers_to_json(BinMappers const &mappers)
{
    json       out   = json::array();
    auto const names = mappers.feature_names();
    for (size_t i = 0; i < mappers.size(); ++i)
    {
        json m;
        m["name"]     = std::string{names[i]};
        json cuts_arr = json::array();
        for (float c : mappers[i].cuts())
        {
            cuts_arr.push_back(c);
        }
        m["cuts"] = std::move(cuts_arr);
        out.push_back(std::move(m));
    }
    return out;
}

BinMappers mappers_from_json(json const &j)
{
    std::vector<BinMapper>   mappers;
    std::vector<std::string> names;
    mappers.reserve(j.size());
    names.reserve(j.size());
    for (auto const &m : j)
    {
        std::vector<float> cuts;
        cuts.reserve(m.at("cuts").size());
        for (auto const &c : m.at("cuts"))
        {
            cuts.push_back(c.get<float>());
        }
        mappers.push_back(BinMapper::from_cuts(std::move(cuts)));
        names.push_back(m.at("name").get<std::string>());
    }
    return BinMappers::from_mappers(std::move(mappers), std::move(names));
}

// ---- Concrete Booster<O,G,Sa> handlers ------------------------------------

template <typename B> bool try_save_as(IBooster const &booster, json &out)
{
    auto const *concrete = dynamic_cast<B const *>(&booster);
    if (concrete == nullptr)
    {
        return false;
    }
    out["init_score"] = concrete->init_score();
    json trees        = json::array();
    for (auto const &t : concrete->trees())
    {
        trees.push_back(tree_to_json(t));
    }
    out["trees"] = std::move(trees);
    return true;
}

template <typename B> bool try_load_into(IBooster &booster, json const &j)
{
    auto *concrete = dynamic_cast<B *>(&booster);
    if (concrete == nullptr)
    {
        return false;
    }
    using TreeT = typename B::tree_type;
    std::vector<TreeT> trees;
    trees.reserve(j.at("trees").size());
    for (auto const &tj : j.at("trees"))
    {
        trees.push_back(tree_from_json<TreeT>(tj));
    }
    concrete->load_state(std::move(trees), j.at("init_score").get<float>());
    return true;
}

bool save_dispatch(IBooster const &booster, DispatchConfig const &disp, json &out)
{
    return with_combo_matching(
        disp,
        [&]<typename Combo>() { return try_save_as<BoosterFor<Combo>>(booster, out); });
}

bool load_dispatch(IBooster &booster, DispatchConfig const &disp, json const &j)
{
    return with_combo_matching(
        disp,
        [&]<typename Combo>() { return try_load_into<BoosterFor<Combo>>(booster, j); });
}

// ---- File I/O -------------------------------------------------------------

std::vector<uint8_t> read_file(std::string const &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("model: cannot open '" + path + "'");
    }
    return std::vector<uint8_t>{std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>()};
}

void write_file(std::string const &path, std::span<uint8_t const> bytes)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("model: cannot write '" + path + "'");
    }
    out.write(reinterpret_cast<char const *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

} // namespace

void save_booster(IBooster const &booster, std::string const &path,
                  BinMappers const &mappers, Config const &cfg)
{
    json root;
    root["magic"]       = std::string{k_magic};
    root["version"]     = k_format_version;
    root["config"]      = cfg;
    root["bin_mappers"] = mappers_to_json(mappers);

    if (!save_dispatch(booster, cfg.dispatch, root))
    {
        throw std::runtime_error(
            "model: save_booster: no impl for (" + cfg.dispatch.objective_name + ", " +
            cfg.dispatch.grower_name + ", " + cfg.dispatch.sampler_name + ")");
    }

    auto const bytes = json::to_msgpack(root);
    write_file(path, bytes);
}

LoadedBooster load_booster(std::string const &path)
{
    auto const bytes = read_file(path);
    auto const root  = json::from_msgpack(bytes);

    if (root.at("magic").get<std::string>() != k_magic)
    {
        throw std::runtime_error("model: bad magic in '" + path + "'");
    }
    if (root.at("version").get<uint32_t>() != k_format_version)
    {
        throw std::runtime_error("model: unsupported format version");
    }

    LoadedBooster out;
    out.cfg     = root.at("config").get<Config>();
    out.mappers = mappers_from_json(root.at("bin_mappers"));
    out.booster = make_booster(out.cfg);

    if (!load_dispatch(*out.booster, out.cfg.dispatch, root))
    {
        throw std::runtime_error(
            "model: load_booster: dispatch triple unknown after make_booster");
    }
    return out;
}

} // namespace bonsai::io

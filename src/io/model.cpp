#include "bonsai/io/model.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/dispatch_config.hpp"
#include "bonsai/config/sampler_config.hpp"
#include "bonsai/config/sections/all.hpp"
#include "bonsai/registry/configurations.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/tree.hpp"

namespace bonsai
{
// namespace bonsai so ADL finds them on the nested types. Member order
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenseTree::Node, feature_id, threshold_or_value,
                                   left, right, default_left)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenseTree::Params, depth, n_leaves)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ObliviousTree::LevelSplit, feature_id, threshold,
                                   default_left)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ObliviousTree::Params, depth, n_leaves)

namespace
{
using json        = nlohmann::json;
namespace cfg_int = config::internal;

inline constexpr std::size_t k_num_sections =
    std::tuple_size_v<std::remove_cvref_t<decltype(cfg_int::all_sections)>>;

constexpr std::array k_persist_skip{std::string_view{"device_id"}};

constexpr bool persist_skip(std::string_view leaf)
{
    return std::ranges::any_of(k_persist_skip,
                               [leaf](std::string_view skip) { return skip == leaf; });
} // namespace cfg_int

template <typename Sub, std::size_t... I>
constexpr std::size_t section_index_impl(std::index_sequence<I...>)
{
    std::size_t idx = k_num_sections;
    ((std::is_same_v<typename std::remove_cvref_t<decltype(std::get<I>(
                         cfg_int::all_sections))>::sub_type,
                     Sub>
          ? (idx = I)
          : std::size_t{0}),
     ...);
    return idx;
}

template <typename Sub>
constexpr std::size_t section_index =
    section_index_impl<Sub>(std::make_index_sequence<k_num_sections>{});

template <typename Sub>
concept ConfigSubStruct = section_index<Sub> < k_num_sections;

template <typename Sub> void section_to_json(json &j, Sub const &s)
{
    std::apply(
        [&](auto const &...f)
        {
            auto emit = [&](auto const &fld)
            {
                if (!persist_skip(fld.leaf))
                {
                    j[std::string{fld.leaf}] = s.*(fld.member);
                }
            };
            (emit(f), ...);
        },
        std::get<section_index<Sub>>(cfg_int::all_sections).fields);
    if constexpr (std::is_same_v<Sub, DataConfig>)
    {
        j["missing_nan"]      = true;
        j["missing_sentinel"] = nullptr;
    }
}

template <typename Sub> void section_from_json(json const &j, Sub &s)
{
    std::apply(
        [&](auto const &...f)
        {
            auto load = [&](auto const &fld)
            {
                if (!persist_skip(fld.leaf))
                {
                    j.at(std::string{fld.leaf}).get_to(s.*(fld.member));
                }
            };
            (load(f), ...);
        },
        std::get<section_index<Sub>>(cfg_int::all_sections).fields);
}
} // namespace

template <ConfigSubStruct Sub> void to_json(json &j, Sub const &s)
{
    section_to_json(j, s);
}
template <ConfigSubStruct Sub> void from_json(json const &j, Sub &s)
{
    section_from_json(j, s);
}

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
constexpr uint32_t         k_format_version = 7;

json tree_to_json(DenseTree const &t)
{
    json out      = t.params();
    out["nodes"]  = t.nodes();
    out["gains"]  = t.split_gains();
    out["covers"] = t.covers();
    return out;
}

template <typename TreeT> TreeT tree_from_json(json const &j, size_t n_features);

constexpr size_t k_max_oblivious_depth = 31;

void validate_dense_tree(DenseTree::Nodes const &nodes, DenseTree::Params const &p,
                         size_t n_features)
{
    if (nodes.empty())
    {
        throw std::runtime_error("dense tree has no nodes");
    }
    if (p.depth > nodes.size())
    {
        throw std::runtime_error("dense tree depth exceeds its node count");
    }
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        DenseTree::Node const &n = nodes[i];
        if (n.feature_id == DenseTree::k_leaf_flag)
        {
            continue;
        }
        if (n.feature_id >= n_features)
        {
            throw std::runtime_error("split names a feature the model lacks");
        }
        if (n.left <= i || n.right <= i || n.left >= nodes.size() ||
            n.right >= nodes.size())
        {
            throw std::runtime_error("node links must point forward");
        }
    }
    std::vector<size_t> depth(nodes.size(), 0);
    for (size_t i = nodes.size(); i-- > 0;)
    {
        DenseTree::Node const &n = nodes[i];
        if (n.feature_id != DenseTree::k_leaf_flag)
        {
            depth[i] = 1 + std::max(depth[n.left], depth[n.right]);
        }
    }
    if (p.depth < depth[0])
    {
        throw std::runtime_error("stated depth understates the tree");
    }
}

template <> DenseTree tree_from_json<DenseTree>(json const &j, size_t n_features)
{
    auto       nodes  = j.at("nodes").get<DenseTree::Nodes>();
    auto const params = j.get<DenseTree::Params>();
    validate_dense_tree(nodes, params, n_features);
    return DenseTree{std::move(nodes), params, j.at("gains").get<std::vector<float>>(),
                     j.at("covers").get<std::vector<float>>()};
}

json tree_to_json(ObliviousTree const &t)
{
    json out      = t.params();
    out["splits"] = t.splits();
    out["leaves"] = t.leaf_table();
    out["gains"]  = t.level_gains();
    out["covers"] = t.leaf_covers();
    return out;
}

template <>
ObliviousTree tree_from_json<ObliviousTree>(json const &j, size_t n_features)
{
    auto splits = j.at("splits").get<ObliviousTree::LevelSplits>();
    auto leaves = j.at("leaves").get<ObliviousTree::LeafTable>();
    if (splits.size() > k_max_oblivious_depth)
    {
        throw std::runtime_error("oblivious depth exceeds the leaf-table cap");
    }
    if (leaves.size() != (size_t{1} << splits.size()))
    {
        throw std::runtime_error("leaf table disagrees with the split count");
    }
    for (auto const &sp : splits)
    {
        if (sp.feature_id >= n_features)
        {
            throw std::runtime_error("split names a feature the model lacks");
        }
    }
    return ObliviousTree{
        std::move(splits),
        std::move(leaves),
        j.at("gains").get<std::vector<float>>(),
        j.value("covers", std::vector<float>{}),
    };
}

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

template <typename B> bool try_save_as(IBooster const &booster, json &out)
{
    auto const *concrete = dynamic_cast<B const *>(&booster);
    if (concrete == nullptr)
    {
        return false;
    }
    if constexpr (requires { concrete->init_scores(); })
    {
        out["init_scores"] = concrete->init_scores(); // multiclass
    }
    else
    {
        out["init_score"] = concrete->init_score();
    }
    json trees = json::array();
    for (auto const &t : concrete->trees())
    {
        trees.push_back(tree_to_json(t));
    }
    out["trees"] = std::move(trees);
    return true;
}

template <typename B>
bool try_load_into(IBooster &booster, json const &j, size_t n_features)
{
    auto *concrete = dynamic_cast<B *>(&booster);
    if (concrete == nullptr)
    {
        return false;
    }
    using TreeT = typename B::tree_type;
    std::vector<TreeT> trees;
    auto const        &tree_array = j.at("trees");
    trees.reserve(tree_array.size());
    for (size_t ti = 0; ti < tree_array.size(); ++ti)
    {
        try
        {
            trees.push_back(tree_from_json<TreeT>(tree_array[ti], n_features));
        }
        catch (std::runtime_error const &e)
        {
            throw std::runtime_error("model: tree " + std::to_string(ti) + ": " +
                                     e.what());
        }
    }
    if constexpr (requires(std::vector<float> v) {
                      concrete->load_state(std::move(trees), std::move(v));
                  })
    {
        concrete->load_state(std::move(trees),
                             j.at("init_scores").get<std::vector<float>>());
    }
    else
    {
        concrete->load_state(std::move(trees), j.at("init_score").get<float>());
    }
    return true;
}

bool save_dispatch(IBooster const &booster, DispatchConfig const &disp, json &out)
{
    return with_combo_matching(
        disp,
        [&]<typename Combo>() { return try_save_as<BoosterFor<Combo>>(booster, out); });
}

bool load_dispatch(IBooster &booster, DispatchConfig const &disp, json const &j,
                   size_t n_features)
{
    return with_combo_matching(
        disp, [&]<typename Combo>()
        { return try_load_into<BoosterFor<Combo>>(booster, j, n_features); });
}

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

    if (!load_dispatch(*out.booster, out.cfg.dispatch, root, out.mappers.size()))
    {
        throw std::runtime_error(
            "model: load_booster: dispatch triple unknown after make_booster");
    }
    return out;
}

} // namespace bonsai::io

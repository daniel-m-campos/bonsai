#include "bonsai/io/model.hpp"

#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/dispatch_config.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/registry/names.hpp"
#include "bonsai/registry/typelists.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/typelist.hpp"

namespace bonsai::io
{

namespace
{

using json = nlohmann::json;

std::string_view constexpr k_magic  = "bonsai01";
uint32_t constexpr k_format_version = 1;

// ---- Tree <-> JSON --------------------------------------------------------

json tree_to_json(DenseTree const &t)
{
    json out        = json::object();
    out["depth"]    = t.params().depth;
    out["n_leaves"] = t.params().n_leaves;
    json nodes      = json::array();
    for (auto const &node : t.nodes())
    {
        std::visit(
            [&nodes](auto const &n)
            {
                using N = std::decay_t<decltype(n)>;
                json j;
                if constexpr (std::is_same_v<N, DenseTree::InternalNode>)
                {
                    j["kind"]         = "internal";
                    j["feature_id"]   = n.feature_id;
                    j["threshold"]    = n.threshold;
                    j["left"]         = n.left;
                    j["right"]        = n.right;
                    j["default_left"] = n.default_left;
                }
                else
                {
                    j["kind"]  = "leaf";
                    j["value"] = n.value;
                }
                nodes.push_back(std::move(j));
            },
            node);
    }
    out["nodes"] = std::move(nodes);
    return out;
}

DenseTree tree_from_json(json const &j)
{
    DenseTree::Nodes nodes;
    nodes.reserve(j.at("nodes").size());
    for (auto const &nj : j.at("nodes"))
    {
        std::string const kind = nj.at("kind").get<std::string>();
        if (kind == "internal")
        {
            DenseTree::InternalNode n{};
            n.feature_id   = nj.at("feature_id").get<feature_id_t>();
            n.threshold    = nj.at("threshold").get<float>();
            n.left         = nj.at("left").get<node_id_t>();
            n.right        = nj.at("right").get<node_id_t>();
            n.default_left = nj.at("default_left").get<bool>();
            nodes.emplace_back(n);
        }
        else if (kind == "leaf")
        {
            nodes.emplace_back(DenseTree::LeafNode{nj.at("value").get<float>()});
        }
        else
        {
            throw std::runtime_error("model: unknown tree node kind '" + kind + "'");
        }
    }
    DenseTree::Params params{};
    params.depth    = j.at("depth").get<size_t>();
    params.n_leaves = j.at("n_leaves").get<size_t>();
    return DenseTree{std::move(nodes), params};
}

// ---- BinMappers <-> JSON --------------------------------------------------

json mappers_to_json(BinMappers const &mappers)
{
    json out         = json::array();
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
    std::vector<BinMapper> mappers;
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
    std::vector<DenseTree> trees;
    trees.reserve(j.at("trees").size());
    for (auto const &tj : j.at("trees"))
    {
        trees.push_back(tree_from_json(tj));
    }
    concrete->load_state(std::move(trees), j.at("init_score").get<float>());
    return true;
}

using Configurations = cartesian_product_t<Objectives, Growers, Samplers>;

template <typename Combo>
using BoosterFor =
    Booster<type_at_t<0, Combo>, type_at_t<1, Combo>, type_at_t<2, Combo>>;

bool save_dispatch(IBooster const &booster, DispatchConfig const &disp, json &out)
{
    bool ok = false;
    for_each_type<Configurations>(
        [&]<typename Combo>()
        {
            if (ok)
            {
                return;
            }
            using O  = type_at_t<0, Combo>;
            using G  = type_at_t<1, Combo>;
            using Sa = type_at_t<2, Combo>;
            static_assert(HasName<O>, "Objective needs impl_name specialization");
            static_assert(HasName<G>, "Grower needs impl_name specialization");
            static_assert(HasName<Sa>, "Sampler needs impl_name specialization");
            if (disp.objective_name != impl_name<O>::value)
            {
                return;
            }
            if (disp.grower_name != impl_name<G>::value)
            {
                return;
            }
            if (disp.sampler_name != impl_name<Sa>::value)
            {
                return;
            }
            ok = try_save_as<BoosterFor<Combo>>(booster, out);
        });
    return ok;
}

bool load_dispatch(IBooster &booster, DispatchConfig const &disp, json const &j)
{
    bool ok = false;
    for_each_type<Configurations>(
        [&]<typename Combo>()
        {
            if (ok)
            {
                return;
            }
            using O  = type_at_t<0, Combo>;
            using G  = type_at_t<1, Combo>;
            using Sa = type_at_t<2, Combo>;
            static_assert(HasName<O>, "Objective needs impl_name specialization");
            static_assert(HasName<G>, "Grower needs impl_name specialization");
            static_assert(HasName<Sa>, "Sampler needs impl_name specialization");
            if (disp.objective_name != impl_name<O>::value)
            {
                return;
            }
            if (disp.grower_name != impl_name<G>::value)
            {
                return;
            }
            if (disp.sampler_name != impl_name<Sa>::value)
            {
                return;
            }
            ok = try_load_into<BoosterFor<Combo>>(booster, j);
        });
    return ok;
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

void write_file(std::string const &path, std::vector<uint8_t> const &bytes)
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
                  BinMappers const &mappers, DispatchConfig const &dispatch,
                  float learning_rate)
{
    json root;
    root["magic"]                 = std::string{k_magic};
    root["version"]               = k_format_version;
    root["dispatch"]["objective"] = dispatch.objective_name;
    root["dispatch"]["grower"]    = dispatch.grower_name;
    root["dispatch"]["sampler"]   = dispatch.sampler_name;
    root["learning_rate"]         = learning_rate;
    root["bin_mappers"]           = mappers_to_json(mappers);

    if (!save_dispatch(booster, dispatch, root))
    {
        throw std::runtime_error("model: save_booster: no impl for (" +
                                 dispatch.objective_name + ", " + dispatch.grower_name +
                                 ", " + dispatch.sampler_name + ")");
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
    out.dispatch.objective_name =
        root.at("dispatch").at("objective").get<std::string>();
    out.dispatch.grower_name  = root.at("dispatch").at("grower").get<std::string>();
    out.dispatch.sampler_name = root.at("dispatch").at("sampler").get<std::string>();
    out.learning_rate         = root.at("learning_rate").get<float>();
    out.mappers               = mappers_from_json(root.at("bin_mappers"));

    // Reconstruct the booster via make_booster with a synthesized config
    // carrying the on-disk dispatch triple and learning rate.
    Config cfg;
    cfg.dispatch                     = out.dispatch;
    cfg.booster_config.learning_rate = out.learning_rate;
    out.booster                      = make_booster(cfg);

    if (!load_dispatch(*out.booster, out.dispatch, root))
    {
        throw std::runtime_error(
            "model: load_booster: dispatch triple unknown after make_booster");
    }
    return out;
}

} // namespace bonsai::io

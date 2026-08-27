#include "bonsai/monotone.hpp"

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace bonsai
{

namespace
{

struct LevelSet
{
    double weighted_sum;
    double weight;
    double plain_sum;
    size_t count;

    double value() const
    {
        return weight > 0.0 ? weighted_sum / weight
                            : plain_sum / static_cast<double>(count);
    }

    void absorb(LevelSet const &other)
    {
        weighted_sum += other.weighted_sum;
        weight += other.weight;
        plain_sum += other.plain_sum;
        count += other.count;
    }
};

std::vector<size_t> bit_positions(std::span<int const> level_directions,
                                  bool                 constrained)
{
    std::vector<size_t> positions;
    size_t const        depth = level_directions.size();
    for (size_t level = 0; level < depth; ++level)
    {
        if ((level_directions[level] != 0) == constrained)
        {
            positions.push_back(depth - 1 - level);
        }
    }
    return positions;
}

std::vector<int> signs_of(std::span<int const> level_directions)
{
    std::vector<int> signs;
    for (int const direction : level_directions)
    {
        if (direction != 0)
        {
            signs.push_back(direction);
        }
    }
    return signs;
}

size_t scatter(size_t bits, std::span<size_t const> positions)
{
    size_t index = 0;
    for (size_t slot = 0; slot < positions.size(); ++slot)
    {
        if (((bits >> slot) & 1U) != 0U)
        {
            index |= size_t{1} << positions[slot];
        }
    }
    return index;
}

size_t oriented_leaf(size_t rank, std::span<size_t const> positions,
                     std::span<int const> signs)
{
    size_t const slots = positions.size();
    size_t       bits  = 0;
    for (size_t slot = 0; slot < slots; ++slot)
    {
        bool const ordered = ((rank >> (slots - 1 - slot)) & 1U) != 0U;
        if (signs[slot] > 0 ? ordered : !ordered)
        {
            bits |= size_t{1} << slot;
        }
    }
    return scatter(bits, positions);
}

void isotonic(std::span<size_t const> chain, std::span<float const> weights,
              std::span<float> leaf_table)
{
    std::vector<LevelSet> blocks;
    blocks.reserve(chain.size());
    for (size_t const leaf : chain)
    {
        double const w = weights[leaf];
        double const y = leaf_table[leaf];
        LevelSet block{.weighted_sum = w * y, .weight = w, .plain_sum = y, .count = 1};
        while (!blocks.empty() && blocks.back().value() > block.value())
        {
            block.absorb(blocks.back());
            blocks.pop_back();
        }
        blocks.push_back(block);
    }
    size_t position = 0;
    for (auto const &block : blocks)
    {
        auto const value = static_cast<float>(block.value());
        for (size_t k = 0; k < block.count; ++k)
        {
            leaf_table[chain[position]] = value;
            ++position;
        }
    }
}

} // namespace

void project_monotone(std::span<int const>   level_directions,
                      std::span<float const> weights, std::span<float> leaf_table)
{
    assert(weights.size() == leaf_table.size());
    assert(leaf_table.empty() || leaf_table.size() == size_t{1}
                                                          << level_directions.size());
    auto const constrained = bit_positions(level_directions, true);
    if (constrained.empty() || leaf_table.empty())
    {
        return;
    }
    auto const free_bits = bit_positions(level_directions, false);
    auto const signs     = signs_of(level_directions);

    size_t const        chain_len = size_t{1} << constrained.size();
    size_t const        groups    = size_t{1} << free_bits.size();
    std::vector<size_t> chain(chain_len);
    for (size_t group = 0; group < groups; ++group)
    {
        size_t const base = scatter(group, free_bits);
        for (size_t rank = 0; rank < chain_len; ++rank)
        {
            chain[rank] = base | oriented_leaf(rank, constrained, signs);
        }
        isotonic(chain, weights, leaf_table);
    }
}

} // namespace bonsai

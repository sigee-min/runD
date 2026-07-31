#pragma once

#include <kernel/program/compute/limit.hpp>

#include <cstddef>

namespace rund::compute::graph {

inline constexpr std::size_t NodeCapacity =
    static_cast<std::size_t>(kernel::kMaxGraphNodeCount);
inline constexpr std::size_t ValueCapacity =
    static_cast<std::size_t>(kernel::kMaxGraphValueCount);
inline constexpr std::size_t BuffersPerNode =
    static_cast<std::size_t>(kernel::kMaxGraphBuffersPerNode);
inline constexpr std::size_t OutputCapacity =
    static_cast<std::size_t>(kernel::kMaxGraphOutputCount);

} // namespace rund::compute::graph

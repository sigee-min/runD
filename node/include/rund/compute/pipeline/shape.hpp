#pragma once

#include <rund/compute/pipeline/capacity.hpp>

#include <cstddef>

namespace rund::compute::detail {

inline constexpr std::size_t PipelineLeafCapacity = 32u;
inline constexpr std::size_t PipelineTransferCapacity =
    PipelineLeafCapacity * 2u;
inline constexpr std::size_t PipelineBindingCapacity =
    PipelineIterationCapacity * PipelineLeafCapacity;
inline constexpr std::size_t PipelineRouteBindingCapacity =
    PipelineRouteCapacity * PipelineLeafCapacity;
inline constexpr std::size_t PipelineResourceCapacity = 2048u;

} // namespace rund::compute::detail

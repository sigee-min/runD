#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute {

inline constexpr std::size_t PipelineStepCapacity = 64u;
inline constexpr std::size_t PipelineIterationCapacity = 1024u;
inline constexpr std::size_t PipelineInnerIterationCapacity = 1024u;
inline constexpr std::size_t PipelineSealedRepetitionCapacity = 1024u;
inline constexpr std::uint64_t PipelineGenerationCapacity =
    std::numeric_limits<std::uint32_t>::max();
inline constexpr std::size_t PipelineRouteCapacity =
    2u * PipelineIterationCapacity + 3u;

} // namespace rund::compute

#pragma once

#include "../publication.hpp"

#include <rund/compute/resource/plan.hpp>

namespace rund::compute::detail {

[[nodiscard]] Status prove_sealed_repetitions(
    const PipelineBuildState &build,
    std::span<const resource::Resource> resource_shapes,
    std::span<const resource::Access> resource_accesses,
    std::span<const std::uint8_t> external_resources,
    std::span<const resource::Access> publication_accesses);

} // namespace rund::compute::detail

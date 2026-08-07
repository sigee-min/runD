#pragma once

#include "prepare.hpp"
#include "resource.hpp"

#include "../../../accel/kernel/publication.hpp"

#include <cstdint>
#include <span>

namespace rund::compute::detail {

class PipelineHash;

[[nodiscard]] Result<PipelineScheduleSuccess>
plan_pipeline_publications(const PipelineBuildState &build,
                           std::span<const std::uint32_t> window_states,
                           PipelineScheduleResources &resources,
                           PipelineMemoryPlan &plan);

// Sole compute-to-backend projection for publication meaning. Native handles
// are resolved separately and never participate in this identity.
[[nodiscard]] node::accel::detail::PreparedKernelPublicationIdentity
project_pipeline_publication_identity(
    const PipelinePublicationPlan &publication,
    const PipelineWindowControl &control, std::uint32_t outer_bound) noexcept;

// Sole v3 public Pipeline-fingerprint serializer for publication meaning.
// Terminal bank/parity detail remains excluded for compatibility: the
// canonical final source ordinal is serialized in the legacy field position.
[[nodiscard]] bool mix_pipeline_publication_public_identity(
    PipelineHash &hash, const PipelinePublicationPlan &publication,
    const PipelineWindowControl &control) noexcept;

} // namespace rund::compute::detail

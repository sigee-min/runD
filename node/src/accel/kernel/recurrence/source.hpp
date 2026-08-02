#pragma once

#include "../recurrence.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] MapRecurrenceSourcePlan PlanMapRecurrenceSource(
    const rund::kernel::LoweringArtifact &artifact, std::uint64_t input_count,
    std::uint64_t output_count,
    std::span<const std::uint64_t> history_pitch_bytes = {}) noexcept;

// Builds a minimal executable recurrence artifact directly from the canonical
// owner. Canonical IR is never copied. `source_reserve_upper_bytes` lets a
// backend extend the proved recurrence upper to its final specialization/ABI
// upper, so the source text is materialized once at the capacity that will be
// moved into its cache; metadata remains a separately bounded ephemeral copy.
[[nodiscard]] bool MaterializeMapRecurrenceArtifact(
    const rund::kernel::LoweringArtifact &canonical,
    const MapRecurrenceSourcePlan &plan, std::uint64_t input_count,
    std::uint64_t output_count,
    std::span<const std::uint64_t> history_pitch_bytes,
    rund::kernel::LoweringArtifact &artifact,
    std::uint64_t source_reserve_upper_bytes = 0u);

[[nodiscard]] bool TransformSource(rund::kernel::LoweringArtifact &artifact,
                                   std::uint64_t input_count,
                                   std::uint64_t output_count,
                                   std::span<const std::uint64_t>
                                       history_pitch_bytes = {});

} // namespace rund::node::accel::detail

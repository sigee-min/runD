#pragma once

#include "../model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail {
struct VirtualPipelineState;
}

namespace rund::compute::detail::graph_reduce {

[[nodiscard]] std::uint64_t
graph_identity(const VirtualRunProjection &) noexcept;
[[nodiscard]] const residency::GraphMaterialization *
graph_materialization(const VirtualRunProjection &, std::uint32_t) noexcept;
[[nodiscard]] bool graph_resource_index(const residency::TiledGraphPlan &,
                                        std::uint32_t, std::size_t &) noexcept;
[[nodiscard]] bool capture_stage_effects(const residency::TiledGraphPlan &,
                                         residency::EpochLease,
                                         StageEffects &) noexcept;
void apply_stage_effects(Ticket &, const StageEffects &) noexcept;
[[nodiscard]] residency::FrameRegion
resource_region(const residency::Pool &, const residency::TiledGraphPlan &,
                std::uint32_t, std::uint32_t) noexcept;
[[nodiscard]] bool resource_cache_regions(
    const residency::Pool &, const residency::TiledGraphPlan &, std::uint32_t,
    std::array<residency::FrameRegion, residency::Pool::BankCount> &,
    std::size_t &) noexcept;
[[nodiscard]] bool copy_keys(std::span<const residency::PageUse>,
                             residency::GraphMaterialization,
                             std::span<residency::CacheKey>) noexcept;
[[nodiscard]] bool project_stage_scratch(const residency::TiledGraphPlan &,
                                         const VirtualRunProjection &,
                                         const residency::Pool &,
                                         const Ticket &, std::size_t,
                                         std::uint32_t,
                                         StageScratch &) noexcept;
[[nodiscard]] Status
project_ticket(VirtualPipelineState &, const VirtualRunProjection &,
               const residency::TiledGraphPlan &, const residency::Pool &,
               std::uint64_t, std::size_t, std::uint32_t, Ticket &) noexcept;
[[nodiscard]] residency::GraphPersistIdentity
persist_identity(const VirtualRunProjection &,
                 const residency::TiledGraphPlan &, std::size_t, const Ticket &,
                 residency::Identity) noexcept;

// The ticket projection is an invocation-local value construction. This seam
// is declarations-only: immutable run/plan inputs and the caller's Ticket
// remain the sole authorities, while the root facade keeps the entry point
// stable.
[[nodiscard]] Status project_ticket_impl(VirtualPipelineState &,
                                         const VirtualRunProjection &,
                                         const residency::TiledGraphPlan &,
                                         const residency::Pool &, std::uint64_t,
                                         std::size_t, std::uint32_t,
                                         Ticket &) noexcept;

} // namespace rund::compute::detail::graph_reduce

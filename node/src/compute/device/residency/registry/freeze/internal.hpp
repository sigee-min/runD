#pragma once

#include "../../registry.hpp"

namespace rund::compute::detail::residency::view_plan_detail {

[[nodiscard]] ViewActivationResult
validate_regions(const std::vector<Authority::Frame> &,
                 std::span<const FrameRegion>) noexcept;
[[nodiscard]] std::uint64_t
apply_regions(std::vector<Authority::Frame> &,
              std::span<const FrameRegion>) noexcept;

[[nodiscard]] bool validate_stream(const std::vector<Authority::Frame> &,
                                   const StreamPlan &, CacheKey,
                                   std::span<const FrameRegion>) noexcept;
void apply_stream(std::vector<Authority::Frame> &, const StreamPlan &, CacheKey,
                  std::span<const FrameRegion>) noexcept;

[[nodiscard]] bool validate_graph(const std::vector<Authority::Frame> &,
                                  const TiledGraphInvocation &,
                                  std::span<const GraphFreezeRequest>) noexcept;
void apply_graph(std::vector<Authority::Frame> &, const TiledGraphInvocation &,
                 std::span<const GraphFreezeRequest>) noexcept;

[[nodiscard]] bool direct_in_regions(const std::vector<Authority::Frame> &,
                                     std::span<const FrameRegion>) noexcept;
[[nodiscard]] bool evicts_frame(const std::vector<Authority::Frame> &,
                                const registry_model::ViewCommitPlan &,
                                std::size_t) noexcept;
[[nodiscard]] bool touches_frame(const std::vector<Authority::Frame> &,
                                 const registry_model::ViewCommitPlan &,
                                 std::size_t) noexcept;

} // namespace rund::compute::detail::residency::view_plan_detail

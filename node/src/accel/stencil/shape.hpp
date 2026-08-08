#pragma once

#include "../kernel/bindings/stencil.hpp"
#include "../primitive/shape.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kStencilPhysicalGroupWidth = 256u;
inline constexpr rund::kernel::u32 kStencilSharedRadiusCap = 8u;
inline constexpr rund::kernel::u32 kStencilSharedElementCapacity =
    kStencilPhysicalGroupWidth + 2u * kStencilSharedRadiusCap;
inline constexpr rund::kernel::u64 kStencilSharedBytesMax =
    static_cast<rund::kernel::u64>(kStencilSharedElementCapacity) *
    sizeof(rund::kernel::u64);

static_assert(kStencilSharedElementCapacity == 272u);
static_assert(kStencilSharedBytesMax == 2176u);

[[nodiscard]] constexpr rund::kernel::u64
StencilPhysicalGroupCount(const rund::kernel::u64 element_count) noexcept {
  return element_count == 0u
             ? 0u
             : 1u + (element_count - 1u) / kStencilPhysicalGroupWidth;
}

[[nodiscard]] constexpr bool
StencilPhysicalGroupsFit(const rund::kernel::u64 element_count,
                         const rund::kernel::u64 maximum_groups) noexcept {
  const rund::kernel::u64 groups = StencilPhysicalGroupCount(element_count);
  return groups != 0u && groups <= maximum_groups &&
         groups <= std::numeric_limits<rund::kernel::u32>::max();
}

[[nodiscard]] constexpr bool
StencilVulkanDispatchFits(const rund::kernel::u64 element_count,
                          const rund::kernel::u64 maximum_groups) noexcept {
  return element_count <= std::numeric_limits<rund::kernel::u32>::max() &&
         StencilPhysicalGroupsFit(element_count, maximum_groups);
}

[[nodiscard]] constexpr rund::kernel::u64
StencilDirectGlobalReads(const rund::kernel::u32 active_lanes,
                         const rund::kernel::u32 radius) noexcept {
  return static_cast<rund::kernel::u64>(active_lanes) *
         (2u * static_cast<rund::kernel::u64>(radius) + 1u);
}

[[nodiscard]] constexpr rund::kernel::u32
StencilLeftHaloDistinctInputs(const rund::kernel::u64 group_base,
                              const rund::kernel::u32 radius) noexcept {
  return static_cast<rund::kernel::u32>(std::min<rund::kernel::u64>(
      group_base, static_cast<rund::kernel::u64>(radius)));
}

[[nodiscard]] constexpr rund::kernel::u32
StencilRightHaloDistinctInputs(const rund::kernel::u64 element_count,
                               const rund::kernel::u64 group_base,
                               const rund::kernel::u32 active_lanes,
                               const rund::kernel::u32 radius) noexcept {
  const rund::kernel::u64 remaining =
      group_base >= element_count ? 0u : element_count - group_base;
  const rund::kernel::u64 covered = std::min<rund::kernel::u64>(
      remaining, static_cast<rund::kernel::u64>(active_lanes));
  const rund::kernel::u64 available = remaining - covered;
  return static_cast<rund::kernel::u32>(std::min<rund::kernel::u64>(
      available, static_cast<rund::kernel::u64>(radius)));
}

[[nodiscard]] constexpr rund::kernel::u64
StencilSharedGlobalReads(const rund::kernel::u64 element_count,
                         const rund::kernel::u64 group_base,
                         const rund::kernel::u32 active_lanes,
                         const rund::kernel::u32 radius) noexcept {
  return static_cast<rund::kernel::u64>(active_lanes) +
         StencilLeftHaloDistinctInputs(group_base, radius) +
         StencilRightHaloDistinctInputs(element_count, group_base, active_lanes,
                                        radius);
}

[[nodiscard]] constexpr bool
StencilUsesSharedHalo(const rund::kernel::u32 radius) noexcept {
  return radius != 0u && radius <= kStencilSharedRadiusCap;
}

[[nodiscard]] bool StencilShapeOk(const rund::kernel::StencilDesc &desc,
                                  const rund::kernel::StencilPlan &plan,
                                  const StencilBinds &bindings) noexcept;

} // namespace rund::node::accel::detail

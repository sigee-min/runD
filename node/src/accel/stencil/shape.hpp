#pragma once

#include "../kernel/bindings/stencil.hpp"
#include "../primitive/shape.hpp"
#include "../range_aggregate/model.hpp"
#include "model.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace rund::node::accel::detail {

class StencilGpuShape final {
public:
  constexpr StencilGpuShape() noexcept = default;

  [[nodiscard]] static constexpr StencilGpuShape
  direct(const rund::kernel::u32 width) noexcept {
    return SupportedWidth(width) ? StencilGpuShape{width, 0u}
                                 : StencilGpuShape{};
  }

  [[nodiscard]] static constexpr StencilGpuShape
  shared(const rund::kernel::u32 width,
         const rund::kernel::u32 radius_cap) noexcept {
    return SupportedWidth(width) && radius_cap != 0u && radius_cap <= width
               ? StencilGpuShape{width, radius_cap}
               : StencilGpuShape{};
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return SupportedWidth(width_) && radius_cap_ <= width_;
  }

  [[nodiscard]] constexpr bool uses_shared_memory() const noexcept {
    return valid() && radius_cap_ != 0u;
  }

  [[nodiscard]] constexpr rund::kernel::u32 width() const noexcept {
    return width_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 radius_cap() const noexcept {
    return radius_cap_;
  }

  [[nodiscard]] constexpr rund::kernel::u32
  shared_element_capacity() const noexcept {
    return uses_shared_memory() ? width_ + 2u * radius_cap_ : 0u;
  }

  [[nodiscard]] constexpr rund::kernel::u64
  shared_bytes(const rund::kernel::u32 element_bytes) const noexcept {
    return static_cast<rund::kernel::u64>(shared_element_capacity()) *
           element_bytes;
  }

  [[nodiscard]] constexpr std::uint64_t identity() const noexcept {
    return (static_cast<std::uint64_t>(width_) << 32u) | radius_cap_;
  }

  friend constexpr bool operator==(const StencilGpuShape &,
                                   const StencilGpuShape &) noexcept = default;

private:
  constexpr StencilGpuShape(const rund::kernel::u32 width,
                            const rund::kernel::u32 radius_cap) noexcept
      : width_(width), radius_cap_(radius_cap) {}

  [[nodiscard]] static constexpr bool
  SupportedWidth(const rund::kernel::u32 width) noexcept {
    return width == 64u || width == 128u || width == 256u;
  }

  rund::kernel::u32 width_{};
  rund::kernel::u32 radius_cap_{};
};

static_assert(sizeof(StencilGpuShape) == 8u);
static_assert(std::is_trivially_copyable_v<StencilGpuShape>);

[[nodiscard]] constexpr rund::kernel::u32
StencilElementBytes(const rund::kernel::StencilElement element) noexcept {
  if (element == rund::kernel::StencilElement::U32) {
    return 4u;
  }
  return element == rund::kernel::StencilElement::U64 ? 8u : 0u;
}

[[nodiscard]] constexpr StencilGpuShape
StencilGpuShapeFromRangeAggregatePlan(const RangeAggregatePlan &plan) noexcept {
  if (!plan.ok()) {
    return {};
  }
  const RangeAggregateCandidate &candidate = plan.candidate();
  if (candidate.disposition() == RangeAggregateCandidateDisposition::Direct) {
    return StencilGpuShape::direct(candidate.width());
  }
  if (candidate.disposition() ==
      RangeAggregateCandidateDisposition::SharedHalo) {
    return StencilGpuShape::shared(candidate.width(),
                                   candidate.radius_capacity());
  }
  // PrefixDifference uses a workgroup-local scan and BlockPrefixSuffix is
  // lane-local.  Neither is a halo allocation, but both retain the frozen
  // physical width as part of their executable identity.
  if (candidate.disposition() ==
          RangeAggregateCandidateDisposition::PrefixDifference ||
      candidate.disposition() ==
          RangeAggregateCandidateDisposition::BlockPrefixSuffix) {
    return StencilGpuShape::direct(candidate.width());
  }
  return {};
}

[[nodiscard]] constexpr bool
StencilRangeUsesGlobalScratch(const RangeAggregatePlan &plan) noexcept {
  return plan.ok() &&
         (plan.candidate().disposition() ==
              RangeAggregateCandidateDisposition::PrefixDifference ||
          plan.candidate().disposition() ==
              RangeAggregateCandidateDisposition::BlockPrefixSuffix);
}

[[nodiscard]] constexpr rund::kernel::u32
StencilRangeDescriptorCount(const RangeAggregatePlan &plan) noexcept {
  // input, output, params plus the two live RangeAggregate temporary roles.
  return StencilRangeUsesGlobalScratch(plan) ? 5u : 3u;
}

[[nodiscard]] constexpr rund::kernel::u64
StencilRangeStaticSharedBytes(const RangeAggregatePlan &plan) noexcept {
  if (!plan.ok()) {
    return 0u;
  }
  const RangeAggregateCandidate &candidate = plan.candidate();
  if (candidate.disposition() ==
      RangeAggregateCandidateDisposition::SharedHalo) {
    return static_cast<rund::kernel::u64>(candidate.width() +
                                          2u * candidate.radius_capacity()) *
           plan.shape().element_bytes();
  }
  // PrefixDifference carries one workgroup scan tree.  It is a different
  // source-shaped resource from a halo and therefore never overloads the
  // halo radius field.
  return candidate.disposition() ==
                 RangeAggregateCandidateDisposition::PrefixDifference
             ? static_cast<rund::kernel::u64>(candidate.width()) *
                   plan.shape().element_bytes()
             : 0u;
}

[[nodiscard]] constexpr bool RangeAggregateStageDispatchFits(
    const RangeAggregatePlan &plan, const std::size_t index,
    const rund::kernel::u64 maximum_groups) noexcept {
  if (!plan.ok() || index >= plan.stage_count()) {
    return false;
  }
  const RangeAggregateStagePlan stage = plan.stage(index);
  const StencilGpuShape shape = StencilGpuShapeFromRangeAggregatePlan(plan);
  return shape.valid() && stage.width == shape.width() && stage.groups != 0u &&
         stage.groups <= maximum_groups &&
         stage.groups <= std::numeric_limits<rund::kernel::u32>::max();
}

// This is a pure ABI projection of the frozen RangeAggregate stage graph.
// It owns no planning policy: all counts originate in RangeAggregatePlan and
// both encoders consume the same exact record.
[[nodiscard]] constexpr StencilParams
StencilRangeStageParams(const RangeAggregatePlan &plan,
                        const std::size_t index) noexcept {
  if (!plan.ok() || index >= plan.stage_count()) {
    return {};
  }
  const RangeAggregateStagePlan stage = plan.stage(index);
  std::uint64_t auxiliary = stage.groups;
  if (stage.disposition == RangeAggregateStageDisposition::BlockPrefixSuffix) {
    const std::uint64_t window = plan.shape().radius() * 2u + 1u;
    auxiliary = window == 0u ? 0u
                             : stage.element_count / window +
                                   static_cast<std::uint64_t>(
                                       stage.element_count % window != 0u);
  }
  return StencilParams{
      .element_count = plan.shape().element_count(),
      .radius = plan.shape().radius(),
      .stage_element_count = stage.element_count,
      .stage_aux_count = auxiliary,
      .stage = static_cast<rund::kernel::u32>(stage.disposition),
  };
}

// Kernel Stencil semantics remain the source of the graph descriptor and
// hash.  This is the one projection that proves a frozen physical plan was
// derived from those unchanged semantics before either backend consumes it.
[[nodiscard]] constexpr bool
StencilRangeAggregatePlanMatches(const rund::kernel::StencilPlan &semantic,
                                 const rund::kernel::ComputeDomain domain,
                                 const RangeAggregatePlan &range) noexcept {
  const std::optional<RangeAggregateShape> shape =
      RangeAggregateShape::from_stencil(semantic, domain);
  return shape.has_value() && range.ok() &&
         range.shape().element_count() == shape->element_count() &&
         range.shape().radius() == shape->radius() &&
         range.shape().element_bytes() == shape->element_bytes() &&
         range.shape().traits().operation() == shape->traits().operation() &&
         range.shape().traits().domain() == shape->traits().domain() &&
         range.shape().traits().arithmetic_law() ==
             shape->traits().arithmetic_law();
}

[[nodiscard]] constexpr rund::kernel::u64
StencilPhysicalGroupCount(const rund::kernel::u64 element_count,
                          const StencilGpuShape shape) noexcept {
  return !shape.valid() || element_count == 0u
             ? 0u
             : 1u + (element_count - 1u) / shape.width();
}

[[nodiscard]] constexpr bool
StencilPhysicalGroupsFit(const rund::kernel::u64 element_count,
                         const rund::kernel::u64 maximum_groups,
                         const StencilGpuShape shape) noexcept {
  const rund::kernel::u64 groups =
      StencilPhysicalGroupCount(element_count, shape);
  return groups != 0u && groups <= maximum_groups &&
         groups <= std::numeric_limits<rund::kernel::u32>::max();
}

[[nodiscard]] constexpr bool
StencilVulkanDispatchFits(const rund::kernel::u64 element_count,
                          const rund::kernel::u64 maximum_groups,
                          const StencilGpuShape shape) noexcept {
  return element_count <= std::numeric_limits<rund::kernel::u32>::max() &&
         StencilPhysicalGroupsFit(element_count, maximum_groups, shape);
}

[[nodiscard]] bool StencilShapeOk(const rund::kernel::StencilDesc &desc,
                                  const rund::kernel::StencilPlan &plan,
                                  const StencilBinds &bindings) noexcept;

} // namespace rund::node::accel::detail

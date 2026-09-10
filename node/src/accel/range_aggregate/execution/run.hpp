#pragma once

#include "model.hpp"

#include <kernel/core/checked.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace rund::node::accel::detail {

// Projects one authenticated resident count into the frozen RangePlan stage
// graph. Candidate, source identity, stage slots, and scratch capacity never
// change; inactive hierarchy slots receive zero groups.
class RangeRun final {
public:
  RangeRun() = delete;

  [[nodiscard]] static constexpr std::optional<RangeRun>
  make(const RangePlan &plan, const rund::kernel::u64 count) noexcept {
    if (!plan.ok() || count > plan.shape().input_count() ||
        (!plan.shape().resident_counted() &&
         count != plan.shape().input_count())) {
      return std::nullopt;
    }
    return RangeRun{plan, count};
  }

  [[nodiscard]] constexpr const RangePlan &plan() const noexcept {
    assert(plan_ != nullptr);
    return *plan_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 count() const noexcept {
    return count_;
  }

  [[nodiscard]] constexpr std::optional<RangeDispatch>
  stage(const std::size_t index) const noexcept {
    if (index >= plan().stage_count()) {
      return std::nullopt;
    }
    const RangeStagePlan frozen = plan().stage(index);
    if (!plan().shape().resident_counted()) {
      const std::optional<RangeParams> params =
          RangeParams::from(plan(), index);
      return params.has_value() ? std::optional<RangeDispatch>{RangeDispatch{
                                      *params, frozen.groups, frozen.width}}
                                : std::nullopt;
    }

    rund::kernel::u64 elements = 0u;
    rund::kernel::u64 groups = 0u;
    switch (frozen.disposition) {
    case RangeStageKind::TiledDifference:
      elements = count_;
      groups = Groups(elements, frozen.width * kRangeTileOutputsPerLane);
      break;
    case RangeStageKind::Direct:
    case RangeStageKind::SharedHalo:
    case RangeStageKind::PrefixWindow:
      elements = count_;
      groups = Groups(elements, frozen.width);
      break;
    case RangeStageKind::BlockWindow:
      elements = count_;
      groups = RangeBlockQueryGroups(
          plan().source_variant(), count_, plan().shape().stride(),
          plan().shape().window_size(), frozen.width);
      break;
    case RangeStageKind::PrefixSequential:
      elements = count_;
      groups = count_ == 0u ? 0u : 1u;
      break;
    case RangeStageKind::PrefixBlock:
    case RangeStageKind::PrefixSummary:
      elements = PrefixCount(frozen.level);
      groups = Groups(elements, frozen.width);
      if (frozen.level != 0u && PrefixCount(static_cast<std::uint8_t>(
                                    frozen.level - 1u)) <= frozen.width) {
        groups = 0u;
      }
      break;
    case RangeStageKind::PrefixFixup:
      elements = PrefixCount(frozen.level);
      groups = Groups(elements, frozen.width);
      if (groups <= 1u) {
        groups = 0u;
      }
      break;
    case RangeStageKind::BlockPrefixSuffix: {
      elements = ActiveSpan();
      const rund::kernel::u64 blocks =
          Groups(elements, plan().shape().window_size());
      groups = RangeBlockGroups(plan().source_variant(), blocks, frozen.width);
      break;
    }
    }
    const rund::kernel::u64 auxiliary =
        frozen.disposition == RangeStageKind::BlockPrefixSuffix
            ? Groups(elements, plan().shape().window_size())
            : groups;
    return RangeDispatch{
        RangeParams{count_, count_, plan().shape().window_size(),
                    plan().shape().stride(), plan().shape().padding(), elements,
                    auxiliary,
                    static_cast<rund::kernel::u32>(frozen.disposition)},
        groups, frozen.width};
  }

  [[nodiscard]] constexpr std::optional<RangeIndirect>
  indirect(const std::size_t index) const noexcept {
    const std::optional<RangeDispatch> dispatch = stage(index);
    if (!dispatch.has_value() ||
        dispatch->groups() > std::numeric_limits<rund::kernel::u32>::max()) {
      return std::nullopt;
    }
    const rund::kernel::u64 work = dispatch->work_items();
    const rund::kernel::u32 active = dispatch->groups() == 0u ? 0u : 1u;
    return RangeIndirect{
        .groups_x = static_cast<rund::kernel::u32>(dispatch->groups()),
        .groups_y = active,
        .groups_z = active,
        .work_items_lo = static_cast<rund::kernel::u32>(work),
        .work_items_hi = static_cast<rund::kernel::u32>(work >> 32u),
    };
  }

private:
  constexpr RangeRun(const RangePlan &plan,
                     const rund::kernel::u64 count) noexcept
      : plan_(&plan), count_(count) {}

  [[nodiscard]] static constexpr rund::kernel::u64
  Groups(const rund::kernel::u64 count,
         const rund::kernel::u32 width) noexcept {
    if (count == 0u) {
      return 0u;
    }
    return width == 0u
               ? 1u
               : count / width +
                     static_cast<rund::kernel::u64>(count % width != 0u);
  }

  [[nodiscard]] constexpr rund::kernel::u64
  PrefixCount(const std::uint8_t level) const noexcept {
    rund::kernel::u64 value = count_;
    const rund::kernel::u32 width = plan().candidate().width();
    for (std::uint8_t cursor = 0u; cursor < level && value != 0u; ++cursor) {
      value = Groups(value, width);
    }
    return value;
  }

  [[nodiscard]] constexpr rund::kernel::u64 ActiveSpan() const noexcept {
    if (count_ == 0u) {
      return 0u;
    }
    rund::kernel::u64 span = 0u;
    const bool ok = rund::kernel::checked::add(
        count_ - 1u, plan().shape().window_size(), span);
    assert(ok);
    return ok ? span : 0u;
  }

  const RangePlan *plan_;
  rund::kernel::u64 count_;
};

} // namespace rund::node::accel::detail

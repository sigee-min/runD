#pragma once

#include "model.hpp"
#include "scratch.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace rund::node::accel::detail {

// The one backend-neutral physical projection of a frozen RangePlan. It
// derives source identity, workgroup shape, dispatch topology, and typed
// temporary bindings without owning native resources or backend state.
class RangeExec final {
public:
  RangeExec() = delete;

  [[nodiscard]] static constexpr std::optional<RangeExec>
  from(const RangePlan &plan) noexcept {
    if (!plan.ok() || (plan.source_variant() != RangeSource::Metal &&
                       plan.source_variant() != RangeSource::Vulkan)) {
      return std::nullopt;
    }
    return RangeExec{plan};
  }

  [[nodiscard]] constexpr const RangePlan &plan() const noexcept {
    assert(plan_ != nullptr);
    return *plan_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 width() const noexcept {
    return plan().candidate().width();
  }

  [[nodiscard]] constexpr rund::kernel::u32
  shared_radius_capacity() const noexcept {
    return plan().candidate().radius_capacity();
  }

  [[nodiscard]] constexpr bool uses_shared_halo() const noexcept {
    return plan().candidate().uses_shared_halo();
  }

  [[nodiscard]] constexpr rund::kernel::u32
  shared_element_capacity() const noexcept {
    assert(uses_shared_halo());
    return width() + 2u * shared_radius_capacity();
  }

  // These are the complete source-shaping inputs. Primitive adapters validate
  // their own public descriptors, while backend source, dispatch, scratch,
  // and cache lookup consume only this projection.
  [[nodiscard]] constexpr RangeOp operation() const noexcept {
    return plan().shape().traits().operation();
  }

  [[nodiscard]] constexpr rund::kernel::ComputeDomain domain() const noexcept {
    return plan().shape().traits().domain();
  }

  [[nodiscard]] constexpr RangeLaw arithmetic_law() const noexcept {
    return plan().shape().traits().arithmetic_law();
  }

  [[nodiscard]] constexpr rund::kernel::u32 element_bytes() const noexcept {
    return plan().shape().element_bytes();
  }

  [[nodiscard]] constexpr bool wide_elements() const noexcept {
    return element_bytes() == 8u;
  }

  [[nodiscard]] constexpr bool signed_extrema() const noexcept {
    return operation() != RangeOp::Sum &&
           plan().shape().traits().signed_domain();
  }

  [[nodiscard]] constexpr bool saturating_sum() const noexcept {
    return operation() == RangeOp::Sum &&
           arithmetic_law() == RangeLaw::Saturating;
  }

  [[nodiscard]] constexpr bool signed_values() const noexcept {
    return signed_extrema() || saturating_sum();
  }

  [[nodiscard]] constexpr RangePath candidate() const noexcept {
    return plan().candidate().disposition();
  }

  [[nodiscard]] constexpr bool uses_global_scratch() const noexcept {
    const RangePath disposition = plan().candidate().disposition();
    return disposition == RangePath::PrefixDifference ||
           disposition == RangePath::BlockPrefixSuffix;
  }

  [[nodiscard]] constexpr std::uint32_t descriptor_count() const noexcept {
    return uses_global_scratch() ? 5u : 3u;
  }

  [[nodiscard]] constexpr rund::kernel::u64
  static_shared_bytes() const noexcept {
    const RangeCandidate &candidate = plan().candidate();

    if (candidate.disposition() == RangePath::TiledDifference) {
      return static_cast<rund::kernel::u64>(2u * width()) * element_bytes();
    }
    if (candidate.disposition() == RangePath::SharedHalo) {
      return static_cast<rund::kernel::u64>(shared_element_capacity()) *
             plan().shape().element_bytes();
    }
    return candidate.disposition() == RangePath::PrefixDifference
               ? static_cast<rund::kernel::u64>(width()) *
                     plan().shape().element_bytes()
               : 0u;
  }

  [[nodiscard]] constexpr bool stage_dispatch_fits(
      const std::size_t index,
      const rund::kernel::u64 maximum_group_count) const noexcept {
    if (index >= plan().stage_count()) {
      return false;
    }
    const RangeStagePlan stage = plan().stage(index);
    return stage.width == width() && stage.groups != 0u &&
           stage.groups <= maximum_group_count &&
           stage.groups <= std::numeric_limits<rund::kernel::u32>::max();
  }

  [[nodiscard]] constexpr bool vulkan_dispatch_fits(
      const rund::kernel::u64 maximum_group_count) const noexcept {
    if (plan().shape().input_count() >
            std::numeric_limits<rund::kernel::u32>::max() ||
        plan().shape().output_count() >
            std::numeric_limits<rund::kernel::u32>::max()) {
      return false;
    }
    for (std::size_t index = 0u; index < plan().stage_count(); ++index) {
      if (plan().stage(index).element_count >
              std::numeric_limits<rund::kernel::u32>::max() ||
          !stage_dispatch_fits(index, maximum_group_count)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] constexpr std::optional<RangeParams>
  stage_params(const std::size_t index) const noexcept {
    return RangeParams::from(plan(), index);
  }

  [[nodiscard]] constexpr std::optional<RangeScratch>
  stage_scratch(const std::size_t index) const noexcept {
    if (index >= plan().stage_count()) {
      return std::nullopt;
    }
    const RangeStagePlan stage = plan().stage(index);
    const auto slot = [](const RangeTempRole role,
                         const std::uint8_t ordinal) constexpr {
      return RangeTempSlot::of(role, ordinal);
    };
    switch (stage.disposition) {
    case RangeStageKind::TiledDifference:
    case RangeStageKind::Direct:
    case RangeStageKind::SharedHalo:
      return RangeScratch::none();
    case RangeStageKind::PrefixSequential:
    case RangeStageKind::PrefixBlock:
      return RangeScratch::pair(
          slot(RangeTempRole::PrefixValues, 0u),
          stage.groups == 1u
              ? slot(RangeTempRole::PrefixValues, 0u)
              : slot(RangeTempRole::BlockSummaries, stage.level));
    case RangeStageKind::PrefixSummary:
      if (stage.level == 0u) {
        return std::nullopt;
      }
      return RangeScratch::pair(
          slot(RangeTempRole::BlockSummaries,
               static_cast<std::uint8_t>(stage.level - 1u)),
          stage.groups == 1u
              ? slot(RangeTempRole::BlockSummaries,
                     static_cast<std::uint8_t>(stage.level - 1u))
              : slot(RangeTempRole::BlockSummaries, stage.level));
    case RangeStageKind::PrefixFixup:
      return RangeScratch::pair(
          stage.level == 0u ? slot(RangeTempRole::PrefixValues, 0u)
                            : slot(RangeTempRole::BlockSummaries,
                                   static_cast<std::uint8_t>(stage.level - 1u)),
          slot(RangeTempRole::BlockSummaries, stage.level));
    case RangeStageKind::PrefixWindow:
      return RangeScratch::pair(slot(RangeTempRole::PrefixValues, 0u),
                                plan().shape().input_count() <= width()
                                    ? slot(RangeTempRole::PrefixValues, 0u)
                                    : slot(RangeTempRole::BlockSummaries, 0u));
    case RangeStageKind::BlockPrefixSuffix:
    case RangeStageKind::BlockWindow:
      return RangeScratch::pair(
          slot(RangeTempRole::ForwardValues, 0u),
          slot(plan().source_variant() == RangeSource::Metal
                   ? RangeTempRole::ForwardValues
                   : RangeTempRole::BackwardValues,
               0u));
    }
    return std::nullopt;
  }

  [[nodiscard]] constexpr RangeIdentity source_identity() const noexcept {
    return plan().source_identity();
  }

  [[nodiscard]] constexpr RangeIdentity execution_identity() const noexcept {
    return plan().execution_identity();
  }

private:
  explicit constexpr RangeExec(const RangePlan &plan) noexcept : plan_(&plan) {}

  const RangePlan *plan_;
};

static_assert(std::is_trivially_copyable_v<RangeParams>);
static_assert(std::is_trivially_copyable_v<RangeTempSlot>);
static_assert(std::is_trivially_copyable_v<RangeScratch>);
static_assert(std::is_trivially_copyable_v<RangeExec>);

// Small projections used by native resource and encoder code.
[[nodiscard]] constexpr bool RangeUsesScratch(const RangePlan &plan) noexcept {
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value() && execution->uses_global_scratch();
}

[[nodiscard]] constexpr std::uint32_t
RangeDescriptorCount(const RangePlan &plan) noexcept {
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value() ? execution->descriptor_count() : 0u;
}

[[nodiscard]] constexpr std::optional<RangeParams>
RangeStageParamsFor(const RangePlan &plan, const std::size_t index) noexcept {
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value() ? execution->stage_params(index) : std::nullopt;
}

} // namespace rund::node::accel::detail

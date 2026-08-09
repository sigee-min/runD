#pragma once

#include "model.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace rund::node::accel::detail {

// This is the backend-neutral physical projection of a selected range plan.
// It borrows the frozen plan. Pipeline execution places every global temporary
// through Pipeline scratch; standalone prepared backend resources retain
// plan-sized temporary buffers. Adapters retain their primitive-specific
// bindings and result authority.
class RangeGpuShape final {
public:
  RangeGpuShape() = delete;

  [[nodiscard]] static constexpr std::optional<RangeGpuShape>
  direct(const rund::kernel::u32 width) noexcept {
    return SupportedWidth(width)
               ? std::optional<RangeGpuShape>{RangeGpuShape{width, 0u}}
               : std::nullopt;
  }

  [[nodiscard]] static constexpr std::optional<RangeGpuShape>
  shared_halo(const rund::kernel::u32 width,
              const rund::kernel::u32 radius_capacity) noexcept {
    return SupportedWidth(width) && radius_capacity != 0u &&
                   radius_capacity <= width
               ? std::optional<RangeGpuShape>{RangeGpuShape{width,
                                                            radius_capacity}}
               : std::nullopt;
  }

  [[nodiscard]] constexpr rund::kernel::u32 width() const noexcept {
    return width_;
  }

  [[nodiscard]] constexpr rund::kernel::u32
  shared_radius_capacity() const noexcept {
    return shared_radius_capacity_;
  }

  [[nodiscard]] constexpr bool uses_shared_halo() const noexcept {
    return shared_radius_capacity_ != 0u;
  }

  [[nodiscard]] constexpr rund::kernel::u32
  shared_element_capacity() const noexcept {
    return width_ + 2u * shared_radius_capacity_;
  }

  [[nodiscard]] constexpr rund::kernel::u64
  shared_halo_bytes(const rund::kernel::u32 element_bytes) const noexcept {
    return static_cast<rund::kernel::u64>(shared_element_capacity()) *
           element_bytes;
  }

  [[nodiscard]] constexpr std::uint64_t identity() const noexcept {
    return (static_cast<std::uint64_t>(width_) << 32u) |
           shared_radius_capacity_;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const RangeGpuShape &, const RangeGpuShape &) = default;

private:
  constexpr RangeGpuShape(const rund::kernel::u32 width,
                          const rund::kernel::u32 radius_capacity) noexcept
      : width_(width), shared_radius_capacity_(radius_capacity) {}

  [[nodiscard]] static constexpr bool
  SupportedWidth(const rund::kernel::u32 width) noexcept {
    return width == 64u || width == 128u || width == 256u;
  }

  rund::kernel::u32 width_;
  rund::kernel::u32 shared_radius_capacity_;
};

static_assert(sizeof(RangeGpuShape) == 8u);
static_assert(std::is_trivially_copyable_v<RangeGpuShape>);

[[nodiscard]] constexpr std::optional<RangeGpuShape>
RangeGpuShapeFor(const RangePlan &plan) noexcept {
  if (!plan.ok() || (plan.source_variant() != RangeSource::Metal &&
                     plan.source_variant() != RangeSource::Vulkan)) {
    return std::nullopt;
  }
  const RangeCandidate &candidate = plan.candidate();
  switch (candidate.disposition()) {
  case RangePath::Direct:
  case RangePath::PrefixDifference:
  case RangePath::BlockPrefixSuffix:
    return RangeGpuShape::direct(candidate.width());
  case RangePath::SharedHalo:
    return RangeGpuShape::shared_halo(candidate.width(),
                                      candidate.radius_capacity());
  }
  return std::nullopt;
}

class RangeParams final {
public:
  RangeParams() = delete;

  [[nodiscard]] static constexpr std::optional<RangeParams>
  from(const RangePlan &plan, const std::size_t index) noexcept {
    if (!plan.ok() || index >= plan.stage_count()) {
      return std::nullopt;
    }
    const RangeStagePlan stage = plan.stage(index);
    std::uint64_t auxiliary = stage.groups;
    if (stage.disposition == RangeStageKind::BlockPrefixSuffix) {
      const std::uint64_t window = plan.shape().window_size();
      auxiliary = window == 0u ? 0u
                               : stage.element_count / window +
                                     static_cast<std::uint64_t>(
                                         stage.element_count % window != 0u);
    }
    return RangeParams{plan.shape().input_count(),
                       plan.shape().output_count(),
                       plan.shape().window_size(),
                       plan.shape().stride(),
                       plan.shape().padding(),
                       stage.element_count,
                       auxiliary,
                       static_cast<rund::kernel::u32>(stage.disposition)};
  }

  [[nodiscard]] constexpr rund::kernel::u64 element_count() const noexcept {
    return input_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 radius() const noexcept {
    return padding_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 input_count() const noexcept {
    return input_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 output_count() const noexcept {
    return output_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 window_size() const noexcept {
    return window_size_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 stride() const noexcept {
    return stride_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 padding() const noexcept {
    return padding_;
  }

  [[nodiscard]] constexpr rund::kernel::u64
  stage_element_count() const noexcept {
    return stage_element_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 stage_aux_count() const noexcept {
    return stage_aux_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 stage() const noexcept {
    return stage_;
  }

private:
  friend class RangeRun;

  constexpr RangeParams(const rund::kernel::u64 input_count,
                        const rund::kernel::u64 output_count,
                        const rund::kernel::u64 window_size,
                        const rund::kernel::u64 stride,
                        const rund::kernel::u64 padding,
                        const rund::kernel::u64 stage_element_count,
                        const rund::kernel::u64 stage_aux_count,
                        const rund::kernel::u32 stage) noexcept
      : input_count_(input_count), output_count_(output_count),
        window_size_(window_size), stride_(stride), padding_(padding),
        stage_element_count_(stage_element_count),
        stage_aux_count_(stage_aux_count), stage_(stage), reserved_(0u) {}

  rund::kernel::u64 input_count_;
  rund::kernel::u64 output_count_;
  rund::kernel::u64 window_size_;
  rund::kernel::u64 stride_;
  rund::kernel::u64 padding_;
  rund::kernel::u64 stage_element_count_;
  rund::kernel::u64 stage_aux_count_;
  rund::kernel::u32 stage_;
  [[maybe_unused]] rund::kernel::u32 reserved_;
};

static_assert(std::is_standard_layout_v<RangeParams>);
static_assert(std::is_trivially_copyable_v<RangeParams>);
static_assert(sizeof(RangeParams) == 64u);
static_assert(alignof(RangeParams) == alignof(rund::kernel::u64));

class RangeDispatch final {
public:
  RangeDispatch() = delete;

  [[nodiscard]] constexpr const RangeParams &params() const noexcept {
    return params_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 groups() const noexcept {
    return groups_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 width() const noexcept {
    return width_;
  }

  [[nodiscard]] constexpr bool active() const noexcept { return groups_ != 0u; }

  [[nodiscard]] constexpr rund::kernel::u64 work_items() const noexcept {
    return groups_ * width_;
  }

private:
  friend class RangeRun;

  constexpr RangeDispatch(const RangeParams params,
                          const rund::kernel::u64 groups,
                          const rund::kernel::u32 width) noexcept
      : params_(params), groups_(groups), width_(width) {}

  RangeParams params_;
  rund::kernel::u64 groups_;
  rund::kernel::u32 width_;
};

// One backend-native indirect row.  The first three words are the Metal and
// Vulkan dispatch ABI.  The exact 64-bit invocation count follows so Pipeline
// telemetry never reconstructs hierarchy work as logical_count * stage_count.
struct RangeIndirect final {
  rund::kernel::u32 groups_x{};
  rund::kernel::u32 groups_y{};
  rund::kernel::u32 groups_z{};
  rund::kernel::u32 work_items_lo{};
  rund::kernel::u32 work_items_hi{};
  std::array<rund::kernel::u32, 3u> reserved{};
};

static_assert(sizeof(RangeIndirect) == 32u);
static_assert(alignof(RangeIndirect) == alignof(rund::kernel::u32));
static_assert(std::is_trivially_copyable_v<RangeIndirect>);

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
    case RangeStageKind::Direct:
    case RangeStageKind::SharedHalo:
    case RangeStageKind::PrefixWindow:
    case RangeStageKind::BlockWindow:
      elements = count_;
      groups = Groups(elements, frozen.width);
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
      groups = Groups(blocks, frozen.width);
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

class RangeControlLayout final {
public:
  RangeControlLayout() = delete;

  [[nodiscard]] static constexpr std::optional<RangeControlLayout>
  from(const RangePlan &plan) noexcept {
    if (!plan.ok() || !plan.shape().resident_counted() ||
        plan.stage_count() == 0u || plan.stage_count() > kRangeStageCap) {
      return std::nullopt;
    }
    rund::kernel::u64 params = 0u;
    rund::kernel::u64 indirect = 0u;
    if (!rund::kernel::checked::mul(plan.stage_count(), sizeof(RangeParams),
                                    params) ||
        !rund::kernel::checked::mul(plan.stage_count(), sizeof(RangeIndirect),
                                    indirect)) {
      return std::nullopt;
    }
    return RangeControlLayout{
        static_cast<rund::kernel::u32>(plan.stage_count()), params, indirect};
  }

  [[nodiscard]] constexpr rund::kernel::u32 stage_count() const noexcept {
    return stage_count_;
  }
  [[nodiscard]] constexpr rund::kernel::u64 params_bytes() const noexcept {
    return params_bytes_;
  }
  [[nodiscard]] constexpr rund::kernel::u64 indirect_bytes() const noexcept {
    return indirect_bytes_;
  }
  [[nodiscard]] static constexpr rund::kernel::u64 status_bytes() noexcept {
    return 2u * sizeof(rund::kernel::u32);
  }

private:
  constexpr RangeControlLayout(const rund::kernel::u32 stage_count,
                               const rund::kernel::u64 params_bytes,
                               const rund::kernel::u64 indirect_bytes) noexcept
      : stage_count_(stage_count), params_bytes_(params_bytes),
        indirect_bytes_(indirect_bytes) {}

  rund::kernel::u32 stage_count_;
  rund::kernel::u64 params_bytes_;
  rund::kernel::u64 indirect_bytes_;
};

class RangeTempSlot final {
public:
  RangeTempSlot() = delete;

  [[nodiscard]] static constexpr RangeTempSlot
  of(const RangeTempRole role, const std::uint8_t ordinal) noexcept {
    return RangeTempSlot{role, ordinal};
  }

  [[nodiscard]] constexpr RangeTempRole role() const noexcept { return role_; }

  [[nodiscard]] constexpr std::uint8_t ordinal() const noexcept {
    return ordinal_;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const RangeTempSlot &, const RangeTempSlot &) = default;

private:
  constexpr RangeTempSlot(const RangeTempRole role,
                          const std::uint8_t ordinal) noexcept
      : role_(role), ordinal_(ordinal) {}

  RangeTempRole role_;
  std::uint8_t ordinal_;
};

enum class RangeScratchKind : std::uint8_t {
  None,
  Pair,
};

class RangeScratch final {
public:
  RangeScratch() = delete;

  [[nodiscard]] static constexpr RangeScratch none() noexcept {
    return RangeScratch{RangeScratchKind::None,
                        RangeTempSlot::of(RangeTempRole::PrefixValues, 0u),
                        RangeTempSlot::of(RangeTempRole::PrefixValues, 0u)};
  }

  [[nodiscard]] static constexpr RangeScratch
  pair(const RangeTempSlot first, const RangeTempSlot second) noexcept {
    return RangeScratch{RangeScratchKind::Pair, first, second};
  }

  [[nodiscard]] constexpr RangeScratchKind disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr bool uses_global_scratch() const noexcept {
    return disposition_ == RangeScratchKind::Pair;
  }

  [[nodiscard]] constexpr RangeTempSlot first() const noexcept {
    assert(uses_global_scratch());
    return first_;
  }

  [[nodiscard]] constexpr RangeTempSlot second() const noexcept {
    assert(uses_global_scratch());
    return second_;
  }

private:
  constexpr RangeScratch(const RangeScratchKind disposition,
                         const RangeTempSlot first,
                         const RangeTempSlot second) noexcept
      : disposition_(disposition), first_(first), second_(second) {}

  RangeScratchKind disposition_;
  RangeTempSlot first_;
  RangeTempSlot second_;
};

class RangeExec final {
public:
  RangeExec() = delete;

  [[nodiscard]] static constexpr std::optional<RangeExec>
  from(const RangePlan &plan) noexcept {
    if (!plan.ok() || (plan.source_variant() != RangeSource::Metal &&
                       plan.source_variant() != RangeSource::Vulkan)) {
      return std::nullopt;
    }
    const std::optional<RangeGpuShape> shape = RangeGpuShapeFor(plan);
    if (!shape.has_value() || plan.stage_count() == 0u ||
        plan.stage_count() > kRangeStageCap) {
      return std::nullopt;
    }
    for (std::size_t index = 0u; index < plan.stage_count(); ++index) {
      const RangeStagePlan stage = plan.stage(index);
      if (stage.width != shape->width() || stage.groups == 0u ||
          stage.groups > std::numeric_limits<rund::kernel::u32>::max()) {
        return std::nullopt;
      }
    }
    return RangeExec{plan, *shape};
  }

  [[nodiscard]] constexpr const RangePlan &plan() const noexcept {
    assert(plan_ != nullptr);
    return *plan_;
  }

  [[nodiscard]] constexpr const RangeGpuShape &shape() const noexcept {
    return shape_;
  }

  // These are the complete source-shaping inputs.  Primitive adapters may
  // validate and retain their own public descriptors, but backend source,
  // dispatch, scratch binding, and cache lookup consume only this projection.
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
    if (candidate.disposition() == RangePath::SharedHalo) {
      return shape_.shared_halo_bytes(plan().shape().element_bytes());
    }
    return candidate.disposition() == RangePath::PrefixDifference
               ? static_cast<rund::kernel::u64>(shape_.width()) *
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
    return stage.width == shape_.width() && stage.groups != 0u &&
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
                                slot(RangeTempRole::PrefixValues, 0u));
    case RangeStageKind::BlockPrefixSuffix:
    case RangeStageKind::BlockWindow:
      return RangeScratch::pair(slot(RangeTempRole::ForwardValues, 0u),
                                slot(RangeTempRole::BackwardValues, 0u));
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
  constexpr RangeExec(const RangePlan &plan, const RangeGpuShape shape) noexcept
      : plan_(&plan), shape_(shape) {}

  const RangePlan *plan_;
  RangeGpuShape shape_;
};

static_assert(std::is_trivially_copyable_v<RangeParams>);
static_assert(std::is_trivially_copyable_v<RangeTempSlot>);
static_assert(std::is_trivially_copyable_v<RangeScratch>);
static_assert(std::is_trivially_copyable_v<RangeExec>);

[[nodiscard]] constexpr bool RangeUsesScratch(const RangePlan &plan) noexcept {
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value() && execution->uses_global_scratch();
}

[[nodiscard]] constexpr std::uint32_t
RangeDescriptorCount(const RangePlan &plan) noexcept {
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value() ? execution->descriptor_count() : 0u;
}

[[nodiscard]] constexpr rund::kernel::u64
RangeStaticSharedBytes(const RangePlan &plan) noexcept {
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value() ? execution->static_shared_bytes() : 0u;
}

[[nodiscard]] constexpr bool
RangeStageDispatchFits(const RangePlan &plan, const std::size_t index,
                       const rund::kernel::u64 maximum_group_count) noexcept {
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value() &&
         execution->stage_dispatch_fits(index, maximum_group_count);
}

[[nodiscard]] constexpr std::optional<RangeParams>
RangeStageParamsFor(const RangePlan &plan, const std::size_t index) noexcept {
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value() ? execution->stage_params(index) : std::nullopt;
}

[[nodiscard]] constexpr rund::kernel::u64
RangePhysicalGroupCount(const rund::kernel::u64 element_count,
                        const RangeGpuShape shape) noexcept {
  return element_count == 0u ? 0u : 1u + (element_count - 1u) / shape.width();
}

[[nodiscard]] constexpr bool
RangePhysicalGroupsFit(const rund::kernel::u64 element_count,
                       const rund::kernel::u64 maximum_group_count,
                       const RangeGpuShape shape) noexcept {
  const rund::kernel::u64 groups =
      RangePhysicalGroupCount(element_count, shape);
  return groups != 0u && groups <= maximum_group_count &&
         groups <= std::numeric_limits<rund::kernel::u32>::max();
}

[[nodiscard]] constexpr bool
RangeVulkanDispatchFits(const rund::kernel::u64 element_count,
                        const rund::kernel::u64 maximum_group_count,
                        const RangeGpuShape shape) noexcept {
  return element_count <= std::numeric_limits<rund::kernel::u32>::max() &&
         RangePhysicalGroupsFit(element_count, maximum_group_count, shape);
}

} // namespace rund::node::accel::detail

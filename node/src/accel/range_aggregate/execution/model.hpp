#pragma once

#include "../model/plan.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace rund::node::accel::detail {

class RangeRun;

// Backend-neutral parameter and dispatch values. These are the host-side
// layouts consumed by both native Range encoders.
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

// The first three words are the Metal and Vulkan dispatch ABI. The exact
// 64-bit invocation count follows so telemetry never reconstructs hierarchy
// work as logical_count * stage_count.
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

} // namespace rund::node::accel::detail

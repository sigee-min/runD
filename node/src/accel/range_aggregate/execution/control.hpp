#pragma once

#include "model.hpp"

#include <kernel/core/checked.hpp>

#include <cstdint>
#include <optional>

namespace rund::node::accel::detail {

// Resident-count control buffers are a separate resource-layout projection;
// native resource owners allocate and publish these exact byte counts.
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

} // namespace rund::node::accel::detail

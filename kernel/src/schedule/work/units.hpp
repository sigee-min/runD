#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel::schedule_detail {

constexpr u64 NormalizeWorkUnits(const u64 work_units) noexcept {
  return work_units == 0u ? 1u : work_units;
}

} // namespace rund::kernel::schedule_detail

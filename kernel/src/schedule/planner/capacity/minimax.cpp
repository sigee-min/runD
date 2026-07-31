#include "local.hpp"

#include <algorithm>

namespace rund::kernel::schedule::planner::capacity {
namespace {

constexpr u64 kRatioScale = 1000000u;

u32 CapacityForSlot(const PartitionRequest& request, const u32 slot) {
  const u32 width = std::max<u32>(1u, request.execution_width);
  return request.worker_capacity_milli[slot % width];
}

u32 UnitsAtLimit(const u64 limit, const u32 capacity, const u32 total_units) {
  const u64 whole = limit / kRatioScale;
  const u64 rem = limit % kRatioScale;
  const u64 base = whole * capacity;
  if (base >= total_units) {
    return total_units;
  }
  const u64 extra = (rem * capacity) / kRatioScale;
  return static_cast<u32>(std::min<u64>(total_units, base + extra));
}

bool Feasible(const PartitionRequest& request,
              const u32 partition_count,
              const u32 partition_units,
              const u64 limit) {
  u64 total = 0u;
  for (u32 slot = 0u; slot < partition_count; ++slot) {
    const u32 units = UnitsAtLimit(limit, CapacityForSlot(request, slot), partition_units);
    if (units == 0u) {
      return false;
    }
    total += units;
    if (total >= partition_units) {
      return true;
    }
  }
  return false;
}

u64 MinimaxLimit(const PartitionRequest& request,
                 const u32 partition_count,
                 const u32 partition_units) {
  u64 low = 0u;
  u64 high = static_cast<u64>(partition_units) * kRatioScale;
  while (low < high) {
    const u64 mid = low + ((high - low) / 2u);
    if (Feasible(request, partition_count, partition_units, mid)) {
      high = mid;
    } else {
      low = mid + 1u;
    }
  }
  return low;
}

bool RatioGreater(const u32 lhs_units,
                  const u32 lhs_capacity,
                  const u32 rhs_units,
                  const u32 rhs_capacity) {
  return static_cast<u64>(lhs_units) * rhs_capacity >
         static_cast<u64>(rhs_units) * lhs_capacity;
}

u32 TrimSlot(const std::vector<Partition>& widths, const PartitionRequest& request) {
  u32 best = 0u;
  bool found = false;
  for (u32 slot = 0u; slot < static_cast<u32>(widths.size()); ++slot) {
    if (widths[slot].end <= 1u) {
      continue;
    }
    if (!found || RatioGreater(widths[slot].end,
                               CapacityForSlot(request, slot),
                               widths[best].end,
                               CapacityForSlot(request, best))) {
      best = slot;
      found = true;
    }
  }
  return found ? best : static_cast<u32>(widths.size());
}

} // namespace

bool BuildMinimaxUnitWidths(std::vector<Partition>& out_widths,
                            const PartitionRequest& request,
                            const u32 partition_count,
                            const u32 partition_units) {
  const u64 limit = MinimaxLimit(request, partition_count, partition_units);
  out_widths.clear();
  u64 total = 0u;
  for (u32 slot = 0u; slot < partition_count; ++slot) {
    const u32 width =
        UnitsAtLimit(limit, CapacityForSlot(request, slot), partition_units);
    if (width == 0u) {
      return false;
    }
    out_widths.push_back(Partition{.worker_index = slot, .end = width});
    total += width;
  }
  while (total > partition_units) {
    const u32 slot = TrimSlot(out_widths, request);
    if (slot >= out_widths.size()) {
      return false;
    }
    out_widths[slot].end -= 1u;
    total -= 1u;
  }
  return total == partition_units;
}

} // namespace rund::kernel::schedule::planner::capacity

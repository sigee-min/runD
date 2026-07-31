#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/bindings/sort.hpp"
#include "../buffer/batch.hpp"

#include <cstring>

namespace rund::node::accel::detail {

[[nodiscard]] inline rund::AccelCheck ReadCpuSortCount(
    const rund::AccelDevice &pick, const rund::kernel::SortPlan &plan,
    const SortBinds &bindings, rund::kernel::u64 &count) {
  count = plan.element_count;
  if (bindings.logical_count_handle == nullptr) {
    return {true, "ok"};
  }
  CpuBufferResult resident{};
  CpuResidentReq req[] = {{.ref = bindings.logical_count,
                           .handle = bindings.logical_count_handle,
                           .usage = rund::kernel::kResidentUsageRead,
                           .out = &resident}};
  LookupCpuResidentBatch(pick, req);
  if (!resident.check.ok || resident.buffer == nullptr) {
    return {false, "compute_sort_invalid"};
  }
  if (plan.count_source == rund::kernel::ComputeCountSource::BufferU64) {
    std::memcpy(&count, resident.buffer->data.data(), sizeof(count));
  } else {
    rund::kernel::u32 narrow = 0u;
    std::memcpy(&narrow, resident.buffer->data.data(), sizeof(narrow));
    count = narrow;
  }
  return count <= plan.element_count
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_bounded_count_invalid"};
}

} // namespace rund::node::accel::detail

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../compact/shape.hpp"
#include "buffer/batch.hpp"

#include <kernel/program/compute/compact/reference.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteCpuCompact(const rund::AccelDevice &pick,
                                   const rund::kernel::CompactDesc &desc,
                                   const rund::kernel::CompactPlan &plan,
                                   const CompactBinds &bindings) {
  if (!pick.check.ok || !CompactShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_compact_invalid"};
  }
  CpuBufferResult flags{};
  CpuBufferResult output{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.flags,
                     .handle = bindings.flags_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &flags},
      CpuResidentReq{.ref = bindings.output,
                     .handle = bindings.output_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!flags.check.ok || !output.check.ok || adapter == nullptr) {
    return rund::AccelCheck{false, "compute_compact_invalid"};
  }
  rund::kernel::u64 output_count = 0u;
  const rund::kernel::CompactResult result =
      rund::kernel::ReferenceCompactIdsU32(
          reinterpret_cast<const rund::kernel::u32 *>(
              flags.buffer->data.data()),
          plan.element_count, plan.output_capacity,
          reinterpret_cast<rund::kernel::u32 *>(output.buffer->data.data()),
          &output_count);
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  const rund::kernel::u64 dispatches =
      2u + (plan.output_capacity < plan.element_count ? 1u : 0u);
  RecordCpuDispatches(*adapter, dispatches);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail

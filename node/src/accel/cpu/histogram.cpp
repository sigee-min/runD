#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../histogram/shape.hpp"
#include "buffer/batch.hpp"

#include <kernel/program/compute/histogram/reference.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck
ExecuteCpuHistogram(const rund::AccelDevice &pick,
                    const rund::kernel::HistogramDesc &desc,
                    const rund::kernel::HistogramPlan &plan,
                    const HistogramBinds &bindings) {
  if (!pick.check.ok || !HistogramShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_histogram_invalid"};
  }
  CpuBufferResult bins{};
  CpuBufferResult counts{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.bins,
                     .handle = bindings.bins_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &bins},
      CpuResidentReq{.ref = bindings.counts,
                     .handle = bindings.counts_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &counts},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!bins.check.ok || !counts.check.ok || adapter == nullptr) {
    return rund::AccelCheck{false, "compute_histogram_invalid"};
  }
  const rund::kernel::HistogramResult result =
      rund::kernel::ReferenceHistogramU32(
          reinterpret_cast<const rund::kernel::u32 *>(bins.buffer->data.data()),
          reinterpret_cast<rund::kernel::u32 *>(counts.buffer->data.data()),
          plan.element_count, plan.bin_count);
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail

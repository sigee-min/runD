#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../domain.hpp"
#include "../kernel/bindings/scan.hpp"
#include "../scan/count.hpp"
#include "buffer/batch.hpp"
#include "scan/canonical.hpp"

#include <kernel/program/compute/scan/reference.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteCpuScan(const rund::AccelDevice &pick,
                                const rund::kernel::ScanPlan &plan,
                                const rund::kernel::ComputeDomain domain,
                                const ScanBinds &bindings) {
  if (!pick.check.ok || !plan.ok) {
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  CpuBufferResult input{};
  CpuBufferResult output{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.input,
                     .handle = bindings.input_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &input},
      CpuResidentReq{.ref = bindings.output,
                     .handle = bindings.output_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!input.check.ok || !output.check.ok || adapter == nullptr) {
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }

  rund::kernel::u64 total = 0u;
  rund::kernel::ScanResult result{};
  if (IsSignedDomain(domain) &&
      plan.element == rund::kernel::ScanElement::U32) {
    const rund::AccelCheck scanned = CanonicalSignedScan(
        reinterpret_cast<const rund::kernel::u32 *>(input.buffer->data.data()),
        reinterpret_cast<rund::kernel::u32 *>(output.buffer->data.data()),
        plan.element_count, plan.op == rund::kernel::ScanOp::InclusiveSum);
    if (!scanned.ok) {
      return scanned;
    }
  } else if (IsSignedDomain(domain)) {
    const rund::AccelCheck scanned = CanonicalSignedScan(
        reinterpret_cast<const rund::kernel::u64 *>(input.buffer->data.data()),
        reinterpret_cast<rund::kernel::u64 *>(output.buffer->data.data()),
        plan.element_count, plan.op == rund::kernel::ScanOp::InclusiveSum);
    if (!scanned.ok) {
      return scanned;
    }
  } else if (plan.element == rund::kernel::ScanElement::U32 &&
             plan.op == rund::kernel::ScanOp::ExclusiveSum) {
    result = rund::kernel::ReferenceExclusiveScanU32(
        reinterpret_cast<const rund::kernel::u32 *>(input.buffer->data.data()),
        reinterpret_cast<rund::kernel::u32 *>(output.buffer->data.data()),
        plan.element_count, &total);
  } else if (plan.element == rund::kernel::ScanElement::U32) {
    result = rund::kernel::ReferenceInclusiveScanU32(
        reinterpret_cast<const rund::kernel::u32 *>(input.buffer->data.data()),
        reinterpret_cast<rund::kernel::u32 *>(output.buffer->data.data()),
        plan.element_count, &total);
  } else if (plan.op == rund::kernel::ScanOp::ExclusiveSum) {
    result = rund::kernel::ReferenceExclusiveScanU64(
        reinterpret_cast<const rund::kernel::u64 *>(input.buffer->data.data()),
        reinterpret_cast<rund::kernel::u64 *>(output.buffer->data.data()),
        plan.element_count, &total);
  } else {
    result = rund::kernel::ReferenceInclusiveScanU64(
        reinterpret_cast<const rund::kernel::u64 *>(input.buffer->data.data()),
        reinterpret_cast<rund::kernel::u64 *>(output.buffer->data.data()),
        plan.element_count, &total);
  }
  if (!IsSignedDomain(domain) && !result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, EncodedScanDispatchCount(plan));
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail

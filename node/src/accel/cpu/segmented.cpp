#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../domain.hpp"
#include "../segmented/shape.hpp"
#include "buffer/batch.hpp"

#include <kernel/program/compute/segmented/scan/reference.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck
ExecuteCpuSegmentedScan(const rund::AccelDevice &pick,
                        const rund::kernel::SegmentedScanDesc &desc,
                        const rund::kernel::SegmentedScanPlan &plan,
                        const rund::kernel::ComputeDomain domain,
                        const SegmentedScanBinds &bindings) {
  if (!pick.check.ok || !SegmentedScanShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
  }
  CpuBufferResult input{};
  CpuBufferResult heads{};
  CpuBufferResult output{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.input,
                     .handle = bindings.input_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &input},
      CpuResidentReq{.ref = bindings.heads,
                     .handle = bindings.heads_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &heads},
      CpuResidentReq{.ref = bindings.output,
                     .handle = bindings.output_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!input.check.ok || !heads.check.ok || !output.check.ok ||
      adapter == nullptr) {
    return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
  }

  const auto *const head_data =
      reinterpret_cast<const rund::kernel::u32 *>(heads.buffer->data.data());
  rund::kernel::SegmentedScanResult result{};
  if (IsSignedDomain(domain) &&
      plan.element == rund::kernel::SegmentedScanElement::U32) {
    result = rund::kernel::ReferenceSignedSegmentedScan(
        reinterpret_cast<const rund::kernel::i32 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::i32 *>(output.buffer->data.data()),
        plan.element_count,
        plan.op == rund::kernel::SegmentedScanOp::InclusiveSum);
  } else if (IsSignedDomain(domain)) {
    result = rund::kernel::ReferenceSignedSegmentedScan(
        reinterpret_cast<const rund::kernel::i64 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::i64 *>(output.buffer->data.data()),
        plan.element_count,
        plan.op == rund::kernel::SegmentedScanOp::InclusiveSum);
  } else if (plan.element == rund::kernel::SegmentedScanElement::U32 &&
             plan.op == rund::kernel::SegmentedScanOp::ExclusiveSum) {
    result = rund::kernel::ReferenceExclusiveSegmentedScanU32(
        reinterpret_cast<const rund::kernel::u32 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::u32 *>(output.buffer->data.data()),
        plan.element_count);
  } else if (plan.element == rund::kernel::SegmentedScanElement::U32) {
    result = rund::kernel::ReferenceInclusiveSegmentedScanU32(
        reinterpret_cast<const rund::kernel::u32 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::u32 *>(output.buffer->data.data()),
        plan.element_count);
  } else if (plan.op == rund::kernel::SegmentedScanOp::ExclusiveSum) {
    result = rund::kernel::ReferenceExclusiveSegmentedScanU64(
        reinterpret_cast<const rund::kernel::u64 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::u64 *>(output.buffer->data.data()),
        plan.element_count);
  } else {
    result = rund::kernel::ReferenceInclusiveSegmentedScanU64(
        reinterpret_cast<const rund::kernel::u64 *>(input.buffer->data.data()),
        head_data,
        reinterpret_cast<rund::kernel::u64 *>(output.buffer->data.data()),
        plan.element_count);
  }
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail

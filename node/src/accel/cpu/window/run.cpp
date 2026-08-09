#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../window/shape.hpp"
#include "../buffer/batch.hpp"
#include <kernel/program/compute/window/reference.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteCpuWindow(const rund::AccelDevice &pick,
                                  const rund::kernel::WindowDesc &desc,
                                  const rund::kernel::WindowPlan &plan,
                                  const RangePlan &range,
                                  const RangeBinds &bindings) {
  if (!pick.check.ok || !WindowShapeOk(desc, plan, bindings) ||
      !WindowRangePlanMatches(plan, range) ||
      range.candidate().disposition() != RangePath::Direct ||
      range.temporary_count() != 0u) {
    return rund::AccelCheck{false, "compute_window_invalid"};
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
  if (!input.check.ok || !output.check.ok || input.buffer == nullptr ||
      output.buffer == nullptr || adapter == nullptr) {
    return rund::AccelCheck{false, "compute_window_invalid"};
  }

  const void *const input_data = input.buffer->data.data();
  void *const output_data = output.buffer->data.data();
  rund::kernel::WindowResult result{};
  switch (plan.domain) {
  case rund::kernel::ComputeDomain::I32:
    result = rund::kernel::ReferenceWindowI32(
        static_cast<const rund::kernel::i32 *>(input_data),
        static_cast<rund::kernel::i32 *>(output_data), plan);
    break;
  case rund::kernel::ComputeDomain::U32:
    result = rund::kernel::ReferenceWindowU32(
        static_cast<const rund::kernel::u32 *>(input_data),
        static_cast<rund::kernel::u32 *>(output_data), plan);
    break;
  case rund::kernel::ComputeDomain::I64:
    result = rund::kernel::ReferenceWindowI64(
        static_cast<const rund::kernel::i64 *>(input_data),
        static_cast<rund::kernel::i64 *>(output_data), plan);
    break;
  case rund::kernel::ComputeDomain::U64:
    result = rund::kernel::ReferenceWindowU64(
        static_cast<const rund::kernel::u64 *>(input_data),
        static_cast<rund::kernel::u64 *>(output_data), plan);
    break;
  case rund::kernel::ComputeDomain::Fixed:
    result = plan.element == rund::kernel::WindowElement::U32
                 ? rund::kernel::ReferenceWindowFixedI32(
                       static_cast<const rund::kernel::i32 *>(input_data),
                       static_cast<rund::kernel::i32 *>(output_data), plan)
                 : rund::kernel::ReferenceWindowFixedI64(
                       static_cast<const rund::kernel::i64 *>(input_data),
                       static_cast<rund::kernel::i64 *>(output_data), plan);
    break;
  }
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, range.stage_count());
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail

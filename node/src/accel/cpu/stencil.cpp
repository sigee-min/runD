#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../domain.hpp"
#include "../stencil/shape.hpp"
#include "buffer/batch.hpp"
#include "stencil/reference.hpp"
namespace rund::node::accel::detail {

rund::AccelCheck ExecuteCpuStencil(const rund::AccelDevice &pick,
                                   const rund::kernel::StencilDesc &desc,
                                   const rund::kernel::StencilPlan &plan,
                                   const rund::kernel::ComputeDomain domain,
                                   const StencilBinds &bindings) {
  if (!pick.check.ok || !StencilShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_stencil_invalid"};
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
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  const bool u32 = plan.element == rund::kernel::StencilElement::U32;
  const bool signed_domain = IsSignedDomain(domain);
  rund::kernel::StencilResult result{};
  if (signed_domain && u32) {
    result = ExecuteSignedStencilReference(
        plan.op,
        reinterpret_cast<const rund::kernel::i32 *>(input.buffer->data.data()),
        reinterpret_cast<rund::kernel::i32 *>(output.buffer->data.data()),
        plan.element_count, plan.radius);
  } else if (signed_domain) {
    result = ExecuteSignedStencilReference(
        plan.op,
        reinterpret_cast<const rund::kernel::i64 *>(input.buffer->data.data()),
        reinterpret_cast<rund::kernel::i64 *>(output.buffer->data.data()),
        plan.element_count, plan.radius);
  } else {
    result = u32 ? ExecuteStencilReference(
                       plan.op,
                       reinterpret_cast<const rund::kernel::u32 *>(
                           input.buffer->data.data()),
                       reinterpret_cast<rund::kernel::u32 *>(
                           output.buffer->data.data()),
                       plan.element_count, plan.radius)
                 : ExecuteStencilReference(
                       plan.op,
                       reinterpret_cast<const rund::kernel::u64 *>(
                           input.buffer->data.data()),
                       reinterpret_cast<rund::kernel::u64 *>(
                           output.buffer->data.data()),
                       plan.element_count, plan.radius);
  }
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}
} // namespace rund::node::accel::detail

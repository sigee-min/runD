#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../domain.hpp"
#include "../reduce/shape.hpp"
#include "buffer/batch.hpp"
#include "reduce/reference.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteCpuReduce(const rund::AccelDevice &pick,
                                  const rund::kernel::ReduceDesc &desc,
                                  const rund::kernel::ReducePlan &plan,
                                  const rund::kernel::ComputeDomain domain,
                                  const ReduceBinds &bindings) {
  if (!pick.check.ok || !ReduceShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_reduce_invalid"};
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
    return rund::AccelCheck{false, "compute_reduce_invalid"};
  }
  const bool u32 = plan.element == rund::kernel::ReduceElement::U32;
  const bool signed_domain = IsSignedDomain(domain);
  const rund::kernel::ReduceResult result =
      signed_domain && u32
          ? ReferenceReduceSigned(desc.op,
                                  reinterpret_cast<const rund::kernel::i32 *>(
                                      input.buffer->data.data()),
                                  reinterpret_cast<rund::kernel::i32 *>(
                                      output.buffer->data.data()),
                                  plan.element_count)
      : signed_domain
          ? ReferenceReduceSigned(desc.op,
                                  reinterpret_cast<const rund::kernel::i64 *>(
                                      input.buffer->data.data()),
                                  reinterpret_cast<rund::kernel::i64 *>(
                                      output.buffer->data.data()),
                                  plan.element_count)
      : u32 ? ReferenceReduceU32(desc.op,
                                 reinterpret_cast<const rund::kernel::u32 *>(
                                     input.buffer->data.data()),
                                 reinterpret_cast<rund::kernel::u32 *>(
                                     output.buffer->data.data()),
                                 plan.element_count)
            : ReferenceReduceU64(desc.op,
                                 reinterpret_cast<const rund::kernel::u64 *>(
                                     input.buffer->data.data()),
                                 reinterpret_cast<rund::kernel::u64 *>(
                                     output.buffer->data.data()),
                                 plan.element_count);
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}
} // namespace rund::node::accel::detail

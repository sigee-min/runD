#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../transform/shape.hpp"
#include "buffer/batch.hpp"

#include <kernel/program/compute/transform/reference.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck
ExecuteCpuTransform(const rund::AccelDevice &pick,
                    const rund::kernel::TransformDesc &desc,
                    const rund::kernel::TransformPlan &plan,
                    const TransformBinds &bindings) {
  if (!pick.check.ok || !TransformShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_transform_invalid"};
  }
  CpuBufferResult input_real{};
  CpuBufferResult input_imag{};
  CpuBufferResult output_real{};
  CpuBufferResult output_imag{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.input_real,
                     .handle = bindings.input_real_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &input_real},
      CpuResidentReq{.ref = bindings.input_imag,
                     .handle = bindings.input_imag_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &input_imag},
      CpuResidentReq{.ref = bindings.output_real,
                     .handle = bindings.output_real_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output_real},
      CpuResidentReq{.ref = bindings.output_imag,
                     .handle = bindings.output_imag_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output_imag},
  };
  LookupCpuResidentBatch(pick, reqs);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!input_real.check.ok || !input_imag.check.ok || !output_real.check.ok ||
      !output_imag.check.ok || adapter == nullptr) {
    return rund::AccelCheck{false, "compute_transform_invalid"};
  }

  const bool wide =
      bindings.input_real->element_bytes == sizeof(rund::kernel::i64);
  const rund::kernel::TransformResult result =
      wide ? rund::kernel::ReferenceFourierSplitI64(
                 reinterpret_cast<const rund::kernel::i64 *>(
                     input_real.buffer->data.data()),
                 reinterpret_cast<const rund::kernel::i64 *>(
                     input_imag.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i64 *>(
                     output_real.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i64 *>(
                     output_imag.buffer->data.data()),
                 plan)
           : rund::kernel::ReferenceFourierSplitI32(
                 reinterpret_cast<const rund::kernel::i32 *>(
                     input_real.buffer->data.data()),
                 reinterpret_cast<const rund::kernel::i32 *>(
                     input_imag.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i32 *>(
                     output_real.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i32 *>(
                     output_imag.buffer->data.data()),
                 plan);
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail

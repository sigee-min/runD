#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../matrix/shape.hpp"
#include "buffer/batch.hpp"

#include <kernel/program/compute/matrix/reference.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteCpuMatrix(const rund::AccelDevice &pick,
                                  const rund::kernel::MatrixDesc &desc,
                                  const rund::kernel::MatrixPlan &plan,
                                  const MatrixBinds &bindings) {
  if (!pick.check.ok || !MatrixShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_matrix_invalid"};
  }
  CpuBufferResult left{};
  CpuBufferResult right{};
  CpuBufferResult output{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.left,
                     .handle = bindings.left_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &left},
      CpuResidentReq{.ref = bindings.right,
                     .handle = bindings.right_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &right},
      CpuResidentReq{.ref = bindings.output,
                     .handle = bindings.output_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output},
  };
  const std::size_t req_count =
      plan.op == rund::kernel::MatrixOp::Transpose ? 2u : 3u;
  if (plan.op == rund::kernel::MatrixOp::Transpose) {
    reqs[1] = reqs[2];
  }
  LookupCpuResidentBatch(pick, reqs, req_count);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!left.check.ok || !output.check.ok || adapter == nullptr ||
      (plan.op != rund::kernel::MatrixOp::Transpose && !right.check.ok)) {
    return rund::AccelCheck{false, "compute_matrix_invalid"};
  }

  const bool wide = plan.element_bytes == sizeof(rund::kernel::i64);
  rund::kernel::MatrixResult result{};
  if (plan.op == rund::kernel::MatrixOp::Transpose) {
    result = wide ? rund::kernel::ReferenceMatrixTransposeI64(
                        reinterpret_cast<const rund::kernel::i64 *>(
                            left.buffer->data.data()),
                        reinterpret_cast<rund::kernel::i64 *>(
                            output.buffer->data.data()),
                        plan)
                  : rund::kernel::ReferenceMatrixTransposeI32(
                        reinterpret_cast<const rund::kernel::i32 *>(
                            left.buffer->data.data()),
                        reinterpret_cast<rund::kernel::i32 *>(
                            output.buffer->data.data()),
                        plan);
  } else {
    result = wide ? rund::kernel::ReferenceMatrixMulI64(
                        reinterpret_cast<const rund::kernel::i64 *>(
                            left.buffer->data.data()),
                        reinterpret_cast<const rund::kernel::i64 *>(
                            right.buffer->data.data()),
                        reinterpret_cast<rund::kernel::i64 *>(
                            output.buffer->data.data()),
                        plan)
                  : rund::kernel::ReferenceMatrixMulI32(
                        reinterpret_cast<const rund::kernel::i32 *>(
                            left.buffer->data.data()),
                        reinterpret_cast<const rund::kernel::i32 *>(
                            right.buffer->data.data()),
                        reinterpret_cast<rund::kernel::i32 *>(
                            output.buffer->data.data()),
                        plan);
  }
  if (!result.ok) {
    return rund::AccelCheck{false, result.reason};
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail

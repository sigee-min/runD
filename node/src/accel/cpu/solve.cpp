#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../solve/shape.hpp"
#include "buffer/batch.hpp"

#include <kernel/program/compute/solve/reference.hpp>

namespace rund::node::accel::detail {

namespace {

[[nodiscard]] rund::AccelCheck
SolveCheck(const rund::kernel::SolveResult &result) {
  rund::AccelCheck check{result.ok, result.reason};
  check.failed_batches = result.failed_batches;
  check.first_failed_batch = result.first_failed_batch;
  check.first_status = static_cast<std::uint32_t>(result.first_status);
  return check;
}

} // namespace

rund::AccelCheck ExecuteCpuSolve(const rund::AccelDevice &pick,
                                 const rund::kernel::SolveDesc &desc,
                                 const rund::kernel::SolvePlan &plan,
                                 const SolveBinds &bindings) {
  if (!pick.check.ok || !SolveShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_solve_invalid"};
  }
  CpuBufferResult primary{};
  CpuBufferResult aux{};
  CpuBufferResult rhs{};
  CpuBufferResult output{};
  CpuBufferResult status{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.primary,
                     .handle = bindings.primary_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &primary},
      CpuResidentReq{.ref = bindings.aux,
                     .handle = bindings.aux_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &aux},
      CpuResidentReq{.ref = bindings.rhs,
                     .handle = bindings.rhs_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &rhs},
      CpuResidentReq{.ref = bindings.output,
                     .handle = bindings.output_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &output},
      CpuResidentReq{.ref = bindings.status,
                     .handle = bindings.status_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &status},
  };
  const bool needs_aux = plan.input == rund::kernel::SolveInput::Factor &&
                         plan.factor == rund::kernel::FactorOp::LU;
  if (!needs_aux) {
    reqs[1] = reqs[2];
    reqs[2] = reqs[3];
    reqs[3] = reqs[4];
  }
  LookupCpuResidentBatch(pick, reqs, needs_aux ? 5u : 4u);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!primary.check.ok || !rhs.check.ok || !output.check.ok ||
      !status.check.ok || adapter == nullptr || (needs_aux && !aux.check.ok)) {
    return rund::AccelCheck{false, "compute_solve_invalid"};
  }

  const bool wide = plan.element_bytes == sizeof(rund::kernel::i64);
  const rund::kernel::SolveResult result =
      wide ? rund::kernel::ReferenceSolveI64(
                 reinterpret_cast<const rund::kernel::i64 *>(
                     primary.buffer->data.data()),
                 needs_aux ? reinterpret_cast<const rund::kernel::u32 *>(
                                 aux.buffer->data.data())
                           : nullptr,
                 reinterpret_cast<const rund::kernel::i64 *>(
                     rhs.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i64 *>(
                     output.buffer->data.data()),
                 reinterpret_cast<rund::kernel::u32 *>(
                     status.buffer->data.data()),
                 plan)
           : rund::kernel::ReferenceSolveI32(
                 reinterpret_cast<const rund::kernel::i32 *>(
                     primary.buffer->data.data()),
                 needs_aux ? reinterpret_cast<const rund::kernel::u32 *>(
                                 aux.buffer->data.data())
                           : nullptr,
                 reinterpret_cast<const rund::kernel::i32 *>(
                     rhs.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i32 *>(
                     output.buffer->data.data()),
                 reinterpret_cast<rund::kernel::u32 *>(
                     status.buffer->data.data()),
                 plan);
  if (!result.ok) {
    return SolveCheck(result);
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return SolveCheck(result);
}

} // namespace rund::node::accel::detail

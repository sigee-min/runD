#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../factor/shape.hpp"
#include "buffer/batch.hpp"

#include <kernel/program/compute/factor/reference.hpp>

namespace rund::node::accel::detail {

namespace {

[[nodiscard]] rund::AccelCheck
FactorCheck(const rund::kernel::FactorResult &result) {
  rund::AccelCheck check{result.ok, result.reason};
  check.failed_batches = result.failed_batches;
  check.first_failed_batch = result.first_failed_batch;
  check.first_status = static_cast<std::uint32_t>(result.first_status);
  return check;
}

} // namespace

rund::AccelCheck ExecuteCpuFactor(const rund::AccelDevice &pick,
                                  const rund::kernel::FactorDesc &desc,
                                  const rund::kernel::FactorPlan &plan,
                                  const FactorBinds &bindings) {
  if (!pick.check.ok || !FactorShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_factor_invalid"};
  }
  CpuBufferResult input{};
  CpuBufferResult factor{};
  CpuBufferResult aux{};
  CpuBufferResult status{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.input,
                     .handle = bindings.input_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &input},
      CpuResidentReq{.ref = bindings.factor,
                     .handle = bindings.factor_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &factor},
      CpuResidentReq{.ref = bindings.aux,
                     .handle = bindings.aux_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &aux},
      CpuResidentReq{.ref = bindings.status,
                     .handle = bindings.status_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &status},
  };
  const bool needs_aux = plan.op == rund::kernel::FactorOp::LU;
  if (!needs_aux) {
    reqs[2] = reqs[3];
  }
  LookupCpuResidentBatch(pick, reqs, needs_aux ? 4u : 3u);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!input.check.ok || !factor.check.ok || !status.check.ok ||
      adapter == nullptr || (needs_aux && !aux.check.ok)) {
    return rund::AccelCheck{false, "compute_factor_invalid"};
  }

  const bool wide = plan.element_bytes == sizeof(rund::kernel::i64);
  const rund::kernel::FactorResult result =
      wide ? rund::kernel::ReferenceFactorI64(
                 reinterpret_cast<const rund::kernel::i64 *>(
                     input.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i64 *>(
                     factor.buffer->data.data()),
                 needs_aux ? reinterpret_cast<rund::kernel::u32 *>(
                                 aux.buffer->data.data())
                           : nullptr,
                 reinterpret_cast<rund::kernel::u32 *>(
                     status.buffer->data.data()),
                 plan)
           : rund::kernel::ReferenceFactorI32(
                 reinterpret_cast<const rund::kernel::i32 *>(
                     input.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i32 *>(
                     factor.buffer->data.data()),
                 needs_aux ? reinterpret_cast<rund::kernel::u32 *>(
                                 aux.buffer->data.data())
                           : nullptr,
                 reinterpret_cast<rund::kernel::u32 *>(
                     status.buffer->data.data()),
                 plan);
  if (!result.ok) {
    return FactorCheck(result);
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return FactorCheck(result);
}

} // namespace rund::node::accel::detail

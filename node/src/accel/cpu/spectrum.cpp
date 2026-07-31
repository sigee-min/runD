#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../spectrum/shape.hpp"
#include "buffer/batch.hpp"

#include <kernel/program/compute/spectrum/reference.hpp>

namespace rund::node::accel::detail {

namespace {

[[nodiscard]] rund::AccelCheck
SpectrumCheck(const rund::kernel::SpectrumResult &result) {
  rund::AccelCheck check{result.ok, result.reason};
  check.failed_batches = result.failed_batches;
  check.first_failed_batch = result.first_failed_batch;
  check.first_status = static_cast<std::uint32_t>(result.first_status);
  return check;
}

} // namespace

rund::AccelCheck ExecuteCpuSpectrum(const rund::AccelDevice &pick,
                                    const rund::kernel::SpectrumDesc &desc,
                                    const rund::kernel::SpectrumPlan &plan,
                                    const SpectrumBinds &bindings) {
  if (!pick.check.ok || !SpectrumShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_spectrum_invalid"};
  }
  CpuBufferResult input{};
  CpuBufferResult values{};
  CpuBufferResult vectors{};
  CpuBufferResult status{};
  CpuResidentReq reqs[] = {
      CpuResidentReq{.ref = bindings.input,
                     .handle = bindings.input_handle,
                     .usage = rund::kernel::kResidentUsageRead,
                     .out = &input},
      CpuResidentReq{.ref = bindings.values,
                     .handle = bindings.values_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &values},
      CpuResidentReq{.ref = bindings.vectors,
                     .handle = bindings.vectors_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &vectors},
      CpuResidentReq{.ref = bindings.status,
                     .handle = bindings.status_handle,
                     .usage = rund::kernel::kResidentUsageWrite,
                     .out = &status},
  };
  const bool has_vectors = plan.vector_count != 0u;
  if (!has_vectors) {
    reqs[2] = reqs[3];
  }
  LookupCpuResidentBatch(pick, reqs, has_vectors ? 4u : 3u);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!input.check.ok || !values.check.ok || !status.check.ok ||
      adapter == nullptr || (has_vectors && !vectors.check.ok)) {
    return rund::AccelCheck{false, "compute_spectrum_invalid"};
  }

  const bool wide = plan.element_bytes == sizeof(rund::kernel::i64);
  const rund::kernel::SpectrumResult result =
      wide ? rund::kernel::ReferenceSpectrumI64(
                 reinterpret_cast<const rund::kernel::i64 *>(
                     input.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i64 *>(
                     values.buffer->data.data()),
                 has_vectors ? reinterpret_cast<rund::kernel::i64 *>(
                                   vectors.buffer->data.data())
                             : nullptr,
                 reinterpret_cast<rund::kernel::u32 *>(
                     status.buffer->data.data()),
                 plan)
           : rund::kernel::ReferenceSpectrumI32(
                 reinterpret_cast<const rund::kernel::i32 *>(
                     input.buffer->data.data()),
                 reinterpret_cast<rund::kernel::i32 *>(
                     values.buffer->data.data()),
                 has_vectors ? reinterpret_cast<rund::kernel::i32 *>(
                                   vectors.buffer->data.data())
                             : nullptr,
                 reinterpret_cast<rund::kernel::u32 *>(
                     status.buffer->data.data()),
                 plan);
  if (!result.ok) {
    return SpectrumCheck(result);
  }
  RecordCpuDispatches(*adapter, plan.pass_count);
  return SpectrumCheck(result);
}

} // namespace rund::node::accel::detail

#include <accel/check.hpp>
#include <accel/context/value.hpp>

#include "local.hpp"

#include "../../../cpu/buffer/batch.hpp"
#include "../../../cpu/local.hpp"
#include <cstddef>
#include <limits>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] rund::AccelCheck Reject(const char *const reason) noexcept {
  return rund::AccelCheck{false, reason};
}

} // namespace

rund::AccelCheck ExecuteCpuMapStep(const rund::AccelDevice &pick,
                                   const PlannedStep &planned,
                                   const DispatchWindowStorage &windows,
                                   const rund::kernel::BindingSet &binds) {
  if (binds.resident_inputs.count >
      static_cast<rund::kernel::u64>(std::numeric_limits<std::size_t>::max())) {
    return Reject("accel_kernel_run_invalid");
  }
  const auto input_count =
      static_cast<std::size_t>(binds.resident_inputs.count);
  if (binds.resident_outputs.count == 0u ||
      binds.resident_outputs.count >
          static_cast<rund::kernel::u64>(
              std::numeric_limits<std::size_t>::max()) ||
      input_count > std::numeric_limits<std::size_t>::max() -
                        static_cast<std::size_t>(binds.resident_outputs.count)) {
    return Reject("accel_kernel_run_invalid");
  }
  const auto output_count =
      static_cast<std::size_t>(binds.resident_outputs.count);
  const auto binding_count = input_count + output_count;
  InlineIndexedStorage<rund::kernel::BufferSpan, kInlineBindingCapacity> inputs;
  InlineIndexedStorage<rund::kernel::OutputSpan, kInlineBindingCapacity>
      outputs;
  InlineIndexedStorage<CpuBufferResult, kInlineBindingCapacity> results;
  InlineIndexedStorage<CpuResidentReq, kInlineBindingCapacity> reqs;
  inputs.resize(input_count);
  outputs.resize(output_count);
  results.resize(binding_count);
  reqs.resize(binding_count);
  for (std::size_t index = 0u; index < input_count; ++index) {
    auto *const req = reqs.get(index);
    if (req == nullptr) {
      return Reject("accel_kernel_run_invalid");
    }
    *req = CpuResidentReq{.ref = binds.resident_inputs.ref(index),
                          .handle = binds.resident_inputs.handle(index),
                          .usage = rund::kernel::kResidentUsageRead,
                          .out = results.get(index)};
  }
  for (std::size_t index = 0u; index < output_count; ++index) {
    auto *const output_req = reqs.get(input_count + index);
    if (output_req == nullptr) {
      return Reject("accel_kernel_run_invalid");
    }
    *output_req =
        CpuResidentReq{.ref = binds.resident_outputs.ref(index),
                       .handle = binds.resident_outputs.handle(index),
                       .usage = rund::kernel::kResidentUsageWrite,
                       .out = results.get(input_count + index)};
  }
  if (!results.valid() || !reqs.valid()) {
    return Reject("accel_kernel_run_invalid");
  }
  LookupCpuResidentBatch(pick, reqs.data(), reqs.size());

  for (std::size_t index = 0u; index < input_count; ++index) {
    const CpuBufferResult *const lookup = results.get(index);
    if (lookup == nullptr || !lookup->check.ok) {
      return Reject(lookup != nullptr ? lookup->check.reason
                                      : "accel_kernel_run_invalid");
    }
    auto *const input = inputs.get(index);
    if (input == nullptr) {
      return Reject("accel_kernel_run_invalid");
    }
    const rund::kernel::ResidentBufferRef *const resident =
        binds.resident_inputs.ref(index);
    if (resident == nullptr) {
      return Reject("accel_kernel_run_invalid");
    }
    *input = rund::kernel::BufferSpan{
        .data = lookup->buffer->data.data() + resident->offset_bytes,
        .element_bytes = resident->element_bytes,
        .stride_bytes = resident->stride_bytes,
        .count = resident->count,
    };
  }
  if (!inputs.valid()) {
    return Reject("accel_kernel_run_invalid");
  }

  for (std::size_t index = 0u; index < output_count; ++index) {
    const CpuBufferResult *const lookup = results.get(input_count + index);
    if (lookup == nullptr || !lookup->check.ok) {
      return Reject(lookup != nullptr ? lookup->check.reason
                                      : "accel_kernel_run_invalid");
    }
    auto *const output = outputs.get(index);
    if (output == nullptr) {
      return Reject("accel_kernel_run_invalid");
    }
    const rund::kernel::ResidentBufferRef *const resident =
        binds.resident_outputs.ref(index);
    if (resident == nullptr) {
      return Reject("accel_kernel_run_invalid");
    }
    *output = rund::kernel::OutputSpan{
        .data = lookup->buffer->data.data() + resident->offset_bytes,
        .element_bytes = resident->element_bytes,
        .stride_bytes = resident->stride_bytes,
        .count = resident->count,
    };
  }
  if (!outputs.valid()) {
    return Reject("accel_kernel_run_invalid");
  }

  rund::kernel::BindingSet cpu_binds = binds;
  cpu_binds.input_buffers = inputs.data();
  cpu_binds.input_buffer_count = static_cast<rund::kernel::u64>(inputs.size());
  cpu_binds.output_buffers = outputs.data();
  cpu_binds.output_buffer_count =
      static_cast<rund::kernel::u64>(outputs.size());
  cpu_binds.staged_output = nullptr;
  cpu_binds.staged_output_stride = 0u;
  cpu_binds.staged_output_count = 0u;
  cpu_binds.resident_inputs = {};
  cpu_binds.resident_outputs = {};
  if (planned.cpu_input == nullptr || planned.artifact == nullptr) {
    return Reject("accel_kernel_run_invalid");
  }
  if (!ExecuteRetainedCpu(
          pick, planned.plan, *planned.cpu_input, planned.artifact->metadata,
          windows.data(),
          static_cast<rund::kernel::u64>(windows.size()), cpu_binds)) {
    return Reject(CpuLastError(pick.backend.context));
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail

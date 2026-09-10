#include "local.hpp"

#include <kernel/core/checked.hpp>

namespace rund::kernel::binding_validation_detail {
namespace {

BindingValidation
ValidateResidentOutputs(const BindingSet &bindings,
                        const BindingObligations &obligations) noexcept {
  if (!bindings.resident_outputs.has_refs() ||
      bindings.resident_outputs.count != obligations.output_buffer_count) {
    return Reject("compute_binding_output_missing");
  }
  u64 output_bytes = 0u;
  for (u64 index = 0u; index < obligations.output_buffer_count; ++index) {
    u64 element_bytes = 0u;
    const BindingValidation expected =
        ExpectedOutputElementBytes(obligations, index, element_bytes);
    if (!expected) {
      return expected;
    }
    const ResidentBufferRef *const resident =
        bindings.resident_outputs.ref(index);
    if (resident == nullptr) {
      return Reject("compute_binding_output_missing");
    }
    const BindingValidation valid = ValidateResidentBuffer(
        *resident, obligations.tile_count, element_bytes, kResidentUsageWrite,
        obligations.allow_resident_stride);
    if (!valid) {
      return valid;
    }
    if (!checked::add(output_bytes, element_bytes)) {
      return Reject("compute_binding_output_bytes_overflow");
    }
    output_bytes += element_bytes;
  }
  return output_bytes == obligations.output_bytes_per_tile
             ? Accept()
             : Reject("compute_binding_output_bytes_mismatch");
}

BindingValidation
ValidateSpannedOutputs(const BindingSet &bindings,
                       const BindingObligations &obligations) noexcept {
  if (bindings.output_buffers == nullptr ||
      bindings.output_buffer_count != obligations.output_buffer_count) {
    return Reject("compute_binding_output_missing");
  }
  u64 output_bytes = 0u;
  for (u64 index = 0u; index < obligations.output_buffer_count; ++index) {
    const OutputSpan &span = bindings.output_buffers[index];
    u64 element_bytes = 0u;
    const BindingValidation expected =
        ExpectedOutputElementBytes(obligations, index, element_bytes);
    if (!expected) {
      return expected;
    }
    if (span.data == nullptr || span.element_bytes != element_bytes ||
        span.stride_bytes < span.element_bytes ||
        !OutputCanAddressTiles(span, obligations.tile_count)) {
      return Reject(span.element_bytes != element_bytes
                        ? "compute_binding_output_bytes_mismatch"
                        : "compute_binding_output_stride_invalid");
    }
    if (!checked::add(output_bytes, element_bytes)) {
      return Reject("compute_binding_output_bytes_overflow");
    }
    output_bytes += element_bytes;
  }
  return output_bytes == obligations.output_bytes_per_tile
             ? Accept()
             : Reject("compute_binding_output_bytes_mismatch");
}

} // namespace

BindingValidation
ValidateOutputs(const BindingSet &bindings,
                const BindingObligations &obligations) noexcept {
  const bool resident = !bindings.resident_outputs.empty();
  const bool staged = bindings.staged_output != nullptr;
  if (obligations.output_bytes_per_tile != 0u) {
    if (staged && resident) {
      return Reject("compute_binding_output_mode_conflict");
    }
    if (resident) {
      return ValidateResidentOutputs(bindings, obligations);
    }
  }
  if (!obligations.validate_staged_output ||
      obligations.output_bytes_per_tile == 0u) {
    return Accept();
  }
  if (bindings.output_buffer_count != 0u) {
    return ValidateSpannedOutputs(bindings, obligations);
  }
  if (bindings.staged_output == nullptr) {
    return Reject(!bindings.resident_inputs.empty()
                      ? "compute_binding_output_missing"
                      : "compute_binding_output_null");
  }
  if (obligations.output_buffer_count != 1u ||
      bindings.staged_output_stride < obligations.output_bytes_per_tile ||
      !OutputCanAddressTiles(bindings)) {
    return Reject("compute_binding_output_stride_invalid");
  }
  return Accept();
}

} // namespace rund::kernel::binding_validation_detail

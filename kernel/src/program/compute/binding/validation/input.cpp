#include "local.hpp"

#include <kernel/core/checked.hpp>

namespace rund::kernel::binding_validation_detail {

BindingValidation
ValidateInputs(const BindingSet &bindings,
               const BindingObligations &obligations) noexcept {
  const bool resident = !bindings.resident_inputs.empty();
  u64 input_bytes = 0u;
  if (resident) {
    if (!bindings.resident_inputs.has_refs()) {
      return Reject("compute_binding_input_null");
    }
    for (u64 index = 0u; index < obligations.input_buffer_count; ++index) {
      const ResidentBufferRef *const value =
          bindings.resident_inputs.ref(index);
      if (value == nullptr) {
        return Reject("compute_binding_input_null");
      }
      u64 element_bytes = value->element_bytes;
      const BindingValidation expected =
          ExpectedInputElementBytes(obligations, index, element_bytes);
      if (!expected) {
        return expected;
      }
      u64 count = obligations.tile_count;
      const BindingValidation count_validation =
          ExpectedInputCount(obligations, index, count);
      if (!count_validation) {
        return count_validation;
      }
      const BindingValidation valid = ValidateResidentBuffer(
          *value, count, element_bytes, kResidentUsageRead,
          obligations.allow_resident_stride);
      if (!valid) {
        return valid;
      }
      if (!checked::add(input_bytes, element_bytes)) {
        return Reject("compute_binding_input_bytes_overflow");
      }
      input_bytes += element_bytes;
    }
  } else if (obligations.input_buffer_count != 0u) {
    if (bindings.input_buffers == nullptr) {
      return Reject("compute_binding_input_null");
    }
    for (u64 index = 0u; index < obligations.input_buffer_count; ++index) {
      const BufferSpan &span = bindings.input_buffers[index];
      u64 element_bytes = span.element_bytes;
      const BindingValidation expected =
          ExpectedInputElementBytes(obligations, index, element_bytes);
      if (!expected) {
        return expected;
      }
      u64 count = obligations.tile_count;
      const BindingValidation count_validation =
          ExpectedInputCount(obligations, index, count);
      if (!count_validation) {
        return count_validation;
      }
      if (span.data == nullptr) {
        return Reject("compute_binding_input_null");
      }
      if (span.element_bytes != element_bytes || span.element_bytes == 0u ||
          span.stride_bytes < span.element_bytes ||
          !SpanCanAddressTiles(span, count)) {
        return span.element_bytes != element_bytes
                   ? Reject("compute_binding_input_bytes_mismatch")
                   : Reject("compute_binding_input_stride_invalid");
      }
      if (!checked::add(input_bytes, element_bytes)) {
        return Reject("compute_binding_input_bytes_overflow");
      }
      input_bytes += element_bytes;
    }
  }
  return input_bytes == obligations.input_bytes_per_tile
             ? Accept()
             : Reject("compute_binding_input_bytes_mismatch");
}

} // namespace rund::kernel::binding_validation_detail

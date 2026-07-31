#include <kernel/program/compute/binding/validation.hpp>
#include <kernel/core/checked.hpp>

namespace rund::kernel {
namespace {

[[nodiscard]] constexpr BindingValidation
Reject(const char *const reason) noexcept {
  return BindingValidation{.reason = reason};
}

[[nodiscard]] constexpr BindingValidation Accept() noexcept {
  return BindingValidation{.ok = true, .reason = "ok"};
}

[[nodiscard]] constexpr bool
SpanCanAddressTiles(const BufferSpan &span, const u64 tile_count) noexcept {
  if (tile_count == 0u || span.count < tile_count) {
    return false;
  }
  if (tile_count == 1u) {
    return true;
  }
  const u64 last_tile = tile_count - 1u;
  if (!checked::mul(last_tile, span.stride_bytes)) {
    return false;
  }
  const u64 offset = last_tile * span.stride_bytes;
  return checked::add(offset, span.element_bytes);
}

[[nodiscard]] constexpr bool
OutputCanAddressTiles(const BindingSet &bindings) noexcept {
  if (bindings.tile_count == 0u ||
      bindings.staged_output_count < bindings.tile_count) {
    return false;
  }
  if (bindings.tile_count == 1u) {
    return true;
  }
  const u64 last_tile = bindings.tile_count - 1u;
  if (!checked::mul(last_tile, bindings.staged_output_stride)) {
    return false;
  }
  const u64 offset = last_tile * bindings.staged_output_stride;
  return checked::add(offset, bindings.output_bytes_per_tile);
}

[[nodiscard]] constexpr bool
OutputCanAddressTiles(const OutputSpan &span, const u64 tile_count) noexcept {
  if (tile_count == 0u || span.count < tile_count) {
    return false;
  }
  if (tile_count == 1u) {
    return true;
  }
  const u64 last_tile = tile_count - 1u;
  if (!checked::mul(last_tile, span.stride_bytes)) {
    return false;
  }
  const u64 offset = last_tile * span.stride_bytes;
  return checked::add(offset, span.element_bytes);
}

[[nodiscard]] constexpr bool
HasResidentInputs(const BindingSet &bindings) noexcept {
  return !bindings.resident_inputs.empty();
}

[[nodiscard]] constexpr bool
HasResidentOutput(const BindingSet &bindings) noexcept {
  return !bindings.resident_outputs.empty();
}

[[nodiscard]] constexpr bool
HasStagedOutputClaim(const BindingSet &bindings) noexcept {
  return bindings.staged_output != nullptr;
}

[[nodiscard]] constexpr bool
InputShapeAvailable(const BindingObligations &obligations) noexcept {
  return obligations.input_element_bytes != nullptr ||
         obligations.input_element_byte_count != 0u ||
         obligations.uniform_input_element_bytes != 0u;
}

[[nodiscard]] constexpr BindingValidation
ExpectedInputElementBytes(const BindingObligations &obligations,
                          const u64 index, u64 &element_bytes) noexcept {
  if (InputShapeAvailable(obligations)) {
    if (obligations.uniform_input_element_bytes != 0u) {
      element_bytes = obligations.uniform_input_element_bytes;
      return Accept();
    }
    if (obligations.input_element_bytes == nullptr ||
        obligations.input_element_byte_count !=
            obligations.input_buffer_count ||
        index >= obligations.input_element_byte_count) {
      return Reject("compute_binding_input_bytes_mismatch");
    }
    element_bytes = obligations.input_element_bytes[index];
    return element_bytes == 0u ? Reject("compute_binding_input_bytes_mismatch")
                               : Accept();
  }
  (void)index;
  return Accept();
}

[[nodiscard]] constexpr BindingValidation
ExpectedInputCount(const BindingObligations &obligations, const u64 index,
                   u64 &count) noexcept {
  if (obligations.input_counts == nullptr &&
      obligations.input_count_count == 0u) {
    count = obligations.tile_count;
    return Accept();
  }
  if (obligations.input_counts == nullptr ||
      obligations.input_count_count != obligations.input_buffer_count ||
      index >= obligations.input_count_count ||
      obligations.input_counts[index] == 0u) {
    return Reject("compute_binding_input_count_mismatch");
  }
  count = obligations.input_counts[index];
  return Accept();
}

[[nodiscard]] constexpr BindingValidation
ExpectedOutputElementBytes(const BindingObligations &obligations,
                           const u64 index, u64 &element_bytes) noexcept {
  if (obligations.uniform_output_element_bytes != 0u) {
    element_bytes = obligations.uniform_output_element_bytes;
    return Accept();
  }
  if (obligations.output_element_bytes == nullptr ||
      obligations.output_element_byte_count !=
          obligations.output_buffer_count ||
      index >= obligations.output_element_byte_count) {
    if (obligations.output_buffer_count == 1u && index == 0u &&
        obligations.output_element_bytes == nullptr &&
        obligations.output_element_byte_count == 0u) {
      element_bytes = obligations.output_bytes_per_tile;
      return element_bytes == 0u
                 ? Reject("compute_binding_output_bytes_mismatch")
                 : Accept();
    }
    return Reject("compute_binding_output_bytes_mismatch");
  }
  element_bytes = obligations.output_element_bytes[index];
  return element_bytes == 0u ? Reject("compute_binding_output_bytes_mismatch")
                             : Accept();
}

} // namespace

BindingValidation ValidateResidentBuffer(const ResidentBufferRef &ref,
                                         const u64 tile_count,
                                         const u64 expected_element_bytes,
                                         const u32 expected_usage,
                                         const bool allow_stride) noexcept {
  if (ref.id == 0u) {
    return Reject("compute_resident_id_invalid");
  }
  if (ref.bytes == 0u) {
    return Reject("compute_resident_bytes_invalid");
  }
  if (ref.usage != expected_usage) {
    return Reject("compute_resident_usage_invalid");
  }
  if (ref.element_bytes != expected_element_bytes) {
    return expected_usage == kResidentUsageWrite
               ? Reject("compute_binding_output_bytes_mismatch")
               : Reject("compute_binding_input_bytes_mismatch");
  }
  if (ref.element_bytes == 0u ||
      (allow_stride ? ref.stride_bytes < ref.element_bytes
                    : ref.stride_bytes != ref.element_bytes) ||
      ref.count < tile_count) {
    return Reject("compute_resident_stride_invalid");
  }

  u64 extent = ref.element_bytes;
  if (tile_count > 1u) {
    const u64 last_tile = tile_count - 1u;
    if (!checked::mul(last_tile, ref.stride_bytes)) {
      return Reject("compute_resident_stride_invalid");
    }
    const u64 offset = last_tile * ref.stride_bytes;
    if (!checked::add(offset, ref.element_bytes)) {
      return Reject("compute_resident_bytes_invalid");
    }
    extent = offset + ref.element_bytes;
  }
  if (!checked::add(ref.offset_bytes, extent) ||
      ref.bytes < ref.offset_bytes + extent) {
    return Reject("compute_resident_bytes_invalid");
  }
  return Accept();
}

BindingValidation
ValidateRuntimeBindings(const BindingSet &bindings,
                        const BindingObligations &obligations) noexcept {
  if (!bindings.ok) {
    return Reject(bindings.reason);
  }
  if (obligations.tile_count == 0u ||
      bindings.tile_count != obligations.tile_count) {
    return Reject("compute_binding_tile_count_mismatch");
  }
  const bool has_resident_inputs = HasResidentInputs(bindings);
  const bool has_resident_output = HasResidentOutput(bindings);
  const bool has_staged_output = HasStagedOutputClaim(bindings);

  if (bindings.resident_inputs.count == 0u && has_resident_inputs) {
    return Reject("compute_binding_input_null");
  }
  if (bindings.resident_outputs.count == 0u && has_resident_output) {
    return Reject("compute_binding_output_missing");
  }

  if (!has_resident_inputs &&
      bindings.input_buffer_count != obligations.input_buffer_count) {
    return Reject("compute_binding_input_count_mismatch");
  }
  if (has_resident_inputs &&
      bindings.resident_inputs.count != obligations.input_buffer_count) {
    return Reject("compute_binding_input_count_mismatch");
  }
  if (bindings.input_bytes_per_tile != obligations.input_bytes_per_tile) {
    return Reject("compute_binding_input_bytes_mismatch");
  }
  if (obligations.validate_staged_output &&
      bindings.output_bytes_per_tile != obligations.output_bytes_per_tile) {
    return Reject("compute_binding_output_bytes_mismatch");
  }
  if (bindings.param_bytes != obligations.param_bytes) {
    return Reject("compute_binding_param_size_mismatch");
  }

  u64 input_bytes = 0u;
  if (has_resident_inputs) {
    if (!bindings.resident_inputs.has_refs()) {
      return Reject("compute_binding_input_null");
    }
    for (u64 index = 0u; index < obligations.input_buffer_count; ++index) {
      const ResidentBufferRef *const resident =
          bindings.resident_inputs.ref(index);
      if (resident == nullptr) {
        return Reject("compute_binding_input_null");
      }
      const ResidentBufferRef &ref = *resident;
      u64 expected_element_bytes = ref.element_bytes;
      const BindingValidation expected_validation =
          ExpectedInputElementBytes(obligations, index, expected_element_bytes);
      if (!expected_validation) {
        return expected_validation;
      }
      u64 expected_count = obligations.tile_count;
      const BindingValidation count_validation =
          ExpectedInputCount(obligations, index, expected_count);
      if (!count_validation) {
        return count_validation;
      }
      const BindingValidation resident_validation = ValidateResidentBuffer(
          ref, expected_count, expected_element_bytes,
          kResidentUsageRead, obligations.allow_resident_stride);
      if (!resident_validation) {
        return resident_validation;
      }
      if (!checked::add(input_bytes, expected_element_bytes)) {
        return Reject("compute_binding_input_bytes_overflow");
      }
      input_bytes += expected_element_bytes;
    }
  } else if (obligations.input_buffer_count != 0u) {
    if (bindings.input_buffers == nullptr) {
      return Reject("compute_binding_input_null");
    }
    for (u64 index = 0u; index < obligations.input_buffer_count; ++index) {
      const BufferSpan &span = bindings.input_buffers[index];
      u64 expected_element_bytes = span.element_bytes;
      const BindingValidation expected_validation =
          ExpectedInputElementBytes(obligations, index, expected_element_bytes);
      if (!expected_validation) {
        return expected_validation;
      }
      u64 expected_count = obligations.tile_count;
      const BindingValidation count_validation =
          ExpectedInputCount(obligations, index, expected_count);
      if (!count_validation) {
        return count_validation;
      }
      if (span.data == nullptr) {
        return Reject("compute_binding_input_null");
      }
      if (span.element_bytes != expected_element_bytes ||
          span.element_bytes == 0u || span.stride_bytes < span.element_bytes ||
          !SpanCanAddressTiles(span, expected_count)) {
        return span.element_bytes != expected_element_bytes
                   ? Reject("compute_binding_input_bytes_mismatch")
                   : Reject("compute_binding_input_stride_invalid");
      }
      if (!checked::add(input_bytes, expected_element_bytes)) {
        return Reject("compute_binding_input_bytes_overflow");
      }
      input_bytes += expected_element_bytes;
    }
  }
  if (input_bytes != obligations.input_bytes_per_tile) {
    return Reject("compute_binding_input_bytes_mismatch");
  }

  if (obligations.param_bytes != bindings.param_data_bytes) {
    return Reject("compute_binding_param_size_mismatch");
  }
  if (obligations.param_bytes != 0u && bindings.param_data == nullptr) {
    return Reject("compute_binding_param_null");
  }

  if (obligations.output_bytes_per_tile != 0u) {
    if (has_staged_output && has_resident_output) {
      return Reject("compute_binding_output_mode_conflict");
    }
    if (has_resident_output) {
      if (!bindings.resident_outputs.has_refs() ||
          bindings.resident_outputs.count != obligations.output_buffer_count) {
        return Reject("compute_binding_output_missing");
      }
      u64 output_bytes = 0u;
      for (u64 index = 0u; index < obligations.output_buffer_count; ++index) {
        u64 expected_element_bytes = 0u;
        const BindingValidation expected = ExpectedOutputElementBytes(
            obligations, index, expected_element_bytes);
        if (!expected) {
          return expected;
        }
        const ResidentBufferRef *const resident =
            bindings.resident_outputs.ref(index);
        if (resident == nullptr) {
          return Reject("compute_binding_output_missing");
        }
        const BindingValidation resident_validation = ValidateResidentBuffer(
            *resident, obligations.tile_count, expected_element_bytes,
            kResidentUsageWrite, obligations.allow_resident_stride);
        if (!resident_validation) {
          return resident_validation;
        }
        if (!checked::add(output_bytes, expected_element_bytes)) {
          return Reject("compute_binding_output_bytes_overflow");
        }
        output_bytes += expected_element_bytes;
      }
      if (output_bytes != obligations.output_bytes_per_tile) {
        return Reject("compute_binding_output_bytes_mismatch");
      }
      return Accept();
    }
  }

  if (obligations.validate_staged_output &&
      obligations.output_bytes_per_tile != 0u) {
    if (bindings.output_buffer_count != 0u) {
      if (bindings.output_buffers == nullptr ||
          bindings.output_buffer_count != obligations.output_buffer_count) {
        return Reject("compute_binding_output_missing");
      }
      u64 output_bytes = 0u;
      for (u64 index = 0u; index < obligations.output_buffer_count; ++index) {
        const OutputSpan &span = bindings.output_buffers[index];
        u64 expected_element_bytes = 0u;
        const BindingValidation expected = ExpectedOutputElementBytes(
            obligations, index, expected_element_bytes);
        if (!expected) {
          return expected;
        }
        if (span.data == nullptr ||
            span.element_bytes != expected_element_bytes ||
            span.stride_bytes < span.element_bytes ||
            !OutputCanAddressTiles(span, obligations.tile_count)) {
          return Reject(span.element_bytes != expected_element_bytes
                            ? "compute_binding_output_bytes_mismatch"
                            : "compute_binding_output_stride_invalid");
        }
        if (!checked::add(output_bytes, expected_element_bytes)) {
          return Reject("compute_binding_output_bytes_overflow");
        }
        output_bytes += expected_element_bytes;
      }
      return output_bytes == obligations.output_bytes_per_tile
                 ? Accept()
                 : Reject("compute_binding_output_bytes_mismatch");
    }
    if (bindings.staged_output == nullptr) {
      return Reject(has_resident_inputs ? "compute_binding_output_missing"
                                        : "compute_binding_output_null");
    }
    if (obligations.output_buffer_count != 1u ||
        bindings.staged_output_stride < obligations.output_bytes_per_tile ||
        !OutputCanAddressTiles(bindings)) {
      return Reject("compute_binding_output_stride_invalid");
    }
  }

  return Accept();
}

} // namespace rund::kernel

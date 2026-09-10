#include "local.hpp"

#include <kernel/core/checked.hpp>

namespace rund::kernel::binding_validation_detail {
namespace {

[[nodiscard]] bool
InputShapeAvailable(const BindingObligations &obligations) noexcept {
  return obligations.input_element_bytes != nullptr ||
         obligations.input_element_byte_count != 0u ||
         obligations.uniform_input_element_bytes != 0u;
}

} // namespace

BindingValidation Reject(const char *const reason) noexcept {
  return BindingValidation{.reason = reason};
}

BindingValidation Accept() noexcept {
  return BindingValidation{.ok = true, .reason = "ok"};
}

bool SpanCanAddressTiles(const BufferSpan &span,
                         const u64 tile_count) noexcept {
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

bool OutputCanAddressTiles(const BindingSet &bindings) noexcept {
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

bool OutputCanAddressTiles(const OutputSpan &span,
                           const u64 tile_count) noexcept {
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

BindingValidation
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

BindingValidation ExpectedInputCount(const BindingObligations &obligations,
                                     const u64 index, u64 &count) noexcept {
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

BindingValidation
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

} // namespace rund::kernel::binding_validation_detail

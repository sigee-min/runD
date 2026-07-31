#pragma once

#include <kernel/program/compute/histogram/model.hpp>

namespace rund::kernel {
namespace histogram_reference_detail {

[[nodiscard]] constexpr HistogramResult Reject(
    const u64 element_count,
    const u64 bin_count,
    const char* const reason) noexcept {
  return HistogramResult{
      .element_count = element_count,
      .bin_count = bin_count,
      .reason = reason,
  };
}

}  // namespace histogram_reference_detail

[[nodiscard]] inline HistogramResult ReferenceHistogramU32(
    const u32* const input,
    u32* const output,
    const u64 element_count,
    const u64 bin_count) noexcept {
  if (element_count == 0u) {
    return histogram_reference_detail::Reject(
        0u, bin_count, "compute_histogram_count_zero");
  }
  if (bin_count == 0u) {
    return histogram_reference_detail::Reject(
        element_count, 0u, "compute_histogram_bin_count_zero");
  }
  if (element_count > static_cast<u64>(~u32{0u})) {
    return histogram_reference_detail::Reject(
        element_count, bin_count, "compute_histogram_count_overflow");
  }
  if (input == nullptr || output == nullptr) {
    return histogram_reference_detail::Reject(
        element_count, bin_count, "compute_histogram_buffer_invalid");
  }

  for (u64 bin = 0u; bin < bin_count; ++bin) {
    output[bin] = 0u;
  }
  for (u64 index = 0u; index < element_count; ++index) {
    const u32 bin = input[index];
    if (static_cast<u64>(bin) >= bin_count) {
      return histogram_reference_detail::Reject(
          element_count, bin_count, "compute_histogram_bin_invalid");
    }
    ++output[bin];
  }
  return HistogramResult{
      .element_count = element_count,
      .bin_count = bin_count,
      .ok = true,
      .reason = "ok",
  };
}

}  // namespace rund::kernel

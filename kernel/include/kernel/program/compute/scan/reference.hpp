#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scan/model.hpp>

namespace rund::kernel {
namespace scan_reference_detail {

[[nodiscard]] constexpr ScanResult Reject(const u64 element_count,
                                          const u64 total,
                                          const char *const reason) noexcept {
  return ScanResult{
      .element_count = element_count,
      .total = total,
      .reason = reason,
  };
}

[[nodiscard]] inline bool
StoreU32TotalOrReject(const u64 element_count, const u64 running,
                      u64 *const total, ScanResult *const result) noexcept {
  if (running <= static_cast<u64>(~u32{0u})) {
    return true;
  }
  *total = running;
  *result = Reject(element_count, running, "compute_scan_sum_overflow");
  return false;
}

} // namespace scan_reference_detail

[[nodiscard]] inline ScanResult
ReferenceExclusiveScanU32(const u32 *const input, u32 *const output,
                          const u64 element_count, u64 *const total) noexcept {
  if (element_count == 0u) {
    return scan_reference_detail::Reject(0u, 0u, "compute_scan_count_zero");
  }
  if (input == nullptr || output == nullptr || total == nullptr) {
    return scan_reference_detail::Reject(element_count, 0u,
                                         "compute_scan_buffer_invalid");
  }

  u64 running = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    ScanResult rejected{};
    if (!scan_reference_detail::StoreU32TotalOrReject(element_count, running,
                                                      total, &rejected)) {
      return rejected;
    }
    output[index] = static_cast<u32>(running);
    const u64 value = static_cast<u64>(input[index]);
    if (!checked::add(running, value)) {
      *total = running;
      return scan_reference_detail::Reject(element_count, running,
                                           "compute_scan_sum_overflow");
    }
    running += value;
  }
  ScanResult rejected{};
  if (!scan_reference_detail::StoreU32TotalOrReject(element_count, running,
                                                    total, &rejected)) {
    return rejected;
  }
  *total = running;
  return ScanResult{
      .element_count = element_count,
      .total = running,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline ScanResult
ReferenceExclusiveScanU64(const u64 *const input, u64 *const output,
                          const u64 element_count, u64 *const total) noexcept {
  if (element_count == 0u) {
    return scan_reference_detail::Reject(0u, 0u, "compute_scan_count_zero");
  }
  if (input == nullptr || output == nullptr || total == nullptr) {
    return scan_reference_detail::Reject(element_count, 0u,
                                         "compute_scan_buffer_invalid");
  }

  u64 running = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    output[index] = running;
    if (!checked::add(running, input[index])) {
      *total = running;
      return scan_reference_detail::Reject(element_count, running,
                                           "compute_scan_sum_overflow");
    }
    running += input[index];
  }
  *total = running;
  return ScanResult{
      .element_count = element_count,
      .total = running,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline ScanResult
ReferenceInclusiveScanU32(const u32 *const input, u32 *const output,
                          const u64 element_count, u64 *const total) noexcept {
  if (element_count == 0u) {
    return scan_reference_detail::Reject(0u, 0u, "compute_scan_count_zero");
  }
  if (input == nullptr || output == nullptr || total == nullptr) {
    return scan_reference_detail::Reject(element_count, 0u,
                                         "compute_scan_buffer_invalid");
  }

  u64 running = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    const u64 value = static_cast<u64>(input[index]);
    if (!checked::add(running, value)) {
      *total = running;
      return scan_reference_detail::Reject(element_count, running,
                                           "compute_scan_sum_overflow");
    }
    running += value;
    ScanResult rejected{};
    if (!scan_reference_detail::StoreU32TotalOrReject(element_count, running,
                                                      total, &rejected)) {
      return rejected;
    }
    output[index] = static_cast<u32>(running);
  }
  *total = running;
  return ScanResult{
      .element_count = element_count,
      .total = running,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline ScanResult
ReferenceInclusiveScanU64(const u64 *const input, u64 *const output,
                          const u64 element_count, u64 *const total) noexcept {
  if (element_count == 0u) {
    return scan_reference_detail::Reject(0u, 0u, "compute_scan_count_zero");
  }
  if (input == nullptr || output == nullptr || total == nullptr) {
    return scan_reference_detail::Reject(element_count, 0u,
                                         "compute_scan_buffer_invalid");
  }

  u64 running = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    if (!checked::add(running, input[index])) {
      *total = running;
      return scan_reference_detail::Reject(element_count, running,
                                           "compute_scan_sum_overflow");
    }
    running += input[index];
    output[index] = running;
  }
  *total = running;
  return ScanResult{
      .element_count = element_count,
      .total = running,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace rund::kernel

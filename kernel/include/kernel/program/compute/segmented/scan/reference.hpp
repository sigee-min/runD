#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/segmented/heads.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>
#include <kernel/program/compute/segmented/signed.hpp>

namespace rund::kernel {
namespace segmented_scan_reference_detail {

[[nodiscard]] constexpr SegmentedScanResult
Reject(const u64 element_count, const u64 segment_count,
       const u64 final_segment_total, const char *const reason) noexcept {
  return SegmentedScanResult{
      .element_count = element_count,
      .segment_count = segment_count,
      .final_segment_total = final_segment_total,
      .reason = reason,
  };
}

[[nodiscard]] inline SegmentedScanResult
RejectPriority(const u32 *const heads, const u64 unread,
               const u64 element_count, const u64 segment_count,
               const u64 total, const char *const numeric_reason) noexcept {
  return !SegmentHeadsValid(heads, unread, element_count)
             ? Reject(element_count, segment_count, total,
                      "compute_segmented_scan_segment_invalid")
             : Reject(element_count, segment_count, total, numeric_reason);
}

[[nodiscard]] inline bool
StoreU32OrReject(const u64 element_count, const u64 segment_count,
                 const u64 running,
                 SegmentedScanResult *const result) noexcept {
  if (running <= static_cast<u64>(~u32{0u})) {
    return true;
  }
  *result = Reject(element_count, segment_count, running,
                   "compute_segmented_scan_sum_overflow");
  return false;
}

template <typename Value, typename Store>
[[nodiscard]] inline SegmentedScanResult
ReferenceSegmentedScan(const Value *const input, const u32 *const segment_heads,
                       Value *const output, const u64 element_count,
                       const bool inclusive, Store store) noexcept {
  if (element_count == 0u) {
    return Reject(0u, 0u, 0u, "compute_segmented_scan_count_zero");
  }
  if (input == nullptr || segment_heads == nullptr || output == nullptr) {
    return Reject(element_count, 0u, 0u,
                  "compute_segmented_scan_buffer_invalid");
  }
  if (!SegmentHeadValid(segment_heads[0], 0u)) {
    return Reject(element_count, 0u, 0u,
                  "compute_segmented_scan_segment_invalid");
  }

  u64 running = 0u;
  u64 segment_count = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    const u32 head = segment_heads[index];
    if (!SegmentHeadValid(head, index)) {
      return Reject(element_count, segment_count, running,
                    "compute_segmented_scan_segment_invalid");
    }
    if (head == 1u) {
      if (!inclusive && segment_count != 0u) {
        Value discarded{};
        SegmentedScanResult rejected{};
        if (!store(element_count, segment_count, running, &discarded,
                   &rejected)) {
          return RejectPriority(segment_heads, index + 1u, element_count,
                                segment_count, running, rejected.reason);
        }
      }
      running = 0u;
      ++segment_count;
    }
    if (!inclusive) {
      SegmentedScanResult rejected{};
      if (!store(element_count, segment_count, running, output + index,
                 &rejected)) {
        return RejectPriority(segment_heads, index + 1u, element_count,
                              segment_count, running, rejected.reason);
      }
    }
    const u64 value = static_cast<u64>(input[index]);
    if (!checked::add(running, value)) {
      return RejectPriority(segment_heads, index + 1u, element_count,
                            segment_count, running,
                            "compute_segmented_scan_sum_overflow");
    }
    running += value;
    if (inclusive) {
      SegmentedScanResult rejected{};
      if (!store(element_count, segment_count, running, output + index,
                 &rejected)) {
        return RejectPriority(segment_heads, index + 1u, element_count,
                              segment_count, running, rejected.reason);
      }
    }
  }
  if (!inclusive) {
    Value discarded{};
    SegmentedScanResult rejected{};
    if (!store(element_count, segment_count, running, &discarded, &rejected)) {
      return rejected;
    }
  }
  return SegmentedScanResult{
      .element_count = element_count,
      .segment_count = segment_count,
      .final_segment_total = running,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace segmented_scan_reference_detail

template <SignedLane Value>
[[nodiscard]] inline SegmentedScanResult
ReferenceSignedSegmentedScan(const Value *const input,
                             const u32 *const segment_heads,
                             Value *const output, const u64 element_count,
                             const bool inclusive) noexcept {
  using segmented_signed_detail::Bits;
  using segmented_signed_detail::Fits;
  using Wide = segmented_signed_detail::Prefix<Value>;
  if (element_count == 0u) {
    return segmented_scan_reference_detail::Reject(
        0u, 0u, 0u, "compute_segmented_scan_count_zero");
  }
  if (input == nullptr || segment_heads == nullptr || output == nullptr) {
    return segmented_scan_reference_detail::Reject(
        element_count, 0u, 0u, "compute_segmented_scan_buffer_invalid");
  }
  if (!SegmentHeadValid(segment_heads[0u], 0u)) {
    return segmented_scan_reference_detail::Reject(
        element_count, 0u, 0u, "compute_segmented_scan_segment_invalid");
  }

  Wide running{};
  u64 segment_count = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    const u32 head = segment_heads[index];
    if (!SegmentHeadValid(head, index)) {
      return segmented_scan_reference_detail::Reject(
          element_count, segment_count, Bits<Value>(running),
          "compute_segmented_scan_segment_invalid");
    }
    if (head == 1u) {
      if (!inclusive && segment_count != 0u && !Fits<Value>(running)) {
        return segmented_scan_reference_detail::RejectPriority(
            segment_heads, index + 1u, element_count, segment_count,
            Bits<Value>(running), "compute_segmented_scan_sum_overflow");
      }
      running = Wide{};
      ++segment_count;
    }
    if (!inclusive) {
      if (!Fits<Value>(running)) {
        return segmented_scan_reference_detail::RejectPriority(
            segment_heads, index + 1u, element_count, segment_count,
            Bits<Value>(running), "compute_segmented_scan_sum_overflow");
      }
      output[index] = static_cast<Value>(running);
    }
    running += static_cast<Wide>(input[index]);
    if (inclusive) {
      if (!Fits<Value>(running)) {
        return segmented_scan_reference_detail::RejectPriority(
            segment_heads, index + 1u, element_count, segment_count,
            Bits<Value>(running), "compute_segmented_scan_sum_overflow");
      }
      output[index] = static_cast<Value>(running);
    }
  }
  if (!inclusive && !Fits<Value>(running)) {
    return segmented_scan_reference_detail::Reject(
        element_count, segment_count, Bits<Value>(running),
        "compute_segmented_scan_sum_overflow");
  }
  return SegmentedScanResult{
      .element_count = element_count,
      .segment_count = segment_count,
      .final_segment_total = Bits<Value>(running),
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline SegmentedScanResult ReferenceExclusiveSegmentedScanU32(
    const u32 *const input, const u32 *const segment_heads, u32 *const output,
    const u64 element_count) noexcept {
  return segmented_scan_reference_detail::ReferenceSegmentedScan(
      input, segment_heads, output, element_count, false,
      [](const u64 count, const u64 segments, const u64 value, u32 *const out,
         SegmentedScanResult *const result) {
        if (!segmented_scan_reference_detail::StoreU32OrReject(count, segments,
                                                               value, result)) {
          return false;
        }
        *out = static_cast<u32>(value);
        return true;
      });
}

[[nodiscard]] inline SegmentedScanResult ReferenceInclusiveSegmentedScanU32(
    const u32 *const input, const u32 *const segment_heads, u32 *const output,
    const u64 element_count) noexcept {
  return segmented_scan_reference_detail::ReferenceSegmentedScan(
      input, segment_heads, output, element_count, true,
      [](const u64 count, const u64 segments, const u64 value, u32 *const out,
         SegmentedScanResult *const result) {
        if (!segmented_scan_reference_detail::StoreU32OrReject(count, segments,
                                                               value, result)) {
          return false;
        }
        *out = static_cast<u32>(value);
        return true;
      });
}

[[nodiscard]] inline SegmentedScanResult ReferenceExclusiveSegmentedScanU64(
    const u64 *const input, const u32 *const segment_heads, u64 *const output,
    const u64 element_count) noexcept {
  return segmented_scan_reference_detail::ReferenceSegmentedScan(
      input, segment_heads, output, element_count, false,
      [](u64, u64, const u64 value, u64 *const out, SegmentedScanResult *) {
        *out = value;
        return true;
      });
}

[[nodiscard]] inline SegmentedScanResult ReferenceInclusiveSegmentedScanU64(
    const u64 *const input, const u32 *const segment_heads, u64 *const output,
    const u64 element_count) noexcept {
  return segmented_scan_reference_detail::ReferenceSegmentedScan(
      input, segment_heads, output, element_count, true,
      [](u64, u64, const u64 value, u64 *const out, SegmentedScanResult *) {
        *out = value;
        return true;
      });
}

} // namespace rund::kernel

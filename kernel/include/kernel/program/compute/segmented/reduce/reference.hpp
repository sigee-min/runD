#pragma once

#include <kernel/program/compute/segmented/reduce/reference/traversal.hpp>
#include <kernel/program/compute/segmented/signed.hpp>

namespace rund::kernel {

template <SignedLane Value>
[[nodiscard]] inline SegmentedReduceResult
ReferenceSignedSegmentedReduce(const Value *const input,
                               const u32 *const segment_heads,
                               Value *const output, const u64 element_count,
                               const ReduceOp operation) noexcept {
  using segmented_signed_detail::Bits;
  using segmented_signed_detail::Fits;
  using Wide = segmented_signed_detail::Total<Value>;
  if (element_count == 0u) {
    return segmented_reduce_reference_detail::Reject(
        0u, 0u, 0u, "compute_segmented_reduce_count_zero");
  }
  if (input == nullptr || segment_heads == nullptr || output == nullptr) {
    return segmented_reduce_reference_detail::Reject(
        element_count, 0u, 0u, "compute_segmented_reduce_buffer_invalid");
  }
  if (operation != ReduceOp::Sum && operation != ReduceOp::CountNonzero &&
      operation != ReduceOp::Min && operation != ReduceOp::Max) {
    return segmented_reduce_reference_detail::Reject(
        element_count, 0u, 0u, "compute_segmented_reduce_op_unsupported");
  }
  if (!SegmentHeadValid(segment_heads[0u], 0u)) {
    return segmented_reduce_reference_detail::Reject(
        element_count, 0u, 0u, "compute_segmented_reduce_segment_invalid");
  }
  const char *const overflow_reason =
      operation == ReduceOp::Sum ? "compute_segmented_reduce_sum_overflow"
                                 : "compute_segmented_reduce_count_overflow";

  Wide running{};
  u64 segment_count = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    const u32 head = segment_heads[index];
    if (!SegmentHeadValid(head, index)) {
      return segmented_reduce_reference_detail::Reject(
          element_count, segment_count, Bits<Value>(running),
          "compute_segmented_reduce_segment_invalid");
    }
    if (head == 1u) {
      if (segment_count != 0u) {
        if (!Fits<Value>(running)) {
          return segmented_reduce_reference_detail::RejectPriority(
              segment_heads, index + 1u, element_count, segment_count,
              Bits<Value>(running), overflow_reason);
        }
        output[segment_count - 1u] = static_cast<Value>(running);
      }
      ++segment_count;
      running = operation == ReduceOp::CountNonzero
                    ? static_cast<Wide>(input[index] != 0 ? 1 : 0)
                    : static_cast<Wide>(input[index]);
      continue;
    }
    if (operation == ReduceOp::Sum) {
      running += static_cast<Wide>(input[index]);
    } else if (operation == ReduceOp::CountNonzero) {
      running += input[index] != 0 ? Wide{1} : Wide{0};
    } else if (operation == ReduceOp::Min) {
      const Wide value = static_cast<Wide>(input[index]);
      running = value < running ? value : running;
    } else {
      const Wide value = static_cast<Wide>(input[index]);
      running = value > running ? value : running;
    }
  }
  if (!Fits<Value>(running)) {
    return segmented_reduce_reference_detail::Reject(
        element_count, segment_count, Bits<Value>(running), overflow_reason);
  }
  output[segment_count - 1u] = static_cast<Value>(running);
  return SegmentedReduceResult{
      .element_count = element_count,
      .segment_count = segment_count,
      .final_segment_total = Bits<Value>(running),
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline SegmentedReduceResult ReferenceSegmentedReduceSumU32(
    const u32 *const input, const u32 *const segment_heads, u32 *const output,
    const u64 element_count) noexcept {
  return segmented_reduce_reference_detail::ReferenceSegmentedReduce(
      input, segment_heads, output, element_count,
      [](const u64 value) {
        return segmented_reduce_reference_detail::Ok(value);
      },
      [](const u64 running, const u64 value) {
        return segmented_reduce_reference_detail::SumOrFail(
            running, value, static_cast<u64>(~u32{0u}));
      },
      [](const u64 count, const u64 segments, const u64 value, u32 *const out,
         SegmentedReduceResult *const result) {
        return segmented_reduce_reference_detail::StoreU32OrReject(
            count, segments, value, out, result,
            "compute_segmented_reduce_sum_overflow");
      });
}

[[nodiscard]] inline SegmentedReduceResult ReferenceSegmentedReduceSumU64(
    const u64 *const input, const u32 *const segment_heads, u64 *const output,
    const u64 element_count) noexcept {
  return segmented_reduce_reference_detail::ReferenceSegmentedReduce(
      input, segment_heads, output, element_count,
      [](const u64 value) {
        return segmented_reduce_reference_detail::Ok(value);
      },
      [](const u64 running, const u64 value) {
        return segmented_reduce_reference_detail::SumOrFail(running, value,
                                                            ~u64{0u});
      },
      [](u64, u64, const u64 value, u64 *const out, SegmentedReduceResult *) {
        *out = value;
        return true;
      });
}

[[nodiscard]] inline SegmentedReduceResult
ReferenceSegmentedReduceCountNonzeroU32(const u32 *const input,
                                        const u32 *const segment_heads,
                                        u32 *const output,
                                        const u64 element_count) noexcept {
  return segmented_reduce_reference_detail::ReferenceSegmentedReduce(
      input, segment_heads, output, element_count,
      [](const u64 value) {
        return segmented_reduce_reference_detail::Ok(value != 0u ? 1u : 0u);
      },
      [](const u64 running, const u64 value) {
        return value == 0u ? segmented_reduce_reference_detail::Ok(running)
                           : segmented_reduce_reference_detail::SumOrFail(
                                 running, 1u, static_cast<u64>(~u32{0u}));
      },
      [](const u64 count, const u64 segments, const u64 value, u32 *const out,
         SegmentedReduceResult *const result) {
        return segmented_reduce_reference_detail::StoreU32OrReject(
            count, segments, value, out, result,
            "compute_segmented_reduce_count_overflow");
      });
}

[[nodiscard]] inline SegmentedReduceResult
ReferenceSegmentedReduceCountNonzeroU64(const u64 *const input,
                                        const u32 *const segment_heads,
                                        u64 *const output,
                                        const u64 element_count) noexcept {
  return segmented_reduce_reference_detail::ReferenceSegmentedReduce(
      input, segment_heads, output, element_count,
      [](const u64 value) {
        return segmented_reduce_reference_detail::Ok(value != 0u ? 1u : 0u);
      },
      [](const u64 running, const u64 value) {
        return value == 0u ? segmented_reduce_reference_detail::Ok(running)
                           : segmented_reduce_reference_detail::SumOrFail(
                                 running, 1u, ~u64{0u});
      },
      [](u64, u64, const u64 value, u64 *const out, SegmentedReduceResult *) {
        *out = value;
        return true;
      });
}

[[nodiscard]] inline SegmentedReduceResult ReferenceSegmentedReduceMinU32(
    const u32 *const input, const u32 *const segment_heads, u32 *const output,
    const u64 element_count) noexcept {
  return segmented_reduce_reference_detail::ReferenceSegmentedReduce(
      input, segment_heads, output, element_count,
      [](const u64 value) {
        return segmented_reduce_reference_detail::Ok(value);
      },
      [](const u64 running, const u64 value) {
        return segmented_reduce_reference_detail::Ok(value < running ? value
                                                                     : running);
      },
      [](u64, u64, const u64 value, u32 *const out, SegmentedReduceResult *) {
        *out = static_cast<u32>(value);
        return true;
      });
}

[[nodiscard]] inline SegmentedReduceResult ReferenceSegmentedReduceMaxU32(
    const u32 *const input, const u32 *const segment_heads, u32 *const output,
    const u64 element_count) noexcept {
  return segmented_reduce_reference_detail::ReferenceSegmentedReduce(
      input, segment_heads, output, element_count,
      [](const u64 value) {
        return segmented_reduce_reference_detail::Ok(value);
      },
      [](const u64 running, const u64 value) {
        return segmented_reduce_reference_detail::Ok(value > running ? value
                                                                     : running);
      },
      [](u64, u64, const u64 value, u32 *const out, SegmentedReduceResult *) {
        *out = static_cast<u32>(value);
        return true;
      });
}

[[nodiscard]] inline SegmentedReduceResult ReferenceSegmentedReduceMinU64(
    const u64 *const input, const u32 *const segment_heads, u64 *const output,
    const u64 element_count) noexcept {
  return segmented_reduce_reference_detail::ReferenceSegmentedReduce(
      input, segment_heads, output, element_count,
      [](const u64 value) {
        return segmented_reduce_reference_detail::Ok(value);
      },
      [](const u64 running, const u64 value) {
        return segmented_reduce_reference_detail::Ok(value < running ? value
                                                                     : running);
      },
      [](u64, u64, const u64 value, u64 *const out, SegmentedReduceResult *) {
        *out = value;
        return true;
      });
}

[[nodiscard]] inline SegmentedReduceResult ReferenceSegmentedReduceMaxU64(
    const u64 *const input, const u32 *const segment_heads, u64 *const output,
    const u64 element_count) noexcept {
  return segmented_reduce_reference_detail::ReferenceSegmentedReduce(
      input, segment_heads, output, element_count,
      [](const u64 value) {
        return segmented_reduce_reference_detail::Ok(value);
      },
      [](const u64 running, const u64 value) {
        return segmented_reduce_reference_detail::Ok(value > running ? value
                                                                     : running);
      },
      [](u64, u64, const u64 value, u64 *const out, SegmentedReduceResult *) {
        *out = value;
        return true;
      });
}

} // namespace rund::kernel

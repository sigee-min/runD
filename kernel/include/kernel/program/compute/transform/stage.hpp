#pragma once

#include <kernel/core/model.hpp>

#include <bit>

namespace rund::kernel::transform_stage {

inline constexpr u64 Lanes = 256u;
inline constexpr u64 LocalStages = std::countr_zero(Lanes);
inline constexpr u64 FirstGlobalSpan = Lanes * 2u;
inline constexpr u64 StagesPerDispatch = 2u;
static_assert(std::has_single_bit(Lanes));

struct Batch final {
  u64 span = 0u;
  u64 stride = 0u;
  u64 next_span = 0u;
  u64 next_stride = 0u;
  u64 bit_count = 0u;
};

[[nodiscard]] constexpr Batch Describe(const u64 elements,
                                       const u64 span) noexcept {
  if (elements == 0u || !std::has_single_bit(elements)) {
    return {};
  }
  if (span == 1u) {
    return Batch{
        .span = span,
        .stride = elements >> 1u,
        .bit_count = static_cast<u64>(std::countr_zero(elements)),
    };
  }
  if (span > elements || !std::has_single_bit(span)) {
    return {};
  }
  const bool paired = span <= (elements >> 1u);
  return Batch{
      .span = span,
      .stride = elements / span,
      .next_span = paired ? span << 1u : 0u,
      .next_stride = paired ? elements / (span << 1u) : 0u,
      .bit_count = static_cast<u64>(std::countr_zero(elements)),
  };
}

[[nodiscard]] constexpr u64 Threads(const u64 elements,
                                    const Batch batch) noexcept {
  if (batch.span == 1u) {
    return elements;
  }
  if (batch.span < 2u) {
    return 0u;
  }
  return batch.next_span == 0u ? elements >> 1u : elements >> 2u;
}

[[nodiscard]] constexpr u64 Next(const u64 elements,
                                 const Batch batch) noexcept {
  return batch.next_span == 0u || batch.next_span == elements
             ? 0u
             : batch.next_span << 1u;
}

[[nodiscard]] constexpr u64 Groups(const u64 items) noexcept {
  return items / Lanes + static_cast<u64>(items % Lanes != 0u);
}

[[nodiscard]] constexpr u64 Reverse(u64 value, const u64 bits) noexcept {
  value = ((value >> 1u) & 0x5555555555555555ull) |
          ((value & 0x5555555555555555ull) << 1u);
  value = ((value >> 2u) & 0x3333333333333333ull) |
          ((value & 0x3333333333333333ull) << 2u);
  value = ((value >> 4u) & 0x0f0f0f0f0f0f0f0full) |
          ((value & 0x0f0f0f0f0f0f0f0full) << 4u);
  value = ((value >> 8u) & 0x00ff00ff00ff00ffull) |
          ((value & 0x00ff00ff00ff00ffull) << 8u);
  value = ((value >> 16u) & 0x0000ffff0000ffffull) |
          ((value & 0x0000ffff0000ffffull) << 16u);
  value = (value >> 32u) | (value << 32u);
  return bits == 0u ? 0u : value >> (64u - bits);
}

[[nodiscard]] constexpr u64 Dispatches(const u64 elements) noexcept {
  if (elements == 0u || !std::has_single_bit(elements)) {
    return 0u;
  }
  const u64 stages = static_cast<u64>(std::countr_zero(elements));
  const u64 global = stages > LocalStages ? stages - LocalStages : 0u;
  return 1u + (global / StagesPerDispatch) +
         static_cast<u64>(global % StagesPerDispatch != 0u);
}

static_assert(Describe(1u << 20u, FirstGlobalSpan).stride == 2'048u);
static_assert(Describe(1u << 20u, FirstGlobalSpan).next_span == 1'024u);
static_assert(Describe(1u << 20u, FirstGlobalSpan).next_stride == 1'024u);
static_assert(Next(1u << 20u, Describe(1u << 20u, FirstGlobalSpan)) == 2'048u);
static_assert(Threads(1u << 20u, Describe(1u << 20u, FirstGlobalSpan)) ==
              1u << 18u);
static_assert(Threads(1u << 20u, Describe(1u << 20u, 1u)) == 1u << 20u);
static_assert(Reverse(0b011u, 3u) == 0b110u);
static_assert(Dispatches(1u << 20u) == 7u);

} // namespace rund::kernel::transform_stage

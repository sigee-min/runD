#pragma once

#include "traits.hpp"

#include <kernel/core/checked.hpp>

#include <cstdint>
#include <limits>
#include <optional>

namespace rund::node::accel::detail {

class RangeShape final {
public:
  RangeShape() = delete;

  // Output q aggregates K logical positions beginning at q*S-P. Clamp
  // repeats an endpoint outside [0,N); Clip excludes those positions. P<K
  // makes the first window intersect the input; the checked last-start test
  // below makes every intervening window intersect as well.
  [[nodiscard]] static constexpr std::optional<RangeShape>
  affine(const RangeTraits traits, const RangeBoundary boundary,
         const rund::kernel::u64 input_count,
         const rund::kernel::u64 output_count,
         const rund::kernel::u64 window_size, const rund::kernel::u64 stride,
         const rund::kernel::u64 padding, const rund::kernel::u32 element_bytes,
         const RangeCount count = RangeCount::Descriptor) noexcept {
    const rund::kernel::u128 last_anchor =
        static_cast<rund::kernel::u128>(
            output_count == 0u ? 0u : output_count - 1u) *
        stride;
    if (!traits.valid() || !KnownBoundary(boundary) || !KnownCount(count) ||
        input_count == 0u || output_count == 0u || window_size == 0u ||
        stride == 0u || padding >= window_size ||
        (count != RangeCount::Descriptor &&
         (input_count != output_count || stride != 1u ||
          window_size - padding - 1u != padding)) ||
        last_anchor > std::numeric_limits<rund::kernel::u64>::max() ||
        last_anchor >= static_cast<rund::kernel::u128>(input_count) + padding ||
        (element_bytes != 4u && element_bytes != 8u) ||
        !DomainWidthMatches(traits.domain(), element_bytes) ||
        input_count >
            std::numeric_limits<rund::kernel::u64>::max() / element_bytes ||
        output_count >
            std::numeric_limits<rund::kernel::u64>::max() / element_bytes) {
      return std::nullopt;
    }
    return RangeShape{traits,       boundary,      input_count,
                      output_count, window_size,   stride,
                      padding,      element_bytes, count};
  }

  [[nodiscard]] constexpr const RangeTraits &traits() const noexcept {
    return traits_;
  }

  [[nodiscard]] constexpr RangeBoundary boundary() const noexcept {
    return boundary_;
  }

  [[nodiscard]] constexpr RangeCount count() const noexcept {
    return static_cast<RangeCount>((element_and_count_ >> 8u) & 0xffu);
  }

  [[nodiscard]] constexpr bool resident_counted() const noexcept {
    return count() != RangeCount::Descriptor;
  }

  [[nodiscard]] constexpr rund::kernel::u64 input_count() const noexcept {
    return input_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 output_count() const noexcept {
    return output_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 window_size() const noexcept {
    return window_size_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 stride() const noexcept {
    return stride_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 padding() const noexcept {
    return padding_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 right_extent() const noexcept {
    return window_size_ - padding_ - 1u;
  }

  [[nodiscard]] constexpr std::optional<rund::kernel::u64>
  affine_span() const noexcept {
    rund::kernel::u64 last_anchor = 0u;
    rund::kernel::u64 span = 0u;
    return rund::kernel::checked::mul(output_count_ - 1u, stride_,
                                      last_anchor) &&
                   rund::kernel::checked::add(last_anchor, window_size_, span)
               ? std::optional<rund::kernel::u64>{span}
               : std::nullopt;
  }

  [[nodiscard]] constexpr bool centered_clamp() const noexcept {
    return boundary_ == RangeBoundary::Clamp && input_count_ == output_count_ &&
           stride_ == 1u && right_extent() == padding_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 element_bytes() const noexcept {
    return element_and_count_ & 0xffu;
  }

  [[nodiscard]] constexpr rund::kernel::u64 payload_bytes() const noexcept {
    return input_count_ * element_bytes();
  }

  [[nodiscard]] constexpr rund::kernel::u64 output_bytes() const noexcept {
    return output_count_ * element_bytes();
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return affine(traits_, boundary_, input_count_, output_count_, window_size_,
                  stride_, padding_, element_bytes(), count())
        .has_value();
  }

private:
  [[nodiscard]] static constexpr bool
  KnownBoundary(const RangeBoundary boundary) noexcept {
    return boundary == RangeBoundary::Clamp || boundary == RangeBoundary::Clip;
  }

  [[nodiscard]] static constexpr bool
  KnownCount(const RangeCount count) noexcept {
    return count == RangeCount::Descriptor || count == RangeCount::U32 ||
           count == RangeCount::U64;
  }

  [[nodiscard]] static constexpr bool
  DomainWidthMatches(const rund::kernel::ComputeDomain domain,
                     const rund::kernel::u32 element_bytes) noexcept {
    if (domain == rund::kernel::ComputeDomain::I32 ||
        domain == rund::kernel::ComputeDomain::U32) {
      return element_bytes == 4u;
    }
    if (domain == rund::kernel::ComputeDomain::I64 ||
        domain == rund::kernel::ComputeDomain::U64) {
      return element_bytes == 8u;
    }
    return domain == rund::kernel::ComputeDomain::Fixed;
  }

  constexpr RangeShape(const RangeTraits traits, const RangeBoundary boundary,
                       const rund::kernel::u64 input_count,
                       const rund::kernel::u64 output_count,
                       const rund::kernel::u64 window_size,
                       const rund::kernel::u64 stride,
                       const rund::kernel::u64 padding,
                       const rund::kernel::u32 element_bytes,
                       const RangeCount count) noexcept
      : traits_(traits), boundary_(boundary),
        element_and_count_(element_bytes |
                           (static_cast<rund::kernel::u32>(count) << 8u)),
        input_count_(input_count), output_count_(output_count),
        window_size_(window_size), stride_(stride), padding_(padding) {}

  RangeTraits traits_;
  RangeBoundary boundary_;
  rund::kernel::u32 element_and_count_;
  rund::kernel::u64 input_count_;
  rund::kernel::u64 output_count_;
  rund::kernel::u64 window_size_;
  rund::kernel::u64 stride_;
  rund::kernel::u64 padding_;
};

} // namespace rund::node::accel::detail

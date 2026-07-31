#pragma once

#include <kernel/program/compute/transform/model.hpp>

namespace rund::kernel {
namespace transform_identity_detail {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr TransformHash Mix(
    const TransformHash hash,
    const u64 value) noexcept {
  const u64 mixed =
      Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi + (hash.lo << 6u) +
                  (hash.lo >> 2u));
  return TransformHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0x6a09e667f3bcc909ull),
  };
}

}  // namespace transform_identity_detail

[[nodiscard]] constexpr TransformHash HashTransform(
    const TransformDesc& desc) noexcept {
  TransformHash hash{
      .hi = 0x243f6a8885a308d3ull,
      .lo = 0x13198a2e03707344ull,
  };
  hash = transform_identity_detail::Mix(hash, static_cast<u64>(desc.op));
  hash = transform_identity_detail::Mix(hash,
                                        static_cast<u64>(desc.direction));
  hash = transform_identity_detail::Mix(hash,
                                        static_cast<u64>(desc.layout));
  hash = transform_identity_detail::Mix(hash,
                                        static_cast<u64>(desc.normalization));
  hash = transform_identity_detail::Mix(hash, desc.element_count);
  hash = transform_identity_detail::Mix(hash, desc.fixed_format.integer_bits);
  hash = transform_identity_detail::Mix(hash, desc.fixed_format.fraction_bits);
  hash = transform_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.rounding));
  hash = transform_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.overflow));
  hash = transform_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.approximation));
  return hash;
}

}  // namespace rund::kernel

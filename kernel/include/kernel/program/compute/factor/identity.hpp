#pragma once

#include <kernel/program/compute/factor/model.hpp>

namespace rund::kernel {
namespace factor_identity_detail {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr FactorHash Mix(const FactorHash hash,
                                       const u64 value) noexcept {
  const u64 mixed =
      Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi + (hash.lo << 6u) +
                  (hash.lo >> 2u));
  return FactorHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0xbb67ae8584caa73bull),
  };
}

}  // namespace factor_identity_detail

[[nodiscard]] constexpr FactorHash HashFactor(
    const FactorDesc& desc) noexcept {
  FactorHash hash{
      .hi = 0x13198a2e03707344ull,
      .lo = 0x243f6a8885a308d3ull,
  };
  hash = factor_identity_detail::Mix(hash, static_cast<u64>(desc.op));
  hash = factor_identity_detail::Mix(hash, static_cast<u64>(desc.layout));
  hash = factor_identity_detail::Mix(hash, static_cast<u64>(desc.output));
  hash = factor_identity_detail::Mix(hash, static_cast<u64>(desc.pivot));
  hash = factor_identity_detail::Mix(hash, desc.rows);
  hash = factor_identity_detail::Mix(hash, desc.cols);
  hash = factor_identity_detail::Mix(hash, desc.batch_count);
  hash = factor_identity_detail::Mix(hash, desc.element_bytes);
  hash = factor_identity_detail::Mix(hash, desc.fixed_format.integer_bits);
  hash = factor_identity_detail::Mix(hash, desc.fixed_format.fraction_bits);
  hash = factor_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.rounding));
  hash = factor_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.overflow));
  hash = factor_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.approximation));
  return hash;
}

}  // namespace rund::kernel

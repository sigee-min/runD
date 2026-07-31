#pragma once

#include <kernel/program/compute/spectrum/model.hpp>

namespace rund::kernel {
namespace spectrum_identity_detail {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr SpectrumHash Mix(const SpectrumHash hash,
                                         const u64 value) noexcept {
  const u64 mixed =
      Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi + (hash.lo << 6u) +
                  (hash.lo >> 2u));
  return SpectrumHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0xbb67ae8584caa73bull),
  };
}

}  // namespace spectrum_identity_detail

[[nodiscard]] constexpr SpectrumHash HashSpectrum(
    const SpectrumDesc& desc) noexcept {
  SpectrumHash hash{
      .hi = 0xc0ac29b7c97c50ddull,
      .lo = 0x3f84d5b5b5470917ull,
  };
  hash = spectrum_identity_detail::Mix(hash, static_cast<u64>(desc.op));
  hash = spectrum_identity_detail::Mix(hash, static_cast<u64>(desc.domain));
  hash = spectrum_identity_detail::Mix(hash, static_cast<u64>(desc.vectors));
  hash = spectrum_identity_detail::Mix(hash, static_cast<u64>(desc.layout));
  hash = spectrum_identity_detail::Mix(hash, desc.rows);
  hash = spectrum_identity_detail::Mix(hash, desc.cols);
  hash = spectrum_identity_detail::Mix(hash, desc.batch_count);
  hash = spectrum_identity_detail::Mix(hash, desc.max_iterations);
  hash = spectrum_identity_detail::Mix(hash, desc.element_bytes);
  hash = spectrum_identity_detail::Mix(hash, desc.fixed_format.integer_bits);
  hash = spectrum_identity_detail::Mix(hash, desc.fixed_format.fraction_bits);
  hash = spectrum_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.rounding));
  hash = spectrum_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.overflow));
  hash = spectrum_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.approximation));
  return hash;
}

}  // namespace rund::kernel

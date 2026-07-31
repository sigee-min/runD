#pragma once

#include <kernel/program/compute/matrix/model.hpp>

namespace rund::kernel {
namespace matrix_identity_detail {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr MatrixHash Mix(const MatrixHash hash,
                                       const u64 value) noexcept {
  const u64 mixed =
      Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi + (hash.lo << 6u) +
                  (hash.lo >> 2u));
  return MatrixHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0xbb67ae8584caa73bull),
  };
}

}  // namespace matrix_identity_detail

[[nodiscard]] constexpr MatrixHash HashMatrix(
    const MatrixDesc& desc) noexcept {
  MatrixHash hash{
      .hi = 0xa4093822299f31d0ull,
      .lo = 0x082efa98ec4e6c89ull,
  };
  hash = matrix_identity_detail::Mix(hash, static_cast<u64>(desc.op));
  hash = matrix_identity_detail::Mix(hash, static_cast<u64>(desc.layout));
  hash = matrix_identity_detail::Mix(hash,
                                     static_cast<u64>(desc.arithmetic));
  hash = matrix_identity_detail::Mix(hash, desc.rows);
  hash = matrix_identity_detail::Mix(hash, desc.cols);
  hash = matrix_identity_detail::Mix(hash, desc.inner);
  hash = matrix_identity_detail::Mix(hash, desc.batch_count);
  hash = matrix_identity_detail::Mix(hash, desc.element_bytes);
  hash = matrix_identity_detail::Mix(hash, desc.fixed_format.integer_bits);
  hash = matrix_identity_detail::Mix(hash, desc.fixed_format.fraction_bits);
  hash = matrix_identity_detail::Mix(
      hash, static_cast<u64>(desc.fixed_format.rounding));
  hash = matrix_identity_detail::Mix(
      hash, static_cast<u64>(desc.fixed_format.overflow));
  hash = matrix_identity_detail::Mix(
      hash, static_cast<u64>(desc.fixed_format.approximation));
  return hash;
}

}  // namespace rund::kernel

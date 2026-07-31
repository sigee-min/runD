#pragma once

#include <kernel/program/compute/solve/model.hpp>

namespace rund::kernel {
namespace solve_identity_detail {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr SolveHash Mix(const SolveHash hash,
                                      const u64 value) noexcept {
  const u64 mixed =
      Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi + (hash.lo << 6u) +
                  (hash.lo >> 2u));
  return SolveHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0xbb67ae8584caa73bull),
  };
}

}  // namespace solve_identity_detail

[[nodiscard]] constexpr SolveHash HashSolve(const SolveDesc& desc) noexcept {
  SolveHash hash{
      .hi = 0x452821e638d01377ull,
      .lo = 0xbe5466cf34e90c6cull,
  };
  hash = solve_identity_detail::Mix(hash, static_cast<u64>(desc.op));
  hash = solve_identity_detail::Mix(hash, static_cast<u64>(desc.input));
  hash = solve_identity_detail::Mix(hash, static_cast<u64>(desc.factor));
  hash = solve_identity_detail::Mix(hash, static_cast<u64>(desc.layout));
  hash = solve_identity_detail::Mix(hash, static_cast<u64>(desc.pivot));
  hash = solve_identity_detail::Mix(hash, desc.rows);
  hash = solve_identity_detail::Mix(hash, desc.rhs_cols);
  hash = solve_identity_detail::Mix(hash, desc.batch_count);
  hash = solve_identity_detail::Mix(hash, desc.element_bytes);
  hash = solve_identity_detail::Mix(hash, desc.fixed_format.integer_bits);
  hash = solve_identity_detail::Mix(hash, desc.fixed_format.fraction_bits);
  hash = solve_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.rounding));
  hash = solve_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.overflow));
  hash = solve_identity_detail::Mix(hash, static_cast<u64>(desc.fixed_format.approximation));
  return hash;
}

}  // namespace rund::kernel

#pragma once

#include <kernel/internal/compute/identity.hpp>
#include <kernel/program/compute/scatter/reduce/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr ScatterReduceHash
HashScatterReduce(const ScatterReduceDesc &desc) noexcept {
  constexpr u64 salt = 0x8f3f73b5cf1c9adeull;
  ScatterReduceHash hash{
      .hi = 0x3c6ef372fe94f82bull,
      .lo = 0xa54ff53a5f1d36f1ull,
  };
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.op), salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.domain), salt);
  hash = internal::MixIdentity(hash, desc.fixed_format.integer_bits, salt);
  hash = internal::MixIdentity(hash, desc.fixed_format.fraction_bits, salt);
  hash = internal::MixIdentity(
      hash, static_cast<u64>(desc.fixed_format.rounding), salt);
  hash = internal::MixIdentity(
      hash, static_cast<u64>(desc.fixed_format.overflow), salt);
  hash = internal::MixIdentity(
      hash, static_cast<u64>(desc.fixed_format.approximation), salt);
  hash = internal::MixIdentity(hash, desc.element_count, salt);
  hash = internal::MixIdentity(hash, desc.output_count, salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.count_source), salt);
  return hash;
}

} // namespace rund::kernel

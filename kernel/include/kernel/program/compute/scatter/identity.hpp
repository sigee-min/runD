#pragma once

#include <kernel/internal/compute/identity.hpp>
#include <kernel/program/compute/scatter/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr ScatterHash HashScatter(
    const ScatterDesc& desc) noexcept {
  constexpr u64 salt = 0x517cc1b727220a95ull;
  ScatterHash hash{
      .hi = 0x2946731f0d5b2d93ull,
      .lo = 0xd5b54a32d192ed03ull,
  };
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.element), salt);
  hash = internal::MixIdentity(hash, desc.element_count, salt);
  hash = internal::MixIdentity(hash, desc.output_count, salt);
  return hash;
}

}  // namespace rund::kernel

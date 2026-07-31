#pragma once

#include <kernel/internal/compute/identity.hpp>
#include <kernel/program/compute/stencil/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr StencilHash HashStencil(
    const StencilDesc& desc) noexcept {
  constexpr u64 salt = 0x517cc1b727220a95ull;
  StencilHash hash{
      .hi = 0x2d358dccaa6c78a5ull,
      .lo = 0x8f5b7d7e2f3a41c9ull,
  };
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.op), salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.element), salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.boundary), salt);
  hash = internal::MixIdentity(hash, desc.element_count, salt);
  hash = internal::MixIdentity(hash, desc.radius, salt);
  return hash;
}

}  // namespace rund::kernel

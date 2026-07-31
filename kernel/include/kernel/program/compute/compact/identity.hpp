#pragma once

#include <kernel/internal/compute/identity.hpp>
#include <kernel/program/compute/compact/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr CompactHash HashCompact(
    const CompactDesc& desc) noexcept {
  constexpr u64 salt = 0x517cc1b727220a95ull;
  CompactHash hash{
      .hi = 0x452821e638d01377ull,
      .lo = 0xbe5466cf34e90c6cull,
  };
  hash = internal::MixIdentity(hash, desc.element_count, salt);
  hash = internal::MixIdentity(hash, desc.output_capacity, salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.flag_bytes), salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.output_bytes), salt);
  return hash;
}

}  // namespace rund::kernel

#pragma once

#include <kernel/internal/compute/identity.hpp>
#include <kernel/program/compute/partition/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr PartitionHash HashPartition(
    const PartitionDesc& desc) noexcept {
  constexpr u64 salt = 0x517cc1b727220a95ull;
  PartitionHash hash{
      .hi = 0x71b811d6df4a07e3ull,
      .lo = 0x16d94b5c21d109a5ull,
  };
  hash = internal::MixIdentity(hash, desc.element_count, salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.flag_bytes), salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.value_bytes), salt);
  return hash;
}

}  // namespace rund::kernel

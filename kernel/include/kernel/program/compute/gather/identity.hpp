#pragma once

#include <kernel/internal/compute/identity.hpp>
#include <kernel/program/compute/gather/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr GatherHash HashGather(
    const GatherDesc& desc) noexcept {
  constexpr u64 salt = 0x517cc1b727220a95ull;
  GatherHash hash{
      .hi = 0x7f4a7c159e3779b9ull,
      .lo = 0x6c8e9cf570932bd5ull,
  };
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.element), salt);
  hash = internal::MixIdentity(hash, desc.element_count, salt);
  hash = internal::MixIdentity(hash, desc.source_count, salt);
  hash = internal::MixIdentity(
      hash, static_cast<u64>(desc.count_source), salt);
  return hash;
}

}  // namespace rund::kernel

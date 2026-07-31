#pragma once

#include <kernel/internal/compute/identity.hpp>
#include <kernel/program/compute/histogram/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr HistogramHash
HashHistogram(const HistogramDesc& desc) noexcept {
  constexpr u64 salt = 0x517cc1b727220a95ull;
  HistogramHash hash{
      .hi = 0x7d1f6b5a38e9a427ull,
      .lo = 0x1f83d9abfb41bd6bull,
  };
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.index), salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.count), salt);
  hash = internal::MixIdentity(hash, desc.element_count, salt);
  hash = internal::MixIdentity(hash, desc.bin_count, salt);
  return hash;
}

}  // namespace rund::kernel

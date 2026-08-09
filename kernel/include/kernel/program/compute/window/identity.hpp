#pragma once

#include <kernel/internal/compute/identity.hpp>
#include <kernel/program/compute/window/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr WindowHash HashWindow(const WindowDesc &desc) noexcept {
  constexpr u64 salt = 0x8f9d1b7c52e3a641ull;
  WindowHash hash{
      .hi = 0x6fd0f15c9b35e8a7ull,
      .lo = 0xb4a2d78317ce509dull,
  };
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.op), salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.element), salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.boundary), salt);
  hash = internal::MixIdentity(hash, static_cast<u64>(desc.domain), salt);
  hash = internal::MixIdentity(hash, desc.fixed_format.integer_bits, salt);
  hash = internal::MixIdentity(hash, desc.fixed_format.fraction_bits, salt);
  hash = internal::MixIdentity(
      hash, static_cast<u64>(desc.fixed_format.rounding), salt);
  hash = internal::MixIdentity(
      hash, static_cast<u64>(desc.fixed_format.overflow), salt);
  hash = internal::MixIdentity(
      hash, static_cast<u64>(desc.fixed_format.approximation), salt);
  hash = internal::MixIdentity(hash, desc.input_count, salt);
  hash = internal::MixIdentity(hash, desc.output_count, salt);
  hash = internal::MixIdentity(hash, desc.window_size, salt);
  hash = internal::MixIdentity(hash, desc.stride, salt);
  return internal::MixIdentity(hash, desc.pad_left, salt);
}

} // namespace rund::kernel

#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class ComputeApi : u8 {
  Metal = 1u,
  Vulkan = 2u,
  Cpu = 3u,
};

[[nodiscard]] constexpr bool ComputeApiValid(const ComputeApi api) noexcept {
  return api == ComputeApi::Metal || api == ComputeApi::Vulkan ||
         api == ComputeApi::Cpu;
}

enum class ComputeScalar : u8 { Lane32 = 1u, Lane64 = 2u };

[[nodiscard]] constexpr bool
ComputeScalarValid(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane32 || scalar == ComputeScalar::Lane64;
}

enum class ComputeDomain : u8 {
  I32 = 1u,
  U32 = 2u,
  I64 = 3u,
  U64 = 4u,
  Fixed = 5u,
};

enum class ComputeBindingAccess : u8 {
  Read = 1u,
  Write = 2u,
};

enum class ComputeRounding : u8 {
  TowardZero = 1u,
  Down = 2u,
  Up = 3u,
  NearestEven = 4u,
};

enum class ComputeOverflow : u8 {
  Saturate = 1u,
  Wrap = 2u,
};

enum class ComputeApproximation : u8 {
  Exact = 1u,
  Deterministic = 2u,
};

struct ComputeFixedFormat final {
  u8 integer_bits = 0u;
  u8 fraction_bits = 0u;
  ComputeRounding rounding = static_cast<ComputeRounding>(0u);
  ComputeOverflow overflow = static_cast<ComputeOverflow>(0u);
  ComputeApproximation approximation = static_cast<ComputeApproximation>(0u);

  [[nodiscard]] constexpr bool
  operator==(const ComputeFixedFormat &) const noexcept = default;
};

[[nodiscard]] constexpr bool
ComputeFixedFormatAbsent(const ComputeFixedFormat format) noexcept {
  return format.integer_bits == 0u && format.fraction_bits == 0u &&
         static_cast<u8>(format.rounding) == 0u &&
         static_cast<u8>(format.overflow) == 0u &&
         static_cast<u8>(format.approximation) == 0u;
}

[[nodiscard]] constexpr u32
ComputeScalarBits(const ComputeScalar scalar) noexcept {
  switch (scalar) {
  case ComputeScalar::Lane32:
    return 32u;
  case ComputeScalar::Lane64:
    return 64u;
  }
  return 0u;
}

[[nodiscard]] constexpr bool
ComputeFixedFormatValid(const ComputeScalar scalar,
                        const ComputeFixedFormat format) noexcept {
  const u32 width = static_cast<u32>(format.integer_bits) +
                    static_cast<u32>(format.fraction_bits);
  return format.integer_bits != 0u && format.fraction_bits != 0u &&
         width == ComputeScalarBits(scalar) &&
         static_cast<u8>(format.rounding) >=
             static_cast<u8>(ComputeRounding::TowardZero) &&
         static_cast<u8>(format.rounding) <=
             static_cast<u8>(ComputeRounding::NearestEven) &&
         static_cast<u8>(format.overflow) >=
             static_cast<u8>(ComputeOverflow::Saturate) &&
         static_cast<u8>(format.overflow) <=
             static_cast<u8>(ComputeOverflow::Wrap) &&
         static_cast<u8>(format.approximation) >=
             static_cast<u8>(ComputeApproximation::Exact) &&
         static_cast<u8>(format.approximation) <=
             static_cast<u8>(ComputeApproximation::Deterministic);
}

[[nodiscard]] constexpr bool
ComputeIntermediateFormatValid(const ComputeFixedFormat format) noexcept {
  if (ComputeFixedFormatAbsent(format)) {
    return true;
  }
  const u32 width = static_cast<u32>(format.integer_bits) +
                    static_cast<u32>(format.fraction_bits);
  return format.integer_bits != 0u && format.fraction_bits != 0u &&
         width <= 128u &&
         static_cast<u8>(format.rounding) >=
             static_cast<u8>(ComputeRounding::TowardZero) &&
         static_cast<u8>(format.rounding) <=
             static_cast<u8>(ComputeRounding::NearestEven) &&
         static_cast<u8>(format.overflow) >=
             static_cast<u8>(ComputeOverflow::Saturate) &&
         static_cast<u8>(format.overflow) <=
             static_cast<u8>(ComputeOverflow::Wrap) &&
         static_cast<u8>(format.approximation) >=
             static_cast<u8>(ComputeApproximation::Exact) &&
         static_cast<u8>(format.approximation) <=
             static_cast<u8>(ComputeApproximation::Deterministic);
}

[[nodiscard]] constexpr ComputeFixedFormat
PrimitiveFixedFormat(const ComputeFixedFormat source,
                     const ComputeApproximation approximation) noexcept {
  ComputeFixedFormat format = source;
  format.approximation = approximation;
  return format;
}

[[nodiscard]] constexpr ComputeScalar
ComputeScalarForBytes(const u32 element_bytes) noexcept {
  switch (element_bytes) {
  case 4u:
    return ComputeScalar::Lane32;
  case 8u:
    return ComputeScalar::Lane64;
  default:
    return static_cast<ComputeScalar>(0u);
  }
}

[[nodiscard]] constexpr bool ComputePrimitiveFixedFormatValid(
    const u32 element_bytes, const ComputeFixedFormat format,
    const ComputeApproximation approximation) noexcept {
  return (element_bytes == 4u || element_bytes == 8u) &&
         ComputeFixedFormatValid(ComputeScalarForBytes(element_bytes),
                                 format) &&
         format.approximation == approximation;
}

enum class ComputeCountSource : u8 {
  Descriptor = 0u,
  BufferU32 = 1u,
  BufferU64 = 2u,
};

[[nodiscard]] constexpr u32
ComputeCountBytes(const ComputeCountSource source) noexcept {
  return source == ComputeCountSource::BufferU32
             ? 4u
             : (source == ComputeCountSource::BufferU64 ? 8u : 0u);
}

struct ComputeCaps {
  ComputeApi api = ComputeApi::Metal;
  u64 device_bytes = 0u;
  u64 staging_bytes = 0u;
  u64 max_window_tiles = 0u;
  u64 storage_alignment = 1u;
  u32 subgroup_width = 0u;
  bool ok = false;
  const char *reason = "compute_caps_invalid";
};

[[nodiscard]] constexpr bool
ComputeStorageAlignmentValid(const u64 alignment) noexcept {
  return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

struct ComputeMap {
  u64 op_hash_hi = 0u;
  u64 op_hash_lo = 0u;
  ComputeApi api = ComputeApi::Metal;
  ComputeScalar scalar = ComputeScalar::Lane32;
  ComputeDomain domain = ComputeDomain::Fixed;
  ComputeFixedFormat fixed_format{};
  u64 input_buffer_count = 0u;
  u64 output_buffer_count = 1u;
  u64 input_bytes_per_tile = 0u;
  u64 output_bytes_per_tile = 0u;
  u64 param_bytes = 0u;
  u64 metadata_bytes_per_tile = 0u;
};

struct ComputeLimit {
  u64 staging_bytes = 0u;
  u64 max_window_tiles = 0u;
};

struct ComputeDispatchPlan {
  u64 bytes_per_tile = 0u;
  u64 staging_bytes = 0u;
  u64 dispatch_window_tiles = 0u;
  u64 dispatch_count = 0u;
  bool ok = false;
  const char *reason = "compute_plan_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

struct ComputePlan {
  u64 phase_id = 0u;
  u64 tile_count = 0u;
  u64 op_hash_hi = 0u;
  u64 op_hash_lo = 0u;
  ComputeApi api = ComputeApi::Metal;
  ComputeScalar scalar = ComputeScalar::Lane32;
  ComputeDomain domain = ComputeDomain::Fixed;
  ComputeFixedFormat fixed_format{};
  u64 input_buffer_count = 0u;
  u64 output_buffer_count = 1u;
  u64 input_bytes_per_tile = 0u;
  u64 output_bytes_per_tile = 0u;
  u64 param_bytes = 0u;
  u64 metadata_bytes_per_tile = 0u;
  u64 bytes_per_tile = 0u;
  u64 staging_bytes = 0u;
  u64 dispatch_window_tiles = 0u;
  u64 dispatch_count = 0u;
  bool fixed_authoritative = false;
  bool ok = false;
  const char *reason = "compute_plan_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

} // namespace rund::kernel

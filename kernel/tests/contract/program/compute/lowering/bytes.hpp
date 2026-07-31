#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace program_compute_contract::lowering_support {

inline constexpr rund::kernel::u8 kI32NumericMode =
    rund::kernel::compute_lowering_detail::DomainModeFor(
        rund::kernel::ComputeScalar::Lane32, rund::kernel::ComputeDomain::I32);
inline constexpr rund::kernel::u8 kI64NumericMode =
    rund::kernel::compute_lowering_detail::DomainModeFor(
        rund::kernel::ComputeScalar::Lane64, rund::kernel::ComputeDomain::I64);

inline void AppendU8(std::vector<rund::kernel::u8>& bytes,
                     const rund::kernel::u8 value) {
  bytes.push_back(value);
}

inline void AppendU32(std::vector<rund::kernel::u8>& bytes,
                      const rund::kernel::u32 value) {
  for (std::size_t index = 0u; index < 4u; ++index) {
    bytes.push_back(static_cast<rund::kernel::u8>(
        (value >> (index * 8u)) & 0xffu));
  }
}

inline void AppendBytes(std::vector<rund::kernel::u8>& bytes,
                        const std::string_view value) {
  AppendU32(bytes, static_cast<rund::kernel::u32>(value.size()));
  for (const char c : value) {
    bytes.push_back(static_cast<rund::kernel::u8>(c));
  }
}

inline void AppendIntegerNumericPolicy(std::vector<rund::kernel::u8>& bytes) {
  for (std::size_t field = 0u; field < 5u; ++field) {
    AppendU8(bytes, 0u);
  }
}

inline void AppendBinding(std::vector<rund::kernel::u8>& bytes,
                          const rund::kernel::u8 kind,
                          const std::string_view name) {
  AppendU8(bytes, kind);
  AppendU8(bytes, kI32NumericMode);
  AppendBytes(bytes, name);
  AppendU32(bytes, 4u);
  AppendU8(bytes, 0u);
  AppendU32(bytes, 0u);
}

inline void AppendBinding(std::vector<rund::kernel::u8>& bytes,
                          const rund::kernel::u8 kind,
                          const std::string_view name,
                          const rund::kernel::u32 element_bytes) {
  AppendU8(bytes, kind);
  AppendU8(bytes, element_bytes == 8u ? kI64NumericMode : kI32NumericMode);
  AppendBytes(bytes, name);
  AppendU32(bytes, element_bytes);
  AppendU8(bytes, 0u);
  AppendU32(bytes, 0u);
}

inline void AppendParamBinding(std::vector<rund::kernel::u8>& bytes,
                               const std::string_view name,
                               const rund::kernel::u32 value) {
  AppendU8(bytes, 1u);
  AppendU8(bytes, kI32NumericMode);
  AppendBytes(bytes, name);
  AppendU32(bytes, 4u);
  AppendU8(bytes, 0u);
  AppendU32(bytes, 4u);
  AppendU32(bytes, value);
}

inline void AppendNode(std::vector<rund::kernel::u8>& bytes,
                       const rund::kernel::IrOp op,
                       const rund::kernel::u32 lhs,
                       const rund::kernel::u32 rhs,
                       const rund::kernel::u32 aux) {
  AppendU8(bytes, static_cast<rund::kernel::u8>(op));
  AppendU32(bytes, lhs);
  AppendU32(bytes, rhs);
  AppendU32(bytes, aux);
  AppendIntegerNumericPolicy(bytes);
}

[[nodiscard]] inline bool ReadU32(const std::vector<rund::kernel::u8>& bytes,
                                  std::size_t& offset,
                                  rund::kernel::u32& value) noexcept {
  if (offset > bytes.size() || bytes.size() - offset < 4u) {
    return false;
  }
  value = static_cast<rund::kernel::u32>(bytes[offset]) |
          (static_cast<rund::kernel::u32>(bytes[offset + 1u]) << 8u) |
          (static_cast<rund::kernel::u32>(bytes[offset + 2u]) << 16u) |
          (static_cast<rund::kernel::u32>(bytes[offset + 3u]) << 24u);
  offset += 4u;
  return true;
}

[[nodiscard]] inline bool WriteU32At(std::vector<rund::kernel::u8>& bytes,
                                     const std::size_t offset,
                                     const rund::kernel::u32 value) noexcept {
  if (offset > bytes.size() || bytes.size() - offset < 4u) {
    return false;
  }
  for (std::size_t index = 0u; index < 4u; ++index) {
    bytes[offset + index] = static_cast<rund::kernel::u8>(
        (value >> (index * 8u)) & 0xffu);
  }
  return true;
}

[[nodiscard]] inline bool SkipBytes(const std::vector<rund::kernel::u8>& bytes,
                                    std::size_t& offset,
                                    const std::size_t count) noexcept {
  if (offset > bytes.size() || bytes.size() - offset < count) {
    return false;
  }
  offset += count;
  return true;
}

[[nodiscard]] inline bool SkipLengthPrefixedBytes(
    const std::vector<rund::kernel::u8>& bytes,
    std::size_t& offset) noexcept {
  rund::kernel::u32 size = 0u;
  return ReadU32(bytes, offset, size) && SkipBytes(bytes, offset, size);
}

[[nodiscard]] inline bool LocateBindingCountOffset(
    const std::vector<rund::kernel::u8>& bytes,
    std::size_t& offset) noexcept {
  offset = 0u;
  return SkipLengthPrefixedBytes(bytes, offset) &&
         SkipLengthPrefixedBytes(bytes, offset) &&
         SkipBytes(bytes, offset, 6u) && bytes.size() - offset >= 4u;
}

[[nodiscard]] inline bool SetBindingCount(std::vector<rund::kernel::u8>& bytes,
                                          const rund::kernel::u32 count) noexcept {
  std::size_t offset = 0u;
  return LocateBindingCountOffset(bytes, offset) &&
         WriteU32At(bytes, offset, count);
}

[[nodiscard]] inline bool SetFirstBindingElementBytes(
    std::vector<rund::kernel::u8>& bytes,
    const rund::kernel::u32 element_bytes) noexcept {
  std::size_t offset = 0u;
  if (!LocateBindingCountOffset(bytes, offset) ||
      !SkipBytes(bytes, offset, 4u) || !SkipBytes(bytes, offset, 2u) ||
      !SkipLengthPrefixedBytes(bytes, offset)) {
    return false;
  }
  return WriteU32At(bytes, offset, element_bytes);
}

[[nodiscard]] inline rund::kernel::ComputeIR RehashIr(
    rund::kernel::ComputeIR ir) {
  const rund::kernel::compute_ir_detail::ComputeIrHash hash =
      rund::kernel::compute_ir_detail::HashComputeIrCanonicalBytes(
          ir.canonical_bytes.empty() ? nullptr : ir.canonical_bytes.data(),
          static_cast<rund::kernel::u64>(ir.canonical_bytes.size()));
  ir.op_hash_hi = hash.hi;
  ir.op_hash_lo = hash.lo;
  return ir;
}

[[nodiscard]] inline rund::kernel::ComputeIR IrFromBytes(
    std::vector<rund::kernel::u8> bytes) {
  rund::kernel::ComputeIR ir{
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = rund::kernel::ComputeDomain::I32,
      .canonical_bytes = std::move(bytes),
      .ok = true,
      .reason = "ok",
  };
  return RehashIr(std::move(ir));
}

[[nodiscard]] inline rund::kernel::ComputeIR IrFromBytes(
    std::vector<rund::kernel::u8> bytes,
    const rund::kernel::ComputeScalar scalar) {
  rund::kernel::ComputeIR ir{
      .scalar = scalar,
      .domain = scalar == rund::kernel::ComputeScalar::Lane64
                    ? rund::kernel::ComputeDomain::I64
                    : rund::kernel::ComputeDomain::I32,
      .canonical_bytes = std::move(bytes),
      .ok = true,
      .reason = "ok",
  };
  return RehashIr(std::move(ir));
}

[[nodiscard]] inline bool ReplaceFirstNodeOp(
    std::vector<rund::kernel::u8>& bytes,
    const rund::kernel::u8 op) noexcept {
  std::size_t offset = 0u;
  if (!SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipBytes(bytes, offset, 6u)) {
    return false;
  }

  rund::kernel::u32 binding_count = 0u;
  if (!ReadU32(bytes, offset, binding_count)) {
    return false;
  }
  for (rund::kernel::u32 binding = 0u; binding < binding_count; ++binding) {
    if (!SkipBytes(bytes, offset, 2u) ||
        !SkipLengthPrefixedBytes(bytes, offset) ||
        !SkipBytes(bytes, offset, 4u) || !SkipBytes(bytes, offset, 1u) ||
        !SkipLengthPrefixedBytes(bytes, offset)) {
      return false;
    }
  }

  rund::kernel::u32 node_count = 0u;
  if (!ReadU32(bytes, offset, node_count) || node_count == 0u ||
      offset >= bytes.size()) {
    return false;
  }
  bytes[offset] = op;
  return true;
}

[[nodiscard]] inline bool SetLastNodeLhs(
    std::vector<rund::kernel::u8>& bytes,
    const rund::kernel::u32 lhs) noexcept {
  std::size_t offset = 0u;
  if (!SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipBytes(bytes, offset, 6u)) {
    return false;
  }

  rund::kernel::u32 binding_count = 0u;
  if (!ReadU32(bytes, offset, binding_count)) {
    return false;
  }
  for (rund::kernel::u32 binding = 0u; binding < binding_count; ++binding) {
    if (!SkipBytes(bytes, offset, 2u) ||
        !SkipLengthPrefixedBytes(bytes, offset) ||
        !SkipBytes(bytes, offset, 4u) || !SkipBytes(bytes, offset, 1u) ||
        !SkipLengthPrefixedBytes(bytes, offset)) {
      return false;
    }
  }

  rund::kernel::u32 node_count = 0u;
  constexpr std::size_t kNodeBytes = 18u;
  if (!ReadU32(bytes, offset, node_count) || node_count == 0u ||
      node_count > (bytes.size() - offset) / kNodeBytes) {
    return false;
  }
  offset += static_cast<std::size_t>(node_count - 1u) * kNodeBytes + 1u;
  return WriteU32At(bytes, offset, lhs);
}

[[nodiscard]] inline bool SetNodeApproximation(
    std::vector<rund::kernel::u8>& bytes,
    const rund::kernel::u32 one_based_node,
    const rund::kernel::ComputeApproximation approximation) noexcept {
  std::size_t offset = 0u;
  if (one_based_node == 0u ||
      !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipBytes(bytes, offset, 6u)) {
    return false;
  }
  rund::kernel::u32 binding_count = 0u;
  if (!ReadU32(bytes, offset, binding_count)) {
    return false;
  }
  for (rund::kernel::u32 binding = 0u; binding < binding_count; ++binding) {
    if (!SkipBytes(bytes, offset, 2u) ||
        !SkipLengthPrefixedBytes(bytes, offset) ||
        !SkipBytes(bytes, offset, 4u) || !SkipBytes(bytes, offset, 1u) ||
        !SkipLengthPrefixedBytes(bytes, offset)) {
      return false;
    }
  }
  rund::kernel::u32 node_count = 0u;
  constexpr std::size_t kNodeBytes = 18u;
  if (!ReadU32(bytes, offset, node_count) || one_based_node > node_count ||
      node_count > (bytes.size() - offset) / kNodeBytes) {
    return false;
  }
  offset += static_cast<std::size_t>(one_based_node - 1u) * kNodeBytes + 17u;
  if (offset >= bytes.size()) {
    return false;
  }
  bytes[offset] = static_cast<rund::kernel::u8>(approximation);
  return true;
}

[[nodiscard]] inline bool WriteFixedFormatAt(
    std::vector<rund::kernel::u8>& bytes, const std::size_t offset,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (offset > bytes.size() || bytes.size() - offset < 5u) {
    return false;
  }
  bytes[offset] = format.integer_bits;
  bytes[offset + 1u] = format.fraction_bits;
  bytes[offset + 2u] = static_cast<rund::kernel::u8>(format.rounding);
  bytes[offset + 3u] = static_cast<rund::kernel::u8>(format.overflow);
  bytes[offset + 4u] = static_cast<rund::kernel::u8>(format.approximation);
  return true;
}

[[nodiscard]] inline bool SetTopLevelFixedFormat(
    std::vector<rund::kernel::u8>& bytes,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  std::size_t offset = 0u;
  return SkipLengthPrefixedBytes(bytes, offset) &&
         SkipLengthPrefixedBytes(bytes, offset) &&
         SkipBytes(bytes, offset, 1u) &&
         WriteFixedFormatAt(bytes, offset, format);
}

[[nodiscard]] inline bool SetNodeFixedFormat(
    std::vector<rund::kernel::u8>& bytes,
    const rund::kernel::u32 one_based_node,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  std::size_t offset = 0u;
  if (one_based_node == 0u ||
      !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipBytes(bytes, offset, 6u)) {
    return false;
  }
  rund::kernel::u32 binding_count = 0u;
  if (!ReadU32(bytes, offset, binding_count)) {
    return false;
  }
  for (rund::kernel::u32 binding = 0u; binding < binding_count; ++binding) {
    if (!SkipBytes(bytes, offset, 2u) ||
        !SkipLengthPrefixedBytes(bytes, offset) ||
        !SkipBytes(bytes, offset, 4u) || !SkipBytes(bytes, offset, 1u) ||
        !SkipLengthPrefixedBytes(bytes, offset)) {
      return false;
    }
  }
  rund::kernel::u32 node_count = 0u;
  constexpr std::size_t kNodeBytes = 18u;
  if (!ReadU32(bytes, offset, node_count) || one_based_node > node_count ||
      node_count > (bytes.size() - offset) / kNodeBytes) {
    return false;
  }
  offset += static_cast<std::size_t>(one_based_node - 1u) * kNodeBytes + 13u;
  return WriteFixedFormatAt(bytes, offset, format);
}

}  // namespace program_compute_contract::lowering_support

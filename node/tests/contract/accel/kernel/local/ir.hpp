#pragma once

#include <kernel/program/compute/ir.hpp>

#include <cstddef>
#include <vector>

namespace node_accel_contract::kernel_case {

[[nodiscard]] inline bool ReadU32(const std::vector<rund::kernel::u8>& bytes,
                                  std::size_t& cursor,
                                  rund::kernel::u32& value) {
  if (cursor + 4u > bytes.size()) { return false; }
  value = static_cast<rund::kernel::u32>(bytes[cursor]) |
          (static_cast<rund::kernel::u32>(bytes[cursor + 1u]) << 8u) |
          (static_cast<rund::kernel::u32>(bytes[cursor + 2u]) << 16u) |
          (static_cast<rund::kernel::u32>(bytes[cursor + 3u]) << 24u);
  cursor += 4u;
  return true;
}

[[nodiscard]] inline bool SkipBytes(const std::vector<rund::kernel::u8>& bytes,
                                    std::size_t& cursor) {
  rund::kernel::u32 size = 0u;
  if (!ReadU32(bytes, cursor, size) || cursor + size > bytes.size()) {
    return false;
  }
  cursor += size;
  return true;
}

[[nodiscard]] inline bool ReplaceFirstNodeOp(
    std::vector<rund::kernel::u8>& bytes,
    const rund::kernel::u8 op) {
  std::size_t offset = 0u;
  if (!SkipBytes(bytes, offset) || !SkipBytes(bytes, offset) ||
      offset + 1u > bytes.size()) {
    return false;
  }
  ++offset;
  rund::kernel::u32 binding_count = 0u;
  if (!ReadU32(bytes, offset, binding_count)) { return false; }
  for (rund::kernel::u32 index = 0u; index < binding_count; ++index) {
    if (offset + 1u > bytes.size()) { return false; }
    ++offset;
    if (!SkipBytes(bytes, offset) || offset + 5u > bytes.size()) {
      return false;
    }
    offset += 5u;
    if (!SkipBytes(bytes, offset)) { return false; }
  }
  rund::kernel::u32 node_count = 0u;
  if (!ReadU32(bytes, offset, node_count) || node_count == 0u ||
      offset >= bytes.size()) {
    return false;
  }
  bytes[offset] = op;
  return true;
}

[[nodiscard]] inline rund::kernel::ComputeIR UnsupportedIr(
    const rund::kernel::ComputeIR& source) {
  rund::kernel::ComputeIR ir = source;
  if (!ReplaceFirstNodeOp(ir.canonical_bytes, 0xffu)) { return {}; }
  const rund::kernel::compute_ir_detail::ComputeIrHash hash =
      rund::kernel::compute_ir_detail::HashComputeIrCanonicalBytes(
          ir.canonical_bytes.data(),
          static_cast<rund::kernel::u64>(ir.canonical_bytes.size()));
  ir.op_hash_hi = hash.hi;
  ir.op_hash_lo = hash.lo;
  return ir;
}

}  // namespace node_accel_contract::kernel_case

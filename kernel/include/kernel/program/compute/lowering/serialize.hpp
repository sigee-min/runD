#pragma once

#include <kernel/program/compute/lowering/model.hpp>
#include <string_view>

namespace rund::kernel {
namespace compute_lowering_detail {

inline void AppendSerializedU8(std::vector<u8> &bytes, const u8 value) {
  bytes.push_back(value);
}

inline void AppendSerializedU32(std::vector<u8> &bytes, const u32 value) {
  bytes.push_back(static_cast<u8>(value & 0xffu));
  bytes.push_back(static_cast<u8>((value >> 8u) & 0xffu));
  bytes.push_back(static_cast<u8>((value >> 16u) & 0xffu));
  bytes.push_back(static_cast<u8>((value >> 24u) & 0xffu));
}

inline void AppendSerializedBytes(std::vector<u8> &bytes,
                                  const std::string_view value) {
  AppendSerializedU32(bytes, static_cast<u32>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}

inline void AppendSerializedPayload(std::vector<u8> &bytes,
                                    const std::vector<u8> &value) {
  AppendSerializedU32(bytes, static_cast<u32>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}

inline void AppendSerializedBinding(std::vector<u8> &bytes,
                                    const ParsedBinding &binding) {
  AppendSerializedU8(bytes, binding.kind);
  AppendSerializedU8(bytes, binding.numeric_mode);
  AppendSerializedBytes(bytes, binding.name);
  AppendSerializedU32(bytes, binding.element_bytes);
  AppendSerializedU8(bytes, binding.floating_point_param ? u8{1u} : u8{0u});
  AppendSerializedPayload(bytes, binding.value_bytes);
}

inline void AppendSerializedNode(std::vector<u8> &bytes,
                                 const ParsedNode &node) {
  AppendSerializedU8(bytes, node.op);
  AppendSerializedU32(bytes, node.lhs);
  AppendSerializedU32(bytes, node.rhs);
  AppendSerializedU32(bytes, node.aux);
  AppendSerializedU8(bytes, node.fixed_format.integer_bits);
  AppendSerializedU8(bytes, node.fixed_format.fraction_bits);
  AppendSerializedU8(bytes, static_cast<u8>(node.fixed_format.rounding));
  AppendSerializedU8(bytes, static_cast<u8>(node.fixed_format.overflow));
  AppendSerializedU8(bytes, static_cast<u8>(node.fixed_format.approximation));
}

} // namespace compute_lowering_detail

} // namespace rund::kernel

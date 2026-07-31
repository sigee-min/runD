#pragma once

#include <kernel/program/compute/ir.hpp>
#include <kernel/program/compute/limit.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace rund::kernel {
namespace compute_lowering_detail {

struct ParsedBinding {
  u8 kind = 0u;
  u8 numeric_mode = 0u;
  std::string name;
  u32 element_bytes = 0u;
  bool floating_point_param = false;
  std::vector<u8> value_bytes;
};

struct ParsedNode {
  u8 op = 0u;
  u32 lhs = 0u;
  u32 rhs = 0u;
  u32 aux = 0u;
  ComputeFixedFormat fixed_format{};
};

struct ParsedIR {
  std::string name;
  u8 scalar_mode = 0u;
  ComputeFixedFormat fixed_format{};
  std::vector<ParsedBinding> bindings;
  std::vector<ParsedNode> nodes;
  bool ok = false;
  const char *reason = "compute_ir_invalid";
};

inline constexpr u8 kParamBindingKind = 1u;
inline constexpr u8 kReadBindingKind = 2u;
inline constexpr u8 kWriteBindingKind = 3u;
inline constexpr u64 kTileIndexMetadataBytes = sizeof(u32);
inline constexpr std::size_t kMinBindingBytes = 15u;
inline constexpr std::size_t kSerializedNodeBytes = 18u;

} // namespace compute_lowering_detail

} // namespace rund::kernel

#pragma once

#include <kernel/program/compute/lowering/format.hpp>

namespace rund::kernel::compute_lowering_detail {

inline void AppendGenericComputeKey(std::string &out, const ArtifactKey &key) {
  out += "api=";
  out += ApiName(key.api);
  out += "\nscalar=";
  out += ScalarName(key.scalar);
  out += "\nop_hash_hi=";
  AppendHex64(out, key.op_hash_hi);
  out += "\nop_hash_lo=";
  AppendHex64(out, key.op_hash_lo);
  out += "\ncanonical_ir_hash_hi=";
  AppendHex64(out, key.canonical_ir_hash_hi);
  out += "\ncanonical_ir_hash_lo=";
  AppendHex64(out, key.canonical_ir_hash_lo);
  out += "\n";
}

[[nodiscard]] inline std::string
CpuPlanText(const ParsedIR &parsed, const ArtifactKey &key,
            const std::vector<BindingLayout> &layouts) {
  std::string out;
  out += "rund.compute.cpu.plan\n";
  AppendGenericComputeKey(out, key);
  out += "operation_hex=";
  out += HexText(parsed.name);
  out += "\n";
  AppendBindingLayout(out, parsed, layouts, "");
  AppendNodeLayout(out, parsed, "");
  return out;
}

} // namespace rund::kernel::compute_lowering_detail

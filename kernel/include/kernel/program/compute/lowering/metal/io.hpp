#pragma once

#include <kernel/program/compute/lowering/fixed/ops.hpp>
#include <kernel/program/compute/lowering/metal/syntax.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

inline void AppendMetalLoadBody(std::string &out, const ComputeScalar scalar,
                                const char *const address_space,
                                const char *const function_name) {
  const u32 scalar_bytes = ScalarBytes(scalar);
  const char *const unsigned_type = MetalUnsignedType(scalar);
  out += "inline ";
  out += MetalType(scalar);
  out += " ";
  out += function_name;
  out += "(const ";
  out += address_space;
  out += " uchar* base, uint byte_offset) {\n";
  out += "  const ";
  out += unsigned_type;
  out += " packed = ";
  for (u32 byte = 0u; byte < scalar_bytes; ++byte) {
    if (byte != 0u) {
      out += " |\n      ";
    }
    out += unsigned_type;
    out += "(base[byte_offset";
    if (byte != 0u) {
      out += " + ";
      out += std::to_string(byte);
      out += "u";
    }
    out += "])";
    if (byte != 0u) {
      out += " << ";
      out += std::to_string(byte * 8u);
      out += "u";
    }
  }
  out += ";\n";
  out += "  return ";
  out += MetalType(scalar);
  out += "(packed);\n";
  out += "}\n";
}

inline void AppendMetalStoreBody(std::string &out, const ComputeScalar scalar) {
  const u32 scalar_bytes = ScalarBytes(scalar);
  const char *const unsigned_type = MetalUnsignedType(scalar);
  out += "inline void ";
  out += MetalStoreFunction(scalar);
  out += "(device uchar* base, uint byte_offset, ";
  out += MetalType(scalar);
  out += " value) {\n";
  out += "  const ";
  out += unsigned_type;
  out += " packed = ";
  out += unsigned_type;
  out += "(value);\n";
  for (u32 byte = 0u; byte < scalar_bytes; ++byte) {
    out += "  base[byte_offset";
    if (byte != 0u) {
      out += " + ";
      out += std::to_string(byte);
      out += "u";
    }
    out += "] = uchar((packed >> ";
    out += std::to_string(byte * 8u);
    out += "u) & ";
    out += (scalar == ComputeScalar::Lane64 ? "0xfful" : "0xffu");
    out += ");\n";
  }
  out += "}\n";
}

inline void AppendMetalHelpers(std::string &out, const ParsedIR &parsed,
                               const ArtifactKey &key) {
  AppendMetalLoadBody(out, key.scalar, "device", MetalLoadFunction(key.scalar));
  if (key.scalar == ComputeScalar::Lane64) {
    for (const ParsedBinding &binding : parsed.bindings) {
      if (binding.kind == 2u && binding.element_bytes == sizeof(u32)) {
        AppendMetalLoadBody(out, ComputeScalar::Lane32, "device",
                            MetalLoadFunction(ComputeScalar::Lane32));
        break;
      }
    }
  }
  AppendMetalLoadBody(out, key.scalar, "constant",
                      MetalParamLoadFunction(key.scalar));
  AppendMetalStoreBody(out, key.scalar);
  AppendMetalStoreBody(out, key.scalar == ComputeScalar::Lane64
                                ? ComputeScalar::Lane32
                                : ComputeScalar::Lane64);
  AppendMetalFixedOpHelpers(out, parsed, key);
}

} // namespace compute_lowering_detail

} // namespace rund::kernel

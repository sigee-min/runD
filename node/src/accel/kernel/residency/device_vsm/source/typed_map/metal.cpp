#include "internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>
#include <kernel/program/compute/lowering/metal/source.hpp>

namespace rund::node::accel::detail::device_vsm_typed_map {

bool append_metal_header(
    std::string &source, const rund::kernel::ArtifactKey &key,
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    std::vector<rund::kernel::compute_lowering_detail::BindingLayout> &layouts,
    const bool emit_param_helpers) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  layouts = lowering::BuildBindingLayout(parsed);
  if (layouts.size() != parsed.bindings.size()) {
    return false;
  }
  auto emitted_key = key;
  emitted_key.variant = rund::kernel::LoweringArtifactVariant::DeviceVsm;
  source += "#include <metal_stdlib>\nusing namespace metal;\n";
  source += "// rund.compute.device_vsm.typed_map\n";
  lowering::AppendKey(source, emitted_key, "// ");
  lowering::AppendBindingLayout(source, parsed, layouts, "// ");
  lowering::AppendNodeLayout(source, parsed, "// ");
  lowering::AppendMetalHelpers(source, parsed, key, emit_param_helpers);
  lowering::AppendMetalIntegerDivHelpers(source, parsed, key);
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const auto &binding = parsed.bindings[index];
    if (binding.kind != lowering::kReadBindingKind &&
        binding.kind != lowering::kWriteBindingKind) {
      continue;
    }
    source += "constant uint " + lowering::BindingBaseSymbol(layouts[index]) +
              " = 0u;\nconstant uint " +
              lowering::BindingStrideSymbol(layouts[index]) + " = " +
              std::to_string(binding.element_bytes) + "u;\n";
  }
  source += "struct RundDeviceVsmConfig { ulong logical_elements; uint "
            "payload_elements; uint page_count; uint width; uint "
            "element_words; };\n";
  source += "kernel void rund_compute_map_";
  lowering::AppendHex64Digits(source, key.op_hash_hi);
  source += "_";
  lowering::AppendHex64Digits(source, key.op_hash_lo);
  source += "_device_vsm(\n    constant uchar* rund_params [[buffer(0)]],\n";
  std::uint32_t data_buffers = 0u;
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const auto &binding = parsed.bindings[index];
    if (binding.kind == lowering::kReadBindingKind) {
      source += "    const device uchar* " + layouts[index].symbol +
                " [[buffer(" + std::to_string(layouts[index].buffer) + ")]],\n";
      data_buffers += 1u;
    }
  }
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const auto &binding = parsed.bindings[index];
    if (binding.kind == lowering::kWriteBindingKind) {
      source += "    device uchar* " + layouts[index].symbol + " [[buffer(" +
                std::to_string(layouts[index].buffer) + ")]],\n";
      data_buffers += 1u;
    }
  }
  const std::uint32_t config_buffer = data_buffers + 1u;
  source += "    constant RundDeviceVsmConfig& config [[buffer(" +
            std::to_string(config_buffer) +
            ")]],\n    device atomic_uint* result [[buffer(" +
            std::to_string(config_buffer + 1u) +
            ")]],\n    uint local [[thread_index_in_threadgroup]]) {\n";
  return true;
}

bool append_metal_value(
    std::string &source, const rund::kernel::ArtifactKey &key,
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout>
        &layouts,
    std::string &mapped) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  const std::string value_type =
      key.scalar == rund::kernel::ComputeScalar::Lane32 ? "uint" : "ulong";
  std::vector<std::string> names(parsed.nodes.size() + 1u);
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const auto &node = parsed.nodes[index];
    if (node.op == static_cast<std::uint8_t>(rund::kernel::IrOp::Write)) {
      if (index + 1u != parsed.nodes.size() || node.lhs >= names.size() ||
          names[node.lhs].empty()) {
        return false;
      }
      mapped = "as_type<" + value_type + ">(" + names[node.lhs] + ")";
      return true;
    }
    lowering::AppendMetalNode(source, parsed, key, layouts, node,
                              static_cast<std::uint32_t>(index + 1u), names);
  }
  return false;
}

void append_metal_store(
    std::string &source,
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout>
        &layouts,
    const std::string &index, const std::string &value,
    const rund::kernel::ComputeScalar scalar) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  for (std::size_t binding = 0u; binding < parsed.bindings.size(); ++binding) {
    if (parsed.bindings[binding].kind != lowering::kWriteBindingKind) {
      continue;
    }
    source += "    " +
              std::string{lowering::MetalStoreFunction(
                  scalar)} +
              "(" + layouts[binding].symbol + ", " +
              lowering::BindingBaseSymbol(layouts[binding]) + " + " +
              (scalar == rund::kernel::ComputeScalar::Lane64 ? "ulong(" :
                                                                "uint(") +
              index + ") * " + lowering::BindingStrideSymbol(layouts[binding]) +
              ", as_type<" +
              (scalar == rund::kernel::ComputeScalar::Lane64 ? "long" : "int") +
              ">(" + value + "));\n";
    return;
  }
}

std::string metal_scratch_value(const rund::kernel::ComputeScalar scalar,
                                const std::string &symbol) {
  if (scalar != rund::kernel::ComputeScalar::Lane64) {
    return "uint(" + symbol + "[scratch_word])";
  }
  return "as_type<ulong>(ulong(" + symbol + "[scratch_word]) | (ulong(" +
         symbol + "[scratch_word + 1u]) << 32u))";
}

} // namespace rund::node::accel::detail::device_vsm_typed_map

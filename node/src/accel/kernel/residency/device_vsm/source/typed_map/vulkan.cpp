#include "internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>
#include <kernel/program/compute/lowering/vulkan/source.hpp>

namespace rund::node::accel::detail::device_vsm_typed_map {

bool append_vulkan_prelude(
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
  source = "#version 450\n";
  if (key.scalar == rund::kernel::ComputeScalar::Lane64) {
    source +=
        "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  source += "// rund.compute.device_vsm.typed_map\n";
  lowering::AppendKey(source, emitted_key, "// ");
  lowering::AppendBindingLayout(source, parsed, layouts, "// ");
  lowering::AppendNodeLayout(source, parsed, "// ");
  source +=
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "layout(push_constant) uniform Dispatch { uint logical_elements; "
      "uint payload_elements; uint page_count; uint width; uint "
      "element_words; } config;\n";
  lowering::AppendVulkanBuffers(source, parsed, layouts);
  lowering::AppendVulkanHelpers(source, parsed, key, layouts,
                                emit_param_helpers);
  lowering::AppendVulkanIntegerDivHelpers(source, parsed, key);
  std::uint32_t data_buffers = 0u;
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const auto &binding = parsed.bindings[index];
    if (binding.kind != lowering::kReadBindingKind &&
        binding.kind != lowering::kWriteBindingKind) {
      continue;
    }
    source += "const uint " + lowering::BindingBaseSymbol(layouts[index]) +
              " = 0u;\nconst uint " +
              lowering::BindingStrideSymbol(layouts[index]) + " = " +
              std::to_string(binding.element_bytes) + "u;\n";
    data_buffers += 1u;
  }
  source += "layout(set = 0, binding = " + std::to_string(data_buffers + 1u) +
            ", std430) buffer RundDeviceVsmResult { uint result[]; };\n";
  if (key.scalar == rund::kernel::ComputeScalar::Lane64) {
    source += "shared uint64_t partial[256];\n"
              "shared uint64_t partial_low[256];\n"
              "shared uint64_t partial_high[256];\n";
  }
  return true;
}

void append_vulkan_main(std::string &source) {
  source += "void main() {\n  const uint local = gl_LocalInvocationID.x;\n";
}

bool append_vulkan_value(
    std::string &source, const rund::kernel::ArtifactKey &key,
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout>
        &layouts,
    std::string &mapped) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  std::vector<std::string> names(parsed.nodes.size() + 1u);
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const auto &node = parsed.nodes[index];
    if (node.op == static_cast<std::uint8_t>(rund::kernel::IrOp::Write)) {
      if (index + 1u != parsed.nodes.size() || node.lhs >= names.size() ||
          names[node.lhs].empty()) {
        return false;
      }
      mapped = names[node.lhs];
      return true;
    }
    lowering::AppendVulkanNode(source, parsed, key, layouts, node,
                               static_cast<std::uint32_t>(index + 1u), names);
  }
  return false;
}

void append_vulkan_store(
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
              lowering::VulkanStoreFunctionName(scalar, layouts[binding]) +
              "(" + lowering::BindingBaseSymbol(layouts[binding]) + " + " +
              "uint(" + index + ") * " +
              lowering::BindingStrideSymbol(layouts[binding]) +
              ", " + value + ");\n";
    return;
  }
}

std::string vulkan_scratch_value(const rund::kernel::ComputeScalar scalar,
                                 const std::string &symbol) {
  if (scalar != rund::kernel::ComputeScalar::Lane64) {
    return symbol + "[scratch_word]";
  }
  return "uint64_t(" + symbol + "[scratch_word]) | (uint64_t(" + symbol +
         "[scratch_word + 1u]) << 32ul)";
}

} // namespace rund::node::accel::detail::device_vsm_typed_map

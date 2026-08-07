#include "../../kernel/backend/source_recipe.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitVulkanPartitionSource(
    Sink &sink, const PartitionStage stage, const rund::kernel::u32 flag_bytes,
    const rund::kernel::u32
        value_bytes) noexcept(noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder source{sink};
  source += "#version 450\n";
  source += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : ";
  source += "require\n";
  source += "layout(local_size_x = 256) in;\n";
  source += "layout(set = 0, binding = 0, std430) readonly buffer Params {\n";
  source += "  uint64_t element_count;\n";
  source += "} params;\n";
  if (stage == PartitionStage::Classify) {
    source += "layout(set = 0, binding = 1, std430) readonly buffer Flags {\n";
    source += flag_bytes == sizeof(rund::kernel::u64) ? "  uint64_t flags[];\n"
                                                      : "  uint flags[];\n";
    source += "};\n";
    source += "layout(set = 0, binding = 2, std430) buffer FalseBits {\n";
    source += "  uint false_bits[];\n";
    source += "};\n";
    source += "void main() {\n";
    source += "  const uint gid = gl_GlobalInvocationID.x;\n";
    source += "  if (uint64_t(gid) >= params.element_count) { return; }\n";
    source += flag_bytes == sizeof(rund::kernel::u64)
                  ? "  const bool true_group = flags[gid] != uint64_t(0);\n"
                  : "  const bool true_group = flags[gid] != 0u;\n";
    source += "  false_bits[gid] = true_group ? 0u : 1u;\n";
    source += "}\n";
    return source.valid();
  }
  source += "layout(set = 0, binding = 1, std430) readonly buffer Flags {\n";
  source += flag_bytes == sizeof(rund::kernel::u64) ? "  uint64_t flags[];\n"
                                                    : "  uint flags[];\n";
  source += "};\n";
  const bool wide = value_bytes == sizeof(rund::kernel::u64);
  source += "layout(set = 0, binding = 2, std430) readonly buffer Values {\n";
  source += wide ? "  uint64_t values[];\n" : "  uint values[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 3, std430) buffer Output {\n";
  source +=
      wide ? "  uint64_t output_values[];\n" : "  uint output_values[];\n";
  source += "};\n";
  source +=
      "layout(set = 0, binding = 4, std430) readonly buffer FalseOffsets {\n";
  source += "  uint false_offsets[];\n";
  source += "};\n";
  source += "shared uint false_total_shared;\n";
  source += "void main() {\n";
  source += "  const uint gid = gl_GlobalInvocationID.x;\n";
  source += "  if (gl_LocalInvocationIndex == 0u) {\n";
  source += "    const uint last = uint(params.element_count - 1ul);\n";
  source += flag_bytes == sizeof(rund::kernel::u64)
                ? "    false_total_shared = false_offsets[last] + "
                  "(flags[last] == uint64_t(0) ? 1u : 0u);\n"
                : "    false_total_shared = false_offsets[last] + "
                  "(flags[last] == 0u ? 1u : 0u);\n";
  source += "  }\n";
  source += "  barrier();\n";
  source += "  if (uint64_t(gid) >= params.element_count) { return; }\n";
  source += "  const uint true_rank = gid - false_offsets[gid];\n";
  source +=
      flag_bytes == sizeof(rund::kernel::u64)
          ? "  const uint target = flags[gid] == uint64_t(0) ? "
            "false_offsets[gid] : "
          : "  const uint target = flags[gid] == 0u ? false_offsets[gid] : ";
  source += "false_total_shared + true_rank;\n";
  source += "  output_values[target] = values[gid];\n";
  source += "}\n";
  return source.valid();
}

} // namespace

std::string VulkanPartitionSource(const PartitionStage stage,
                                  const rund::kernel::u32 flag_bytes,
                                  const rund::kernel::u32 value_bytes) {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [&](auto &sink) noexcept(noexcept(EmitVulkanPartitionSource(
                        sink, stage, flag_bytes, value_bytes))) {
    return EmitVulkanPartitionSource(sink, stage, flag_bytes, value_bytes);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool VulkanPartitionSourceBytes(const PartitionStage stage,
                                const rund::kernel::u32 flag_bytes,
                                const rund::kernel::u32 value_bytes,
                                std::uint64_t &bytes) noexcept {
  const auto emit = [&](auto &sink) noexcept(noexcept(EmitVulkanPartitionSource(
                        sink, stage, flag_bytes, value_bytes))) {
    return EmitVulkanPartitionSource(sink, stage, flag_bytes, value_bytes);
  };
  return backend_source_recipe::bytes(emit, bytes);
}
#endif

} // namespace rund::node::accel::detail

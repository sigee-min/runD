#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_source {
namespace {

[[nodiscard]] bool install_dispatch(std::string &source,
                                    const std::uint64_t binding,
                                    const std::uint32_t input_count) {
  constexpr std::string_view dispatch =
      "layout(push_constant) uniform RundDispatch {\n"
      "  uint tile_count;\n"
      "  uint iterations;\n"
      "} rund_dispatch;\n";
  const std::string replacement =
      "layout(push_constant) uniform RundDeviceVsmDispatch {\n"
      "  uint logical_elements;\n"
      "  uint payload_elements;\n"
      "  uint page_count;\n"
      "  uint width;\n"
      "  uint element_words;\n"
      "} rund_vsm;\n"
      "layout(set = 0, binding = " +
      std::to_string(binding) +
      ", std430) buffer RundDeviceVsmResult { uint "
      "rund_vsm_result[]; };\n"
      "layout(set = 0, binding = " +
      std::to_string(binding + 1u) +
      ", std430) buffer RundDeviceVsmRingState { uint "
      "rund_vsm_ring_state[]; };\n"
      "layout(set = 0, binding = " +
      std::to_string(binding + 2u) +
      ", std430) buffer RundDeviceVsmRingScratch { uint "
      "rund_vsm_ring_scratch[]; };\n";
  std::string aliases = replacement;
  for (std::uint32_t input = 0u; input < input_count; ++input) {
    aliases +=
        "layout(set = 0, binding = " + std::to_string(binding + 3u + input) +
        ", std430) buffer RundDeviceVsmInputScratch" + std::to_string(input) +
        " { uint rund_vsm_input_scratch_" + std::to_string(input) + "[]; };\n";
  }
  for (std::uint32_t input = 0u; input < input_count; ++input) {
    aliases += "layout(set = 0, binding = " +
               std::to_string(binding + 3u + input_count + input) +
               ", std430) readonly buffer RundDeviceVsmInputAlias" +
               std::to_string(input) + " { uint rund_vsm_input_alias_" +
               std::to_string(input) + "[]; };\n";
  }
  aliases += "layout(set = 0, binding = " +
             std::to_string(binding + 3u + 2u * input_count) +
             ", std430) buffer RundDeviceVsmOutputAlias { uint "
             "rund_vsm_output_alias[]; };\n";
  return replace_one(source, dispatch, aliases);
}

[[nodiscard]] bool install_controller(std::string &source,
                                      const std::uint32_t input_count) {
  constexpr std::string_view entry =
      "void main() {\n"
      "  const uint gid = gl_GlobalInvocationID.x;\n"
      "  if (gid >= rund_dispatch.tile_count) { return; }\n";
  std::string controller =
      "void main() {\n"
      "  const uint rund_local = gl_LocalInvocationID.x;\n"
      "  const uint rund_worker = gl_WorkGroupID.x;\n"
      "  for (uint rund_page = rund_worker; rund_page < "
      "rund_vsm.page_count; rund_page += rund_vsm.width) {\n"
      "    const uint rund_page_begin = rund_page * "
      "rund_vsm.payload_elements;\n"
      "    const uint rund_remaining = rund_vsm.logical_elements - "
      "rund_page_begin;\n"
      "    const uint rund_page_elements = min(rund_remaining, "
      "rund_vsm.payload_elements);\n"
      "    const uint rund_slot = rund_page % rund_vsm.width;\n"
      "    const uint rund_turn = rund_page / rund_vsm.width;\n"
      "    if (rund_local == 0u) {\n"
      "      const uint rund_prior_turn = "
      "atomicAdd(rund_vsm_ring_state[rund_slot], 1u);\n"
      "      if (rund_prior_turn != rund_turn) "
      "atomicAdd(rund_vsm_result[6], 1u);\n"
      "    }\n"
      "    memoryBarrierBuffer(); barrier();\n"
      "    const uint rund_payload_words = "
      "rund_vsm.payload_elements * rund_vsm.element_words;\n"
      "    for (uint rund_page_local = rund_local; rund_page_local < "
      "rund_page_elements; rund_page_local += gl_WorkGroupSize.x) {\n"
      "      const uint rund_global_word = "
      "(rund_page_begin + rund_page_local) * rund_vsm.element_words;\n"
      "      const uint rund_scratch_word = "
      "rund_slot * rund_payload_words + "
      "rund_page_local * rund_vsm.element_words;\n"
      "      for (uint rund_word = 0u; rund_word < "
      "rund_vsm.element_words; ++rund_word) {\n";
  for (std::uint32_t input = 0u; input < input_count; ++input) {
    controller += "        rund_vsm_input_scratch_" + std::to_string(input) +
                  "[rund_scratch_word + rund_word] = "
                  "rund_vsm_input_alias_" +
                  std::to_string(input) + "[rund_global_word + rund_word];\n";
  }
  controller +=
      "      }\n"
      "    }\n"
      "    memoryBarrierBuffer(); barrier();\n"
      "    for (uint rund_page_local = rund_local; rund_page_local < "
      "rund_page_elements; rund_page_local += gl_WorkGroupSize.x) {\n"
      "      const uint gid = rund_slot * rund_vsm.payload_elements + "
      "rund_page_local;\n";
  return replace_one(source, entry, controller);
}

[[nodiscard]] bool close_controller(std::string &source,
                                    const std::uint32_t input_count) {
  const std::size_t end = source.rfind("}\n");
  if (end == std::string::npos) {
    return false;
  }
  std::string replacement =
      "    }\n"
      "    memoryBarrierBuffer();\n"
      "    barrier();\n"
      "    for (uint rund_page_local = rund_local; "
      "rund_page_local < rund_page_elements; "
      "rund_page_local += gl_WorkGroupSize.x) {\n"
      "      const uint rund_global_word = "
      "(rund_page_begin + rund_page_local) * "
      "rund_vsm.element_words;\n"
      "      const uint rund_scratch_word = "
      "rund_slot * rund_payload_words + "
      "rund_page_local * rund_vsm.element_words;\n"
      "      for (uint rund_word = 0u; rund_word < "
      "rund_vsm.element_words; ++rund_word) {\n"
      "        rund_vsm_output_alias[rund_global_word + "
      "rund_word] = rund_vsm_ring_scratch[rund_scratch_word + "
      "rund_word];\n"
      "      }\n"
      "    }\n"
      "    memoryBarrierBuffer(); barrier();\n"
      "    if (rund_local == 0u) {\n"
      "      atomicAdd(rund_vsm_result[0], 1u);\n"
      "      atomicAdd(rund_vsm_result[1], 1u);\n"
      "      atomicAdd(rund_vsm_result[2], 1u);\n"
      "      atomicAdd(rund_vsm_result[3], 1u);\n"
      "      atomicAdd(rund_vsm_result[4], 1u);\n"
      "      atomicAdd(rund_vsm_result[5], 1u);\n"
      "      if (rund_page >= rund_vsm.width) {\n"
      "        atomicAdd(rund_vsm_result[10], 1u);\n"
      "      }\n"
      "      const uint rund_schedule = 3u * (rund_page + 1u) + "
      "5u * (rund_slot + 1u) + 7u * (rund_turn + 1u) + "
      "11u * rund_page_elements;\n"
      "      atomicAdd(rund_vsm_result[11], rund_schedule);\n"
      "      atomicAdd(rund_vsm_result[12], ";
  replacement += std::to_string(input_count + 1u);
  replacement += "u);\n"
                 "    }\n"
                 "    memoryBarrierBuffer();\n"
                 "    barrier();\n"
                 "  }\n"
                 "}\n";
  source.replace(end, 2u, replacement);
  return true;
}

} // namespace

bool transform_vulkan(std::string &source, const std::uint64_t binding,
                      const std::uint32_t input_count) {
  return install_dispatch(source, binding, input_count) &&
         install_controller(source, input_count) &&
         close_controller(source, input_count);
}

} // namespace rund::node::accel::detail::device_vsm_source

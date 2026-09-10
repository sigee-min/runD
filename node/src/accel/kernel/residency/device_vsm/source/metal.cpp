#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_source {
namespace {

[[nodiscard]] bool rename_entry(std::string &source) {
  const std::string function = "rund_compute_map_";
  const std::size_t function_at = source.find(function);
  if (function_at == std::string::npos ||
      source.find(function, function_at + function.size()) !=
          std::string::npos) {
    return false;
  }
  const std::size_t name_end = source.find("(\n", function_at);
  if (name_end == std::string::npos) {
    return false;
  }
  source.insert(name_end, "_device_vsm");
  return true;
}

[[nodiscard]] bool install_controller(std::string &source,
                                      const std::uint64_t binding,
                                      const std::uint32_t input_count) {
  const std::string entry = "    uint gid [[thread_position_in_grid]]) {\n";
  std::string replacement =
      "    constant RundDeviceVsmConfig& rund_vsm [[buffer(" +
      std::to_string(binding) +
      ")]],\n"
      "    device atomic_uint* rund_vsm_result [[buffer(" +
      std::to_string(binding + 1u) +
      ")]],\n"
      "    device atomic_uint* rund_vsm_ring_state [[buffer(" +
      std::to_string(binding + 2u) +
      ")]],\n"
      "    device uint* rund_vsm_ring_scratch [[buffer(" +
      std::to_string(binding + 3u) + ")]],\n";
  for (std::uint32_t input = 0u; input < input_count; ++input) {
    replacement += "    device uint* rund_vsm_input_scratch_" +
                   std::to_string(input) + " [[buffer(" +
                   std::to_string(binding + 4u + input) + ")]],\n";
  }
  for (std::uint32_t input = 0u; input < input_count; ++input) {
    replacement += "    const device uint* rund_vsm_input_alias_" +
                   std::to_string(input) + " [[buffer(" +
                   std::to_string(binding + 4u + input_count + input) +
                   ")]],\n";
  }
  replacement +=
      "    device uint* rund_vsm_output_alias [[buffer(" +
      std::to_string(binding + 4u + 2u * input_count) +
      ")]],\n"
      "    uint rund_local [[thread_index_in_threadgroup]],\n"
      "    uint rund_worker [[threadgroup_position_in_grid]],\n"
      "    uint rund_group_width [[threads_per_threadgroup]]) {\n"
      "  for (uint rund_page = rund_worker; rund_page < "
      "rund_vsm.page_count; rund_page += rund_vsm.width) {\n"
      "    const ulong rund_page_begin = ulong(rund_page) * "
      "ulong(rund_vsm.payload_elements);\n"
      "    const ulong rund_remaining = rund_vsm.logical_elements - "
      "rund_page_begin;\n"
      "    const uint rund_page_elements = uint(min("
      "rund_remaining, ulong(rund_vsm.payload_elements)));\n"
      "    const uint rund_slot = rund_page % rund_vsm.width;\n"
      "    const uint rund_turn = rund_page / rund_vsm.width;\n"
      "    if (rund_local == 0u) {\n"
      "      const uint rund_prior_turn = atomic_fetch_add_explicit("
      "&rund_vsm_ring_state[rund_slot], 1u, memory_order_relaxed);\n"
      "      if (rund_prior_turn != rund_turn) "
      "atomic_fetch_add_explicit(&rund_vsm_result[6], 1u, "
      "memory_order_relaxed);\n"
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n"
      "    const uint rund_payload_words = "
      "rund_vsm.payload_elements * rund_vsm.element_words;\n"
      "    for (uint rund_page_local = rund_local; rund_page_local < "
      "rund_page_elements; rund_page_local += rund_group_width) {\n"
      "      const uint rund_global_word = "
      "uint(rund_page_begin + ulong(rund_page_local)) * "
      "rund_vsm.element_words;\n"
      "      const uint rund_scratch_word = "
      "rund_slot * rund_payload_words + "
      "rund_page_local * rund_vsm.element_words;\n"
      "      for (uint rund_word = 0u; rund_word < "
      "rund_vsm.element_words; ++rund_word) {\n";
  for (std::uint32_t input = 0u; input < input_count; ++input) {
    replacement += "        rund_vsm_input_scratch_" + std::to_string(input) +
                   "[rund_scratch_word + rund_word] = "
                   "rund_vsm_input_alias_" +
                   std::to_string(input) + "[rund_global_word + rund_word];\n";
  }
  replacement +=
      "      }\n"
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n"
      "    for (uint rund_page_local = rund_local; rund_page_local < "
      "rund_page_elements; rund_page_local += rund_group_width) {\n"
      "      const ulong gid = ulong(rund_slot) * "
      "ulong(rund_vsm.payload_elements) + ulong(rund_page_local);\n";
  return replace_one(source, entry, replacement);
}

[[nodiscard]] bool close_controller(std::string &source,
                                    const std::uint32_t input_count) {
  const std::size_t end = source.rfind("}\n");
  if (end == std::string::npos) {
    return false;
  }
  std::string replacement =
      "    }\n"
      "    threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n"
      "    for (uint rund_page_local = rund_local; "
      "rund_page_local < rund_page_elements; "
      "rund_page_local += rund_group_width) {\n"
      "      const uint rund_global_word = "
      "uint(rund_page_begin + ulong(rund_page_local)) * "
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
      "    threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n"
      "    if (rund_local == 0u) {\n"
      "      atomic_fetch_add_explicit(&rund_vsm_result[0], 1u, "
      "memory_order_relaxed);\n"
      "      atomic_fetch_add_explicit(&rund_vsm_result[1], 1u, "
      "memory_order_relaxed);\n"
      "      atomic_fetch_add_explicit(&rund_vsm_result[2], 1u, "
      "memory_order_relaxed);\n"
      "      atomic_fetch_add_explicit(&rund_vsm_result[3], 1u, "
      "memory_order_relaxed);\n"
      "      atomic_fetch_add_explicit(&rund_vsm_result[4], 1u, "
      "memory_order_relaxed);\n"
      "      atomic_fetch_add_explicit(&rund_vsm_result[5], 1u, "
      "memory_order_relaxed);\n"
      "      if (rund_page >= rund_vsm.width) {\n"
      "        atomic_fetch_add_explicit(&rund_vsm_result[10], "
      "1u, memory_order_relaxed);\n"
      "      }\n"
      "      const uint rund_schedule = 3u * (rund_page + 1u) + "
      "5u * (rund_slot + 1u) + 7u * (rund_turn + 1u) + "
      "11u * rund_page_elements;\n"
      "      atomic_fetch_add_explicit(&rund_vsm_result[11], "
      "rund_schedule, memory_order_relaxed);\n"
      "      atomic_fetch_add_explicit(&rund_vsm_result[12], ";
  replacement += std::to_string(input_count + 1u);
  replacement += "u, memory_order_relaxed);\n"
                 "    }\n"
                 "    threadgroup_barrier(mem_flags::mem_device | "
                 "mem_flags::mem_threadgroup);\n"
                 "  }\n"
                 "}\n";
  source.replace(end, 2u, replacement);

  const std::string declarations = std::string{DeviceVariant} +
                                   "\nstruct RundDeviceVsmConfig {\n"
                                   "  ulong logical_elements;\n"
                                   "  uint payload_elements;\n"
                                   "  uint page_count;\n"
                                   "  uint width;\n"
                                   "  uint element_words;\n"
                                   "};";
  return replace_one(source, DeviceVariant, declarations);
}

} // namespace

bool transform_metal(std::string &source, const std::uint64_t binding,
                     const std::uint32_t input_count) {
  return rename_entry(source) &&
         install_controller(source, binding, input_count) &&
         close_controller(source, input_count);
}

} // namespace rund::node::accel::detail::device_vsm_source

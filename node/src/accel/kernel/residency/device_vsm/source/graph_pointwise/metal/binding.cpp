#include "internal.hpp"

#include <string_view>

namespace rund::node::accel::detail::device_vsm_graph_pointwise::metal {

bool append_map_binding(std::string &source, const bool active) {
  if (!active) {
    return true;
  }
  const std::string old = "constant uchar* rund_params [[buffer(0)]],\n";
  const std::string replacement =
      "const device uint* graph_page_map [[buffer(0)]],\n";
  const std::size_t at = source.find(old);
  if (at == std::string::npos) {
    return false;
  }
  source.replace(at, old.size(), replacement);
  return true;
}

bool append_ring_bindings(std::string &source, const std::size_t data_buffers,
                          const std::size_t input_count) {
  constexpr std::string_view Local =
      "    uint local [[thread_index_in_threadgroup]]) {\n";
  const std::size_t at = source.find(Local);
  if (at == std::string::npos ||
      source.find(Local, at + Local.size()) != std::string::npos) {
    return false;
  }
  const std::size_t config = data_buffers + 1u;
  std::string replacement = "    device atomic_uint* ring_state [[buffer(" +
                            std::to_string(config + 2u) +
                            ")]],\n"
                            "    device uint* ring_scratch [[buffer(" +
                            std::to_string(config + 3u) + ")]],\n";
  for (std::size_t input = 0u; input < input_count; ++input) {
    replacement += "    device uint* input_scratch_" + std::to_string(input) +
                   " [[buffer(" + std::to_string(config + 4u + input) +
                   ")]],\n";
  }
  for (std::size_t input = 0u; input < input_count; ++input) {
    replacement += "    const device uint* input_alias_" +
                   std::to_string(input) + " [[buffer(" +
                   std::to_string(config + 4u + input_count + input) + ")]],\n";
  }
  replacement += "    device uint* output_alias [[buffer(" +
                 std::to_string(config + 4u + 2u * input_count) +
                 ")]],\n"
                 "    uint local [[thread_index_in_threadgroup]]) {\n";
  source.replace(at, Local.size(), replacement);
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise::metal

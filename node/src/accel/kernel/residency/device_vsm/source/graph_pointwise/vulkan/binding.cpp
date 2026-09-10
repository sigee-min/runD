#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_pointwise::vulkan {

bool install_map_binding(std::string &source, const bool active) {
  if (!active) {
    return true;
  }
  const std::string old =
      "layout(set = 0, binding = 0, std430) readonly buffer RundParams {\n"
      "  uint rund_params_data[];\n};\n";
  const std::string replacement =
      "layout(set = 0, binding = 0, std430) readonly buffer "
      "RundGraphPointwisePageMap { uint graph_page_map[]; };\n";
  const std::size_t at = source.find(old);
  if (at == std::string::npos) {
    return false;
  }
  source.replace(at, old.size(), replacement);
  return true;
}

bool append_storage_bindings(std::string &source,
                             const std::size_t data_buffers,
                             const std::size_t input_count) {
  source += "layout(set = 0, binding = " + std::to_string(data_buffers + 2u) +
            ", std430) buffer RundDeviceVsmRingState { uint "
            "ring_state[]; };\n"
            "layout(set = 0, binding = " +
            std::to_string(data_buffers + 3u) +
            ", std430) buffer RundDeviceVsmRingScratch { uint "
            "ring_scratch[]; };\n";
  for (std::size_t input = 0u; input < input_count; ++input) {
    source += "layout(set = 0, binding = " +
              std::to_string(data_buffers + 4u + input) +
              ", std430) buffer RundDeviceVsmInputScratch" +
              std::to_string(input) + " { uint input_scratch_" +
              std::to_string(input) + "[]; };\n";
  }
  for (std::size_t input = 0u; input < input_count; ++input) {
    source += "layout(set = 0, binding = " +
              std::to_string(data_buffers + 4u + input_count + input) +
              ", std430) readonly buffer RundDeviceVsmInputAlias" +
              std::to_string(input) + " { uint input_alias_" +
              std::to_string(input) + "[]; };\n";
  }
  source += "layout(set = 0, binding = " +
            std::to_string(data_buffers + 4u + 2u * input_count) +
            ", std430) buffer RundDeviceVsmOutputAlias { uint "
            "output_alias[]; };\n";
  return true;
}

MapRows collect_map_rows(const DeviceVsmPageMap &page_map) {
  MapRows result{};
  result.fill(DeviceVsmPageMapRowCapacity);
  if (!device_vsm_page_map_active(page_map)) {
    return result;
  }
  const std::size_t row_count = page_map.words[3u] < DeviceVsmPageMapRowCapacity
                                    ? page_map.words[3u]
                                    : DeviceVsmPageMapRowCapacity;
  for (std::size_t row_index = 0u; row_index < row_count; ++row_index) {
    DeviceVsmPageMapRow row{};
    if (!device_vsm_page_map_load_row(page_map, row_index, row)) {
      continue;
    }
    if (row.external_slot < result.size()) {
      result[row.external_slot] = row_index;
    }
  }
  return result;
}

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise::vulkan

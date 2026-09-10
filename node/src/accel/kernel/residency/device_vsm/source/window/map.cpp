#include "internal.hpp"

#include "../typed_map/scalar.hpp"

namespace rund::node::accel::detail::device_vsm_window_source {
namespace {

[[nodiscard]] const char *map_function_name(const MapSlot slot,
                                            const std::uint32_t ordinal) {
  if (slot == MapSlot::Before) {
    return ordinal == 0u   ? "rund_window_before"
           : ordinal == 1u ? "rund_window_before_second"
                           : "rund_window_before_third";
  }
  return ordinal == 0u   ? "rund_window_after"
         : ordinal == 1u ? "rund_window_after_second"
                         : "rund_window_after_third";
}

} // namespace

void append_map_expression(std::string &source, const DeviceVsmWindowMap &map,
                           const MapSlot slot, const std::uint32_t ordinal,
                           const std::string_view value,
                           const std::string_view index) {
  if (!map.active()) {
    source.append(value);
    return;
  }
  if (map.kind == DeviceVsmWindowMapKind::CanonicalTotalU32) {
    source += map_function_name(slot, ordinal);
    source += "(uint(";
    source.append(value);
    source += "), uint(";
    source.append(index);
    source += "))";
    return;
  }
  source += "(";
  source.append(value);
  source +=
      map.kind == DeviceVsmWindowMapKind::AddWrapU32Immediate ? " + " : " * ";
  source += std::to_string(map.immediate);
  source += "u)";
}

void append_map_chain_expression(std::string &source,
                                 const DeviceVsmWindowFusion &fusion,
                                 const MapSlot slot,
                                 const std::string_view value,
                                 const std::string_view index) {
  const DeviceVsmWindowMap &first =
      slot == MapSlot::Before ? fusion.before : fusion.after;
  const DeviceVsmWindowMap &second =
      slot == MapSlot::Before ? fusion.before_second : fusion.after_second;
  const DeviceVsmWindowMap &third =
      slot == MapSlot::Before ? fusion.before_third : fusion.after_third;
  if (!second.active()) {
    append_map_expression(source, first, slot, 0u, value, index);
    return;
  }
  std::string first_value;
  append_map_expression(first_value, first, slot, 0u, value, index);
  if (!third.active()) {
    append_map_expression(source, second, slot, 1u, first_value, index);
    return;
  }
  std::string second_value;
  append_map_expression(second_value, second, slot, 1u, first_value, index);
  append_map_expression(source, third, slot, 2u, second_value, index);
}

namespace {

[[nodiscard]] bool append_metal_function(std::string &source,
                                         const DeviceVsmWindowMap &map,
                                         const DeviceVsmWindowMapSource &owner,
                                         const char *const name) {
  return map.kind != DeviceVsmWindowMapKind::CanonicalTotalU32 ||
         (owner.artifact != nullptr && owner.input != nullptr &&
          device_vsm_typed_map::append_metal_u32_scalar_function(
              source, *owner.artifact, *owner.input, name));
}

[[nodiscard]] bool append_vulkan_function(std::string &source,
                                          const DeviceVsmWindowMap &map,
                                          const DeviceVsmWindowMapSource &owner,
                                          const char *const name) {
  return map.kind != DeviceVsmWindowMapKind::CanonicalTotalU32 ||
         (owner.artifact != nullptr && owner.input != nullptr &&
          device_vsm_typed_map::append_vulkan_u32_scalar_function(
              source, *owner.artifact, *owner.input, name));
}

} // namespace

bool append_metal_map_functions(std::string &source,
                                const DeviceVsmWindowFusion &fusion,
                                const DeviceVsmWindowMapSources &owners) {
  return append_metal_function(source, fusion.before, owners.before,
                               "rund_window_before") &&
         append_metal_function(source, fusion.before_second,
                               owners.before_second,
                               "rund_window_before_second") &&
         append_metal_function(source, fusion.before_third, owners.before_third,
                               "rund_window_before_third") &&
         append_metal_function(source, fusion.after, owners.after,
                               "rund_window_after") &&
         append_metal_function(source, fusion.after_second, owners.after_second,
                               "rund_window_after_second") &&
         append_metal_function(source, fusion.after_third, owners.after_third,
                               "rund_window_after_third");
}

bool append_vulkan_map_functions(std::string &source,
                                 const DeviceVsmWindowFusion &fusion,
                                 const DeviceVsmWindowMapSources &owners) {
  return append_vulkan_function(source, fusion.before, owners.before,
                                "rund_window_before") &&
         append_vulkan_function(source, fusion.before_second,
                                owners.before_second,
                                "rund_window_before_second") &&
         append_vulkan_function(source, fusion.before_third,
                                owners.before_third,
                                "rund_window_before_third") &&
         append_vulkan_function(source, fusion.after, owners.after,
                                "rund_window_after") &&
         append_vulkan_function(source, fusion.after_second,
                                owners.after_second,
                                "rund_window_after_second") &&
         append_vulkan_function(source, fusion.after_third, owners.after_third,
                                "rund_window_after_third");
}

} // namespace rund::node::accel::detail::device_vsm_window_source

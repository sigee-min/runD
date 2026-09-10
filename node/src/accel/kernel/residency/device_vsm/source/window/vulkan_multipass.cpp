#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_window_source {
namespace {

void append_increment(std::string &source) {
  source += R"GLSL(    for (uint word = 0u; word < 6u; ++word) {
      atomicAdd(rund_vsm_result[word], 1u);
    }
    const uint rund_first_page = page == 0u ? 0u : page - 1u;
    const uint rund_last_page = min(page + 1u, rund_vsm.page_count - 1u);
    const uint rund_page_elements = end - begin;
    const uint rund_footprint =
        3u * (page + 1u) + 5u * rund_first_page + 7u * rund_last_page +
        11u * rund_page_elements;
    if (page + 1u < rund_vsm.page_count) {
      atomicAdd(rund_vsm_result[8], 1u);
    }
    atomicAdd(rund_vsm_result[9], rund_footprint);
)GLSL";
}

void append_prefix(std::string &source, const DeviceVsmWindowFusion &fusion) {
  source += R"GLSL(  uint prefix = 0u;
  for (uint index = 0u; index < params.input_count; ++index) {
    const uint rund_sample = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input_values[index]", "index");
  source += R"GLSL(;
    prefix += rund_sample;
    input_values[index] = prefix;
  }
  for (uint page = 0u; page < rund_vsm.page_count; ++page) {
    const uint begin = page * rund_vsm.payload_elements;
    const uint end = min(begin + rund_vsm.payload_elements,
                         uint(params.output_count));
    for (uint index = begin; index < end; ++index) {
      const uint64_t anchor = uint64_t(index) * params.stride;
      const uint64_t left = anchor < params.padding ? uint64_t(0)
                                                    : anchor - params.padding;
      const uint64_t right_width = params.window_size - params.padding;
      const uint64_t right =
          anchor >= params.input_count
              ? params.input_count - uint64_t(1)
              : (right_width >= params.input_count - anchor
                     ? params.input_count - uint64_t(1)
                     : anchor + right_width - uint64_t(1));
      uint value = input_values[uint(right)];
      if (left != uint64_t(0)) { value -= input_values[uint(left - 1ul)]; }
      output_values[index] = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value", "index");
  source += R"GLSL(;
    }
)GLSL";
  append_increment(source);
  source += "  }\n";
}

void append_extreme(std::string &source, const RangeExec &execution,
                    const DeviceVsmWindowFusion &fusion) {
  const char *const combine =
      execution.operation() == RangeOp::Minimum ? "min" : "max";
  source += R"GLSL(  const uint block_width = uint(params.window_size);
  for (uint block = 0u; block < params.input_count; block += block_width) {
    const uint end = min(block + block_width, uint(params.input_count));
    uint value = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input_values[end - 1u]", "end - 1u");
  source += R"GLSL(;
    output_values[end - 1u] = value;
    for (uint cursor = end - 1u; cursor > block; --cursor) {
      const uint rund_sample = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input_values[cursor - 1u]", "cursor - 1u");
  source += ";\n      value = ";
  source += combine;
  source += R"GLSL((rund_sample, value);
      output_values[cursor - 1u] = value;
    }
  }
  for (uint block = 0u; block < params.input_count; block += block_width) {
    const uint end = min(block + block_width, uint(params.input_count));
    uint value = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input_values[block]", "block");
  source += R"GLSL(;
    input_values[block] = value;
    for (uint cursor = block + 1u; cursor < end; ++cursor) {
      const uint rund_sample = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input_values[cursor]", "cursor");
  source += ";\n      value = ";
  source += combine;
  source += R"GLSL((value, rund_sample);
      input_values[cursor] = value;
    }
  }
  for (uint page_cursor = rund_vsm.page_count; page_cursor != 0u;
       --page_cursor) {
    const uint page = page_cursor - 1u;
    const uint begin = page * rund_vsm.payload_elements;
    const uint end = min(begin + rund_vsm.payload_elements,
                         uint(params.output_count));
    for (uint cursor = end; cursor > begin; --cursor) {
      const uint index = cursor - 1u;
      const uint64_t anchor = uint64_t(index) * params.stride;
      const uint64_t left = anchor < params.padding ? uint64_t(0)
                                                    : anchor - params.padding;
      const uint64_t right_width = params.window_size - params.padding;
      const uint64_t right =
          anchor >= params.input_count
              ? params.input_count - uint64_t(1)
              : (right_width >= params.input_count - anchor
                     ? params.input_count - uint64_t(1)
                     : anchor + right_width - uint64_t(1));
      uint value;
      if (left / uint64_t(block_width) == right / uint64_t(block_width)) {
        value = left % uint64_t(block_width) == uint64_t(0)
                    ? input_values[uint(right)]
                    : output_values[uint(left)];
      } else {
        value = )GLSL";
  source += combine;
  source += R"GLSL((output_values[uint(left)], input_values[uint(right)]);
      }
      output_values[index] = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value", "index");
  source += R"GLSL(;
    }
)GLSL";
  append_increment(source);
  source += "  }\n";
}

} // namespace

bool emit_vulkan_multipass(const RangeExec &execution,
                           const rund::kernel::ArtifactKey &key,
                           const DeviceVsmWindowFusion &fusion,
                           const DeviceVsmWindowMapSources &sources,
                           std::string &source) noexcept {
  try {
    source.clear();
    source += R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// artifact_variant=device_vsm
layout(local_size_x = )GLSL";
    source += std::to_string(execution.width());
    source += R"GLSL(, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t input_count;
  uint64_t output_count;
  uint64_t window_size;
  uint64_t stride;
  uint64_t padding;
  uint64_t stage_element_count;
  uint64_t stage_aux_count;
  uint stage;
  uint reserved;
} params;
layout(set = 0, binding = 1, std430) buffer Input {
  uint input_values[];
};
layout(set = 0, binding = 2, std430) buffer Output {
  uint output_values[];
};
layout(set = 0, binding = 3, std430) buffer RundDeviceVsmResult {
  uint rund_vsm_result[];
};
layout(push_constant) uniform RundDeviceVsmDispatch {
  uint logical_elements;
  uint payload_elements;
  uint page_count;
  uint width;
} rund_vsm;

 )GLSL";
    if (!append_vulkan_map_functions(source, fusion, sources)) {
      return false;
    }
    source += "// entry=";
    source += entry_name(key);
    source += R"GLSL(
void main() {
  const uint rund_local = gl_LocalInvocationID.x;
  const uint rund_worker = gl_WorkGroupID.x;
  if (rund_worker != 0u || rund_local != 0u) { return; }
)GLSL";
    if (execution.candidate() == RangePath::PrefixDifference) {
      append_prefix(source, fusion);
    } else {
      append_extreme(source, execution, fusion);
    }
    source += "}\n";
    return !entry_name(key).empty();
  } catch (...) {
    source.clear();
    return false;
  }
}

} // namespace rund::node::accel::detail::device_vsm_window_source

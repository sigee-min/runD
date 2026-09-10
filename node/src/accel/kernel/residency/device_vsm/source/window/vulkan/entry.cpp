#include "internal.hpp"

#include "../../../../../../vulkan/range/source/algebra.hpp"

namespace rund::node::accel::detail::device_vsm_window_source {

bool emit_vulkan(const RangeExec &execution,
                 const rund::kernel::ArtifactKey &key,
                 const DeviceVsmWindowFusion &fusion,
                 const DeviceVsmWindowRingPlan &ring,
                 const DeviceVsmWindowMapSources &sources,
                 std::string &source) noexcept {
  try {
    source.clear();
    if (ring.gpu_owned) {
      source += "// artifact_variant=device_vsm_window_ring\n";
      return vulkan::emit_ring(execution, key, fusion, ring, sources, source);
    }
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
)GLSL";
    const char *const scalar =
        VulkanRangeScalar(execution.wide_elements(), execution.signed_values());
    source +=
        "layout(set = 0, binding = 1, std430) readonly buffer Input {\n  ";
    source += scalar;
    source += " input_values[];\n};\n";
    source += "layout(set = 0, binding = 2, std430) buffer Output {\n  ";
    source += scalar;
    source += R"GLSL( output_values[];
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
    if (execution.saturating_sum() &&
        !vulkan::append_saturating_algebra(source)) {
      return false;
    }
    if (execution.uses_shared_halo()) {
      source += "shared ";
      source += scalar;
      source += " range_tile[";
      source += std::to_string(execution.shared_element_capacity());
      source += "];\n";
    }
    source += "// entry=";
    source += entry_name(key);
    source += R"GLSL(
void main() {
  const uint rund_local = gl_LocalInvocationID.x;
  const uint rund_worker = gl_WorkGroupID.x;
  for (uint rund_page = rund_worker; rund_page < rund_vsm.page_count;
       rund_page += rund_vsm.width) {
    const uint rund_page_begin = rund_page * rund_vsm.payload_elements;
    const uint rund_remaining = uint(params.output_count) - rund_page_begin;
    const uint rund_page_elements =
        min(rund_remaining, rund_vsm.payload_elements);
)GLSL";
    if (execution.uses_shared_halo()) {
      vulkan::append_shared(source, execution, scalar, fusion);
    } else {
      vulkan::append_direct(source, execution, scalar, fusion);
    }
    vulkan::append_counters(source);
    return true;
  } catch (...) {
    source.clear();
    return false;
  }
}

} // namespace rund::node::accel::detail::device_vsm_window_source

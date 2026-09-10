#include "internal.hpp"

#include "../../../../../../vulkan/range/source/algebra.hpp"

namespace rund::node::accel::detail::device_vsm_window_source::vulkan {

[[nodiscard]] bool emit_ring(
    const RangeExec &execution, const rund::kernel::ArtifactKey &key,
    const DeviceVsmWindowFusion &fusion,
    const DeviceVsmWindowRingPlan &plan,
    const DeviceVsmWindowMapSources &sources, std::string &source) {
  const bool signed_i32 =
      !execution.wide_elements() &&
      execution.domain() == rund::kernel::ComputeDomain::I32;
  const char *const scalar =
      signed_i32 ? "int"
                 : VulkanRangeScalar(execution.wide_elements(),
                                     execution.signed_values());
  const char *const identity =
      execution.operation() == RangeOp::Minimum
          ? (signed_i32 ? "2147483647" : "0xffffffffu")
      : execution.operation() == RangeOp::Maximum
          ? (signed_i32 ? "(-2147483647 - 1)" : "0u")
          : "0";
  source += R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
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
  source += "layout(set = 0, binding = 1, std430) readonly buffer Input { ";
  source += scalar;
  source += " input_values[]; };\n";
  source += "layout(set = 0, binding = 2, std430) buffer Output { ";
  source += scalar;
  source += " output_values[]; };\n";
  source += R"GLSL(layout(set = 0, binding = 3, std430) buffer Result {
  uint result_values[];
};
layout(set = 0, binding = 4, std430) buffer RingState {
  uint state_values[];
};
layout(set = 0, binding = 5, std430) buffer ScratchA { )GLSL";
  source += scalar;
  source += R"GLSL( scratch_a[]; };
layout(set = 0, binding = 6, std430) buffer ScratchB { )GLSL";
  source += scalar;
  source += R"GLSL( scratch_b[]; };
layout(push_constant) uniform RundDeviceVsmDispatch {
  uint logical_elements;
  uint payload_elements;
  uint page_count;
  uint width;
  uint element_words;
  uint epoch;
  uint slot;
  uint phase;
  uint frame_elements;
  uint halo_elements;
} rund_vsm;

)GLSL";
  if (!append_vulkan_map_functions(source, fusion, sources)) {
    return false;
  }
  if (execution.saturating_sum()) {
    if (!append_saturating_algebra(source)) {
      return false;
    }
  }
  if (signed_i32 && execution.operation() == RangeOp::Sum &&
      !execution.saturating_sum()) {
    source += R"GLSL(
int rund_i32_sum(const int left, const int right) {
  const uint left_bits =
      left < 0 ? 0xffffffffu - uint(-(left + 1)) : uint(left);
  const uint right_bits =
      right < 0 ? 0xffffffffu - uint(-(right + 1)) : uint(right);
  const uint bits = left_bits + right_bits;
  const uint magnitude_bits = bits & 0x7fffffffu;
  const int magnitude = int(magnitude_bits);
  return (bits & 0x80000000u) != 0u
             ? (-2147483647 - 1) + magnitude
             : magnitude;
}

)GLSL";
  }
  source += "// entry=";
  source += entry_name(key);
  source += R"GLSL(
void main() {
  const uint local = gl_LocalInvocationID.x;
  const uint group_width = gl_WorkGroupSize.x;
  const uint64_t logical = uint64_t(rund_vsm.logical_elements);
  for (uint epoch = 0u; epoch < rund_vsm.page_count; ++epoch) {
    const uint slot = epoch & 1u;
    const uint previous = slot ^ 1u;
    const uint64_t page_begin = uint64_t(epoch) *
                                uint64_t(rund_vsm.payload_elements);
    const uint count = page_begin < logical
                           ? uint(min(uint64_t(rund_vsm.payload_elements),
                                      logical - page_begin))
                           : 0u;
    for (uint frame_local = local; frame_local < rund_vsm.frame_elements;
         frame_local += group_width) {
      const uint64_t raw = page_begin + uint64_t(frame_local);
      uint64_t index = raw < uint64_t(rund_vsm.halo_elements)
                           ? uint64_t(0)
                           : raw - uint64_t(rund_vsm.halo_elements);
      if (index >= logical) {
        index = logical == uint64_t(0) ? uint64_t(0) : logical - uint64_t(1);
      }
      )GLSL";
  source += scalar;
  source += " value = ";
  source += identity;
  source += R"GLSL(;
      if (epoch != 0u && frame_local < rund_vsm.halo_elements) {
        const uint tail = uint(uint64_t(rund_vsm.payload_elements) +
                               uint64_t(frame_local));
        value = previous == 0u ? scratch_a[tail] : scratch_b[tail];
      } else {
        value = input_values[uint(index)];
        value = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::Before, "value",
                              "index");
  source += R"GLSL(;
      }
      if (slot == 0u) {
        scratch_a[frame_local] = value;
      } else {
        scratch_b[frame_local] = value;
      }
    }
    memoryBarrierBuffer();
    barrier();
    if (local == 0u) {
      state_values[slot] = epoch + 1u;
    }
    memoryBarrierBuffer();
    barrier();
    if (state_values[slot] != epoch + 1u) {
      atomicOr(result_values[7], epoch);
    }
    for (uint page_local = local; page_local < count;
         page_local += group_width) {
      const uint center = rund_vsm.halo_elements + page_local;
      )GLSL";
  source += scalar;
  source += " value = ";
  source += identity;
  source += R"GLSL(;
      bool seeded = false;
      for (uint64_t offset = uint64_t(0); offset < params.window_size; ++offset) {
        const uint index = uint(uint64_t(center) + offset -
                                uint64_t(rund_vsm.halo_elements));
        const )GLSL";
  source += scalar;
  source += R"GLSL( item = slot == 0u ? scratch_a[index] : scratch_b[index];
        if (!seeded) { value = item; seeded = true; } else {
)GLSL";
  if (execution.operation() == RangeOp::Minimum) {
    source += "          value = min(value, item);\n";
  } else if (execution.operation() == RangeOp::Maximum) {
    source += "          value = max(value, item);\n";
  } else if (execution.saturating_sum()) {
    append_update(source, execution);
  } else if (signed_i32) {
    source += "          value = rund_i32_sum(value, item);\n";
  } else {
    source += "          value += item;\n";
  }
  source += R"GLSL(      }
    }
    output_values[uint(page_begin + uint64_t(page_local))] = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value",
                              "page_begin + uint64_t(page_local)");
  source += R"GLSL(;
    }
    memoryBarrierBuffer();
    barrier();
    if (local == 0u) {
      atomicAdd(result_values[0], 1u);
      atomicAdd(result_values[1], 1u);
      atomicAdd(result_values[2], 1u);
      atomicAdd(result_values[3], 1u);
      atomicAdd(result_values[4], 1u);
      atomicAdd(result_values[5], 1u);
    }
    memoryBarrierBuffer();
    barrier();
  }
  if (local == 0u) {
    result_values[10] = )GLSL";
  source += std::to_string(plan.page_count);
  source += R"GLSL(u;
    result_values[11] = )GLSL";
  source += std::to_string(plan.schedule_checksum);
  source += R"GLSL(u;
    result_values[12] = )GLSL";
  source += std::to_string(plan.page_count);
  source += R"GLSL(u;
  }
  memoryBarrierBuffer();
  barrier();
}
)GLSL";
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_window_source::vulkan

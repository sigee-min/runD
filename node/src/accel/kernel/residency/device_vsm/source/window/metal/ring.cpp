#include "internal.hpp"

#include "../../../../../../metal/range/source/algebra.hpp"

namespace rund::node::accel::detail::device_vsm_window_source::metal {

[[nodiscard]] bool emit_ring(
    const RangeExec &execution, const rund::kernel::ArtifactKey &key,
    const DeviceVsmWindowFusion &fusion,
    const DeviceVsmWindowRingPlan &plan,
    const DeviceVsmWindowMapSources &sources, std::string &source) {
  const bool signed_i32 =
      !execution.wide_elements() &&
      execution.domain() == rund::kernel::ComputeDomain::I32;
  const char *const type = signed_i32 ? "int" : scalar_type(execution);
  const char *const identity =
      execution.operation() == RangeOp::Minimum
          ? (signed_i32 ? "2147483647" : "0xffffffffu")
      : execution.operation() == RangeOp::Maximum
          ? (signed_i32 ? "(-2147483647 - 1)" : "0u")
          : "0";
  source += R"MSL(#include <metal_stdlib>
using namespace metal;

struct RangeParams {
  ulong input_count;
  ulong output_count;
  ulong window_size;
  ulong stride;
  ulong padding;
  ulong stage_element_count;
  ulong stage_aux_count;
  uint stage;
  uint reserved;
};
struct RingConfig {
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
  };
)MSL";
  if (!append_metal_map_functions(source, fusion, sources)) {
    return false;
  }
  if (execution.saturating_sum()) {
    AppendMetalRangeSaturatingAlgebra(source);
  }
  source += "kernel void ";
  source += entry_name(key);
  source += "(\n";
  source += "    constant RangeParams& params [[buffer(0)]],\n    device const ";
  source += type;
  source += R"MSL(* input [[buffer(1)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(2)]],
    constant RingConfig& config [[buffer(3)]],
    device atomic_uint* result [[buffer(4)]],
    device atomic_uint* state [[buffer(5)]],
    device )MSL";
  source += type;
  source += R"MSL(* scratch_a [[buffer(6)]],
    device )MSL";
  source += type;
  source += R"MSL(* scratch_b [[buffer(7)]],
    uint local [[thread_index_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]) {
  const ulong logical = ulong(config.logical_elements);
  for (uint epoch = 0u; epoch < config.page_count; ++epoch) {
    const uint slot = epoch & 1u;
    const uint previous = slot ^ 1u;
    const ulong page_begin = ulong(epoch) * ulong(config.payload_elements);
    const uint count = page_begin < logical
                           ? uint(min(ulong(config.payload_elements),
                                      logical - page_begin))
                           : 0u;
    for (uint frame_local = local; frame_local < config.frame_elements;
         frame_local += group_width) {
      const ulong raw = page_begin + ulong(frame_local);
      ulong index = raw < ulong(config.halo_elements)
                        ? 0ul
                        : raw - ulong(config.halo_elements);
      if (index >= logical) {
        index = logical == 0ul ? 0ul : logical - 1ul;
      }
      )MSL";
  source += type;
  source += " value = ";
  source += identity;
  source += R"MSL(;
      if (epoch != 0u && frame_local < config.halo_elements) {
        const ulong tail = ulong(config.payload_elements) +
                           ulong(frame_local);
        value = previous == 0u ? scratch_a[uint(tail)] : scratch_b[uint(tail)];
      } else {
        value = input[uint(index)];
        value = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "value", "index");
  source += R"MSL(;
      }
      if (slot == 0u) { scratch_a[frame_local] = value; }
      else { scratch_b[frame_local] = value; }
    }
    threadgroup_barrier(mem_flags::mem_device |
                        mem_flags::mem_threadgroup);
    if (local == 0u) {
      atomic_store_explicit(&state[slot], epoch + 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device |
                        mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&state[slot], memory_order_relaxed) != epoch + 1u) {
      atomic_fetch_or_explicit(&result[7], epoch, memory_order_relaxed);
    }
    for (uint page_local = local; page_local < count;
         page_local += group_width) {
      const uint center = config.halo_elements + page_local;
      )MSL";
  source += type;
  source += " value = ";
  source += identity;
  source += R"MSL(;
      bool seeded = false;
      for (ulong offset = 0ul; offset < params.window_size; ++offset) {
        const uint sample_index = uint(ulong(center) + offset -
                                       ulong(config.halo_elements));
        const )MSL";
  source += type;
  source += R"MSL( sample = slot == 0u ? scratch_a[sample_index]
                                     : scratch_b[sample_index];
        if (!seeded) { value = sample; seeded = true; } else {
)MSL";
  if (execution.operation() == RangeOp::Minimum) {
    source += "          value = min(value, sample);\n";
  } else if (execution.operation() == RangeOp::Maximum) {
    source += "          value = max(value, sample);\n";
  } else if (execution.saturating_sum()) {
    source += MetalRangeUpdate(execution.operation(), true);
  } else if (signed_i32) {
    source += "          value = as_type<int>(as_type<uint>(value) + "
              "as_type<uint>(sample));\n";
  } else {
    source += "          value += sample;\n";
  }
  source += R"MSL(        }
      }
      output[uint(page_begin + ulong(page_local))] = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value",
                              "page_begin + ulong(page_local)");
  source += R"MSL(;
    }
    threadgroup_barrier(mem_flags::mem_device |
                        mem_flags::mem_threadgroup);
    if (local == 0u) {
      atomic_fetch_add_explicit(&result[0], 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(&result[1], 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(&result[2], 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(&result[3], 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(&result[4], 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(&result[5], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device |
                        mem_flags::mem_threadgroup);
  }
  if (local == 0u) {
    atomic_store_explicit(&result[10], )MSL";
  source += std::to_string(plan.page_count);
  source += R"MSL(u, memory_order_relaxed);
    atomic_store_explicit(&result[11], )MSL";
  source += std::to_string(plan.schedule_checksum);
  source += R"MSL(u, memory_order_relaxed);
    atomic_store_explicit(&result[12], )MSL";
  source += std::to_string(plan.page_count);
  source += R"MSL(u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device |
                      mem_flags::mem_threadgroup);
}
)MSL";
  return !entry_name(key).empty();
}

} // namespace rund::node::accel::detail::device_vsm_window_source::metal

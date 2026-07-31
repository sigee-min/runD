#include "model.hpp"

#include "../../../scatter/reduce/model.hpp"

#include <cstdint>
#include <string>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

std::string MetalScatterReduceKey(const rund::kernel::ScatterReducePlan &plan) {
  return "scatter_reduce." + std::to_string(static_cast<unsigned>(plan.op)) +
         "." + std::to_string(static_cast<unsigned>(plan.domain)) + "." +
         std::to_string(plan.element_bytes) + "." +
         std::to_string(static_cast<unsigned>(plan.fixed_format.overflow));
}

std::string
MetalScatterReduceSource(const rund::kernel::ScatterReducePlan &plan) {
  const bool wide = plan.element_bytes == 8u;
  const bool signed_value = plan.domain == rund::kernel::ComputeDomain::I32 ||
                            plan.domain == rund::kernel::ComputeDomain::I64 ||
                            plan.domain == rund::kernel::ComputeDomain::Fixed;
  const bool fixed_saturate =
      plan.domain == rund::kernel::ComputeDomain::Fixed &&
      plan.fixed_format.overflow == rund::kernel::ComputeOverflow::Saturate;
  const bool parallel_fold = rund::kernel::ScatterReduceFoldParallel(plan);
  const std::string bits = wide ? "ulong" : "uint";
  const std::string signed_bits = wide ? "long" : "int";
  const std::string max_bits = wide ? "0x7ffffffffffffffful" : "0x7fffffffu";
  const std::string min_bits = wide ? "0x8000000000000000ul" : "0x80000000u";
  const std::string all_bits = wide ? "0xfffffffffffffffful" : "0xffffffffu";

  std::string source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ScatterReduceParams {
  ulong element_count;
  ulong output_count;
  uint count_source;
  uint reserved;
  uint value_base;
  uint index_base;
  uint count_base;
  uint output_base;
};

inline ulong rund_scatter_reduce_count(device const uint* words,
                                       constant ScatterReduceParams& params) {
  if (params.count_source == 0u) { return params.element_count; }
  if (params.count_source == 1u) { return ulong(words[0]); }
  return ulong(words[0]) | (ulong(words[1]) << 32u);
}

kernel void rund_scatter_reduce_control(
    device const uint* indices [[buffer(1)]],
    device const uint* count_words [[buffer(2)]],
    device atomic_uint* status [[buffer(4)]],
    device uint* indirect [[buffer(5)]],
    constant ScatterReduceParams& params [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]) {
  const ulong logical = rund_scatter_reduce_count(count_words, params);
  uint local_invalid = 0xffffffffu;
  if (logical <= params.element_count) {
    for (ulong ordinal = ulong(tid); ordinal < logical; ordinal += 256u) {
      if (ulong(indices[ordinal]) >= params.output_count) {
        local_invalid = min(local_invalid, uint(ordinal));
      }
    }
  }
  threadgroup uint invalids[256];
  invalids[tid] = local_invalid;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (tid < stride) {
      invalids[tid] = min(invalids[tid], invalids[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid != 0u) { return; }
  atomic_store_explicit(&status[0], 0u, memory_order_relaxed);
  atomic_store_explicit(&status[1], uint(min(logical, 0xfffffffful)),
                        memory_order_relaxed);
  atomic_store_explicit(&status[2], 0u, memory_order_relaxed);
  atomic_store_explicit(&status[3], 0u, memory_order_relaxed);
  indirect[0] = 0u;
  indirect[1] = 1u;
  indirect[2] = 1u;
  indirect[3] = 0u;
  indirect[4] = 1u;
  indirect[5] = 1u;
  if (logical > params.element_count) {
    atomic_store_explicit(&status[0], 1u, memory_order_relaxed);
    atomic_store_explicit(&status[1], uint(params.element_count),
                          memory_order_relaxed);
    return;
  }
  if (invalids[0] != 0xffffffffu) {
    atomic_store_explicit(&status[0], 2u, memory_order_relaxed);
    atomic_store_explicit(&status[1], invalids[0], memory_order_relaxed);
    return;
  }
  indirect[0] = uint((params.output_count + 255u) / 256u);
)MSL";
  source += parallel_fold ? "  indirect[3] = uint((logical + 255u) / 256u);\n"
                          : "  indirect[3] = 1u;\n";
  source += "}\n";

  source += "inline " + bits + " rund_scatter_reduce_fold(" + bits + " lhs, " +
            bits + " rhs) {\n";
  if (plan.op == rund::kernel::ScatterReduceOp::Sum) {
    if (fixed_saturate) {
      source += "  const " + signed_bits + " a = as_type<" + signed_bits +
                ">(lhs);\n";
      source += "  const " + signed_bits + " b = as_type<" + signed_bits +
                ">(rhs);\n";
      source += "  if (b > 0 && a > as_type<" + signed_bits + ">(" + max_bits +
                ") - b) { return " + max_bits + "; }\n";
      source += "  if (b < 0 && a < as_type<" + signed_bits + ">(" + min_bits +
                ") - b) { return " + min_bits + "; }\n";
      source += "  return lhs + rhs;\n";
    } else {
      source += "  return lhs + rhs;\n";
    }
  } else if (plan.op == rund::kernel::ScatterReduceOp::Min) {
    source += signed_value
                  ? "  return as_type<" + signed_bits + ">(rhs) < as_type<" +
                        signed_bits + ">(lhs) ? rhs : lhs;\n"
                  : "  return rhs < lhs ? rhs : lhs;\n";
  } else {
    source += signed_value
                  ? "  return as_type<" + signed_bits + ">(rhs) > as_type<" +
                        signed_bits + ">(lhs) ? rhs : lhs;\n"
                  : "  return rhs > lhs ? rhs : lhs;\n";
  }
  source += "}\n\n";

  std::string identity = wide ? "0ul" : "0u";
  if (plan.op == rund::kernel::ScatterReduceOp::Min) {
    identity = signed_value ? max_bits : all_bits;
  } else if (plan.op == rund::kernel::ScatterReduceOp::Max && signed_value) {
    identity = min_bits;
  }
  source += "kernel void rund_scatter_reduce_initialize(\n";
  source += "    device " + bits + "* output [[buffer(3)]],\n";
  source += R"MSL(    device uint* counts [[buffer(7)]],
    device const uint* indirect [[buffer(5)]],
    constant ScatterReduceParams& params [[buffer(6)]],
    uint target [[thread_position_in_grid]]) {
  if (indirect[0] == 0u || ulong(target) >= params.output_count) { return; }
)MSL";
  source += "  output[target] = " + identity + ";\n";
  source += R"MSL(  counts[target] = 0u;
}

)MSL";
  source += "kernel void rund_scatter_reduce_fold_sources(\n";
  source += "    device const " + bits + "* values [[buffer(0)]],\n";
  source += R"MSL(    device const uint* indices [[buffer(1)]],
    device const uint* count_words [[buffer(2)]],
)MSL";
  source += "    device " + bits + "* output [[buffer(3)]],\n";
  source += R"MSL(    device atomic_uint* status [[buffer(4)]],
    device const uint* indirect [[buffer(5)]],
    constant ScatterReduceParams& params [[buffer(6)]],
    device uint* counts [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong logical = rund_scatter_reduce_count(count_words, params);
  if (indirect[3] == 0u) { return; }
)MSL";
  if (parallel_fold) {
    source += R"MSL(  const ulong ordinal = ulong(gid);
  if (ordinal >= logical) { return; }
  const uint target = indices[ordinal];
  device atomic_uint* contributor_counts =
      reinterpret_cast<device atomic_uint*>(counts);
  const uint prior = atomic_fetch_add_explicit(
      &contributor_counts[target], 1u, memory_order_relaxed);
  if (prior != 0u) {
    atomic_fetch_add_explicit(&status[2], 1u, memory_order_relaxed);
  }
  device atomic_uint* atomic_output =
      reinterpret_cast<device atomic_uint*>(output);
)MSL";
    if (plan.op == rund::kernel::ScatterReduceOp::Sum) {
      source += R"MSL(  atomic_fetch_add_explicit(
      &atomic_output[target], uint(values[ordinal]), memory_order_relaxed);
)MSL";
    } else if (plan.op == rund::kernel::ScatterReduceOp::Min) {
      source += signed_value ? R"MSL(  device atomic_int* signed_atomic_output =
      reinterpret_cast<device atomic_int*>(output);
  atomic_fetch_min_explicit(
      &signed_atomic_output[target], as_type<int>(uint(values[ordinal])),
      memory_order_relaxed);
)MSL"
                             : R"MSL(  atomic_fetch_min_explicit(
      &atomic_output[target], uint(values[ordinal]), memory_order_relaxed);
)MSL";
    } else {
      source += signed_value ? R"MSL(  device atomic_int* signed_atomic_output =
      reinterpret_cast<device atomic_int*>(output);
  atomic_fetch_max_explicit(
      &signed_atomic_output[target], as_type<int>(uint(values[ordinal])),
      memory_order_relaxed);
)MSL"
                             : R"MSL(  atomic_fetch_max_explicit(
      &atomic_output[target], uint(values[ordinal]), memory_order_relaxed);
)MSL";
    }
  } else {
    source += R"MSL(  if (gid != 0u) { return; }
  for (ulong ordinal = 0u; ordinal < logical; ++ordinal) {
    const uint target = indices[ordinal];
    if (counts[target] != 0u) {
      atomic_fetch_add_explicit(&status[2], 1u, memory_order_relaxed);
    }
    ++counts[target];
    output[target] = rund_scatter_reduce_fold(output[target], values[ordinal]);
  }
)MSL";
  }
  source += "}\n";
  return source;
}

#endif

} // namespace rund::node::accel::detail

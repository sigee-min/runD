#include "model.hpp"
#include "source/parallel.hpp"

#include "../../../kernel/backend/source/storage.hpp"
#include "../../../scatter/reduce/model.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

struct ScatterReduceSourceSemantics final {
  bool wide{};
  bool signed_comparison{};
  bool saturating_sum{};
  bool parallel_fold{};
};

[[nodiscard]] ScatterReduceSourceSemantics
ScatterReduceSemantics(const rund::kernel::ScatterReducePlan &plan) noexcept {
  const bool sum = plan.op == rund::kernel::ScatterReduceOp::Sum;
  const bool signed_value = plan.domain == rund::kernel::ComputeDomain::I32 ||
                            plan.domain == rund::kernel::ComputeDomain::I64 ||
                            plan.domain == rund::kernel::ComputeDomain::Fixed;
  return ScatterReduceSourceSemantics{
      .wide = plan.element_bytes == 8u,
      .signed_comparison = !sum && signed_value,
      .saturating_sum =
          sum && plan.domain == rund::kernel::ComputeDomain::Fixed &&
          plan.fixed_format.overflow == rund::kernel::ComputeOverflow::Saturate,
      .parallel_fold = rund::kernel::ScatterReduceFoldParallel(plan),
  };
}

} // namespace

std::string MetalScatterReduceKey(const rund::kernel::ScatterReducePlan &plan) {
  const ScatterReduceSourceSemantics semantics = ScatterReduceSemantics(plan);
  std::string key = "scatter_reduce.";
  key += std::to_string(static_cast<unsigned>(plan.op));
  key += semantics.wide ? ".64" : ".32";
  if (plan.op == rund::kernel::ScatterReduceOp::Sum) {
    key += semantics.saturating_sum ? ".saturate" : ".modular";
  } else {
    key += semantics.signed_comparison ? ".signed" : ".unsigned";
  }
  key += semantics.parallel_fold ? ".parallel" : ".serial";
  return key;
}

template <typename Sink>
[[nodiscard]] bool EmitMetalScatterReduceSource(
    Sink &source, const rund::kernel::ScatterReducePlan
                      &plan) noexcept(noexcept(source += std::string_view{})) {
  const ScatterReduceSourceSemantics semantics = ScatterReduceSemantics(plan);
  const bool wide = semantics.wide;
  const bool signed_value = semantics.signed_comparison;
  const bool fixed_saturate = semantics.saturating_sum;
  const bool parallel_fold = semantics.parallel_fold;
  const std::string_view bits = wide ? "ulong" : "uint";
  const std::string_view signed_bits = wide ? "long" : "int";
  const std::string_view max_bits =
      wide ? "0x7ffffffffffffffful" : "0x7fffffffu";
  const std::string_view min_bits =
      wide ? "0x8000000000000000ul" : "0x80000000u";
  const std::string_view all_bits =
      wide ? "0xfffffffffffffffful" : "0xffffffffu";

  source += R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ScatterReduceParams {
  ulong element_count;
  ulong output_count;
  uint count_source;
  uint validation_groups;
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
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 grid [[threadgroups_per_grid]]) {
  const ulong logical = rund_scatter_reduce_count(count_words, params);
  uint local_invalid = 0xffffffffu;
  const bool chunks = grid.x > 1u;
  if (logical <= params.element_count) {
    if (!chunks && params.validation_groups > 1u) {
      for (uint partial = tid; partial < params.validation_groups;
           partial += 256u) {
        local_invalid = min(local_invalid, atomic_load_explicit(&status[4u + partial], memory_order_relaxed));
      }
    } else {
      const ulong stride = ulong(grid.x) * 256u;
      for (ulong ordinal = ulong(group.x) * 256u + tid; ordinal < logical;
           ordinal += stride) {
        if (ulong(indices[ordinal]) >= params.output_count) {
          local_invalid = min(local_invalid, uint(ordinal));
        }
      }
    }
  }
  threadgroup atomic_uint first_invalid;
  if (tid == 0u) {
    atomic_store_explicit(&first_invalid, 0xffffffffu, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint group_invalid = simd_min(local_invalid);
  if (simd_prefix_exclusive_sum(1u) == 0u && group_invalid != 0xffffffffu) {
    atomic_fetch_min_explicit(&first_invalid, group_invalid, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0u) { return; }
  const uint invalid = atomic_load_explicit(&first_invalid, memory_order_relaxed);
  if (chunks) {
    atomic_store_explicit(&status[4u + group.x], invalid, memory_order_relaxed);
    return;
  }
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
  if (invalid != 0xffffffffu) {
    atomic_store_explicit(&status[0], 2u, memory_order_relaxed);
    atomic_store_explicit(&status[1], invalid, memory_order_relaxed);
    return;
  }
  indirect[0] = uint((params.output_count + 255u) / 256u);
)MSL";
  source += parallel_fold ? "  indirect[3] = uint((logical + 255u) / 256u);\n"
                          : "  indirect[3] = 1u;\n";
  source += "}\n";

  if (!parallel_fold) {
    source += "inline ";
    source += bits;
    source += " rund_scatter_reduce_fold(";
    source += bits;
    source += " lhs, ";
    source += bits;
    source += " rhs) {\n";
    if (plan.op == rund::kernel::ScatterReduceOp::Sum) {
      if (fixed_saturate) {
        source += "  const ";
        source += signed_bits;
        source += " a = as_type<";
        source += signed_bits;
        source += ">(lhs);\n  const ";
        source += signed_bits;
        source += " b = as_type<";
        source += signed_bits;
        source += ">(rhs);\n  if (b > 0 && a > as_type<";
        source += signed_bits;
        source += ">(";
        source += max_bits;
        source += ") - b) { return ";
        source += max_bits;
        source += "; }\n  if (b < 0 && a < as_type<";
        source += signed_bits;
        source += ">(";
        source += min_bits;
        source += ") - b) { return ";
        source += min_bits;
        source += "; }\n";
        source += "  return lhs + rhs;\n";
      } else {
        source += "  return lhs + rhs;\n";
      }
    } else if (plan.op == rund::kernel::ScatterReduceOp::Min) {
      if (signed_value) {
        source += "  return as_type<";
        source += signed_bits;
        source += ">(rhs) < as_type<";
        source += signed_bits;
        source += ">(lhs) ? rhs : lhs;\n";
      } else {
        source += "  return rhs < lhs ? rhs : lhs;\n";
      }
    } else {
      if (signed_value) {
        source += "  return as_type<";
        source += signed_bits;
        source += ">(rhs) > as_type<";
        source += signed_bits;
        source += ">(lhs) ? rhs : lhs;\n";
      } else {
        source += "  return rhs > lhs ? rhs : lhs;\n";
      }
    }
    source += "}\n\n";
  }

  std::string_view identity = wide ? "0ul" : "0u";
  if (plan.op == rund::kernel::ScatterReduceOp::Min) {
    identity = signed_value ? max_bits : all_bits;
  } else if (plan.op == rund::kernel::ScatterReduceOp::Max && signed_value) {
    identity = min_bits;
  }
  source += "kernel void rund_scatter_reduce_initialize(\n";
  source += "    device ";
  source += bits;
  source += "* output [[buffer(3)]],\n";
  source += R"MSL(    device uint* counts [[buffer(7)]],
    device const uint* indirect [[buffer(5)]],
    constant ScatterReduceParams& params [[buffer(6)]],
    uint target [[thread_position_in_grid]]) {
  if (indirect[0] == 0u || ulong(target) >= params.output_count) { return; }
)MSL";
  source += "  output[target] = ";
  source += identity;
  source += ";\n";
  source += R"MSL(  counts[target] = 0u;
}

)MSL";
  source += "kernel void rund_scatter_reduce_fold_sources(\n";
  source += "    device const ";
  source += bits;
  source += "* values [[buffer(0)]],\n";
  source += R"MSL(    device const uint* indices [[buffer(1)]],
    device const uint* count_words [[buffer(2)]],
)MSL";
  source += "    device ";
  source += bits;
  source += "* output [[buffer(3)]],\n";
  source += R"MSL(    device atomic_uint* status [[buffer(4)]],
    device const uint* indirect [[buffer(5)]],
    constant ScatterReduceParams& params [[buffer(6)]],
    device uint* counts [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong logical = rund_scatter_reduce_count(count_words, params);
  if (indirect[3] == 0u) { return; }
)MSL";
  if (parallel_fold) {
    if (!AppendMetalScatterReduceParallelFold(source, plan.op, signed_value)) {
      return false;
    }
  } else {
    source += R"MSL(  if (gid != 0u) { return; }
  uint conflicts = 0u;
  for (ulong ordinal = 0u; ordinal < logical; ++ordinal) {
    const uint target = indices[ordinal];
    conflicts += uint(counts[target] != 0u);
    ++counts[target];
    output[target] = rund_scatter_reduce_fold(output[target], values[ordinal]);
  }
  atomic_store_explicit(&status[2], conflicts, memory_order_relaxed);
)MSL";
  }
  source += "}\n";
  return source.valid();
}

std::string
MetalScatterReduceSource(const rund::kernel::ScatterReducePlan &plan) {
  const auto emit = [&plan](auto &sink) noexcept(
                        noexcept(EmitMetalScatterReduceSource(sink, plan))) {
    return EmitMetalScatterReduceSource(sink, plan);
  };
  return backend_source_recipe::materialize(emit);
}

bool MetalScatterReduceSourceUpperBytes(
    const rund::kernel::ScatterReducePlan &plan,
    std::uint64_t &upper) noexcept {
  const auto emit = [&plan](backend_source_recipe::CountSink &sink) noexcept {
    return EmitMetalScatterReduceSource(sink, plan);
  };
  return backend_source_recipe::bytes(emit, upper);
}

#endif

} // namespace rund::node::accel::detail

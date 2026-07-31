#include "model.hpp"

#include "../../../scatter/reduce/model.hpp"

#include <cstdint>
#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] std::string
Declarations(const rund::kernel::ScatterReducePlan &plan,
             const std::uint32_t local_width) {
  const bool signed_atomic =
      rund::kernel::ScatterReduceFoldParallel(plan) &&
      plan.op != rund::kernel::ScatterReduceOp::Sum &&
      (plan.domain == rund::kernel::ComputeDomain::I32 ||
       plan.domain == rund::kernel::ComputeDomain::Fixed);
  const std::string bits =
      plan.element_bytes == 8u ? "uint64_t" : (signed_atomic ? "int" : "uint");
  std::string source = "#version 450\n";
  source +=
      "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  source += "layout(local_size_x = " + std::to_string(local_width) + ") in;\n";
  source += R"GLSL(layout(set=0,binding=0,std430) readonly buffer Params {
  uint64_t element_count;
  uint64_t output_count;
  uint count_source;
  uint reserved;
  uint value_base;
  uint index_base;
  uint count_base;
  uint output_base;
} params;
)GLSL";
  source += "layout(set=0,binding=1,std430) readonly buffer Values { " + bits +
            " values[]; };\n";
  source +=
      R"GLSL(layout(set=0,binding=2,std430) readonly buffer Indices { uint indices[]; };
layout(set=0,binding=3,std430) readonly buffer CountWords { uint count_words[]; };
)GLSL";
  source += "layout(set=0,binding=4,std430) buffer Output { " + bits +
            " output_values[]; };\n";
  source +=
      R"GLSL(layout(set=0,binding=5,std430) buffer Status { uint status[]; };
layout(set=0,binding=6,std430) buffer Indirect { uint indirect_args[]; };
layout(set=0,binding=7,std430) buffer Counts { uint counts[]; };

uint64_t logical_count() {
  if (params.count_source == 0u) { return params.element_count; }
  if (params.count_source == 1u) {
    return uint64_t(count_words[params.count_base]);
  }
  return uint64_t(count_words[params.count_base]) |
         (uint64_t(count_words[params.count_base + 1u]) << 32u);
}
)GLSL";
  return source;
}

} // namespace

std::string
VulkanScatterReduceSource(const rund::kernel::ScatterReducePlan &plan,
                          const VulkanScatterReduceStage stage) {
  if (stage == VulkanScatterReduceStage::Control) {
    std::string source = Declarations(plan, kScatterReduceWidth);
    source += R"GLSL(shared uint invalids[256];

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint64_t logical = logical_count();
  uint local_invalid = 0xffffffffu;
  if (logical <= params.element_count) {
    for (uint ordinal = tid; uint64_t(ordinal) < logical;) {
      if (uint64_t(indices[params.index_base + ordinal]) >=
          params.output_count) {
        local_invalid = min(local_invalid, ordinal);
      }
      if (ordinal > 0xffffffffu - 256u) { break; }
      ordinal += 256u;
    }
  }
  invalids[tid] = local_invalid;
  barrier();
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (tid < stride) {
      invalids[tid] = min(invalids[tid], invalids[tid + stride]);
    }
    barrier();
  }
  if (tid != 0u) { return; }
  status[0] = 0u;
  status[1] = uint(min(logical, uint64_t(0xffffffffu)));
  status[2] = 0u;
  status[3] = 0u;
  indirect_args[0] = 0u;
  indirect_args[1] = 1u;
  indirect_args[2] = 1u;
  indirect_args[3] = 0u;
  indirect_args[4] = 1u;
  indirect_args[5] = 1u;
  if (logical > params.element_count) {
    status[0] = 1u;
    status[1] = uint(params.element_count);
    return;
  }
  if (invalids[0] != 0xffffffffu) {
    status[0] = 2u;
    status[1] = invalids[0];
    return;
  }
  indirect_args[0] = uint((params.output_count + 255u) / 256u);
)GLSL";
    source += rund::kernel::ScatterReduceFoldParallel(plan)
                  ? "  indirect_args[3] = uint((logical + 255u) / 256u);\n"
                  : "  indirect_args[3] = 1u;\n";
    source += "}\n";
    return source;
  }

  const bool wide = plan.element_bytes == 8u;
  const bool signed_value = plan.domain == rund::kernel::ComputeDomain::I32 ||
                            plan.domain == rund::kernel::ComputeDomain::I64 ||
                            plan.domain == rund::kernel::ComputeDomain::Fixed;
  const bool fixed_saturate =
      plan.domain == rund::kernel::ComputeDomain::Fixed &&
      plan.fixed_format.overflow == rund::kernel::ComputeOverflow::Saturate;
  const bool parallel_fold = rund::kernel::ScatterReduceFoldParallel(plan);
  const bool signed_atomic = parallel_fold &&
                             plan.op != rund::kernel::ScatterReduceOp::Sum &&
                             signed_value;
  const std::string bits = wide ? "uint64_t" : (signed_atomic ? "int" : "uint");
  const std::string signed_bits = wide ? "int64_t" : "int";
  const std::string max_bits =
      wide ? "uint64_t(0x7ffffffffffffffful)" : "0x7fffffffu";
  const std::string min_bits =
      wide ? "uint64_t(0x8000000000000000ul)" : "0x80000000u";
  const std::string all_bits =
      wide ? "uint64_t(0xfffffffffffffffful)" : "0xffffffffu";
  std::string identity = wide ? "uint64_t(0u)" : "0u";
  if (plan.op == rund::kernel::ScatterReduceOp::Min) {
    identity =
        signed_atomic ? "2147483647" : (signed_value ? max_bits : all_bits);
  } else if (plan.op == rund::kernel::ScatterReduceOp::Max && signed_value) {
    identity = signed_atomic ? "(-2147483647 - 1)" : min_bits;
  }

  if (stage == VulkanScatterReduceStage::Init) {
    std::string source = Declarations(plan, kScatterReduceWidth);
    source += R"GLSL(void main() {
  const uint target = gl_GlobalInvocationID.x;
  if (uint64_t(target) >= params.output_count) { return; }
)GLSL";
    source +=
        "  output_values[params.output_base + target] = " + identity + ";\n";
    source += "  counts[target] = 0u;\n}\n";
    return source;
  }

  std::string source =
      Declarations(plan, parallel_fold ? kScatterReduceWidth : 1u);
  source += bits + " reduce_value(" + bits + " lhs, " + bits + " rhs) {\n";
  if (plan.op == rund::kernel::ScatterReduceOp::Sum) {
    if (fixed_saturate) {
      source += "  const " + signed_bits + " a = " + signed_bits + "(lhs);\n";
      source += "  const " + signed_bits + " b = " + signed_bits + "(rhs);\n";
      source += "  if (b > 0 && a > " + signed_bits + "(" + max_bits +
                ") - b) { return " + max_bits + "; }\n";
      source += "  if (b < 0 && a < " + signed_bits + "(" + min_bits +
                ") - b) { return " + min_bits + "; }\n";
    }
    source += "  return lhs + rhs;\n";
  } else if (plan.op == rund::kernel::ScatterReduceOp::Min) {
    source += signed_value ? "  return " + signed_bits + "(rhs) < " +
                                 signed_bits + "(lhs) ? rhs : lhs;\n"
                           : "  return rhs < lhs ? rhs : lhs;\n";
  } else {
    source += signed_value ? "  return " + signed_bits + "(rhs) > " +
                                 signed_bits + "(lhs) ? rhs : lhs;\n"
                           : "  return rhs > lhs ? rhs : lhs;\n";
  }
  source += R"GLSL(}

void main() {
  const uint64_t logical = logical_count();
)GLSL";
  if (parallel_fold) {
    source += R"GLSL(  const uint ordinal = gl_GlobalInvocationID.x;
  if (uint64_t(ordinal) >= logical) { return; }
  const uint target = indices[params.index_base + ordinal];
  if (atomicAdd(counts[target], 1u) != 0u) {
    atomicAdd(status[2], 1u);
  }
)GLSL";
    if (plan.op == rund::kernel::ScatterReduceOp::Sum) {
      source +=
          "  atomicAdd(output_values[params.output_base + target], "
          "values[params.value_base + ordinal]);\n";
    } else if (plan.op == rund::kernel::ScatterReduceOp::Min) {
      source +=
          "  atomicMin(output_values[params.output_base + target], "
          "values[params.value_base + ordinal]);\n";
    } else {
      source +=
          "  atomicMax(output_values[params.output_base + target], "
          "values[params.value_base + ordinal]);\n";
    }
  } else {
    source += R"GLSL(  if (gl_GlobalInvocationID.x != 0u) { return; }
  for (uint ordinal = 0u; uint64_t(ordinal) < logical; ++ordinal) {
    const uint target = indices[params.index_base + ordinal];
    if (counts[target] != 0u) { ++status[2]; }
    ++counts[target];
    output_values[params.output_base + target] =
        reduce_value(output_values[params.output_base + target],
                     values[params.value_base + ordinal]);
  }
)GLSL";
  }
  source += "}\n";
  return source;
}

#endif

} // namespace rund::node::accel::detail

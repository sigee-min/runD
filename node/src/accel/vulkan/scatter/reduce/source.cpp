#include "model.hpp"

#include "../../../kernel/backend/source_recipe.hpp"
#include "../../../scatter/reduce/model.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitDeclarations(
    Sink &sink, const rund::kernel::ScatterReducePlan &plan,
    const std::uint32_t
        local_width) noexcept(noexcept(sink.append(std::string_view{}))) {
  const bool signed_atomic =
      rund::kernel::ScatterReduceFoldParallel(plan) &&
      plan.op != rund::kernel::ScatterReduceOp::Sum &&
      (plan.domain == rund::kernel::ComputeDomain::I32 ||
       plan.domain == rund::kernel::ComputeDomain::Fixed);
  const char *const bits =
      plan.element_bytes == 8u ? "uint64_t" : (signed_atomic ? "int" : "uint");
  backend_source_recipe::SourceBuilder source{sink};
  source += "#version 450\n";
  source +=
      "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  source += "layout(local_size_x = ";
  source.decimal(local_width);
  source += R"GLSL() in;
layout(set=0,binding=0,std430) readonly buffer Params {
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
  source += "layout(set=0,binding=1,std430) readonly buffer Values { ";
  source += bits;
  source += R"GLSL( values[]; };
layout(set=0,binding=2,std430) readonly buffer Indices { uint indices[]; };
layout(set=0,binding=3,std430) readonly buffer CountWords { uint count_words[]; };
layout(set=0,binding=4,std430) buffer Output { )GLSL";
  source += bits;
  source += R"GLSL( output_values[]; };
layout(set=0,binding=5,std430) buffer Status { uint status[]; };
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
  return source.valid();
}

[[nodiscard]] const char *
ScatterReduceIdentity(const rund::kernel::ScatterReducePlan &plan,
                      const bool wide, const bool signed_value,
                      const bool signed_atomic) noexcept {
  if (plan.op == rund::kernel::ScatterReduceOp::Min) {
    return signed_atomic
               ? "2147483647"
               : (signed_value ? (wide ? "uint64_t(0x7ffffffffffffffful)"
                                       : "0x7fffffffu")
                               : (wide ? "uint64_t(0xfffffffffffffffful)"
                                       : "0xffffffffu"));
  }
  if (plan.op == rund::kernel::ScatterReduceOp::Max && signed_value) {
    return signed_atomic
               ? "(-2147483647 - 1)"
               : (wide ? "uint64_t(0x8000000000000000ul)" : "0x80000000u");
  }
  return wide ? "uint64_t(0u)" : "0u";
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanScatterReduceSource(
    Sink &sink, const rund::kernel::ScatterReducePlan &plan,
    const VulkanScatterReduceStage
        stage) noexcept(noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder source{sink};
  if (stage == VulkanScatterReduceStage::Control) {
    if (!EmitDeclarations(sink, plan, kScatterReduceWidth)) {
      return false;
    }
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
    return source.valid();
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
  const char *const bits = wide ? "uint64_t" : (signed_atomic ? "int" : "uint");
  const char *const signed_bits = wide ? "int64_t" : "int";
  const char *const max_bits =
      wide ? "uint64_t(0x7ffffffffffffffful)" : "0x7fffffffu";
  const char *const min_bits =
      wide ? "uint64_t(0x8000000000000000ul)" : "0x80000000u";
  const char *const identity =
      ScatterReduceIdentity(plan, wide, signed_value, signed_atomic);

  if (stage == VulkanScatterReduceStage::Init) {
    if (!EmitDeclarations(sink, plan, kScatterReduceWidth)) {
      return false;
    }
    source += R"GLSL(void main() {
  const uint target = gl_GlobalInvocationID.x;
  if (uint64_t(target) >= params.output_count) { return; }
  output_values[params.output_base + target] = )GLSL";
    source += identity;
    source += R"GLSL(;
  counts[target] = 0u;
}
)GLSL";
    return source.valid();
  }

  if (!EmitDeclarations(sink, plan, parallel_fold ? kScatterReduceWidth : 1u)) {
    return false;
  }
  source += bits;
  source += " reduce_value(";
  source += bits;
  source += " lhs, ";
  source += bits;
  source += " rhs) {\n";
  if (plan.op == rund::kernel::ScatterReduceOp::Sum) {
    if (fixed_saturate) {
      source += "  const ";
      source += signed_bits;
      source += " a = ";
      source += signed_bits;
      source += "(lhs);\n  const ";
      source += signed_bits;
      source += " b = ";
      source += signed_bits;
      source += "(rhs);\n  if (b > 0 && a > ";
      source += signed_bits;
      source += "(";
      source += max_bits;
      source += ") - b) { return ";
      source += max_bits;
      source += "; }\n  if (b < 0 && a < ";
      source += signed_bits;
      source += "(";
      source += min_bits;
      source += ") - b) { return ";
      source += min_bits;
      source += "; }\n";
    }
    source += "  return lhs + rhs;\n";
  } else if (plan.op == rund::kernel::ScatterReduceOp::Min) {
    if (signed_value) {
      source += "  return ";
      source += signed_bits;
      source += "(rhs) < ";
      source += signed_bits;
      source += "(lhs) ? rhs : lhs;\n";
    } else {
      source += "  return rhs < lhs ? rhs : lhs;\n";
    }
  } else if (signed_value) {
    source += "  return ";
    source += signed_bits;
    source += "(rhs) > ";
    source += signed_bits;
    source += "(lhs) ? rhs : lhs;\n";
  } else {
    source += "  return rhs > lhs ? rhs : lhs;\n";
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
      source += "  atomicAdd(output_values[params.output_base + target], "
                "values[params.value_base + ordinal]);\n";
    } else if (plan.op == rund::kernel::ScatterReduceOp::Min) {
      source += "  atomicMin(output_values[params.output_base + target], "
                "values[params.value_base + ordinal]);\n";
    } else {
      source += "  atomicMax(output_values[params.output_base + target], "
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
  return source.valid();
}

} // namespace

std::string
VulkanScatterReduceSource(const rund::kernel::ScatterReducePlan &plan,
                          const VulkanScatterReduceStage stage) {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [&](auto &sink) noexcept(noexcept(
                        EmitVulkanScatterReduceSource(sink, plan, stage))) {
    return EmitVulkanScatterReduceSource(sink, plan, stage);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool VulkanScatterReduceSourceBytes(const rund::kernel::ScatterReducePlan &plan,
                                    const VulkanScatterReduceStage stage,
                                    std::uint64_t &bytes) noexcept {
  const auto emit = [&](auto &sink) noexcept(noexcept(
                        EmitVulkanScatterReduceSource(sink, plan, stage))) {
    return EmitVulkanScatterReduceSource(sink, plan, stage);
  };
  return backend_source_recipe::bytes(emit, bytes);
}

#endif

} // namespace rund::node::accel::detail

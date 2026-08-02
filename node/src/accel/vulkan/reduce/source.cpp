#include "../../domain.hpp"
#include "local.hpp"
#include "source/op.hpp"
#include "../kernel/source_recipe.hpp"

#include <kernel/program/compute/reduce/wide.hpp>
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitVulkanReduceSource(
    Sink &sink, const rund::kernel::ReduceOp op,
    const rund::kernel::ReduceElement element,
    const rund::kernel::u64 block_size,
    const rund::kernel::ComputeDomain domain)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  const bool u64 = element == rund::kernel::ReduceElement::U64;
  const bool signed_domain = IsSignedDomain(domain);
  VulkanSourceTextSink source{sink};
  source += "#version 450\n";
  source +=
      "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  source += "layout(local_size_x = ";
  source.decimal(block_size);
  source += ") in;\n";
  source += "#define RUND_REDUCE_NARROW_CHUNK ";
  source.decimal(rund::kernel::kReduceNarrowChunkItems);
  source += "u\n";
  source += "layout(set = 0, binding = 0, std430) buffer Params {\n";
  source += "  uint64_t input_offset;\n";
  source += "  uint64_t output_offset;\n";
  source += "  uint64_t input_count;\n";
  source += "  uint64_t grid_size;\n";
  source += "  uint final_pass;\n";
  source += "  uint initial_pass;\n";
  source += "  uint count_words;\n";
  source += "} params;\n";
  source += "layout(set = 0, binding = 1, std430) readonly buffer Input {\n";
  source += u64 ? "  uint64_t input_values[];\n" : "  uint input_values[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 2, std430) buffer Partial {\n";
  source += (op == rund::kernel::ReduceOp::Sum ||
             op == rund::kernel::ReduceOp::CountNonzero)
                ? "  uint64_t partial_values[];\n"
                : (u64 ? "  uint64_t partial_values[];\n"
                       : "  uint partial_values[];\n");
  source += "};\n";
  source += "layout(set = 0, binding = 3, std430) buffer Output {\n";
  source += u64 ? "  uint64_t output_values[];\n" : "  uint output_values[];\n";
  source += "};\n";
  source += "layout(set = 0, binding = 4, std430) buffer Status { uint "
            "status[]; };\n";
  source += "layout(set = 0, binding = 5, std430) readonly buffer "
            "LogicalCount { uint logical_count[]; };\n";
  if (op == rund::kernel::ReduceOp::Sum ||
      op == rund::kernel::ReduceOp::CountNonzero) {
    source += "#define RUND_REDUCE_I64 int64_t\n"
              "#define RUND_REDUCE_U64 uint64_t\n";
    source.append(rund::kernel::ReduceWideSource);
    source += R"GLSL(
shared RundWide sums[)GLSL";
    source.decimal(block_size);
    source += R"GLSL(];
void main() {
  const uint tid = gl_LocalInvocationID.x;
  RundWide acc = rund_wide_zero();
  if (params.initial_pass != 0u) {
    uint64_t resident_count = params.input_count;
    if (params.count_words == 1u) {
      resident_count = uint64_t(logical_count[0]);
    }
    if (params.count_words == 2u) {
      resident_count = (uint64_t(logical_count[1]) << 32u) |
                       uint64_t(logical_count[0]);
    }
    const uint64_t active_count = min(resident_count, params.input_count);
    if (resident_count > params.input_count && tid == 0u &&
        gl_WorkGroupID.x == 0u) {
      atomicOr(status[0], 2u);
    }
    const uint64_t stride = params.grid_size * uint64_t()GLSL";
    source.decimal(block_size);
    source += ");\n";
    if (op == rund::kernel::ReduceOp::CountNonzero) {
      source += "    RundPair count = rund_pair_zero();\n";
      source += "    for (uint64_t index = uint64_t(gl_WorkGroupID.x) * "
                "uint64_t(";
      source.decimal(block_size);
      source += R"GLSL() + uint64_t(tid); index < active_count;
         index += stride) {
      count = rund_pair_add(
          count, input_values[uint(index)] != 0 ? 1u : 0u, 0u);
    }
    acc = rund_wide_pair_unsigned(count);
)GLSL";
    } else if (!u64) {
      source += "    for (uint64_t index = uint64_t(gl_WorkGroupID.x) * "
                "uint64_t(";
      source.decimal(block_size);
      source += R"GLSL() + uint64_t(tid); index < active_count;) {
)GLSL";
      source += "      RundPair narrow = rund_pair_zero();\n";
      source += R"GLSL(      uint items = 0u;
      do {
)GLSL";
      source += signed_domain
                    ? "        int value = int(input_values[uint(index)]);\n   "
                      "     narrow = rund_pair_add(narrow, uint(value), value "
                      "< 0 ? 0xffffffffu : 0u);\n"
                    : "        narrow = rund_pair_add(narrow, "
                      "input_values[uint(index)], 0u);\n";
      source += R"GLSL(        index += stride;
        ++items;
      } while (items < RUND_REDUCE_NARROW_CHUNK && index < active_count);
      acc = rund_wide_add(acc, )GLSL";
      source += signed_domain ? "rund_wide_pair_signed(narrow));\n"
                              : "rund_wide_pair_unsigned(narrow));\n";
      source += "    }\n";
    } else {
      source += "    for (uint64_t index = uint64_t(gl_WorkGroupID.x) * "
                "uint64_t(";
      source.decimal(block_size);
      source += R"GLSL() + uint64_t(tid); index < active_count;
         index += stride) {
      acc = rund_wide_add(acc, rund_wide_)GLSL";
      source += signed_domain ? "i" : "u";
      source += "64";
      source += "(";
      source += signed_domain ? "int64_t" : "uint64_t";
      source += "(input_values[uint(index)])));\n";
      source += "    }\n";
    }
    source += R"GLSL(  } else {
    for (uint index = tid; index < uint(params.input_count);
         index += )GLSL";
    source.decimal(block_size);
    source += R"GLSL(u) {
      const uint word = index << 1u;
      acc = rund_wide_add(
          acc, rund_wide_make(partial_values[word], partial_values[word + 1u]));
    }
  }
  sums[tid] = acc;
  barrier();
  uint width = )GLSL";
    source.decimal(block_size);
    source += R"GLSL(u;
  while (width > 1u) {
    const uint next = (width + 1u) >> 1u;
    if (tid < width - next) {
      sums[tid] = rund_wide_add(sums[tid], sums[tid + next]);
    }
    barrier();
    width = next;
  }
  if (tid != 0u) { return; }
  acc = sums[0];
  if (params.final_pass == 0u) {
    const uint word = gl_WorkGroupID.x << 1u;
    partial_values[word] = acc.lo;
    partial_values[word + 1u] = acc.hi;
    return;
  }
  if (!rund_wide_fits_)GLSL";
    source += signed_domain ? "i" : "u";
    source += u64 ? "64" : "32";
    source += "(acc)) { atomicOr(status[0], 1u); }\n"
              "  output_values[0] = ";
    source +=
        u64 ? "uint64_t(rund_wide_low64(acc));\n" : "rund_wide_low32(acc);\n";
    source += "}\n";
    return source.ok();
  }
  source += "shared uint64_t sums[";
  source.decimal(block_size);
  source += "];\nshared uint overflows[";
  source.decimal(block_size);
  source += "];\nvoid main() {\n";
  source += "  const uint tid = gl_LocalInvocationID.x;\n";
  source += "  const uint64_t index = uint64_t(gl_WorkGroupID.x) * uint64_t(";
  source.decimal(block_size);
  source += ") + uint64_t(tid);\n";
  source += "  uint64_t resident_count = params.input_count;\n";
  source += "  if (params.count_words == 1u) { resident_count = "
            "uint64_t(logical_count[0]); }\n";
  source += "  if (params.count_words == 2u) { resident_count = "
            "(uint64_t(logical_count[1]) << 32u) | "
            "uint64_t(logical_count[0]); }\n";
  source += "  const uint64_t active_count = params.initial_pass != 0u ? "
            "min(resident_count, params.input_count) : params.input_count;\n";
  source += "  if (params.initial_pass != 0u && resident_count > "
            "params.input_count && tid == 0u && gl_WorkGroupID.x == 0u) "
            "{ atomicOr(status[0], 2u); }\n";
  if (op == rund::kernel::ReduceOp::Min || op == rund::kernel::ReduceOp::Max) {
    source += "  if (params.initial_pass != 0u && active_count == 0ul && "
              "tid == 0u && gl_WorkGroupID.x == 0u) { atomicOr(status[0], "
              "4u); }\n";
  }
  source += "  const uint source_index = uint(params.input_offset + index);\n";
  source += "  const uint partial_index = uint(params.output_offset + "
            "uint64_t(gl_WorkGroupID.x));\n";
  source += VulkanReduceInit(op, u64, signed_domain);
  source += "  overflows[tid] = 0u;\n";
  source += "  barrier();\n";
  source += "  uint width = ";
  source.decimal(block_size);
  source += "u;\n";
  source += "  while (width > 1u) {\n";
  source += "    const uint next = (width + 1u) >> 1u;\n";
  source += "    if (tid < width - next) {\n";
  source += "      const uint64_t rhs = sums[tid + next];\n";
  source += VulkanReduceCombine(op, u64, signed_domain);
  source += VulkanReduceOverflow(op, u64, signed_domain);
  source += "      overflows[tid] |= overflows[tid + next];\n";
  source += "      sums[tid] = combined;\n";
  source += "    }\n";
  source += "    barrier();\n";
  source += "    width = next;\n";
  source += "  }\n";
  source += "  if (tid == 0u) {\n";
  source += "    if (overflows[0] != 0u) { atomicOr(status[0], 1u); }\n";
  source += "    if (params.final_pass != 0u) { output_values[0] = ";
  source += u64 ? "sums[0]" : "uint(sums[0])";
  source += "; }\n";
  source += "    else { partial_values[partial_index] = ";
  source += u64 ? "sums[0]" : "uint(sums[0])";
  source += "; }\n";
  source += "  }\n}\n";
  return source.ok();
}

} // namespace

std::string VulkanReduceSource(const rund::kernel::ReduceOp op,
                               const rund::kernel::ReduceElement element,
                               const rund::kernel::u64 block_size,
                               const rund::kernel::ComputeDomain domain) {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [&](auto &sink)
      noexcept(noexcept(EmitVulkanReduceSource(sink, op, element, block_size,
                                                domain))) {
    return EmitVulkanReduceSource(sink, op, element, block_size, domain);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool VulkanReduceSourceBytes(const rund::kernel::ReduceOp op,
                             const rund::kernel::ReduceElement element,
                             const rund::kernel::u64 block_size,
                             const rund::kernel::ComputeDomain domain,
                             std::uint64_t &bytes) noexcept {
  const auto emit = [&](auto &sink)
      noexcept(noexcept(EmitVulkanReduceSource(sink, op, element, block_size,
                                                domain))) {
    return EmitVulkanReduceSource(sink, op, element, block_size, domain);
  };
  return backend_source_recipe::bytes(emit, bytes);
}
#endif
} // namespace rund::node::accel::detail

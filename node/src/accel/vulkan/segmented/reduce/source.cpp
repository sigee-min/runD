#include "model.hpp"

#include "../../../domain.hpp"
#include "../../../kernel/backend/source_recipe.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] const char *WideHelpers() noexcept {
  return R"GLSL(
struct RundWide { uint64_t lo; uint64_t hi; };
RundWide rund_wide_add(RundWide lhs, RundWide rhs) {
  uint64_t lo = lhs.lo + rhs.lo;
  return RundWide(lo, lhs.hi + rhs.hi + (lo < lhs.lo ? 1ul : 0ul));
}
RundWide rund_wide_i32(int value) {
  return RundWide(uint64_t(int64_t(value)),
                  value < 0 ? uint64_t(-1l) : 0ul);
}
RundWide rund_wide_u32(uint value) {
  return RundWide(uint64_t(value), 0ul);
}
RundWide rund_wide_i64(int64_t value) {
  return RundWide(uint64_t(value), value < 0l ? uint64_t(-1l) : 0ul);
}
RundWide rund_wide_u64(uint64_t value) {
  return RundWide(value, 0ul);
}
bool rund_wide_fits_i32(RundWide value) {
  return (value.hi == 0ul && value.lo <= 0x7ffffffful) ||
         (value.hi == uint64_t(-1l) && value.lo >= 0xffffffff80000000ul);
}
bool rund_wide_fits_u32(RundWide value) {
  return value.hi == 0ul && value.lo <= 0xfffffffful;
}
bool rund_wide_fits_i64(RundWide value) {
  return (value.hi == 0ul && value.lo <= 0x7ffffffffffffffful) ||
         (value.hi == uint64_t(-1l) && value.lo >= 0x8000000000000000ul);
}
bool rund_wide_fits_u64(RundWide value) {
  return value.hi == 0ul;
}
)GLSL";
}

template <typename Sink>
[[nodiscard]] bool
EmitHeader(Sink &sink) noexcept(noexcept(sink.append(std::string_view{}))) {
  return sink.append("#version 450\n") &&
         sink.append("#extension "
                     "GL_EXT_shader_explicit_arithmetic_types_int64 : "
                     "require\n") &&
         AppendSegmentedReduceShaderModel(sink);
}

template <typename Sink>
[[nodiscard]] bool EmitSegmentedClassifySource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return EmitHeader(sink) && sink.append(R"GLSL(
layout(local_size_x = RUND_SEGMENT_INDEX_WIDTH) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t count;
  uint64_t block_count;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Heads { uint heads[]; };
layout(set = 0, binding = 2, std430) buffer Counts { uint counts[]; };
layout(set = 0, binding = 5, std430) buffer Status { uint status; };
shared uint partial[RUND_SEGMENT_INDEX_WIDTH];
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint blocks = uint(params.block_count);
  const uint groups = min(blocks, RUND_SEGMENT_MAX_GROUPS);
  for (uint block = gl_WorkGroupID.x; block < blocks; block += groups) {
    const uint index = block * RUND_SEGMENT_INDEX_WIDTH + lane;
    const uint head = uint64_t(index) < params.count ? heads[index] : 0u;
    if (index == 0u && head != 1u) { atomicOr(status, RUND_SEGMENT_INVALID); }
    if (uint64_t(index) < params.count && head > 1u) {
      atomicOr(status, RUND_SEGMENT_INVALID);
    }
    partial[lane] = head != 0u ? 1u : 0u;
    barrier();
    for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
         stride >>= 1u) {
      if (lane < stride) { partial[lane] += partial[lane + stride]; }
      barrier();
    }
    if (lane == 0u) { counts[block] = partial[0]; }
  }
}
)GLSL");
}

template <typename Sink>
[[nodiscard]] bool EmitSegmentedPrefixSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return EmitHeader(sink) && sink.append(R"GLSL(
layout(local_size_x = RUND_SEGMENT_INDEX_WIDTH) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t count;
  uint64_t block_count;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Counts {
  uint counts[];
};
layout(set = 0, binding = 2, std430) buffer Offsets { uint offsets[]; };
layout(set = 0, binding = 3, std430) buffer SegmentCount { uint segments; };
layout(set = 0, binding = 4, std430) buffer Dispatch { uint dispatch[]; };
shared uint partial[RUND_SEGMENT_INDEX_WIDTH];
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint blocks = uint(params.block_count);
  const uint quotient = blocks / RUND_SEGMENT_INDEX_WIDTH;
  const uint remainder = blocks % RUND_SEGMENT_INDEX_WIDTH;
  const uint begin = quotient * lane + min(lane, remainder);
  const uint end = begin + quotient + (lane < remainder ? 1u : 0u);
  uint local = 0u;
  for (uint block = begin; block < end; ++block) {
    offsets[block] = local;
    local += counts[block];
  }
  partial[lane] = local;
  barrier();
  for (uint stride = 1u; stride < RUND_SEGMENT_INDEX_WIDTH; stride <<= 1u) {
    const uint index = (lane + 1u) * (stride << 1u) - 1u;
    if (index < RUND_SEGMENT_INDEX_WIDTH) {
      partial[index] += partial[index - stride];
    }
    barrier();
  }
  if (lane == 0u) { partial[RUND_SEGMENT_INDEX_WIDTH - 1u] = 0u; }
  barrier();
  for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
       stride >>= 1u) {
    const uint index = (lane + 1u) * (stride << 1u) - 1u;
    if (index < RUND_SEGMENT_INDEX_WIDTH) {
      const uint left = partial[index - stride];
      partial[index - stride] = partial[index];
      partial[index] += left;
    }
    barrier();
  }
  const uint base = partial[lane];
  for (uint block = begin; block < end; ++block) { offsets[block] += base; }
  memoryBarrierBuffer();
  barrier();
  if (lane == 0u) {
    const uint last = blocks - 1u;
    segments = offsets[last] + counts[last];
    const uint groups =
        segments / RUND_SEGMENT_TEAMS_PER_GROUP +
        (segments % RUND_SEGMENT_TEAMS_PER_GROUP != 0u ? 1u : 0u);
    dispatch[0] = min(groups, RUND_SEGMENT_MAX_GROUPS);
    dispatch[1] = 1u;
    dispatch[2] = 1u;
  }
}
)GLSL");
}

template <typename Sink>
[[nodiscard]] bool EmitSegmentedScatterSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return EmitHeader(sink) && sink.append(R"GLSL(
layout(local_size_x = RUND_SEGMENT_INDEX_WIDTH) in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t count;
  uint64_t block_count;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Heads { uint heads[]; };
layout(set = 0, binding = 2, std430) readonly buffer Offsets {
  uint offsets[];
};
layout(set = 0, binding = 3, std430) buffer Starts { uint starts[]; };
shared uint partial[RUND_SEGMENT_INDEX_WIDTH];
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint blocks = uint(params.block_count);
  const uint groups = min(blocks, RUND_SEGMENT_MAX_GROUPS);
  for (uint block = gl_WorkGroupID.x; block < blocks; block += groups) {
    const uint index = block * RUND_SEGMENT_INDEX_WIDTH + lane;
    const uint head = uint64_t(index) < params.count ? heads[index] : 0u;
    partial[lane] = head != 0u ? 1u : 0u;
    barrier();
    for (uint stride = 1u; stride < RUND_SEGMENT_INDEX_WIDTH; stride <<= 1u) {
      const uint offset = (lane + 1u) * (stride << 1u) - 1u;
      if (offset < RUND_SEGMENT_INDEX_WIDTH) {
        partial[offset] += partial[offset - stride];
      }
      barrier();
    }
    if (lane == 0u) { partial[RUND_SEGMENT_INDEX_WIDTH - 1u] = 0u; }
    barrier();
    for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
         stride >>= 1u) {
      const uint offset = (lane + 1u) * (stride << 1u) - 1u;
      if (offset < RUND_SEGMENT_INDEX_WIDTH) {
        const uint left = partial[offset - stride];
        partial[offset - stride] = partial[offset];
        partial[offset] += left;
      }
      barrier();
    }
    if (head != 0u) { starts[offsets[block] + partial[lane]] = index; }
    barrier();
  }
}
)GLSL");
}

template <typename Sink>
[[nodiscard]] bool EmitSegmentedReduceSource(
    Sink &sink, const rund::kernel::SegmentedReducePlan &plan,
    const rund::kernel::ComputeDomain
        domain) noexcept(noexcept(sink.append(std::string_view{}))) {
  const bool wide = plan.element_bytes == sizeof(rund::kernel::u64);
  const bool signed_domain = IsSignedDomain(domain);
  const char *const storage = wide ? "uint64_t" : "uint";
  const char *const value = wide ? (signed_domain ? "int64_t" : "uint64_t")
                                 : (signed_domain ? "int" : "uint");
  if (!EmitHeader(sink)) {
    return false;
  }
  backend_source_recipe::SourceBuilder source{sink};
  source += R"GLSL(
layout(local_size_x = RUND_SEGMENT_INDEX_WIDTH) in;
)GLSL";
  source += WideHelpers();
  source += R"GLSL(
layout(set = 0, binding = 0, std430) readonly buffer Params { uint64_t count; } params;
layout(set = 0, binding = 1, std430) readonly buffer Input { )GLSL";
  source += storage;
  source += " input_values[]; };\n";
  source += "layout(set = 0, binding = 2, std430) readonly buffer "
            "Starts { uint starts[]; };\n";
  source += "layout(set = 0, binding = 3, std430) readonly buffer "
            "SegmentCount { uint segments; };\n";
  source += "layout(set = 0, binding = 4, std430) buffer Output { ";
  source += storage;
  source += " output_values[]; };\n";
  source += "layout(set = 0, binding = 5, std430) buffer Status { uint status; "
            "};\n";
  if (plan.op == rund::kernel::ReduceOp::Sum ||
      plan.op == rund::kernel::ReduceOp::CountNonzero) {
    source += R"GLSL(
shared RundWide partial[RUND_SEGMENT_INDEX_WIDTH];
void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint team = tid / RUND_SEGMENT_TEAM_WIDTH;
  const uint lane = tid % RUND_SEGMENT_TEAM_WIDTH;
  const uint groups = min(
      segments / RUND_SEGMENT_TEAMS_PER_GROUP +
          (segments % RUND_SEGMENT_TEAMS_PER_GROUP != 0u ? 1u : 0u),
      RUND_SEGMENT_MAX_GROUPS);
  if (groups == 0u) { return; }
  for (uint base = gl_WorkGroupID.x * RUND_SEGMENT_TEAMS_PER_GROUP;
       base < segments;
       base += groups * RUND_SEGMENT_TEAMS_PER_GROUP) {
    const uint slot = base + team;
    const bool live = slot < segments;
    const uint begin = live ? starts[slot] : 0u;
    const uint end = !live ? 0u :
        (slot + 1u < segments ? starts[slot + 1u] : uint(params.count));
    RundWide acc = RundWide(0ul, 0ul);
    for (uint index = begin + lane; index < end;
         index += RUND_SEGMENT_TEAM_WIDTH) {
)GLSL";
    if (plan.op == rund::kernel::ReduceOp::CountNonzero) {
      source += "      if (input_values[index] != 0) { acc = "
                "rund_wide_add(acc, RundWide(1ul, 0ul)); }\n";
    } else {
      source += "      acc = rund_wide_add(acc, rund_wide_";
      source += signed_domain ? "i" : "u";
      source += wide ? "64" : "32";
      source += "(";
      source += value;
      source += "(input_values[index])));\n";
    }
    source += R"GLSL(
    }
    partial[tid] = acc;
    barrier();
    for (uint stride = RUND_SEGMENT_TEAM_WIDTH >> 1u; stride > 0u;
         stride >>= 1u) {
      if (lane < stride) {
        partial[tid] = rund_wide_add(partial[tid], partial[tid + stride]);
      }
      barrier();
    }
    if (live && lane == 0u) {
      const RundWide total = partial[tid];
      if (!rund_wide_fits_)GLSL";
    source += signed_domain ? "i" : "u";
    source += wide ? "64" : "32";
    source += "(total)) { atomicOr(status, ";
    source += plan.op == rund::kernel::ReduceOp::CountNonzero
                  ? "RUND_SEGMENT_COUNT_OVERFLOW"
                  : "RUND_SEGMENT_SUM_OVERFLOW";
    source += "); }\n      output_values[slot] = ";
    source += storage;
    source += R"GLSL((total.lo);
    }
    barrier();
  }
}
)GLSL";
    return source.valid();
  }
  const char *const maximum =
      wide ? (signed_domain ? "int64_t(0x7fffffffffffffffUL)"
                            : "uint64_t(0xffffffffffffffffUL)")
           : (signed_domain ? "int(0x7fffffffU)" : "uint(0xffffffffU)");
  const char *const minimum =
      wide ? (signed_domain ? "(-int64_t(0x7fffffffffffffffUL) - int64_t(1))"
                            : "uint64_t(0ul)")
           : (signed_domain ? "(-int(0x7fffffffU) - 1)" : "uint(0u)");
  source += "shared ";
  source += value;
  source += R"GLSL( partial[RUND_SEGMENT_INDEX_WIDTH];
void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint team = tid / RUND_SEGMENT_TEAM_WIDTH;
  const uint lane = tid % RUND_SEGMENT_TEAM_WIDTH;
  const uint groups = min(
      segments / RUND_SEGMENT_TEAMS_PER_GROUP +
          (segments % RUND_SEGMENT_TEAMS_PER_GROUP != 0u ? 1u : 0u),
      RUND_SEGMENT_MAX_GROUPS);
  if (groups == 0u) { return; }
  for (uint base = gl_WorkGroupID.x * RUND_SEGMENT_TEAMS_PER_GROUP;
       base < segments;
       base += groups * RUND_SEGMENT_TEAMS_PER_GROUP) {
    const uint slot = base + team;
    const bool live = slot < segments;
    const uint begin = live ? starts[slot] : 0u;
    const uint end = !live ? 0u :
        (slot + 1u < segments ? starts[slot + 1u] : uint(params.count));
)GLSL";
  source += "    ";
  source += value;
  source += " acc = ";
  source += plan.op == rund::kernel::ReduceOp::Min ? maximum : minimum;
  source += R"GLSL(;
    for (uint index = begin + lane; index < end;
         index += RUND_SEGMENT_TEAM_WIDTH) {
      acc = )GLSL";
  source += plan.op == rund::kernel::ReduceOp::Min ? "min" : "max";
  source += "(acc, ";
  source += value;
  source += R"GLSL((input_values[index]));
    }
    partial[tid] = acc;
    barrier();
    for (uint stride = RUND_SEGMENT_TEAM_WIDTH >> 1u; stride > 0u;
         stride >>= 1u) {
      if (lane < stride) {
        partial[tid] = )GLSL";
  source += plan.op == rund::kernel::ReduceOp::Min ? "min" : "max";
  source += R"GLSL((partial[tid], partial[tid + stride]);
      }
      barrier();
    }
    if (live && lane == 0u) { output_values[slot] = )GLSL";
  source += storage;
  source += R"GLSL((partial[tid]); }
    barrier();
  }
}
)GLSL";
  return source.valid();
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanSegmentedReduceSource(
    Sink &sink, const rund::kernel::SegmentedReducePlan &plan,
    const rund::kernel::ComputeDomain domain,
    const VulkanSegmentedReduceStage
        stage) noexcept(noexcept(sink.append(std::string_view{}))) {
  switch (stage) {
  case VulkanSegmentedReduceStage::Classify:
    return EmitSegmentedClassifySource(sink);
  case VulkanSegmentedReduceStage::Prefix:
    return EmitSegmentedPrefixSource(sink);
  case VulkanSegmentedReduceStage::Scatter:
    return EmitSegmentedScatterSource(sink);
  case VulkanSegmentedReduceStage::Reduce:
    return EmitSegmentedReduceSource(sink, plan, domain);
  }
  return false;
}

} // namespace

std::string
VulkanSegmentedReduceSource(const rund::kernel::SegmentedReducePlan &plan,
                            const rund::kernel::ComputeDomain domain,
                            const VulkanSegmentedReduceStage stage) {
  std::uint64_t exact_bytes = 0u;
  const auto emit =
      [&](auto &sink) noexcept(noexcept(
          EmitVulkanSegmentedReduceSource(sink, plan, domain, stage))) {
        return EmitVulkanSegmentedReduceSource(sink, plan, domain, stage);
      };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool VulkanSegmentedReduceSourceBytes(
    const rund::kernel::SegmentedReducePlan &plan,
    const rund::kernel::ComputeDomain domain,
    const VulkanSegmentedReduceStage stage, std::uint64_t &bytes) noexcept {
  const auto emit =
      [&](auto &sink) noexcept(noexcept(
          EmitVulkanSegmentedReduceSource(sink, plan, domain, stage))) {
        return EmitVulkanSegmentedReduceSource(sink, plan, domain, stage);
      };
  return backend_source_recipe::bytes(emit, bytes);
}

#endif

} // namespace rund::node::accel::detail

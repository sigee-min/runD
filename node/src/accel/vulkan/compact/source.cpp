#include "local.hpp"

#include "../../kernel/backend/source_recipe.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitCompactBindings(Sink &sink, const bool flags,
                                       const bool counts, const bool offsets,
                                       const bool output)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return (!flags || sink.append(R"glsl(layout(set = 0, binding = 1, std430) readonly buffer Flags {
  uint flags[];
};
)glsl")) &&
         (!counts || sink.append(R"glsl(layout(set = 0, binding = 2, std430) buffer Counts {
  uint counts[];
};
)glsl")) &&
         (!offsets || sink.append(R"glsl(layout(set = 0, binding = 3, std430) buffer Offsets {
  uint offsets[];
};
)glsl")) &&
         (!output || sink.append(R"glsl(layout(set = 0, binding = 4, std430) buffer Output {
  uint output_ids[];
};
)glsl"));
}

template <typename Sink>
[[nodiscard]] bool EmitCompactStage(Sink &sink, const CompactStage stage)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  switch (stage) {
  case CompactStage::Classify:
    return EmitCompactBindings(sink, true, true, false, false) &&
           sink.append(R"glsl(shared uint local_values[kCompactBlockSize];
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint index = gl_GlobalInvocationID.x;
  uint selected = 0u;
  if (uint64_t(index) < params.element_count) {
    selected = flags[index] != 0u ? 1u : 0u;
  }
  local_values[lane] = selected;
  barrier();
  for (uint width = kCompactBlockSize >> 1u; width != 0u; width >>= 1u) {
    if (lane < width) { local_values[lane] += local_values[lane + width]; }
    barrier();
  }
  if (lane == 0u) { counts[gl_WorkGroupID.x] = local_values[0]; }
}
)glsl");
  case CompactStage::Prefix:
    return EmitCompactBindings(sink, false, true, true, false) &&
           sink.append(R"glsl(shared uint local_values[kCompactBlockSize];
shared uint local_carry;
shared uint local_base;
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint block_count = uint((params.element_count - uint64_t(1)) / uint64_t(kCompactBlockSize)) + 1u;
  if (lane == 0u) { local_carry = 0u; }
  barrier();
  for (uint tile = 0u; tile < block_count; tile += kCompactBlockSize) {
    const uint block = tile + lane;
    const uint value = block < block_count ? counts[block] : 0u;
    local_values[lane] = value;
    if (lane == 0u) { local_base = local_carry; }
    barrier();
    for (uint step = 1u; step < kCompactBlockSize; step <<= 1u) {
      const uint addend = lane >= step ? local_values[lane - step] : 0u;
      barrier();
      local_values[lane] += addend;
      barrier();
    }
    if (block < block_count) {
      offsets[block] = local_base + local_values[lane] - value;
    }
    barrier();
    if (lane == 0u) {
      const uint remaining = block_count - tile;
      const uint last = min(remaining, kCompactBlockSize) - 1u;
      local_carry = local_base + local_values[last];
    }
    barrier();
  }
  if (lane == 0u) {
    status = uint64_t(local_carry) > params.output_capacity ? 1u : 0u;
  }
}
)glsl");
  case CompactStage::Scatter:
    return EmitCompactBindings(sink, true, false, true, true) &&
           sink.append(R"glsl(shared uint local_values[kCompactBlockSize];
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint index = gl_GlobalInvocationID.x;
  uint selected = 0u;
  if (uint64_t(index) < params.element_count) {
    selected = flags[index] != 0u ? 1u : 0u;
  }
  local_values[lane] = selected;
  barrier();
  for (uint step = 1u; step < kCompactBlockSize; step <<= 1u) {
    const uint addend = lane >= step ? local_values[lane - step] : 0u;
    barrier();
    local_values[lane] += addend;
    barrier();
  }
  if (selected != 0u) {
    const uint target = offsets[gl_WorkGroupID.x] + local_values[lane] - 1u;
    if (uint64_t(target) < params.output_capacity) {
      output_ids[target] = index;
    }
  }
}
)glsl");
  }
  return false;
}

template <typename Sink>
[[nodiscard]] bool EmitVulkanCompactSource(Sink &sink,
                                           const CompactStage stage)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return sink.append(R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = )glsl") &&
         backend_source_recipe::append_decimal(sink, block::VulkanCompact) &&
         sink.append(") in;\nconst uint kCompactBlockSize = ") &&
         backend_source_recipe::append_decimal(sink, block::VulkanCompact) &&
         sink.append(R"glsl(u;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t element_count;
  uint64_t output_capacity;
} params;
layout(set = 0, binding = 5, std430) buffer Status {
  uint status;
};
)glsl") &&
         EmitCompactStage(sink, stage);
}

} // namespace

std::string VulkanCompactSource(const CompactStage stage) {
  return backend_source_recipe::materialize(
      [&](auto &sink) { return EmitVulkanCompactSource(sink, stage); });
}

bool VulkanCompactSourceBytes(const CompactStage stage,
                              std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanCompactSource(sink, stage);
      },
      bytes);
}
#endif

} // namespace rund::node::accel::detail

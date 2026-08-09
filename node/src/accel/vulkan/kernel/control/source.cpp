#include "source.hpp"

#include "../control.hpp"

#include "../../../../hash/fnv.hpp"
#include "../../../kernel/backend/phase_source.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <kernel/program/compute/artifact.hpp>

#include <string_view>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] std::uint64_t
PipelineSourceIdentity(const std::string_view source) noexcept {
  ::rund::node::hash_detail::Fnv hash{};
  for (const unsigned char byte : source) {
    hash.Byte(byte);
  }
  return hash.Finish();
}

[[nodiscard]] constexpr std::string_view CanonicalSource() noexcept {
  return R"GLSL(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer RawStatus {
  uint raw_status[];
};
layout(set = 0, binding = 1, std430) buffer CanonicalStatus {
  uint canonical_status[];
};
layout(push_constant) uniform CanonicalParams {
  uint first;
  uint count;
  uint rule;
  uint success;
  uint mapping_count;
  uint invalid_reason;
  uint raw_values[8];
  uint reasons[8];
} p;
void main() {
  const uint index = gl_GlobalInvocationID.x;
  if (index >= p.count) { return; }
  const uint raw = raw_status[index];
  uint reason = 0u;
  if (raw != p.success) {
    uint key = raw;
    if (p.rule == 3u) { key &= 1u; }
    reason = p.invalid_reason;
    if (p.rule == 4u) { reason = p.reasons[0]; }
    for (uint map = 0u; map < p.mapping_count; ++map) {
      const bool match = p.rule == 2u
                             ? (raw & p.raw_values[map]) != 0u
                             : key == p.raw_values[map];
      if (p.rule != 4u && match) { reason = p.reasons[map]; break; }
    }
  }
  canonical_status[p.first + index] = reason;
}
)GLSL";
}

inline constexpr std::string_view VulkanReduceStatusPreamble =
    R"GLSL(#version 450
)GLSL";

inline constexpr std::string_view VulkanReduceStatusBody = R"GLSL(
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer CanonicalStatus {
  uint status_values[];
};
layout(set = 0, binding = 1, std430) buffer ControlSummary {
  uint control[];
};
layout(push_constant) uniform Params {
  uint first;
  uint count;
  uint declared_step;
  uint phase;
  uint failed_outer_window;
  uint failed_inner_iteration;
  uint failed_nested_phase;
  uint reserved;
  uint generation_stride;
  uint declared_step_count;
} p;
shared uint ordinals[256];
shared uint reasons[256];
void main() {
  const uint lane = gl_LocalInvocationID.x;
  if (p.phase == 2u) {
    if (lane == 0u) {
      control[1] = 0u;
      control[2] = 0xffffffffu;
      control[3] = 0u;
      for (uint index = 4u; index < 18u; ++index) { control[index] = 0u; }
      control[18] = 0xffffffffu;
      control[19] = 0xffffffffu;
      control[20] = 0xffffffffu;
      control[21] = 0xffffffffu;
      control[22] = rund_pipeline_phase_none;
      control[23] = 0u;
      for (uint index = 24u; index < 32u; ++index) { control[index] = 0u; }
    }
    return;
  }
  if (p.phase == 1u) {
    if (lane == 0u) {
      control[0] += p.generation_stride;
      if (control[1] == 0u) {
        control[2] = 0xffffffffu;
        control[3] = p.declared_step_count;
      } else {
        control[3] = control[2];
      }
    }
    return;
  }
  const bool valid_failed_phase =
      rund_pipeline_phase_valid(p.failed_nested_phase);
  if (!valid_failed_phase) {
    if (lane == 0u && control[1] == 0u) {
      control[1] = rund_pipeline_reason_invalid;
      control[2] = p.declared_step;
      control[20] = 0xffffffffu;
      control[21] = 0xffffffffu;
      control[22] = rund_pipeline_phase_none;
    }
    return;
  }
  if (control[1] != 0u) { return; }
  uint best = 0xffffffffu;
  uint reason = 0u;
  for (uint index = lane; index < p.count; index += 256u) {
    const uint candidate = status_values[p.first + index];
    if (candidate != 0u && index < best) {
      best = index;
      reason = candidate;
    }
  }
  ordinals[lane] = best;
  reasons[lane] = reason;
  barrier();
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (lane < stride && ordinals[lane + stride] < ordinals[lane]) {
      ordinals[lane] = ordinals[lane + stride];
      reasons[lane] = reasons[lane + stride];
    }
    barrier();
  }
  if (lane == 0u) {
    if (ordinals[0] != 0xffffffffu) {
      control[1] = reasons[0];
      control[2] = p.declared_step;
      control[20] = p.failed_outer_window;
      control[21] = p.failed_inner_iteration;
      control[22] = p.failed_nested_phase;
    }
  }
}
)GLSL";

template <typename Sink>
[[nodiscard]] bool EmitVulkanReduceStatusSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return sink.append(VulkanReduceStatusPreamble) &&
         EmitPipelineNestedPhaseContract(
             sink, PipelineNestedPhaseSourceLanguage::Vulkan) &&
         sink.append(VulkanReduceStatusBody);
}

[[nodiscard]] std::string_view ReduceSource() noexcept {
  static const auto source = backend_source_recipe::materialize_fixed<
      VulkanReduceStatusPreamble.size() + VulkanReduceStatusBody.size() +
      1024u>(
      [](auto &sink) noexcept { return EmitVulkanReduceStatusSource(sink); });
  return source.text();
}

} // namespace

rund::kernel::ComputePlan
VulkanPipelineControlPlan(const std::string_view source) noexcept {
  const std::uint64_t identity = PipelineSourceIdentity(source);
  return rund::kernel::ComputePlan{
      .op_hash_hi = identity,
      .op_hash_lo = identity ^ 0x9e3779b97f4a7c15ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

std::string_view VulkanCanonicalStatusSourceText() noexcept {
  return CanonicalSource();
}

std::string_view VulkanReduceStatusSourceText() noexcept {
  return ReduceSource();
}

} // namespace rund::node::accel::detail

#endif

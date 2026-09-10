#include "source.hpp"

#include "../pipeline/source/artifact.hpp"
#include "../../../kernel/backend/phase/source.hpp"
#include "../../../kernel/backend/source/storage.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

inline constexpr std::string_view VulkanWindowPreamble = R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
)GLSL";

inline constexpr std::string_view VulkanWindowBody = R"GLSL(
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Terminal0 {
  uint terminal0[];
};
layout(set = 0, binding = 1, std430) readonly buffer Terminal1 {
  uint terminal1[];
};
layout(set = 0, binding = 2, std430) readonly buffer Terminal2 {
  uint terminal2[];
};
layout(set = 0, binding = 3, std430) readonly buffer Count {
  uint counts[];
};
layout(set = 0, binding = 4, std430) buffer States {
  uvec2 states[];
};
layout(set = 0, binding = 5, std430) buffer Arguments {
  uint argument_words[];
};
layout(set = 0, binding = 6, std430) readonly buffer Owners {
  uint owners[];
};
layout(set = 0, binding = 7, std430) buffer Control {
  uint control[];
};
layout(push_constant) uniform WindowParams {
  uint64_t count_offset_words;
  uint64_t terminal_offset_words[3];
  uint maximum;
  uint tile;
  uint iteration;
  uint expected;
  uint state;
  uint has_terminal;
  uint command_count;
  uint phase;
  uint declared_step;
  uint overflow_reason;
  uint inner_bound;
  uint inner_advance;
} p;
shared uint enabled;
shared uint fresh;
uint64_t load64(uint word) {
  return uint64_t(control[word]) | (uint64_t(control[word + 1u]) << 32u);
}
void store64(uint word, uint64_t value) {
  control[word] = uint(value);
  control[word + 1u] = uint(value >> 32u);
}
void add64(uint word, uint64_t value) {
  const uint64_t current = load64(word);
  const uint64_t maximum = uint64_t(0xfffffffffffffffful);
  store64(word, value > maximum - current ? maximum : current + value);
}
void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint raw_phase = p.phase;
  uint phase = rund_pipeline_phase_none;
  bool preflight = false;
  const bool valid_phase = rund_pipeline_phase_parameter_decode(
      raw_phase, phase, preflight);
  if (lane == 0u) {
    const uvec2 current = states[p.state];
    const bool failed = control[1] != 0u;
    if (!valid_phase) {
      if (!failed) {
        control[1] = rund_pipeline_reason_invalid;
        control[2] = p.declared_step;
        control[20] = 0xffffffffu;
        control[21] = 0xffffffffu;
        control[22] = rund_pipeline_phase_none;
      }
      states[p.state].y = p.iteration + 1u;
      enabled = 0u;
      fresh = 1u;
    } else if (phase == rund_pipeline_phase_action) {
      enabled = current.y == 0u && !failed ? 1u : 0u;
      fresh = current.y == 0u && failed ? 1u : 0u;
      if (enabled != 0u) { add64(28u, uint64_t(p.inner_advance)); }
    } else if (phase == rund_pipeline_phase_fold) {
      if (current.y == 0u && p.inner_advance != 0u) {
        add64(28u, uint64_t(p.inner_advance));
      }
      enabled = current.y == 0u && !failed ? 1u : 0u;
      fresh = current.y == 0u && failed ? 1u : 0u;
      if (enabled != 0u) {
        add64(12u, 1ul);
        add64(24u, 1ul);
      }
    } else if (phase == rund_pipeline_phase_seed && !preflight) {
      enabled = current.y == 0u && !failed ? 1u : 0u;
      fresh = current.y == 0u && failed ? 1u : 0u;
    } else {
      const uint items = counts[uint(p.count_offset_words)];
      const uint64_t base = uint64_t(p.iteration) * uint64_t(p.tile);
      uint terminal = terminal0[uint(p.terminal_offset_words[0])];
      if (current.x == 1u) {
        terminal = terminal1[uint(p.terminal_offset_words[1])];
      } else if (current.x == 2u) {
        terminal = terminal2[uint(p.terminal_offset_words[2])];
      }
      const bool ended = p.has_terminal != 0u && terminal == p.expected;
      const bool overflow = current.y == 0u && items > p.maximum;
      if (overflow && !failed) {
        control[1] = p.overflow_reason;
        control[2] = p.declared_step;
        store64(18u, uint64_t(p.maximum));
        control[20] = phase == rund_pipeline_phase_seed ? p.iteration
                                                        : 0xffffffffu;
        control[21] = 0xffffffffu;
        control[22] = phase == rund_pipeline_phase_seed
                          ? rund_pipeline_phase_seed
                          : rund_pipeline_phase_none;
      }
      enabled = current.y == 0u && control[1] == 0u &&
                        base < uint64_t(items) && !ended
                    ? 1u
                    : 0u;
      fresh = current.y == 0u && enabled == 0u ? 1u : 0u;
      if (phase == rund_pipeline_phase_seed && control[1] == 0u &&
          enabled == 0u) {
        add64(14u, 1ul);
        add64(26u, 1ul);
        add64(30u, uint64_t(p.inner_bound));
      }
    }
  }
  barrier();
  if (fresh != 0u) {
    for (uint index = lane; index < p.command_count; index += 256u) {
      if (owners[index] == p.state) {
        const uint word = index * 3u;
        argument_words[word] = 0u;
        argument_words[word + 1u] = 0u;
        argument_words[word + 2u] = 0u;
      }
    }
  }
  barrier();
  if (lane == 0u) {
    if (enabled != 0u) {
      if (phase == rund_pipeline_phase_none ||
          phase == rund_pipeline_phase_fold) {
        states[p.state].x = 1u + (p.iteration & 1u);
      }
    } else if (states[p.state].y == 0u) {
      states[p.state].y = p.iteration + 1u;
    }
  }
}
)GLSL";

template <typename Sink>
[[nodiscard]] bool EmitVulkanWindowSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return sink.append(VulkanWindowPreamble) &&
         EmitPipelineNestedPhaseContract(
             sink, PipelineNestedPhaseSourceLanguage::Vulkan) &&
         sink.append(VulkanWindowBody);
}

[[nodiscard]] std::string_view WindowSource() noexcept {
  static const auto source =
      backend_source_recipe::materialize_fixed<VulkanWindowPreamble.size() +
                                               VulkanWindowBody.size() + 1024u>(
          [](auto &sink) noexcept { return EmitVulkanWindowSource(sink); });
  return source.text();
}

[[nodiscard]] constexpr std::string_view GateSource() noexcept {
  return R"GLSL(#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Source {
  uint source_words[];
};
layout(set = 0, binding = 1, std430) buffer Target {
  uint target_words[];
};
layout(set = 0, binding = 2, std430) readonly buffer States {
  uvec2 states[];
};
layout(push_constant) uniform GateParams {
  uint state;
  uint source_word;
  uint target_word;
} p;
void main() {
  const bool enabled = states[p.state].y == 0u;
  target_words[p.target_word] = enabled ? source_words[p.source_word] : 0u;
  target_words[p.target_word + 1u] =
      enabled ? source_words[p.source_word + 1u] : 0u;
  target_words[p.target_word + 2u] =
      enabled ? source_words[p.source_word + 2u] : 0u;
}
)GLSL";
}

} // namespace

rund::kernel::LoweringArtifact VulkanWindowArtifact() {
  return VulkanSourceArtifact(
      [](auto &sink) noexcept { return EmitVulkanWindowSource(sink); });
}

rund::kernel::LoweringArtifact VulkanGateArtifact() {
  return VulkanFixedSourceArtifact(GateSource());
}

std::string_view VulkanWindowSourceText() noexcept { return WindowSource(); }

bool VulkanWindowSourceBytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanWindowSource(sink);
      },
      bytes);
}

std::string_view VulkanGateSourceText() noexcept { return GateSource(); }

#endif

} // namespace rund::node::accel::detail

#include "internal.hpp"

namespace rund::node::accel::detail::vulkan_sliding_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

std::string_view gate_source() noexcept {
  return R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer DescriptorPayload {
  uint descriptor_words[];
};
layout(set = 0, binding = 1, std430) readonly buffer OriginalArguments {
  uint original_words[];
};
layout(set = 0, binding = 2, std430) readonly buffer ArgumentOwners {
  uint argument_owners[];
};
layout(set = 0, binding = 3, std430) buffer LiveArguments {
  uint live_words[];
};
layout(set = 0, binding = 4, std430) buffer ControlSummary {
  uint control[];
};

const uint row_words = 58u;
const uint published_word = 58u;
const uint accepted_word = 116u;
const uint reason_word = 117u;
const uint observed_generation_word = 118u;
const uint owner_word = 0u;
const uint plan_word = 2u;
const uint token_word = 4u;
const uint run_word = 6u;
const uint coordinate_word = 8u;
const uint turn_word = 10u;
const uint read_mask_word = 12u;
const uint write_mask_word = 14u;
const uint descriptor_generation_word = 16u;
const uint generation_stride_word = 18u;
const uint argument_count_word = 19u;
const uint local_count_word = 20u;
const uint stride_word = 21u;
const uint slot_word = 22u;
const uint invalid_reason_word = 23u;
const uint step_count_word = 24u;
const uint control_generation_word = 25u;
const uint local_word = 26u;

shared uint structurally_valid;
shared uint selected_mask;
shared uint approved;

uint64_t load_u64(uint base, uint word) {
  return uint64_t(descriptor_words[base + word]) |
         (uint64_t(descriptor_words[base + word + 1u]) << 32u);
}

void main() {
  const uint lane = gl_LocalInvocationID.x;
  if (lane == 0u) {
    bool valid = descriptor_words.length() >= 120u && control.length() >= 32u;
    for (uint word = 0u; valid && word < row_words; ++word) {
      valid = descriptor_words[word] ==
              descriptor_words[published_word + word];
    }
    const uint argument_count = descriptor_words[argument_count_word];
    const uint local_count = descriptor_words[local_count_word];
    const uint stride = descriptor_words[stride_word];
    const uint slot = descriptor_words[slot_word];
    const uint step_count = descriptor_words[step_count_word];
    const uint64_t owner = load_u64(0u, owner_word);
    const uint64_t plan = load_u64(0u, plan_word);
    const uint64_t token = load_u64(0u, token_word);
    const uint64_t run = load_u64(0u, run_word);
    const uint64_t coordinate = load_u64(0u, coordinate_word);
    const uint64_t turn = load_u64(0u, turn_word);
    const uint64_t read_mask = load_u64(0u, read_mask_word);
    const uint64_t write_mask = load_u64(0u, write_mask_word);
    const uint64_t descriptor_generation =
        load_u64(0u, descriptor_generation_word);
    valid = valid && owner != 0ul && plan != 0ul && token != 0ul &&
            run != 0ul && descriptor_generation != 0ul &&
            descriptor_words[generation_stride_word] != 0u &&
            descriptor_words[control_generation_word] >=
                descriptor_words[generation_stride_word] &&
            control[0] == descriptor_words[control_generation_word] -
                              descriptor_words[generation_stride_word] &&
            argument_count != 0u && local_count != 0u && local_count <= 32u &&
            stride != 0u && stride <= 4u && slot < stride &&
            step_count != 0u &&
            argument_owners.length() == argument_count &&
            original_words.length() == argument_count * 3u &&
            live_words.length() == argument_count * 3u &&
            turn <= (0xfffffffffffffffful - uint64_t(slot)) /
                        uint64_t(stride) &&
            coordinate == turn * uint64_t(stride) + uint64_t(slot);
    const uint64_t active_mask =
        local_count == 32u ? 0xfffffffful
                           : (uint64_t(1u) << local_count) - 1ul;
    valid = valid && ((read_mask | write_mask) != 0ul) &&
            ((read_mask | write_mask) & ~active_mask) == 0ul;
    for (uint local = 0u; valid && local < local_count; ++local) {
      const uint value = descriptor_words[local_word + local];
      valid = value < step_count;
      for (uint prior = 0u; valid && prior < local; ++prior) {
        valid = descriptor_words[local_word + prior] != value;
      }
    }
    structurally_valid = valid ? 1u : 0u;
    selected_mask = 0u;
    approved = 0u;
    descriptor_words[accepted_word] = 0u;
    descriptor_words[reason_word] = descriptor_words[invalid_reason_word];
    descriptor_words[observed_generation_word] =
        descriptor_words[published_word + descriptor_generation_word];
    descriptor_words[observed_generation_word + 1u] =
        descriptor_words[published_word + descriptor_generation_word + 1u];
  }
  barrier();

  const uint argument_count = descriptor_words[argument_count_word];
  const uint local_count = descriptor_words[local_count_word];
  if (structurally_valid != 0u) {
    for (uint argument = lane; argument < argument_count; argument += 256u) {
      const uint owner = argument_owners[argument];
      for (uint local = 0u; local < local_count; ++local) {
        if (descriptor_words[local_word + local] == owner) {
          atomicOr(selected_mask, 1u << local);
          break;
        }
      }
    }
  }
  barrier();

  if (lane == 0u) {
    const uint active_bits = local_count == 32u
                                 ? 0xffffffffu
                                 : ((1u << local_count) - 1u);
    if (structurally_valid != 0u && selected_mask == active_bits) {
      approved = 1u;
      descriptor_words[accepted_word] = 1u;
      descriptor_words[reason_word] = 0u;
    } else if (control[1] == 0u) {
      control[1] = descriptor_words[invalid_reason_word];
      control[2] = 0xffffffffu;
      control[3] = 0u;
    }
  }
  barrier();

  for (uint argument = lane; argument < argument_count; argument += 256u) {
    bool selected = false;
    const uint owner = argument_owners[argument];
    for (uint local = 0u; local < local_count; ++local) {
      selected = selected || descriptor_words[local_word + local] == owner;
    }
    const uint source = argument * 3u;
    live_words[source] =
        approved != 0u && selected ? original_words[source] : 0u;
    live_words[source + 1u] =
        approved != 0u && selected ? original_words[source + 1u] : 0u;
    live_words[source + 2u] =
        approved != 0u && selected ? original_words[source + 2u] : 0u;
  }
}
)GLSL";
}

#endif

} // namespace rund::node::accel::detail::vulkan_sliding_detail

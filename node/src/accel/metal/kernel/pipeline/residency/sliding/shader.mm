#include "internal.hpp"

namespace rund::node::accel::detail::metal_residency_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

namespace {

constexpr const char *SlidingGateSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint row_words = 58u;
constant uint published_word = 58u;
constant uint accepted_word = 116u;
constant uint reason_word = 117u;
constant uint observed_generation_word = 118u;
constant uint owner_word = 0u;
constant uint plan_word = 2u;
constant uint token_word = 4u;
constant uint run_word = 6u;
constant uint coordinate_word = 8u;
constant uint turn_word = 10u;
constant uint read_mask_word = 12u;
constant uint write_mask_word = 14u;
constant uint descriptor_generation_word = 16u;
constant uint generation_stride_word = 18u;
constant uint argument_count_word = 19u;
constant uint local_count_word = 20u;
constant uint stride_word = 21u;
constant uint slot_word = 22u;
constant uint invalid_reason_word = 23u;
constant uint step_count_word = 24u;
constant uint control_generation_word = 25u;
constant uint local_word = 26u;

inline ulong load_u64(device const uint *words, uint base) {
  return ulong(words[base]) | (ulong(words[base + 1u]) << 32u);
}

kernel void rund_metal_residency_sliding_gate(
    device uint *descriptor [[buffer(0)]],
    device uint *guard [[buffer(1)]],
    device uint *control [[buffer(2)]]) {
  bool valid = true;
  for (uint word = 0u; valid && word < row_words; ++word) {
    valid = descriptor[word] == descriptor[published_word + word];
  }
  const uint argument_count = descriptor[argument_count_word];
  const uint local_count = descriptor[local_count_word];
  const uint stride = descriptor[stride_word];
  const uint slot = descriptor[slot_word];
  const uint step_count = descriptor[step_count_word];
  const ulong owner = load_u64(descriptor, owner_word);
  const ulong plan = load_u64(descriptor, plan_word);
  const ulong token = load_u64(descriptor, token_word);
  const ulong run = load_u64(descriptor, run_word);
  const ulong coordinate = load_u64(descriptor, coordinate_word);
  const ulong turn = load_u64(descriptor, turn_word);
  const ulong read_mask = load_u64(descriptor, read_mask_word);
  const ulong write_mask = load_u64(descriptor, write_mask_word);
  const ulong descriptor_generation =
      load_u64(descriptor, descriptor_generation_word);
  valid = valid && owner != 0ul && plan != 0ul && token != 0ul &&
          run != 0ul && descriptor_generation != 0ul &&
          descriptor[generation_stride_word] != 0u &&
          descriptor[control_generation_word] >=
              descriptor[generation_stride_word] &&
          control[0] == descriptor[control_generation_word] -
                            descriptor[generation_stride_word] &&
          argument_count != 0u && argument_count == step_count &&
          local_count != 0u && local_count <= 32u && stride != 0u &&
          stride <= 4u && slot < stride && step_count != 0u &&
          turn <= (0xfffffffffffffffful - ulong(slot)) / ulong(stride) &&
          coordinate == turn * ulong(stride) + ulong(slot);
  const ulong active_mask =
      local_count == 32u ? 0xfffffffful : (1ul << local_count) - 1ul;
  valid = valid && (((read_mask | write_mask) & ~active_mask) == 0ul);
  for (uint local = 0u; valid && local < local_count; ++local) {
    const uint value = descriptor[local_word + local];
    valid = value < step_count;
    for (uint prior = 0u; valid && prior < local; ++prior) {
      valid = descriptor[local_word + prior] != value;
    }
  }
  descriptor[accepted_word] = valid ? 1u : 0u;
  descriptor[reason_word] = valid ? 0u : descriptor[invalid_reason_word];
  descriptor[observed_generation_word] =
      descriptor[published_word + descriptor_generation_word];
  descriptor[observed_generation_word + 1u] =
      descriptor[published_word + descriptor_generation_word + 1u];
  guard[0] = valid && ((read_mask | write_mask) != 0ul) ? 0u : 1u;
  if (!valid) {
    control[1] = descriptor[invalid_reason_word];
    control[2] = 0xffffffffu;
    control[3] = 0u;
  }
}
)MSL";

} // namespace

const char *GateSource() noexcept { return SlidingGateSource; }

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding

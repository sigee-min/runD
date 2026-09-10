#include "internal.hpp"

#include <rund/compute/reason.hpp>

#include <algorithm>
#include <atomic>
#include <limits>

namespace rund::node::accel::detail::metal_residency_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

namespace {

void Store64(std::array<std::uint32_t, MetalResidencySlidingRowWords> &row,
             const std::size_t word, const std::uint64_t value) noexcept {
  row[word] = static_cast<std::uint32_t>(value);
  row[word + 1u] = static_cast<std::uint32_t>(value >> 32u);
}

[[nodiscard]] bool SameRun(const MetalResidencySlidingGate &gate,
                           const BackendResidencySlidingDescriptor &row) {
  return gate.authenticated && gate.owner == row.owner &&
         gate.plan_identity == row.plan_identity && gate.token == row.token &&
         gate.run_generation == row.generation && gate.stride == row.stride &&
         gate.slot == row.slot;
}

[[nodiscard]] bool
ValidateDescriptor(const MetalSequence &sequence,
                   const MetalResidencySlidingGate &gate,
                   const BackendResidencySlidingDescriptor &descriptor,
                   const std::span<const std::uint32_t> locals) noexcept {
  if (descriptor.owner != &sequence || descriptor.plan_identity == 0u ||
      descriptor.token == 0u || descriptor.generation == 0u ||
      descriptor.descriptor_generation == 0u ||
      descriptor.control_generation == 0u || descriptor.stride == 0u ||
      descriptor.stride > ResidencySlidingCapacity ||
      descriptor.slot >= descriptor.stride || locals.empty() ||
      locals.size() > ResidencyWindowLocalCapacity ||
      locals.size() > sequence.residency_steps.size() ||
      descriptor.turn >
          (std::numeric_limits<std::uint64_t>::max() - descriptor.slot) /
              descriptor.stride ||
      descriptor.coordinate !=
          descriptor.turn * descriptor.stride + descriptor.slot) {
    return false;
  }
  const std::uint64_t active =
      locals.size() == std::numeric_limits<std::uint32_t>::digits
          ? std::numeric_limits<std::uint32_t>::max()
          : (std::uint64_t{1u} << locals.size()) - 1u;
  if (((descriptor.read_mask | descriptor.write_mask) & ~active) != 0u) {
    return false;
  }
  for (std::size_t local = 0u; local < locals.size(); ++local) {
    if (locals[local] >= sequence.residency_steps.size()) {
      return false;
    }
    for (std::size_t prior = 0u; prior < local; ++prior) {
      if (locals[prior] == locals[local]) {
        return false;
      }
    }
  }
  if (!gate.authenticated || !SameRun(gate, descriptor)) {
    return descriptor.turn == 0u && descriptor.coordinate == descriptor.slot;
  }
  return gate.last_turn != std::numeric_limits<std::uint64_t>::max() &&
         descriptor.turn == gate.last_turn + 1u &&
         gate.last_coordinate <=
             std::numeric_limits<std::uint64_t>::max() - descriptor.stride &&
         descriptor.coordinate == gate.last_coordinate + descriptor.stride &&
         descriptor.descriptor_generation > gate.last_descriptor_generation;
}

} // namespace

bool BuildPayload(MetalSequence &sequence, MetalResidencySlidingGate &gate,
                  const BackendResidencySlidingDescriptor &descriptor,
                  const std::span<const std::uint32_t> locals,
                  MetalResidencySlidingPayload &payload) noexcept {
  if (!ValidateDescriptor(sequence, gate, descriptor, locals) ||
      sequence.control_generation_stride != 1u ||
      sequence.residency_steps.empty() ||
      sequence.residency_steps.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  std::array<std::uint32_t, MetalResidencySlidingRowWords> expected{};
  Store64(expected, OwnerWord,
          static_cast<std::uint64_t>(
              reinterpret_cast<std::uintptr_t>(descriptor.owner)));
  Store64(expected, PlanWord, descriptor.plan_identity);
  Store64(expected, TokenWord, descriptor.token);
  Store64(expected, RunWord, descriptor.generation);
  Store64(expected, CoordinateWord, descriptor.coordinate);
  Store64(expected, TurnWord, descriptor.turn);
  Store64(expected, ReadMaskWord, descriptor.read_mask);
  Store64(expected, WriteMaskWord, descriptor.write_mask);
  Store64(expected, DescriptorGenerationWord, descriptor.descriptor_generation);
  expected[GenerationStrideWord] = sequence.control_generation_stride;
  expected[ArgumentCountWord] =
      static_cast<std::uint32_t>(sequence.residency_steps.size());
  expected[LocalCountWord] = static_cast<std::uint32_t>(locals.size());
  expected[StrideWord] = descriptor.stride;
  expected[SlotWord] = descriptor.slot;
  expected[InvalidReasonWord] =
      static_cast<std::uint32_t>(rund::compute::Reason::CompletionInvalid);
  expected[StepCountWord] =
      static_cast<std::uint32_t>(sequence.residency_steps.size());
  expected[ControlGenerationWord] = descriptor.control_generation;
  std::copy(locals.begin(), locals.end(),
            expected.begin() + MetalResidencySlidingLocalWord);
  std::copy(expected.begin(), expected.end(), payload.words.begin());
  const bool stale =
      gate.force_stale_once.exchange(false, std::memory_order_acq_rel);
  const auto &published = stale ? gate.published_row : expected;
  std::copy(published.begin(), published.end(),
            payload.words.begin() + MetalResidencySlidingPublishedWord);
  if (!stale) {
    gate.published_row = expected;
  }
  payload.words[MetalResidencySlidingAcceptedWord] = 0u;
  payload.words[MetalResidencySlidingReasonWord] = expected[InvalidReasonWord];
  payload.words[MetalResidencySlidingObservedGenerationWord] = 0u;
  payload.words[MetalResidencySlidingObservedGenerationWord + 1u] = 0u;
  return true;
}

void CommitDescriptor(MetalResidencySlidingGate &gate,
                      const BackendResidencySlidingDescriptor &row) noexcept {
  gate.owner = row.owner;
  gate.plan_identity = row.plan_identity;
  gate.token = row.token;
  gate.run_generation = row.generation;
  gate.last_coordinate = row.coordinate;
  gate.last_turn = row.turn;
  gate.last_descriptor_generation = row.descriptor_generation;
  gate.stride = row.stride;
  gate.slot = row.slot;
  gate.authenticated = true;
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding

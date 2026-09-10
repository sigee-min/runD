#include "../mode.hpp"
#include "../generated_indirect/map.hpp"
#include "../../../../buffer/create/memory.hpp"
#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail::vulkan_sliding_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void store64(std::array<std::uint32_t, VulkanResidencySlidingRowWords> &row,
             const std::size_t word, const std::uint64_t value) noexcept {
  row[word] = static_cast<std::uint32_t>(value);
  row[word + 1u] = static_cast<std::uint32_t>(value >> 32u);
}

[[nodiscard]] bool
same_run(const VulkanResidencySlidingGate &gate,
         const BackendResidencySlidingDescriptor &row) noexcept {
  return gate.authenticated && gate.owner == row.owner &&
         gate.plan_identity == row.plan_identity && gate.token == row.token &&
         gate.run_generation == row.generation && gate.stride == row.stride &&
         gate.slot == row.slot;
}

[[nodiscard]] bool
validate_descriptor(const VulkanPipeline &pipeline,
                    const VulkanResidencySlidingGate &gate,
                    const BackendResidencySlidingDescriptor &descriptor,
                    const std::span<const std::uint32_t> locals) noexcept {
  if (descriptor.owner != &pipeline || descriptor.plan_identity == 0u ||
      descriptor.token == 0u || descriptor.generation == 0u ||
      descriptor.descriptor_generation == 0u || descriptor.stride == 0u ||
      descriptor.control_generation == 0u ||
      descriptor.stride > ResidencySlidingCapacity ||
      descriptor.slot >= descriptor.stride || locals.empty() ||
      locals.size() > ResidencyWindowLocalCapacity ||
      locals.size() > pipeline.residency->steps.size() ||
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
  if ((descriptor.read_mask | descriptor.write_mask) == 0u ||
      ((descriptor.read_mask | descriptor.write_mask) & ~active) != 0u) {
    return false;
  }
  for (std::size_t local = 0u; local < locals.size(); ++local) {
    if (locals[local] >= pipeline.residency->steps.size()) {
      return false;
    }
    for (std::size_t prior = 0u; prior < local; ++prior) {
      if (locals[prior] == locals[local]) {
        return false;
      }
    }
  }
  if (!gate.authenticated || !same_run(gate, descriptor)) {
    return descriptor.turn == 0u && descriptor.coordinate == descriptor.slot;
  }
  return descriptor.slot == gate.last_coordinate % descriptor.stride &&
         gate.last_turn != std::numeric_limits<std::uint64_t>::max() &&
         descriptor.turn == gate.last_turn + 1u &&
         gate.last_coordinate <=
             std::numeric_limits<std::uint64_t>::max() - descriptor.stride &&
         descriptor.coordinate == gate.last_coordinate + descriptor.stride &&
         descriptor.descriptor_generation > gate.last_descriptor_generation;
}

} // namespace

bool coherent_storage(const VulkanBuffer &buffer) noexcept {
  return VulkanHostCoherentBufferReady(buffer) &&
         (buffer.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0u;
}

bool build_payload(VulkanPipeline &pipeline, VulkanResidencySlidingGate &gate,
                   const BackendResidencySlidingDescriptor &descriptor,
                   const std::span<const std::uint32_t> locals,
                   VulkanResidencySlidingPayload &payload) noexcept {
  if (pipeline.residency == nullptr ||
      !validate_descriptor(pipeline, gate, descriptor, locals) ||
      gate.descriptor.mapped == nullptr ||
      gate.descriptor.bytes < sizeof(payload) ||
      pipeline.control.generation_stride == 0u) {
    return false;
  }
  const auto access =
      vulkan_generated_indirect_detail::map_access(*pipeline.residency);
  if (!access.usable()) {
    return false;
  }
  const bool generated = access.selected;
  if (!generated && (gate.argument_owners.bytes /
                             static_cast<VkDeviceSize>(sizeof(std::uint32_t)) !=
                         pipeline.residency->argument_owners.size() ||
                     pipeline.residency->argument_owners.size() >
                         std::numeric_limits<std::uint32_t>::max())) {
    return false;
  }
  std::array<std::uint32_t, VulkanResidencySlidingRowWords> expected{};
  store64(expected, OwnerWord,
          static_cast<std::uint64_t>(
              reinterpret_cast<std::uintptr_t>(descriptor.owner)));
  store64(expected, PlanWord, descriptor.plan_identity);
  store64(expected, TokenWord, descriptor.token);
  store64(expected, RunWord, descriptor.generation);
  store64(expected, CoordinateWord, descriptor.coordinate);
  store64(expected, TurnWord, descriptor.turn);
  store64(expected, ReadMaskWord, descriptor.read_mask);
  store64(expected, WriteMaskWord, descriptor.write_mask);
  store64(expected, DescriptorGenerationWord, descriptor.descriptor_generation);
  expected[GenerationStrideWord] = pipeline.control.generation_stride;
  expected[ArgumentCountWord] =
      generated ? 0u
                : static_cast<std::uint32_t>(
                      pipeline.residency->argument_owners.size());
  expected[LocalCountWord] = static_cast<std::uint32_t>(locals.size());
  expected[StrideWord] = descriptor.stride;
  expected[SlotWord] = descriptor.slot;
  expected[InvalidReasonWord] =
      static_cast<std::uint32_t>(rund::compute::Reason::CompletionInvalid);
  expected[StepCountWord] =
      static_cast<std::uint32_t>(pipeline.residency->steps.size());
  expected[ControlGenerationWord] = descriptor.control_generation;
  std::copy(locals.begin(), locals.end(),
            expected.begin() + VulkanResidencySlidingLocalWord);
  std::copy(expected.begin(), expected.end(), payload.words.begin());
  const bool stale =
      gate.force_stale_once.exchange(false, std::memory_order_acq_rel);
  const auto &published = stale ? gate.published_row : expected;
  std::copy(published.begin(), published.end(),
            payload.words.begin() + VulkanResidencySlidingPublishedWord);
  if (!stale) {
    gate.published_row = expected;
  }
  payload.words[VulkanResidencySlidingAcceptedWord] = 0u;
  payload.words[VulkanResidencySlidingReasonWord] = expected[InvalidReasonWord];
  payload.words[VulkanResidencySlidingObservedGenerationWord] = 0u;
  payload.words[VulkanResidencySlidingObservedGenerationWord + 1u] = 0u;
  return true;
}

void commit_descriptor(VulkanResidencySlidingGate &gate,
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

} // namespace rund::node::accel::detail::vulkan_sliding_detail

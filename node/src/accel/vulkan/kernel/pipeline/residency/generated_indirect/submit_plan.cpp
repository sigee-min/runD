#include "internal.hpp"
#include "map.hpp"

#include <array>
#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace impl {

bool graph_active_rows(const std::span<const std::uint32_t> locals,
                       const std::size_t capacity,
                       const VulkanResidencyMode mode,
                       std::size_t &active_count,
                       std::size_t &row_count) noexcept {
  active_count = 0u;
  row_count = 0u;
  if (capacity == 0u || capacity > PreparedPipelineStepCapacity ||
      locals.empty() || locals.size() > capacity) {
    return false;
  }
  std::array<bool, PreparedPipelineStepCapacity> seen{};
  for (const std::uint32_t local : locals) {
    if (local >= capacity || seen[local]) {
      return false;
    }
    seen[local] = true;
  }
  active_count = locals.size();
  return active_rows(mode, active_count, row_count);
}

bool build_plan(const VulkanPipeline &pipeline,
                const std::span<const std::uint32_t> locals,
                VulkanResidencyGraphSubmitPlan &plan) noexcept {
  plan = {};
  const VulkanResidencySelection *const selection = pipeline.residency.get();
  if (selection != nullptr &&
      selection->mode == VulkanResidencyMode::GraphStageSequence) {
    std::size_t active_count = 0u;
    std::size_t row_count = 0u;
    if (!graph_active_rows(locals, selection->graph_sequence.local_count,
                           VulkanResidencyMode::GraphStageSequence,
                           active_count, row_count) ||
        selection->graph_sequence_command_count != 2u ||
        selection->graph_generated_phase !=
            VulkanResidencyGraphGeneratedPhase::Recorded ||
        selection->graph_sequence_submitted_count != 0u ||
        !sequence_proof_matches(pipeline, *selection)) {
      return false;
    }
    std::array<bool, PreparedPipelineStepCapacity> seen{};
    std::array<VkCommandBuffer, PreparedPipelineStepCapacity> commands{};
    std::size_t count = 0u;
    if (selection->prefix.buffer == VK_NULL_HANDLE ||
        selection->suffix.buffer == VK_NULL_HANDLE) {
      return false;
    }
    for (const std::uint32_t local : locals) {
      if (local >= 2u || seen[local] ||
          selection->graph_sequence_frames[local].submitted ||
          !selection->graph_sequence_frames[local].command_ready ||
          selection->graph_sequence_frames[local].command.buffer ==
              VK_NULL_HANDLE) {
        return false;
      }
      seen[local] = true;
      commands[count++] =
          selection->graph_sequence_frames[local].command.buffer;
    }
    if (count != active_count) {
      return false;
    }
    const auto distinct_command = [&](const VkCommandBuffer value,
                                      const std::size_t prior_count) noexcept {
      if (value == VK_NULL_HANDLE || value == selection->prefix.buffer ||
          value == selection->suffix.buffer) {
        return false;
      }
      for (std::size_t index = 0u; index < prior_count; ++index) {
        if (commands[index] == value) {
          return false;
        }
      }
      return true;
    };
    if (selection->prefix.buffer == selection->suffix.buffer) {
      return false;
    }
    for (std::size_t index = 0u; index < count; ++index) {
      if (!distinct_command(commands[index], index)) {
        return false;
      }
    }
    std::uint64_t next_submit_seq{};
    std::uint64_t next_submit_total{};
    if (!rund::kernel::checked::add(selection->submit_seq, 1u,
                                    next_submit_seq) ||
        !rund::kernel::checked::add(selection->submit_total, row_count,
                                    next_submit_total)) {
      return false;
    }
    plan.commands[0u] = selection->prefix.buffer;
    for (std::size_t index = 0u; index < count; ++index) {
      plan.commands[index + 1u] = commands[index];
    }
    plan.commands[count + 1u] = selection->suffix.buffer;
    for (const std::uint32_t local : locals) {
      plan.local_mask[local] = true;
    }
    plan.command_count = count + 2u;
    plan.active_count = active_count;
    plan.next_submit_seq = next_submit_seq;
    plan.next_submit_total = next_submit_total;
    plan.valid = true;
    return true;
  }
  if (selection == nullptr ||
      selection->mode != VulkanResidencyMode::GraphStageGeneratedIndirect ||
      selection->graph_generated.local_count == 0u ||
      selection->graph_generated_command_count !=
          selection->graph_generated.local_count ||
      selection->graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::Recorded ||
      selection->graph_generated_generation == 0u ||
      selection->graph_generated_submitted_count != 0u) {
    return false;
  }
  std::size_t active_count = 0u;
  std::size_t row_count = 0u;
  if (!graph_active_rows(locals, selection->graph_generated.local_count,
                         VulkanResidencyMode::GraphStageGeneratedIndirect,
                         active_count, row_count)) {
    return false;
  }
  if (selection->prefix.buffer == VK_NULL_HANDLE ||
      selection->suffix.buffer == VK_NULL_HANDLE ||
      selection->prefix.buffer == selection->suffix.buffer) {
    return false;
  }
  std::array<VkCommandBuffer, PreparedPipelineStepCapacity> temporary{};
  std::array<bool, PreparedPipelineStepCapacity> seen{};
  std::size_t temporary_count = 0u;
  const auto expected_generation = [&](const std::uint32_t local,
                                       std::uint64_t &value) noexcept {
    return rund::kernel::checked::add(static_cast<std::uint64_t>(local), 1u,
                                      value);
  };
  for (const std::uint32_t local : locals) {
    std::uint64_t expected{};
    if (local >= selection->graph_generated.local_count ||
        !selection->graph_generated_frames[local].ready ||
        selection->graph_generated_frames[local].command.buffer ==
            VK_NULL_HANDLE ||
        selection->graph_generated_frames[local].quarantined ||
        selection->graph_generated_frames[local].submitted ||
        selection->graph_generated_frames[local].generation == 0u ||
        !expected_generation(local, expected) ||
        selection->graph_generated_frames[local].generation != expected ||
        selection->graph_generated_frames[local]
                .expected_descriptor_generation != expected) {
      return false;
    }
    if (seen[local]) {
      return false;
    }
    seen[local] = true;
    const VkCommandBuffer frame_command =
        selection->graph_generated_frames[local].command.buffer;
    if (frame_command == selection->prefix.buffer ||
        frame_command == selection->suffix.buffer) {
      return false;
    }
    for (std::size_t prior = 0u; prior < temporary_count; ++prior) {
      if (temporary[prior] == frame_command ||
          temporary[prior] == selection->prefix.buffer ||
          temporary[prior] == selection->suffix.buffer) {
        return false;
      }
    }
    temporary[temporary_count++] =
        selection->graph_generated_frames[local].command.buffer;
  }
  std::uint64_t next_submit_seq{};
  std::uint64_t next_submit_total{};
  if (!rund::kernel::checked::add(selection->submit_seq, 1u, next_submit_seq) ||
      !rund::kernel::checked::add(selection->submit_total,
                                  static_cast<std::uint64_t>(row_count),
                                  next_submit_total)) {
    return false;
  }
  plan.commands[0u] = selection->prefix.buffer;
  for (std::size_t index = 0u; index < temporary_count; ++index) {
    plan.commands[index + 1u] = temporary[index];
  }
  plan.commands[temporary_count + 1u] = selection->suffix.buffer;
  for (const std::uint32_t local : locals) {
    plan.local_mask[local] = true;
  }
  plan.command_count = temporary_count + 2u;
  plan.active_count = active_count;
  plan.next_submit_seq = next_submit_seq;
  plan.next_submit_total = next_submit_total;
  plan.valid = true;
  return true;
}

} // namespace impl

rund::AccelCheck
build_submit_plan(const VulkanPipeline &pipeline,
                  const std::span<const std::uint32_t> locals,
                  VulkanResidencyGraphSubmitPlan &plan) noexcept {
  plan = {};
  const VulkanResidencySelection *const selection = pipeline.residency.get();
  if (selection == nullptr ||
      (selection->mode != VulkanResidencyMode::GraphStageGeneratedIndirect &&
       selection->mode != VulkanResidencyMode::GraphStageSequence) ||
      !selection->ready ||
      selection->quarantined.load(std::memory_order_acquire)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  const bool proof_ok =
      selection->mode == VulkanResidencyMode::GraphStageGeneratedIndirect
          ? impl::graph_proof_matches(pipeline, *selection)
          : impl::sequence_proof_matches(pipeline, *selection);
  if (!proof_ok || !impl::build_plan(pipeline, locals, plan)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  return {true, "ok"};
}

rund::AccelCheck select(const VulkanPipeline &pipeline,
                        const std::span<const std::uint32_t> locals,
                        const std::span<VkCommandBuffer> commands,
                        std::size_t &command_count,
                        std::uint64_t &dispatch_count,
                        std::uint64_t &control_count, Plan &plan) noexcept {
  command_count = 0u;
  dispatch_count = 0u;
  control_count = 0u;
  plan = {};
  const VulkanResidencySelection *const selection = pipeline.residency.get();
  if (selection == nullptr) {
    return {true, "ok"};
  }
  const auto map = map_access(*selection);
  if (map.selected || (!map.valid && !map.graph)) {
    return {false, "accel_vulkan_pipeline_selection_unavailable"};
  }
  if (!vulkan_residency_detail::owns(selection->mode)) {
    return {true, "ok"};
  }
  plan.generated = true;
  const rund::AccelCheck built =
      build_submit_plan(pipeline, locals, plan.graph);
  if (!built.ok) {
    return built;
  }
  std::size_t units = 0u;
  if (!impl::active_rows(selection->mode, plan.graph.active_count, units) ||
      commands.size() < plan.graph.command_count) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  command_count = plan.graph.command_count;
  for (std::size_t index = 0u; index < command_count; ++index) {
    commands[index] = plan.graph.commands[index];
  }
  dispatch_count = units;
  control_count = units;
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail

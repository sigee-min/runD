#include "../../../buffer/access.hpp"
#include "../../../buffer/create.hpp"

#include "local.hpp"
#include "mode.hpp"
#include "graph_direct.hpp"
#include "plan.hpp"

#include "../../../command/capture.hpp"
#include "generated_indirect/internal.hpp"
#include "generated_indirect/map.hpp"

#include "../prepare/record.hpp"

#include "../../../../kernel/recurrence.hpp"
#include "../../../command/resources.hpp"
#include "../../../map/api.hpp"
#include "../../../map/local.hpp"
#include "../../ops/table.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
CreateSelectionCommand(VulkanPipeline &pipeline, VulkanCommand &command,
                       const VulkanPipelineRecordSlice slice) noexcept {
  const rund::AccelCheck created = CreateCommand(
      pipeline.adapter->device, pipeline.adapter->compute_queue_family, command,
      CommandKind::ReusablePrimary);
  if (!created.ok) {
    return created;
  }
  return RecordVulkanPipeline(pipeline, command, CommandKind::ReusablePrimary,
                              false, nullptr, nullptr, &slice);
}

[[nodiscard]] rund::AccelCheck
PrepareSelectionArguments(VulkanPipeline &pipeline,
                          VulkanResidencySelection &selection,
                          VulkanDispatchCapture &capture) noexcept {
  std::uint64_t dispatches = 0u;
  for (std::size_t index = 0u; index < pipeline.record->entries.size();
       ++index) {
    VulkanKernelResources *resources = nullptr;
    std::uint64_t direct = 0u;
    std::uint64_t indirect = 0u;
    if (!SelectionEntry(pipeline, index, resources)) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    const rund::AccelCheck described =
        DescribeVulkanRouteDispatches(*resources, direct, indirect);
    if (!described.ok) {
      return described;
    }
    // The core Vulkan 1.1 path gates every direct dispatch through one
    // retained indirect arena. A primitive whose own indirect count is
    // device-generated requires a distinct copy/gate recipe and is not
    // admitted by this exact suppression owner.
    if (indirect != 0u ||
        direct > std::numeric_limits<std::uint64_t>::max() - dispatches) {
      return rund::AccelCheck{false,
                              "accel_vulkan_pipeline_selection_unavailable"};
    }
    dispatches += direct;
  }
  if (dispatches == 0u ||
      dispatches > std::numeric_limits<std::size_t>::max() ||
      dispatches > std::numeric_limits<VkDeviceSize>::max() /
                       sizeof(VkDispatchIndirectCommand)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  try {
    selection.original_arguments.resize(static_cast<std::size_t>(dispatches));
    selection.argument_owners.resize(static_cast<std::size_t>(dispatches));
  } catch (...) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const VkDeviceSize bytes =
      static_cast<VkDeviceSize>(dispatches) * sizeof(VkDispatchIndirectCommand);
  if (!CreateVulkanBuffer(*pipeline.adapter, bytes,
                          VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          selection.arguments) ||
      selection.arguments.mapped == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  std::fill(selection.argument_owners.begin(), selection.argument_owners.end(),
            std::numeric_limits<std::uint32_t>::max());
  capture = VulkanDispatchCapture{
      .arguments = selection.arguments.buffer,
      .mapped =
          static_cast<VkDispatchIndirectCommand *>(selection.arguments.mapped),
      .original = selection.original_arguments.data(),
      .owners = selection.argument_owners.data(),
      .capacity = selection.original_arguments.size(),
  };
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck
PrepareVulkanPipelineResidency(VulkanPipeline &pipeline) noexcept {
  const rund::AccelCheck generation =
      MintResidencyPreparationGeneration(pipeline);
  if (!generation.ok) {
    return generation;
  }
  ResidencyPlan plan{};
  const rund::AccelCheck admission = BuildResidencyPlan(pipeline, plan);
  if (!admission.ok) {
    return rund::AccelCheck{true, "ok"};
  }
  const VulkanResidencyMode mode =
      vulkan_residency_detail::mode(plan.candidate);
  const bool bounded = plan.bounded;
  const bool generated = vulkan_residency_detail::owns(mode);
  if (pipeline.residency != nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_residency_owner_duplicate"};
  }
  std::shared_ptr<VulkanResidencySelection> selection;
  try {
    selection = std::make_shared<VulkanResidencySelection>();
    selection->steps.reserve(pipeline.record->entries.size());
    selection->steps.resize(pipeline.record->entries.size());
  } catch (...) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  selection->adapter = pipeline.adapter;
  selection->mode = mode;
  const VulkanResidencyAdmissionCandidate final_candidate = plan.candidate;
  selection->graph_direct = plan.graph_direct;
  selection->graph_generated = plan.graph_generated;
  selection->graph_sequence = plan.graph_sequence;
  selection->bounded = bounded;
  VulkanDispatchCapture capture{};
  rund::AccelCheck recorded =
      bounded && !generated
          ? PrepareSelectionArguments(pipeline, *selection, capture)
          : rund::AccelCheck{true, "ok"};
  if (!recorded.ok) {
    return recorded;
  }
  if (generated) {
    recorded = CreateSelectionCommand(
        pipeline, selection->prefix,
        VulkanPipelineRecordSlice{
            .first = 0u, .count = 0u, .open = true, .close = false});
  } else {
    recorded = CreateSelectionCommand(pipeline, selection->prefix,
                                      VulkanPipelineRecordSlice{.open = true});
  }
  for (std::size_t index = 0u;
       recorded.ok && !generated && index < selection->steps.size(); ++index) {
    VulkanKernelResources *resources = nullptr;
    std::uint64_t control_count = 0u;
    if (!SelectionEntry(pipeline, index, resources) ||
        !StepControlCount(pipeline, index, control_count)) {
      recorded = rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      break;
    }
    VulkanResidencyStep &step = selection->steps[index];
    step.declared_step = static_cast<std::uint32_t>(index);
    step.dispatch_count = resources->dispatch_count;
    step.control_count = control_count;
    step.reset_count = resources->reset_count;
    step.reset_bytes = resources->reset_bytes;
    recorded = CreateSelectionCommand(
        pipeline, step.command,
        VulkanPipelineRecordSlice{
            .first = index,
            .count = 1u,
            .capture = bounded ? &capture : nullptr,
            .capture_owner = static_cast<std::uint32_t>(index),
        });
  }
  if (recorded.ok && !generated) {
    recorded = CreateSelectionCommand(
        pipeline, selection->suffix,
        VulkanPipelineRecordSlice{.first = pipeline.record->entries.size(),
                                  .close = true});
  }
  if (recorded.ok && generated) {
    recorded = vulkan_generated_indirect_detail::prepare(pipeline, *selection);
    if (recorded.ok) {
      std::array<std::uint32_t, PreparedPipelineStepCapacity> all_locals{};
      for (std::size_t index = 0u; index < selection->steps.size(); ++index) {
        all_locals[index] = static_cast<std::uint32_t>(index);
      }
      recorded = vulkan_generated_indirect_detail::record(
                     pipeline, *selection,
                     std::span<const std::uint32_t>{all_locals.data(),
                                                    selection->steps.size()})
                     ? rund::AccelCheck{true, "ok"}
                     : rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      if (recorded.ok) {
        recorded = CreateSelectionCommand(
            pipeline, selection->suffix,
            VulkanPipelineRecordSlice{.first = pipeline.record->entries.size(),
                                      .count = 0u,
                                      .open = false,
                                      .close = true});
      }
    }
  }
  if (!recorded.ok) {
    return recorded;
  }
  if (bounded && !generated &&
      (capture.failed || capture.cursor != capture.capacity ||
       capture.indirect_count != 0u ||
       !UploadVulkanBuffer(
           selection->arguments, selection->original_arguments.data(),
           static_cast<VkDeviceSize>(selection->original_arguments.size() *
                                     sizeof(VkDispatchIndirectCommand))))) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  const auto map = vulkan_generated_indirect_detail::map_access(*selection);
  if (!generated && !map.recognized) {
    return rund::AccelCheck{false,
                            "accel_vulkan_pipeline_selection_unavailable"};
  }
  if (map.direct || map.selected) {
    const rund::AccelCheck sliding =
        PrepareVulkanResidencySlidingGate(pipeline, *selection);
    if (!sliding.ok) {
      if (map.selected) {
        return sliding;
      }
      DestroyVulkanResidencySlidingGate(*selection);
    }
  }
  selection->host_bytes = ::rund::detail::counter::SaturatingAdd(
      sizeof(VulkanResidencySelection),
      ::rund::detail::counter::SaturatingMultiply(
          static_cast<std::uint64_t>(selection->steps.capacity()),
          sizeof(VulkanResidencyStep)));
  selection->host_bytes = ::rund::detail::counter::SaturatingAdd(
      selection->host_bytes,
      ::rund::detail::counter::SaturatingMultiply(
          static_cast<std::uint64_t>(selection->original_arguments.capacity()),
          sizeof(VkDispatchIndirectCommand)));
  selection->host_bytes = ::rund::detail::counter::SaturatingAdd(
      selection->host_bytes,
      ::rund::detail::counter::SaturatingMultiply(
          static_cast<std::uint64_t>(selection->argument_owners.capacity()),
          sizeof(std::uint32_t)));
  selection->host_bytes = ::rund::detail::counter::SaturatingAdd(
      selection->host_bytes,
      ::rund::detail::counter::SaturatingMultiply(
          static_cast<std::uint64_t>(
              selection->sliding.descriptor_leases.capacity()),
          sizeof(VulkanCollectiveDescriptorLease)));
  if (selection->host_bytes == std::numeric_limits<std::uint64_t>::max()) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  selection->ready = true;
  pipeline.residency = std::move(selection);
  pipeline.residency_admission.final_candidate = final_candidate;
  pipeline.residency_admission.selected_mode = pipeline.residency->mode;
  pipeline.residency_admission.final_selected = true;
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail

#include "state.hpp"

#include "prepare/capacity.hpp"
#include "prepare/context.hpp"
#include "residency/local.hpp"

#include "../../../kernel/backend/exception.hpp"
#include "../../../kernel/backend/pipeline/failure.hpp"

#include <rund/counter.hpp>

#include <limits>
#include <mutex>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] rund::AccelCheck PrepareVulkanPipelineImpl(
    const std::span<const BackendBatchEntry> templates,
    const std::span<const BackendBatchEntry> entries,
    const std::span<const std::uint8_t> barriers,
    const std::span<const TileTransducer> transducers,
    const std::span<const NestedAggregate>,
    const std::span<const BackendPublish> publications,
    PreparedKernelTemplateRegistry &registry,
    PreparedPipelineStatusLayout &status, const bool profile_steps,
    PreparedPipelineMemoryMeter *const memory_meter,
    std::shared_ptr<void> &prepared, PreparedPipelineMemory &memory,
    PreparedPipelineFailureContext &failure_context) {
  prepared.reset();
  memory = {};
  failure_context.stage(PreparedPipelineFailureStage::BackendAdmission);
  if (templates.empty() || entries.empty() ||
      entries.size() != barriers.size() ||
      templates.size() != status.active_step_count ||
      entries.size() != status.command_count) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    failure_context.occurrence_route(entries[index]);
    if (entries[index].occurrence_index != index ||
        entries[index].template_index >= templates.size() ||
        (entries[index].transducer != NoTileTransducer &&
         entries[index].transducer >= transducers.size()) ||
        entries[index].run != templates[entries[index].template_index].run ||
        entries[index].prepared !=
            templates[entries[index].template_index].prepared) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  }
  failure_context.clear_route();

  VulkanPipelinePreparation preparation{
      .templates = templates,
      .entries = entries,
      .barriers = barriers,
      .transducers = transducers,
      .publications = publications,
      .registry = &registry,
      .status = &status,
      .profile_steps = profile_steps,
      .memory_meter = memory_meter,
      .prepared = &prepared,
      .memory = &memory,
      .failure_context = &failure_context,
      .recurrence = BuildMapRecurrence(entries, barriers),
  };
  if (preparation.recurrence.invalid()) {
    return rund::AccelCheck{false, preparation.recurrence.reason};
  }
  if (preparation.recurrence.ready() && !transducers.empty()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }

  if (!preparation.recurrence.ready()) {
    const rund::AccelCheck described = DescribeVulkanPipelineCapacity(
        templates, preparation.description_capacity);
    if (!described.ok) {
      return described;
    }
    const PreparedKernelPipelineReservation &limit = registry.limit;
    if (!limit.ok ||
        preparation.description_capacity.step_count >
            limit.backend_step_description_count ||
        preparation.description_capacity.status_source_count >
            limit.backend_status_source_count ||
        preparation.description_capacity.status_entry_count >
            limit.backend_status_entry_count ||
        preparation.description_capacity.telemetry_source_count >
            limit.backend_telemetry_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }

  rund::AccelCheck ready =
      PrepareVulkanPipelineAllocation(preparation);
  if (!ready.ok) {
    return ready;
  }
  ready = PrepareVulkanPipelineDescriptions(preparation);
  if (!ready.ok) {
    return ready;
  }
  ready = PrepareVulkanPipelineOccurrences(preparation);
  if (!ready.ok) {
    return ready;
  }

  std::shared_ptr<VulkanPipeline> &pipeline = preparation.pipeline;
  failure_context.stage(PreparedPipelineFailureStage::BackendAllocation);
  std::scoped_lock lock{pipeline->submission.mutex, pipeline->adapter->mutex};
  if (pipeline->dispatch_count == 0u) {
    const std::uint64_t bytes =
        sizeof(VulkanPipeline) +
        (pipeline->profile == nullptr ? 0u : sizeof(VulkanPipelineProfile));
    memory.host = PreparedMemory{
        .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
    prepared = std::static_pointer_cast<void>(pipeline);
    return rund::AccelCheck{true, "ok"};
  }

  for (std::size_t index = 0u; index < preparation.canonical.size(); ++index) {
    const VulkanPipelineCanonicalStatus &source =
        preparation.canonical[index];
    if (source.active_program >= status.active_step_count) {
      // An out-of-range owner is malformed input, not a real template
      // coordinate. Keep all route evidence unknown.
      failure_context.clear_route();
      return FailVulkanPipeline(pipeline,
                                "accel_kernel_primitive_unsupported");
    }
    failure_context.template_node_route(source.active_program,
                                        source.source_node);
    if (source.source_node == PreparedPipelineNoStep ||
        source.source.raw == nullptr ||
        source.source.raw->buffer == VK_NULL_HANDLE) {
      return FailVulkanPipeline(pipeline,
                                "accel_kernel_primitive_unsupported");
    }
  }
  failure_context.clear_route();

  ready = PrepareVulkanPipelinePublication(preparation);
  if (!ready.ok) {
    return ready;
  }
  ready = CaptureVulkanPipeline(preparation);
  if (!ready.ok) {
    return ready;
  }
  ready = PrepareVulkanPipelineResidencyMemory(preparation);
  if (!ready.ok) {
    return ready;
  }

  std::uint64_t bytes =
      sizeof(VulkanPipeline) +
      (pipeline->profile == nullptr ? 0u : sizeof(VulkanPipelineProfile)) +
      pipeline->control.descriptor_leases.capacity() *
          sizeof(VulkanCollectiveDescriptorLease);
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, VulkanRecurrenceHostBytes(*pipeline));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, VulkanPipelinePublishHostBytes(pipeline->publish));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, VulkanWindowHostBytes(pipeline->window));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, VulkanPipelineRecordHostBytes(*pipeline->record));
  if (pipeline->residency != nullptr) {
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, pipeline->residency->host_bytes);
  }
  if (bytes == std::numeric_limits<std::uint64_t>::max()) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  memory.host = PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
  prepared = std::static_pointer_cast<void>(pipeline);
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck PrepareVulkanPipeline(
    const std::span<const BackendBatchEntry> templates,
    const std::span<const BackendBatchEntry> entries,
    const std::span<const std::uint8_t> barriers,
    const std::span<const TileTransducer> transducers,
    const std::span<const NestedAggregate> aggregates,
    const std::span<const BackendPublish> publications,
    PreparedKernelTemplateRegistry &registry,
    PreparedPipelineStatusLayout &status, const bool profile_steps,
    std::shared_ptr<void> &prepared, PreparedPipelineMemory &memory,
    PreparedPipelineMemoryMeter *const memory_meter,
    rund::AccelRunFacts &preparation, PreparedPipelineFailure &failure) {
  PreparedPipelineFailureContext failure_context{};
  failure = {};
  preparation = {};
  preparation.preparation_evidence =
      rund::AccelPreparationEvidenceSource::BackendGlobalOnly;
  rund::AccelCheck result{};
  try {
    result = PrepareVulkanPipelineImpl(
        templates, entries, barriers, transducers, aggregates, publications,
        registry, status, profile_steps, memory_meter, prepared, memory,
        failure_context);
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    result = rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!result.ok) {
    failure = failure_context.failure(result.reason);
  }
  return result;
}

#endif

} // namespace rund::node::accel::detail

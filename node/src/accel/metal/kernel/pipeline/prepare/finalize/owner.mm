#include "internal.hpp"

#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck FinalizeMetalOwner(
    MetalPipelineBuild &build,
    const MetalPipelineFinalizeProjection &projection,
    std::shared_ptr<void> &prepared, PreparedPipelineMemory &memory) {
  std::uint64_t recurrence_host_bytes = 0u;
  std::uint64_t recurrence_device_bytes = 0u;
  std::uint64_t recurrence_reused_bytes = 0u;
  const auto account_recurrence = [&](const std::shared_ptr<void> &resource) {
    const auto *const map =
        static_cast<const MetalMapEncodeResources *>(resource.get());
    if (map == nullptr || map->prepared == nullptr) {
      return;
    }
    std::uint64_t host = sizeof(MetalMapEncodeResources);
    for (const std::uint64_t bytes : {
             static_cast<std::uint64_t>(map->windows.capacity()) *
                 sizeof(rund::kernel::ComputeDispatchWindow),
             static_cast<std::uint64_t>(
                 map->resident.overflow_inputs.capacity()) *
                 sizeof(MetalResidentBufferResult),
             static_cast<std::uint64_t>(
                 map->resident.overflow_outputs.capacity()) *
                 sizeof(MetalResidentBufferResult)}) {
      host = ::rund::detail::counter::SaturatingAdd(host, bytes);
    }
    // Immutable Map template vectors and native pipelines belong to the
    // pipeline-global registry and are observed there exactly once across
    // primary/alternate streams. A history proof, in contrast, is route-owned
    // and remains live through this encoded recurrence group.
    if (map->binding_owner != nullptr) {
      host = ::rund::detail::counter::SaturatingAdd(
          host, sizeof(MapRecurrenceHistory));
    }
    recurrence_host_bytes =
        ::rund::detail::counter::SaturatingAdd(recurrence_host_bytes, host);
    recurrence_device_bytes = ::rund::detail::counter::SaturatingAdd(
        recurrence_device_bytes, map->param.bytes);
    if (map->param.reused) {
      recurrence_reused_bytes = ::rund::detail::counter::SaturatingAdd(
          recurrence_reused_bytes, map->param.bytes);
    }
  };
  account_recurrence(build.pipeline->recurrence);
  for (const std::shared_ptr<void> &resource : build.pipeline->transducers) {
    account_recurrence(resource);
  }
  const std::uint64_t host_bytes =
      sizeof(MetalSequence) +
      build.pipeline->spatial_window.locals.capacity() *
          sizeof(MetalSpatialWindowLocalProof) +
      build.pipeline->command_chunks.capacity() * sizeof(MetalIcbChunk) +
      build.pipeline->residency_steps.capacity() *
          sizeof(MetalResidencyStepRange) +
      projection.native_window_capacity * sizeof(MetalWindow) +
      build.pipeline->trace_commands.capacity() * sizeof(std::uint64_t) +
      build.pipeline->residency.capacity() * sizeof(id<MTLResource>) +
      build.pipeline->pipelines.capacity() *
          sizeof(id<MTLComputePipelineState>) +
      build.pipeline->telemetry.capacity() *
          sizeof(MetalPipelineTelemetryRecord) +
      build.pipeline->step_evidence.capacity() *
          sizeof(PreparedPipelineStepEvidence) +
      build.pipeline->transducers.capacity() * sizeof(std::shared_ptr<void>) +
      recurrence_host_bytes;
  const std::uint64_t device_bytes =
      projection.icb_device_bytes +
      static_cast<std::uint64_t>([build.pipeline->parameters allocatedSize]) +
      static_cast<std::uint64_t>([build.pipeline->raw_status allocatedSize]) +
      static_cast<std::uint64_t>([build.pipeline->control allocatedSize]) +
      static_cast<std::uint64_t>([build.pipeline->states allocatedSize]) +
      static_cast<std::uint64_t>([build.pipeline->guard_zero allocatedSize]) +
      static_cast<std::uint64_t>(
          [build.pipeline->step_control allocatedSize]) +
      recurrence_device_bytes;
  if (build.profile_steps) {
    build.pipeline->instrumentation_byte_count = static_cast<std::uint64_t>(
        [build.pipeline->step_control allocatedSize]);
  }
  build.pipeline->retained_bytes = host_bytes + device_bytes;
  std::uint64_t cold_workspace_bytes = projection.identity_index_bytes;
  const auto account_workspace = [&](const std::uint64_t count,
                                     const std::uint64_t element) {
    cold_workspace_bytes = ::rund::detail::counter::SaturatingAdd(
        cold_workspace_bytes,
        ::rund::detail::counter::SaturatingMultiply(count, element));
  };
  account_workspace(projection.native_window_capacity, sizeof(MetalWindow));
  account_workspace(build.status_bindings.capacity(),
                    sizeof(MetalPipelineStatusBindingRecord));
  account_workspace(build.status_sources.capacity(),
                    sizeof(MetalPipelineStatusSourceMeta));
  account_workspace(build.status_entries.capacity(),
                    sizeof(MetalPipelineStatusEntryMeta));
  account_workspace(build.status_resets.capacity(),
                    sizeof(MetalPipelineResetMeta));
  account_workspace(build.telemetry_steps.capacity(),
                    sizeof(PreparedProgramStatusSlice));
  account_workspace(build.captured.parameters.capacity(), sizeof(std::byte));
  account_workspace(build.captured.command_bindings.capacity(),
                    sizeof(MetalCommandBinding));
  account_workspace(build.captured.commands.capacity(), sizeof(MetalCommand));
  const std::uint64_t host_peak =
      ::rund::detail::counter::SaturatingAdd(host_bytes, cold_workspace_bytes);
  memory.host = PreparedMemory{.current = host_bytes,
                               .peak = host_peak,
                               .cumulative = host_peak,
                               .budget = host_peak};
  memory.device = PreparedMemory{.current = device_bytes,
                                 .peak = device_bytes,
                                 .cumulative = device_bytes,
                                 .reused = recurrence_reused_bytes,
                                 .budget = device_bytes};
  prepared = std::static_pointer_cast<void>(build.pipeline);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail

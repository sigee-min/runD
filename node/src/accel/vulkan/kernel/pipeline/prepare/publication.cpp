#include "../../../adapter/error.hpp"

#include "context.hpp"
#include "../telemetry.hpp"
#include "../../../../kernel/backend/pipeline/failure.hpp"
#include <kernel/core/checked.hpp>
#include <limits>
#include <rund/counter.hpp>
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanPipelinePublication(VulkanPipelinePreparation &preparation) {
  auto &pipeline = preparation.pipeline;
  auto &canonical = preparation.canonical;
  auto &recurrence_staging = preparation.recurrence_staging;
  auto &transducer_staging = preparation.transducer_staging;
  auto &window_dispatches = preparation.window_dispatches;
  auto &window_gate_count = preparation.window_gate_count;
  auto &status_command_sources = preparation.status_command_sources;
  auto &telemetry_command_count = preparation.telemetry_command_count;
  auto &encoded_work_command_count = preparation.encoded_work_command_count;
  const bool profile_steps = preparation.profile_steps;
  const auto &entries = preparation.entries;
  const auto &publications = preparation.publications;
  auto &registry = *preparation.registry;
  auto &status = *preparation.status;
  auto &memory = *preparation.memory;
  const rund::AccelCheck control_ready = PrepareVulkanPipelineControl(
      *pipeline->adapter, canonical, status, pipeline->telemetry.size(),
      profile_steps, pipeline->control, memory);
  if (!control_ready.ok) {
    return FailVulkanPipeline(pipeline, control_ready.reason);
  }
  if (status_command_sources >
      (std::numeric_limits<std::uint64_t>::max() - 2u) / 2u) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  pipeline->control.command_count = 2u + 2u * status_command_sources;
  const rund::AccelCheck window_ready = PrepareVulkanWindow(
      *pipeline->adapter, entries, window_dispatches, window_gate_count, status,
      pipeline->control.summary, pipeline->window);
  if (!window_ready.ok) {
    return FailVulkanPipeline(pipeline, window_ready.reason);
  }
  const std::uint64_t window_bytes = ::rund::detail::counter::SaturatingAdd(
      pipeline->window.states.bytes,
      ::rund::detail::counter::SaturatingAdd(
          pipeline->window.arguments.bytes,
          ::rund::detail::counter::SaturatingAdd(
              pipeline->window.original_arguments.bytes,
              pipeline->window.owners.bytes)));
  accumulate_memory(memory.staging, PreparedMemory{.current = window_bytes,
                                                   .peak = window_bytes,
                                                   .cumulative = window_bytes,
                                                   .budget = window_bytes});
  const rund::AccelCheck publish_ready = PrepareVulkanPipelinePublish(
      *pipeline->adapter, publications, status, pipeline->control,
      pipeline->window, pipeline->publish);
  if (!publish_ready.ok) {
    return FailVulkanPipeline(pipeline, publish_ready.reason);
  }
  std::uint64_t seed_preflight_count = 0u;
  std::uint64_t canonicalize_count = 0u;
  std::uint64_t terminal_publish_count = 0u;
  std::uint64_t window_publish_count = 0u;
  std::uint64_t window_transition_count = 0u;
  for (const VulkanPipelinePublishRoute &publication :
       pipeline->publish.routes) {
    terminal_publish_count = ::rund::detail::counter::SaturatingAdd(
        terminal_publish_count,
        static_cast<std::uint64_t>(
            publication.params.kind ==
            static_cast<std::uint32_t>(
                PreparedKernelPublicationKind::Terminal)));
  }
  for (const VulkanWindowRoute &window : pipeline->window.routes) {
    BackendWindowPhase phase{};
    if (!DecodeBackendWindowPhase(window.params.phase, phase)) {
      return FailVulkanPipeline(pipeline, "accel_kernel_run_invalid");
    }
    window_transition_count = ::rund::detail::counter::SaturatingAdd(
        window_transition_count,
        static_cast<std::uint64_t>(phase != BackendWindowPhase::NestedAction ||
                                   window.params.inner_advance != 0u));
    seed_preflight_count = ::rund::detail::counter::SaturatingAdd(
        seed_preflight_count,
        static_cast<std::uint64_t>(phase == BackendWindowPhase::NestedSeed));
  }
  for (const BackendBatchEntry &entry : entries) {
    const BackendWindow *const window = entry.recurrence.window;
    if (window != nullptr && window->phase == BackendWindowPhase::NestedFold) {
      for (const VulkanPipelinePublishRoute &publication :
           pipeline->publish.routes) {
        window_publish_count = ::rund::detail::counter::SaturatingAdd(
            window_publish_count,
            static_cast<std::uint64_t>(
                publication.params.kind ==
                    static_cast<std::uint32_t>(
                        PreparedKernelPublicationKind::Window) &&
                publication.params.state == window->state));
      }
    }
    if (window == nullptr || !window->advances_outer_state() ||
        window->outer_iteration + 1u != window->outer_bound) {
      continue;
    }
    for (const VulkanPipelinePublishRoute &publication :
         pipeline->publish.routes) {
      canonicalize_count = ::rund::detail::counter::SaturatingAdd(
          canonicalize_count,
          static_cast<std::uint64_t>(
              publication.params.kind ==
                  static_cast<std::uint32_t>(
                      PreparedKernelPublicationKind::Terminal) &&
              publication.params.state == window->state));
    }
  }
  const std::uint64_t publication_dispatches =
      ::rund::detail::counter::SaturatingAdd(
          terminal_publish_count,
          ::rund::detail::counter::SaturatingAdd(window_publish_count,
                                                 canonicalize_count));
  const std::uint64_t window_control_dispatches =
      ::rund::detail::counter::SaturatingAdd(window_transition_count,
                                             seed_preflight_count);
  const std::uint64_t control_dispatches =
      ::rund::detail::counter::SaturatingAdd(window_control_dispatches,
                                             publication_dispatches);
  if (window_control_dispatches == std::numeric_limits<std::uint64_t>::max() ||
      publication_dispatches == std::numeric_limits<std::uint64_t>::max() ||
      control_dispatches == std::numeric_limits<std::uint64_t>::max() ||
      !registry.limit.ok ||
      window_control_dispatches >
          registry.limit.backend_window_control_command_count ||
      window_gate_count > registry.limit.backend_indirect_dispatch_count ||
      publication_dispatches >
          registry.limit.backend_publication_command_count ||
      control_dispatches > std::numeric_limits<std::uint64_t>::max() -
                               pipeline->dispatch_count) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  pipeline->dispatch_count += control_dispatches;
  accumulate_memory(memory.staging, recurrence_staging);
  accumulate_memory(memory.staging, transducer_staging);
  if (!PrepareVulkanTelemetry(*pipeline)) {
    const char *const reason = VulkanLastError(pipeline->adapter);
    return FailVulkanPipeline(pipeline,
                              reason == nullptr || reason[0] == '\0'
                                  ? "accel_vulkan_descriptor_unavailable"
                                  : reason);
  }
  pipeline->control.command_count = ::rund::detail::counter::SaturatingAdd(
      pipeline->control.command_count, telemetry_command_count);
  if (pipeline->control.command_count ==
      std::numeric_limits<std::uint64_t>::max()) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  std::uint64_t encoded_command_count = encoded_work_command_count;
  if (!rund::kernel::checked::add(encoded_command_count, control_dispatches,
                                  encoded_command_count) ||
      !rund::kernel::checked::add(encoded_command_count,
                                  pipeline->control.command_count,
                                  encoded_command_count) ||
      !registry.limit.ok ||
      encoded_command_count > registry.limit.backend_command_count) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }

  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail

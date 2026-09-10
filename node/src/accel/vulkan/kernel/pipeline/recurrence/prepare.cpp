#include "internal.hpp"

#include "../../../buffer/create/telemetry.hpp"
#include "../../../map/api.hpp"
#include "../../../map/local.hpp"

#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
PrepareRecurrenceMap(PreparedKernelTemplateRegistry &registry,
                     VulkanAdapter &adapter, const rund::AccelDevice &pick,
                     const BackendRun &owner, const MapRecurrence &recurrence,
                     const std::uint64_t group_capacity,
                     std::shared_ptr<void> &resource) {
  std::shared_ptr<VulkanMapRecurrenceTemplate> recurrence_template;
  const rund::AccelCheck acquired = AcquireVulkanRecurrenceTemplate(
      registry, adapter, pick, owner, recurrence, group_capacity,
      recurrence_template);
  if (!acquired.ok || recurrence_template == nullptr) {
    return acquired.ok
               ? rund::AccelCheck{false, "accel_kernel_template_invalid"}
               : acquired;
  }
  const rund::AccelCheck ready = PrepareVulkanMapProvedRoute(
      pick, recurrence.plan, recurrence.windows, recurrence.window_count,
      recurrence.bindings, recurrence.first->control,
      recurrence.history != nullptr, recurrence_template->prepared,
      recurrence_template->descriptors, resource,
      static_cast<std::uint32_t>(recurrence.iterations));
  auto *const map = static_cast<VulkanMapEncodeResources *>(resource.get());
  if (!ready.ok || map == nullptr || map->adapter != &adapter ||
      map->prepared == nullptr || map->controlled() || map->windows.empty() ||
      map->prepared->plan.dispatch_count != map->windows.size()) {
    return ready.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                    : ready;
  }
  map->binding_owner = recurrence.history;
  return rund::AccelCheck{true, "ok"};
}

} // namespace

[[nodiscard]] rund::AccelCheck PrepareVulkanRecurrence(
    const std::span<const BackendBatchEntry> entries,
    const MapRecurrence &recurrence, PreparedKernelTemplateRegistry &registry,
    PreparedPipelineStatusLayout &status, VulkanPipeline &pipeline,
    PreparedMemory &staging_memory) {
  if (!ValidRecurrence(recurrence) || entries.empty() ||
      !ValidRecurrenceReservation(
          registry, status, 1u, recurrence.history == nullptr ? 0u : 1u,
          recurrence.history == nullptr ? status.generation_stride : 0u,
          recurrence.history == nullptr ? 0u : status.generation_stride, 1u)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    const BackendBatchEntry &entry = entries[index];
    VulkanKernelContext context{};
    const rund::AccelCheck valid =
        entry.run == nullptr || entry.run->pick == nullptr
            ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
            : ValidateVulkanKernelContext(*entry.run->pick, context);
    if (!valid.ok || context.adapter == nullptr ||
        (pipeline.adapter != nullptr && pipeline.adapter != context.adapter) ||
        !SetPreparedProgramStatusSlice(status,
                                       static_cast<std::uint32_t>(index), 0u)) {
      return valid.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                      : valid;
    }
    pipeline.adapter = context.adapter;
  }
  const BackendRun *const owner = entries.front().run;
  if (pipeline.adapter == nullptr || owner == nullptr ||
      owner->pick == nullptr ||
      !RuntimeRecurrenceMatchesPlan(
          *owner, recurrence, 1u,
          recurrence.history == nullptr ? 0u : 1u,
          pipeline.adapter->storage_align)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::uint64_t group_capacity = status.generation_stride;
  std::scoped_lock lock{pipeline.adapter->mutex};
  const VulkanMemoryStats before = pipeline.adapter->staging_memory;
  rund::AccelCheck ready{};
  {
    ready =
        PrepareRecurrenceMap(registry, *pipeline.adapter, *owner->pick, *owner,
                             recurrence, group_capacity, pipeline.recurrence);
  }
  if (!ready.ok) {
    return ready;
  }
  pipeline.dispatch_count = recurrence.window_count;
  staging_memory =
      VulkanPreparedMemory(before, pipeline.adapter->staging_memory,
                           pipeline.adapter->caps.staging_bytes);
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
PrepareVulkanTransducers(const std::span<const BackendBatchEntry> templates,
                         const std::span<const TileTransducer> transducers,
                         PreparedKernelTemplateRegistry &registry,
                         const PreparedPipelineStatusLayout &status,
                         VulkanPipeline &pipeline,
                         PreparedMemory &staging_memory) {
  staging_memory = {};
  pipeline.transducers.clear();
  if (transducers.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (pipeline.adapter == nullptr ||
      transducers.size() > PreparedPipelineStepCapacity) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }

  struct RouteDemand final {
    const BackendRun *owner{};
    std::uint64_t group_count{};
    std::uint64_t history_group_count{};
  };
  struct TemplateDemand final {
    std::size_t representative{};
    std::uint64_t group_count{};
    std::uint64_t group_capacity{};
  };
  std::array<RouteDemand, PreparedPipelineStepCapacity> route_demands{};
  std::array<TemplateDemand, PreparedPipelineStepCapacity> template_demands{};
  std::array<std::size_t, PreparedPipelineStepCapacity> route_demand_indices{};
  std::array<std::size_t, PreparedPipelineStepCapacity>
      template_demand_indices{};
  std::size_t route_demand_count = 0u;
  std::size_t template_demand_count = 0u;
  std::uint64_t history_group_count = 0u;

  for (std::size_t index = 0u; index < transducers.size(); ++index) {
    const TileTransducer &transducer = transducers[index];
    const MapRecurrence &recurrence = transducer.recurrence;
    if (!ValidRecurrence(recurrence) ||
        recurrence.iterations != transducer.template_count ||
        transducer.template_first >= templates.size() ||
        transducer.template_count >
            templates.size() - transducer.template_first) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const BackendBatchEntry &owner = templates[transducer.template_first];
    VulkanKernelContext context{};
    const rund::AccelCheck valid =
        owner.run == nullptr || owner.run->pick == nullptr
            ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
            : ValidateVulkanKernelContext(*owner.run->pick, context);
    if (!valid.ok || context.adapter != pipeline.adapter) {
      return valid.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                      : valid;
    }

    std::size_t route_demand = 0u;
    while (route_demand < route_demand_count &&
           route_demands[route_demand].owner != owner.run) {
      ++route_demand;
    }
    if (route_demand == route_demand_count) {
      route_demands[route_demand] = RouteDemand{.owner = owner.run};
      ++route_demand_count;
    }
    RouteDemand &route = route_demands[route_demand];
    if (!rund::kernel::checked::add(route.group_count, 1u,
                                   route.group_count) ||
        (recurrence.history != nullptr &&
         (!rund::kernel::checked::add(route.history_group_count, 1u,
                                      route.history_group_count) ||
          !rund::kernel::checked::add(history_group_count, 1u,
                                      history_group_count)))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    route_demand_indices[index] = route_demand;

    std::size_t template_demand = 0u;
    for (; template_demand < template_demand_count; ++template_demand) {
      const TileTransducer &representative =
          transducers[template_demands[template_demand].representative];
      const BackendBatchEntry &representative_owner =
          templates[representative.template_first];
      if (representative_owner.run != nullptr &&
          SameRuntimeRecurrenceTemplate(
              *owner.run, recurrence, *representative_owner.run,
              representative.recurrence, pipeline.adapter->storage_align)) {
        break;
      }
    }
    if (template_demand == template_demand_count) {
      template_demands[template_demand] =
          TemplateDemand{.representative = index};
      ++template_demand_count;
    }
    if (!rund::kernel::checked::add(
            template_demands[template_demand].group_count, 1u,
            template_demands[template_demand].group_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    template_demand_indices[index] = template_demand;
  }

  for (std::size_t index = 0u; index < transducers.size(); ++index) {
    const TileTransducer &transducer = transducers[index];
    const BackendBatchEntry &owner = templates[transducer.template_first];
    const RouteDemand &route = route_demands[route_demand_indices[index]];
    if (route.owner != owner.run || route.group_count == 0u ||
        !RuntimeRecurrenceMatchesPlan(
            *owner.run, transducer.recurrence, route.group_count,
            route.history_group_count, pipeline.adapter->storage_align)) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  }

  std::uint64_t terminal_template_group_capacity = 0u;
  std::uint64_t history_template_group_capacity = 0u;
  for (std::size_t index = 0u; index < template_demand_count; ++index) {
    TemplateDemand &demand = template_demands[index];
    if (demand.group_count == 0u ||
        !rund::kernel::checked::mul(demand.group_count,
                                    status.generation_stride,
                                    demand.group_capacity)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const MapRecurrence &representative =
        transducers[demand.representative].recurrence;
    std::uint64_t &variant_capacity = representative.history == nullptr
                                          ? terminal_template_group_capacity
                                          : history_template_group_capacity;
    if (!rund::kernel::checked::add(variant_capacity, demand.group_capacity,
                                    variant_capacity)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (!ValidRecurrenceReservation(
          registry, status, transducers.size(), history_group_count,
          terminal_template_group_capacity, history_template_group_capacity,
          template_demand_count)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  {
    pipeline.transducers.resize(transducers.size());
  }

  std::scoped_lock lock{pipeline.adapter->mutex};
  const VulkanMemoryStats before = pipeline.adapter->staging_memory;
  {
    for (std::size_t index = 0u; index < transducers.size(); ++index) {
      const TileTransducer &transducer = transducers[index];
      const BackendBatchEntry &owner = templates[transducer.template_first];
      const std::uint64_t group_capacity =
          template_demands[template_demand_indices[index]].group_capacity;
      if (group_capacity == 0u) {
        pipeline.transducers.clear();
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      const rund::AccelCheck ready = PrepareRecurrenceMap(
          registry, *pipeline.adapter, *owner.run->pick, *owner.run,
          transducer.recurrence, group_capacity, pipeline.transducers[index]);
      if (!ready.ok) {
        pipeline.transducers.clear();
        return ready;
      }
    }
  }
  staging_memory =
      VulkanPreparedMemory(before, pipeline.adapter->staging_memory,
                           pipeline.adapter->caps.staging_bytes);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail

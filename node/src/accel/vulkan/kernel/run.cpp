#include "../../kernel/backend/execute.hpp"
#include "../../kernel/backend/template_plan.hpp"
#include "../../kernel/recurrence/plan.hpp"
#include "../../kernel/status.hpp"

#include "../collective/chunk.hpp"
#include "../compact/local.hpp"
#include "../descriptor.hpp"
#include "../gather/local.hpp"
#include "../histogram/local.hpp"
#include "../kernel.hpp"
#include "../map/api.hpp"
#include "../map/local.hpp"
#include "../map/source_upper.hpp"
#include "../numeric/source.hpp"
#include "../numeric/state.hpp"
#include "../partition/local.hpp"
#include "../reduce/local.hpp"
#include "../scan/local.hpp"
#include "../scan/source.hpp"
#include "../scatter/local.hpp"
#include "../scatter/reduce/model.hpp"
#include "../segmented/local.hpp"
#include "../segmented/reduce/model.hpp"
#include "../sort/local/state.hpp"
#include "../stencil/local.hpp"
#include "manifest.hpp"
#include "ops/prepare.hpp"
#include "pipeline/capacity.hpp"
#include "pipeline/evidence.hpp"
#include "pipeline/recurrence.hpp"
#include "pipeline/source.hpp"
#include "pipeline/state.hpp"
#include "reset_source.hpp"

#include "../../primitive/block.hpp"
#include "../../sort/block/vulkan.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool AddVulkanHostBytes(std::uint64_t &target,
                                      const std::uint64_t count,
                                      const std::uint64_t element) noexcept {
  std::uint64_t bytes = 0u;
  return backend_template_plan::product(count, element, bytes) &&
         backend_template_plan::add(target, bytes);
}

template <class T>
[[nodiscard]] bool
AddVulkanVectorStorage(std::uint64_t &target,
                       const std::vector<T> &values) noexcept {
  return AddVulkanHostBytes(target,
                            static_cast<std::uint64_t>(values.capacity()),
                            sizeof(T));
}

[[nodiscard]] bool ObserveVulkanMapTemplate(
    const VulkanMapTemplateResources &prepared,
    std::uint64_t &bytes) noexcept {
  if (prepared.adapter == nullptr || prepared.pipeline == nullptr ||
      !prepared.plan.ok ||
      prepared.input_plans.size() != prepared.plan.input_buffer_count ||
      prepared.input_layouts.size() != prepared.plan.input_buffer_count ||
      prepared.output_layouts.size() != prepared.plan.output_buffer_count ||
      prepared.checks.size() > prepared.plan.input_buffer_count ||
      !backend_template_plan::add(bytes,
                                  sizeof(VulkanMapTemplateResources)) ||
      !AddVulkanVectorStorage(bytes, prepared.input_plans) ||
      !AddVulkanVectorStorage(bytes, prepared.input_layouts) ||
      !AddVulkanVectorStorage(bytes, prepared.output_layouts) ||
      !AddVulkanVectorStorage(bytes, prepared.checks)) {
    return false;
  }
  return true;
}

[[nodiscard]] bool ObserveVulkanMapDescriptorArena(
    const VulkanMapDescriptorArena &arena, const std::uint64_t expected_sets,
    std::uint64_t &bytes) noexcept {
  if (arena.adapter == nullptr || arena.pool == VK_NULL_HANDLE ||
      arena.sets.size() != expected_sets || arena.next > arena.sets.size() ||
      !backend_template_plan::add(bytes, sizeof(VulkanMapDescriptorArena)) ||
      !AddVulkanVectorStorage(bytes, arena.sets)) {
    return false;
  }
  for (const VkDescriptorSet set : arena.sets) {
    if (set == VK_NULL_HANDLE) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ObserveVulkanProgramTemplate(
    const VulkanKernelProgramTemplate &program,
    std::uint64_t &bytes) noexcept {
  if (program.kind != VulkanKernelTemplateKind::Program ||
      program.signature == nullptr || !program.route_demand.valid() ||
      program.signature->steps == nullptr ||
      program.steps.size() != program.signature->step_count ||
      program.steps.empty() ||
      (program.reset_set_count != 0u) !=
          (program.reset_pipeline != nullptr) ||
      (program.view_set_count != 0u) != (program.view_pipeline != nullptr) ||
      !backend_template_plan::add(bytes,
                                  sizeof(VulkanKernelProgramTemplate)) ||
      !AddVulkanVectorStorage(bytes, program.steps) ||
      !AddVulkanVectorStorage(bytes, program.descriptor_dependencies)) {
    return false;
  }

  for (std::size_t index = 0u; index < program.steps.size(); ++index) {
    const VulkanKernelProgramStepTemplate &step = program.steps[index];
    if (step.immutable == nullptr || !step.manifest.ok ||
        program.signature->steps[index].step == nullptr) {
      return false;
    }
    bool first_owner = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (program.steps[prior].immutable.get() == step.immutable.get()) {
        first_owner = false;
        break;
      }
    }
    if (!first_owner) {
      continue;
    }
    if (step.ops.encode == EncodeVulkanMap) {
      const auto *const map =
          static_cast<const VulkanMapTemplateResources *>(step.immutable.get());
      if (map == nullptr || !ObserveVulkanMapTemplate(*map, bytes)) {
        return false;
      }
      continue;
    }
    const auto *const pipelines =
        static_cast<const VulkanKernelImmutablePipelines *>(
            step.immutable.get());
    if (pipelines == nullptr ||
        !pipelines->ready(program.signature->steps[index].step->kind(),
                          step.manifest) ||
        !backend_template_plan::add(
            bytes, sizeof(VulkanKernelImmutablePipelines))) {
      return false;
    }
  }

  for (std::size_t index = 0u;
       index < program.descriptor_dependencies.size(); ++index) {
    const VulkanKernelDescriptorDependency &dependency =
        program.descriptor_dependencies[index];
    std::uint64_t expected_capacity = 0u;
    if (dependency.identity == nullptr || dependency.descriptor_count == 0u ||
        dependency.sets_per_route == 0u || dependency.set_capacity == 0u ||
        dependency.source_node == NoNode ||
        !rund::kernel::checked::mul(dependency.sets_per_route,
                                    program.route_demand.capacity,
                                    expected_capacity) ||
        expected_capacity != dependency.set_capacity) {
      return false;
    }
    if (dependency.kind ==
        VulkanKernelDescriptorDependencyKind::Collective) {
      if (dependency.map_arena != nullptr) {
        return false;
      }
      continue;
    }
    if (dependency.kind != VulkanKernelDescriptorDependencyKind::Map ||
        dependency.map_arena == nullptr) {
      return false;
    }
    bool first_owner = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const VulkanKernelDescriptorDependency &previous =
          program.descriptor_dependencies[prior];
      if (previous.map_arena.get() == dependency.map_arena.get()) {
        first_owner = false;
        break;
      }
    }
    if (first_owner &&
        !ObserveVulkanMapDescriptorArena(*dependency.map_arena,
                                         dependency.set_capacity, bytes)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ObserveVulkanRecurrenceTemplate(
    const VulkanMapRecurrenceTemplate &recurrence,
    std::uint64_t &bytes) noexcept {
  std::uint64_t expected_sets = 0u;
  if (recurrence.signature == nullptr) {
    return false;
  }
  const MapRecurrencePreparationPlan preparation =
      PlanMapRecurrencePreparation(*recurrence.signature, 1u,
                                   recurrence.history ? 1u : 0u);
  const MapRecurrenceSourcePlan &source = recurrence.history
                                              ? preparation.history_source
                                              : preparation.terminal_source;
  if (recurrence.kind != VulkanKernelTemplateKind::MapRecurrence ||
      !preparation.eligible() || !source.ok ||
      source.history != recurrence.history ||
      (recurrence.history ? preparation.history_group_count == 0u
                          : preparation.terminal_group_count() == 0u) ||
      recurrence.group_capacity == 0u ||
      recurrence.prepared == nullptr || recurrence.descriptors == nullptr ||
      recurrence.prepared->control_pipeline != nullptr ||
      recurrence.prepared->check_pipeline != nullptr ||
      !recurrence.prepared->checks.empty() ||
      !rund::kernel::checked::mul(recurrence.group_capacity,
                                  recurrence.prepared->plan.dispatch_count,
                                  expected_sets) ||
      expected_sets != recurrence.descriptor_set_capacity ||
      !backend_template_plan::add(bytes,
                                  sizeof(VulkanMapRecurrenceTemplate)) ||
      !ObserveVulkanMapTemplate(*recurrence.prepared, bytes) ||
      !ObserveVulkanMapDescriptorArena(*recurrence.descriptors, expected_sets,
                                       bytes)) {
    return false;
  }
  return recurrence.prepared->adapter == recurrence.descriptors->adapter;
}

[[nodiscard]] std::uint64_t UniqueVulkanMapCheckCount(
    const rund::kernel::LoweringArtifact &artifact) noexcept {
  std::uint64_t count = 0u;
  for (std::size_t index = 0u; index < artifact.metadata.read_routes.size();
       ++index) {
    bool first = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (artifact.metadata.read_routes[prior].index ==
          artifact.metadata.read_routes[index].index) {
        first = false;
        break;
      }
    }
    count += first ? 1u : 0u;
  }
  return count;
}

[[nodiscard]] bool VulkanNumericSourceBytes(const rund::kernel::NodeKind kind,
                                            const bool wide,
                                            std::uint64_t &bytes) noexcept {
  switch (kind) {
  case rund::kernel::NodeKind::Transform:
    return wide ? TransformSource64Bytes(bytes) : TransformSourceBytes(bytes);
  case rund::kernel::NodeKind::Matrix:
    return wide ? MatrixSource64Bytes(bytes) : MatrixSourceBytes(bytes);
  case rund::kernel::NodeKind::Factor:
    return wide ? FactorSource64Bytes(bytes) : FactorSourceBytes(bytes);
  case rund::kernel::NodeKind::Solve:
    return wide ? SolveSource64Bytes(bytes) : SolveSourceBytes(bytes);
  case rund::kernel::NodeKind::Spectrum:
    return wide ? SpectrumSource64Bytes(bytes) : SpectrumSourceBytes(bytes);
  default:
    return false;
  }
}

[[nodiscard]] bool
PlanVulkanViewCaptureCount(const KernelViewLayout *const views,
                           const VulkanAdapter &adapter,
                           std::uint64_t &count) noexcept {
  count = 0u;
  if (views == nullptr) {
    return true;
  }
  for (const KernelViewSlot &view : *views) {
    const rund::kernel::ResidentBufferRef ref{
        .bytes = view.backing_bytes,
        .offset_bytes = view.offset_bytes,
        .element_bytes = view.element_bytes,
        .stride_bytes = view.stride_bytes,
        .count = view.count,
        .usage = view.usage,
    };
    std::uint64_t begin = 0u;
    while (begin < ref.count) {
      StorageRange range{};
      if (!PlanStoragePage(adapter, ref, begin, range) || range.count == 0u ||
          !rund::kernel::checked::add(begin, range.count, begin) ||
          !rund::kernel::checked::add(count, 1u, count)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] rund::AccelCheck
PlanVulkanStepStructure(const KernelExecutionStep &step,
                        const rund::kernel::ComputePlan &plan,
                        const BoundStep *const bound, const KernelViewLayout *,
                        const std::uint64_t max_dispatch_groups,
                        PreparedKernelRouteReservation &reservation) noexcept {
  const PreparedBackendManifest manifest =
      BuildVulkanBackendManifest(step, plan, bound, max_dispatch_groups);
  if (!manifest.ok) {
    return rund::AccelCheck{false, manifest.reason};
  }
  // The shared planner already charged plan.dispatch_count. Vulkan's capture
  // manifest is the exact physical primitive command authority, so retain only
  // its additional stages here; reset and View commands have separate owners.
  std::uint64_t primitive_dispatch_count = 0u;
  if (!rund::kernel::checked::add(manifest.capture_direct_dispatch_count,
                                  manifest.capture_indirect_dispatch_count,
                                  primitive_dispatch_count) ||
      primitive_dispatch_count < plan.dispatch_count ||
      !backend_template_plan::add(reservation.dispatch_count,
                                  primitive_dispatch_count -
                                      plan.dispatch_count)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t route = 0u;
  std::uint64_t map_descriptor_sets = 0u;
  std::uint64_t map_check_binding_count = 0u;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    const std::uint64_t check_count = UniqueVulkanMapCheckCount(step.artifact);
    route = sizeof(VulkanMapEncodeResources);
    if (!backend_template_plan::add(reservation.template_host_bytes,
                                    sizeof(VulkanMapTemplateResources)) ||
        !backend_template_plan::add(reservation.template_host_bytes,
                                    sizeof(VulkanMapDescriptorArena)) ||
        !AddVulkanHostBytes(reservation.template_host_bytes,
                            plan.input_buffer_count, sizeof(InputWindowPlan)) ||
        !AddVulkanHostBytes(reservation.template_host_bytes,
                            plan.input_buffer_count,
                            sizeof(VulkanMapBindingLayout)) ||
        !AddVulkanHostBytes(reservation.template_host_bytes,
                            plan.output_buffer_count,
                            sizeof(VulkanMapBindingLayout)) ||
        !AddVulkanHostBytes(reservation.template_host_bytes, check_count,
                            sizeof(VulkanMapCheck)) ||
        !AddVulkanHostBytes(route, plan.input_buffer_count,
                            sizeof(VulkanResidentBufferResult)) ||
        !AddVulkanHostBytes(route, plan.output_buffer_count,
                            sizeof(VulkanResidentBufferResult)) ||
        !AddVulkanHostBytes(route, check_count, sizeof(std::uint64_t)) ||
        !AddVulkanHostBytes(route, plan.dispatch_count,
                            sizeof(VkDescriptorSet))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (check_count != 0u &&
        !rund::kernel::checked::add(check_count, 3u, map_check_binding_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    map_descriptor_sets = plan.dispatch_count;
    break;
  }
  case rund::kernel::NodeKind::Scan:
    route =
        sizeof(VulkanKernelScanResources) + sizeof(VulkanScanEncodeResources);
    break;
  case rund::kernel::NodeKind::SegmentedScan:
    route = sizeof(VulkanSegmentedScanEncodeResources);
    break;
  case rund::kernel::NodeKind::SegmentedReduce:
    route = sizeof(VulkanSegmentedReduceResources);
    break;
  case rund::kernel::NodeKind::Sort:
    route = sizeof(VulkanSortEncodeResources);
    break;
  case rund::kernel::NodeKind::Compact:
    route = sizeof(VulkanCompactEncodeResources);
    break;
  case rund::kernel::NodeKind::Gather:
    route = sizeof(VulkanGatherEncodeResources);
    break;
  case rund::kernel::NodeKind::Histogram:
    route = sizeof(VulkanHistogramEncodeResources);
    break;
  case rund::kernel::NodeKind::Partition:
    route = sizeof(VulkanPartitionEncodeResources) +
            sizeof(VulkanScanEncodeResources);
    break;
  case rund::kernel::NodeKind::Reduce:
    route = sizeof(VulkanReduceEncodeResources);
    if (!AddVulkanHostBytes(
            route, step.operation.get<operation::Reduce>().plan.pass_count,
            sizeof(VkDescriptorSet))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    break;
  case rund::kernel::NodeKind::Scatter:
    route = sizeof(VulkanScatterEncodeResources);
    break;
  case rund::kernel::NodeKind::ScatterReduce:
    route = sizeof(VulkanScatterReduceResources);
    break;
  case rund::kernel::NodeKind::Stencil:
    route = sizeof(VulkanStencilEncodeResources);
    break;
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    route = sizeof(VulkanNumericPrepared);
    break;
  }
  const std::uint64_t immutable_host_bytes =
      step.kind() == rund::kernel::NodeKind::Map
          ? 0u
          : sizeof(VulkanKernelImmutablePipelines);
  std::uint64_t cache_owner_host_bytes = 0u;
  if (step.kind() == rund::kernel::NodeKind::Map) {
    if (manifest.source_library_dependency_count == 0u ||
        !AddVulkanHostBytes(cache_owner_host_bytes,
                            manifest.source_library_dependency_count - 1u,
                            sizeof(VulkanCollectivePipeline)) ||
        !backend_template_plan::add(cache_owner_host_bytes,
                                    sizeof(VulkanCachedPipeline))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } else if (!AddVulkanHostBytes(cache_owner_host_bytes,
                                 manifest.source_library_dependency_count,
                                 sizeof(VulkanCollectivePipeline))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t collective_descriptor_sets = 0u;
  if (map_descriptor_sets > manifest.descriptor_set_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  collective_descriptor_sets =
      manifest.descriptor_set_count - map_descriptor_sets;
  reservation.source_transient_bytes = std::max(
      reservation.source_transient_bytes, manifest.cold_source_transient_bytes);
  return AddVulkanHostBytes(reservation.host_transient_bytes,
                            manifest.status_source_count,
                            sizeof(VulkanPipelineCanonicalStatus)) &&
                 AddVulkanHostBytes(reservation.host_transient_bytes,
                                    map_check_binding_count,
                                    sizeof(VulkanStorageBinding)) &&
                 AddVulkanHostBytes(reservation.host_transient_bytes,
                                    manifest.descriptor_dependency_count,
                                    sizeof(VulkanKernelDescriptorDependency)) &&
                 AddVulkanHostBytes(reservation.host_transient_bytes, 2u,
                                    sizeof(PreparedProgramStatusSlice)) &&
                 backend_template_plan::add(reservation.route_host_bytes,
                                            route) &&
                 AddVulkanHostBytes(
                     reservation.route_host_bytes, collective_descriptor_sets,
                     sizeof(VkDescriptorSet) + sizeof(std::uint8_t)) &&
                 AddVulkanHostBytes(reservation.route_host_bytes,
                                    manifest.descriptor_lease_count,
                                    sizeof(VulkanCollectiveDescriptorLease)) &&
                 backend_template_plan::add(reservation.template_host_bytes,
                                            immutable_host_bytes) &&
                 backend_template_plan::add(reservation.template_host_bytes,
                                            cache_owner_host_bytes) &&
                 AddVulkanHostBytes(reservation.template_host_bytes,
                                    manifest.descriptor_dependency_count,
                                    sizeof(VulkanKernelDescriptorDependency)) &&
                 AddVulkanHostBytes(
                     reservation.template_host_bytes,
                     manifest.descriptor_dependency_count -
                         static_cast<std::uint64_t>(
                             step.kind() == rund::kernel::NodeKind::Map),
                     sizeof(VkDescriptorPool)) &&
                 backend_template_plan::add(reservation.template_source_bytes,
                                            manifest.cold_cache_source_bytes) &&
                 backend_template_plan::add(
                     reservation.template_host_bytes,
                     manifest.cold_cache_source_storage_bytes) &&
                 backend_template_plan::add(
                     reservation.template_native_allocation_count,
                     manifest.cold_cache_native_object_count) &&
                 backend_template_plan::add(reservation.descriptor_set_count,
                                            manifest.descriptor_set_count) &&
                 backend_template_plan::add(
                     reservation.descriptor_count,
                     manifest.descriptor_binding_count) &&
                 backend_template_plan::add(
                     reservation.capture_direct_dispatch_count,
                     manifest.capture_direct_dispatch_count) &&
                 backend_template_plan::add(
                     reservation.capture_indirect_dispatch_count,
                     manifest.capture_indirect_dispatch_count) &&
                 backend_template_plan::add(reservation.status_source_count,
                                            manifest.status_source_count) &&
                 backend_template_plan::add(reservation.status_entry_count,
                                            manifest.status_entry_count) &&
                 backend_template_plan::add(reservation.status_command_count,
                                            manifest.status_command_count) &&
                 backend_template_plan::add(reservation.status_parameter_bytes,
                                            manifest.status_parameter_bytes) &&
                 backend_template_plan::add(reservation.telemetry_source_count,
                                            manifest.telemetry_source_count)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_pipeline_capacity"};
}

[[nodiscard]] backend_template_plan::BackendShape
VulkanBackendShape(const std::uint64_t alignment,
                   const std::uint64_t max_dispatch_groups) noexcept {
  return backend_template_plan::BackendShape{
      .storage_alignment = alignment,
      .max_dispatch_groups = max_dispatch_groups,
      .reset_dispatch_window = max_dispatch_groups * 256u,
      .template_capacity = PreparedPipelineStepCapacity,
      .route_header_bytes = sizeof(VulkanKernelResources),
      .route_step_bytes = sizeof(VulkanKernelEntry),
      .route_inline_step_capacity = kInlineBoundStepCapacity,
      .template_header_bytes = sizeof(VulkanKernelProgramTemplate),
      .template_step_bytes = sizeof(VulkanKernelProgramStepTemplate),
      .template_step_capacity = kVulkanPipelineTemplateStepCapacity,
      .plan_step = PlanVulkanStepStructure,
  };
}

[[nodiscard]] rund::AccelCheck CompleteVulkanRouteCaptureStructure(
    const KernelViewLayout *const views, const std::uint64_t reset_count,
    const VulkanAdapter &adapter,
    PreparedKernelRouteReservation &reservation) noexcept {
  std::uint64_t view_dispatch_count = 0u;
  std::uint64_t auxiliary_set_count = 0u;
  std::uint64_t auxiliary_binding_count = 0u;
  std::uint64_t auxiliary_dependency_count = 0u;
  std::uint64_t lease_bytes = 0u;
  std::uint64_t reset_source_storage = 0u;
  std::uint64_t reset_route_descriptor_host = 0u;
  std::uint64_t reset_template_descriptor_host = 0u;
  std::uint64_t reset_native_objects = 0u;
  std::uint64_t view_source_storage = 0u;
  std::uint64_t view_route_descriptor_host = 0u;
  std::uint64_t view_template_descriptor_host = 0u;
  std::uint64_t view_native_objects = 0u;
  if (!PlanVulkanViewCaptureCount(views, adapter, view_dispatch_count) ||
      !rund::kernel::checked::add(reset_count, view_dispatch_count,
                                  auxiliary_set_count) ||
      !rund::kernel::checked::mul(view_dispatch_count, 2u,
                                  auxiliary_binding_count) ||
      !rund::kernel::checked::add(auxiliary_binding_count, reset_count,
                                  auxiliary_binding_count) ||
      !rund::kernel::checked::add(reset_count == 0u ? 0u : 1u,
                                  view_dispatch_count == 0u ? 0u : 1u,
                                  auxiliary_dependency_count) ||
      !rund::kernel::checked::mul(auxiliary_set_count,
                                  sizeof(VulkanCollectiveDescriptorLease),
                                  lease_bytes) ||
      (reset_count != 0u &&
       (!backend_source_recipe::string_external_storage_upper_bytes(
            VulkanResetSourceText().size(), reset_source_storage) ||
        !rund::kernel::checked::mul(
            reset_count, sizeof(VkDescriptorSet) + sizeof(std::uint8_t),
            reset_route_descriptor_host) ||
        !rund::kernel::checked::add(reset_template_descriptor_host,
                                    sizeof(VulkanKernelDescriptorDependency) +
                                        sizeof(VkDescriptorPool) +
                                        sizeof(VulkanCollectivePipeline),
                                    reset_template_descriptor_host) ||
        !rund::kernel::checked::add(reset_count, 4u, reset_native_objects))) ||
      (view_dispatch_count != 0u &&
       (!backend_source_recipe::string_external_storage_upper_bytes(
            VulkanViewSourceText().size(), view_source_storage) ||
        !rund::kernel::checked::mul(
            view_dispatch_count, sizeof(VkDescriptorSet) + sizeof(std::uint8_t),
            view_route_descriptor_host) ||
        !rund::kernel::checked::add(view_template_descriptor_host,
                                    sizeof(VulkanKernelDescriptorDependency) +
                                        sizeof(VkDescriptorPool) +
                                        sizeof(VulkanCollectivePipeline),
                                    view_template_descriptor_host) ||
        !rund::kernel::checked::add(view_dispatch_count, 4u,
                                    view_native_objects))) ||
      !backend_template_plan::add(reservation.capture_direct_dispatch_count,
                                  reservation.reset_dispatch_count) ||
      !backend_template_plan::add(reservation.capture_direct_dispatch_count,
                                  view_dispatch_count) ||
      // View commands are encoded body work. Reset remains exclusively in
      // reset_dispatch_count even though both participate in capture gating.
      !backend_template_plan::add(reservation.dispatch_count,
                                  view_dispatch_count) ||
      !backend_template_plan::add(reservation.route_host_bytes, lease_bytes) ||
      !backend_template_plan::add(reservation.route_host_bytes,
                                  reset_route_descriptor_host) ||
      !backend_template_plan::add(reservation.route_host_bytes,
                                  view_route_descriptor_host) ||
      !AddVulkanHostBytes(reservation.host_transient_bytes,
                          auxiliary_dependency_count,
                          sizeof(VulkanKernelDescriptorDependency)) ||
      !backend_template_plan::add(reservation.template_host_bytes,
                                  reset_template_descriptor_host) ||
      !backend_template_plan::add(reservation.template_host_bytes,
                                  reset_source_storage) ||
      !backend_template_plan::add(reservation.template_host_bytes,
                                  view_template_descriptor_host) ||
      !backend_template_plan::add(reservation.template_host_bytes,
                                  view_source_storage) ||
      !backend_template_plan::add(
          reservation.template_source_bytes,
          reset_count == 0u ? 0u : VulkanResetSourceText().size()) ||
      !backend_template_plan::add(
          reservation.template_source_bytes,
          view_dispatch_count == 0u ? 0u : VulkanViewSourceText().size()) ||
      !backend_template_plan::add(reservation.template_native_allocation_count,
                                  reset_native_objects) ||
      !backend_template_plan::add(reservation.template_native_allocation_count,
                                  view_native_objects) ||
      !backend_template_plan::add(reservation.descriptor_set_count,
                                  auxiliary_set_count) ||
      !backend_template_plan::add(reservation.descriptor_count,
                                  auxiliary_binding_count)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.source_transient_bytes =
      std::max(reservation.source_transient_bytes,
               std::max(reset_source_storage, view_source_storage));
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] bool PlanVulkanCaptureManifest(
    const KernelExecutionStep &step, const rund::kernel::ComputePlan &plan,
    const BoundStep *const bound, const std::uint64_t max_dispatch_groups,
    PreparedBackendManifest &manifest) noexcept {
  using rund::kernel::checked::add;
  using rund::kernel::checked::mul;
  if (max_dispatch_groups == 0u || plan.dispatch_count == 0u) {
    return false;
  }
  const bool has_checks = !step.artifact.metadata.read_routes.empty();
  const bool controlled = (bound != nullptr ? bound->control.active()
                                            : (step.control.has_count() ||
                                               step.control.has_predicate())) ||
                          has_checks;
  const std::uint32_t control_iteration =
      bound == nullptr ? step.control.iteration
                       : bound->control.control.iteration;
  std::uint64_t direct = 0u;
  std::uint64_t indirect = 0u;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map:
    direct = plan.dispatch_count;
    if (controlled && has_checks && !add(direct, 1u, direct)) {
      return false;
    }
    indirect = controlled ? plan.dispatch_count : 0u;
    break;
  case rund::kernel::NodeKind::Scan: {
    const auto &active = step.operation.get<operation::Scan>().plan;
    direct = ScanDispatches(active.pass_count, active.block_count,
                            max_dispatch_groups);
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto &active = step.operation.get<operation::SegmentedScan>().plan;
    direct = ScanDispatches(active.pass_count, active.block_count,
                            max_dispatch_groups);
    break;
  }
  case rund::kernel::NodeKind::SegmentedReduce:
    direct = 3u;
    indirect = 1u;
    break;
  case rund::kernel::NodeKind::Sort: {
    const auto &active = step.operation.get<operation::Sort>().plan;
    const std::uint64_t blocks =
        CeilGroups(active.element_count, kVulkanSortBlockSize);
    const std::uint64_t chunks = CeilGroups(blocks, max_dispatch_groups);
    const std::uint64_t fixed = blocks == 1u ? 1u : 2u;
    std::uint64_t pass_direct = 0u;
    std::uint64_t pass_indirect = 0u;
    if (blocks == 0u || chunks == 0u || active.radix_pass_count == 0u ||
        !mul(active.radix_pass_count, fixed, pass_direct) ||
        !add(pass_direct, 1u, direct) ||
        !mul(active.radix_pass_count, chunks, pass_indirect) ||
        !mul(pass_indirect, 2u, indirect)) {
      return false;
    }
    break;
  }
  case rund::kernel::NodeKind::Gather:
    direct = 1u;
    indirect = 1u;
    break;
  case rund::kernel::NodeKind::Partition: {
    const auto &active = step.operation.get<operation::Partition>().plan;
    const std::uint64_t blocks =
        CeilGroups(active.element_count, block::VulkanPartition);
    const std::uint64_t passes =
        active.element_count > block::VulkanPartition ? 2u : 1u;
    const std::uint64_t scan =
        ScanDispatches(passes, blocks, max_dispatch_groups);
    if (scan == 0u || !add(scan, 2u, direct)) {
      return false;
    }
    break;
  }
  case rund::kernel::NodeKind::ScatterReduce:
    direct = 1u;
    indirect = 2u;
    break;
  default:
    direct = plan.dispatch_count;
    break;
  }
  if (direct == 0u) {
    return false;
  }
  manifest.capture_direct_dispatch_count = direct;
  manifest.capture_indirect_dispatch_count = indirect;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map:
    manifest.status_source_count = controlled ? 1u : 0u;
    manifest.telemetry_source_count = controlled ? 1u : 0u;
    break;
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
    break;
  default:
    manifest.status_source_count = 1u;
    break;
  }
  if (step.kind() == rund::kernel::NodeKind::Gather ||
      step.kind() == rund::kernel::NodeKind::ScatterReduce ||
      ((step.kind() == rund::kernel::NodeKind::Scan ||
        step.kind() == rund::kernel::NodeKind::Sort) &&
       control_iteration != 0u)) {
    manifest.telemetry_source_count = 1u;
  }
  switch (step.kind()) {
  case rund::kernel::NodeKind::Factor:
    manifest.status_entry_count =
        step.operation.get<operation::Factor>().plan.status_count;
    break;
  case rund::kernel::NodeKind::Solve:
    manifest.status_entry_count =
        step.operation.get<operation::Solve>().plan.status_count;
    break;
  case rund::kernel::NodeKind::Spectrum:
    manifest.status_entry_count =
        step.operation.get<operation::Spectrum>().plan.status_count;
    break;
  default:
    manifest.status_entry_count = manifest.status_source_count;
    break;
  }
  if ((manifest.status_source_count == 0u) !=
      (manifest.status_entry_count == 0u)) {
    return false;
  }
  if (!mul(manifest.status_source_count, 2u,
           manifest.status_command_count) ||
      !mul(manifest.status_source_count,
           VulkanPipelineStatusSourceParameterBytes,
           manifest.status_parameter_bytes)) {
    return false;
  }
  return true;
}

} // namespace

PreparedBackendManifest
BuildVulkanBackendManifest(const KernelExecutionStep &step,
                           const rund::kernel::ComputePlan &plan,
                           const BoundStep *const bound,
                           const std::uint64_t max_dispatch_groups) noexcept {
  PreparedBackendManifest manifest{};
  const bool has_checks = !step.artifact.metadata.read_routes.empty();
  const bool controlled = (bound != nullptr ? bound->control.active()
                                            : (step.control.has_count() ||
                                               step.control.has_predicate())) ||
                          has_checks;
  const auto scan_stages = [](const std::uint64_t passes) {
    return passes == 1u ? std::uint64_t{1u} : std::uint64_t{3u};
  };
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    const std::uint64_t checks = UniqueVulkanMapCheckCount(step.artifact);
    const std::uint64_t check_stage = has_checks ? 1u : 0u;
    const std::uint64_t control_stage = controlled ? 1u : 0u;
    manifest.source_build_count = 1u + control_stage + check_stage;
    manifest.source_library_dependency_count = manifest.source_build_count;
    manifest.pipeline_stage_count = manifest.source_build_count;
    manifest.descriptor_dependency_count = manifest.pipeline_stage_count;
    manifest.descriptor_lease_count = control_stage + check_stage;
    std::uint64_t bindings_per_window = 0u;
    std::uint64_t window_bindings = 0u;
    std::uint64_t check_bindings = 0u;
    if (!rund::kernel::checked::add(plan.input_buffer_count,
                                    plan.output_buffer_count,
                                    bindings_per_window) ||
        !rund::kernel::checked::add(bindings_per_window, 1u + control_stage,
                                    bindings_per_window) ||
        !rund::kernel::checked::mul(plan.dispatch_count, bindings_per_window,
                                    window_bindings) ||
        !rund::kernel::checked::add(checks, 3u, check_bindings) ||
        !rund::kernel::checked::mul(check_bindings, check_stage,
                                    check_bindings) ||
        !rund::kernel::checked::add(plan.dispatch_count, control_stage,
                                    manifest.descriptor_set_count) ||
        !rund::kernel::checked::add(manifest.descriptor_set_count, check_stage,
                                    manifest.descriptor_set_count) ||
        !rund::kernel::checked::add(window_bindings, 4u * control_stage,
                                    manifest.descriptor_binding_count) ||
        !rund::kernel::checked::add(manifest.descriptor_binding_count,
                                    check_bindings,
                                    manifest.descriptor_binding_count)) {
      return manifest;
    }
    std::uint64_t main_source = 0u;
    std::uint64_t final_main_source = 0u;
    std::uint64_t raw_source_transient = 0u;
    if (!backend_template_plan::map_source_upper(step, plan, main_source,
                                                 raw_source_transient) ||
        (controlled && !VulkanControlledMapSourceUpperBytes(
                           plan, main_source, final_main_source)) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e6d6170ull,
                          .source_upper_bytes =
                              controlled ? final_main_source : main_source,
                          .pipeline_stage_count = 1u,
                      })) {
      return manifest;
    }
    if (controlled &&
        !backend_source_recipe::string_external_storage_upper_bytes(
            main_source, raw_source_transient)) {
      return manifest;
    }
    manifest.cold_source_transient_bytes = raw_source_transient;
    if (controlled && !AddPreparedBackendCacheDependency(
                          manifest, PreparedBackendCacheDependency{
                                        .source_recipe = 0x76756c6b2e637472ull,
                                        .source_upper_bytes =
                                            VulkanMapControlSourceText().size(),
                                        .pipeline_stage_count = 1u,
                                    })) {
      return manifest;
    }
    if (has_checks) {
      std::uint64_t check_source = 0u;
      if (!VulkanMapCheckSourceUpperBytes(step.artifact, check_source) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e63686bull,
                            .source_upper_bytes = check_source,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    // The main Map artifact is moved into the cached pipeline. Control and
    // bounds-check collective pipelines copy their source, so their caller
    // storage is the only additional full-source allocation live during cache
    // publication. Dependency zero is always the moved main artifact.
    for (std::size_t index = 1u; index < manifest.cache_dependency_entry_count;
         ++index) {
      manifest.cold_source_transient_bytes = std::max(
          manifest.cold_source_transient_bytes,
          manifest.source_dependencies[index].source_storage_upper_bytes);
    }
    break;
  }
  case rund::kernel::NodeKind::Scan: {
    const auto &active = step.operation.get<operation::Scan>();
    const std::uint64_t stages = scan_stages(active.plan.pass_count);
    manifest = PreparedBackendManifest{
        .source_build_count = stages,
        .source_library_dependency_count = stages,
        .pipeline_stage_count = stages,
        .descriptor_set_count = stages,
        .descriptor_binding_count = 6u * stages,
        .descriptor_lease_count = stages,
        .descriptor_dependency_count = stages,
    };
    const bool inclusive = active.desc.op == rund::kernel::ScanOp::InclusiveSum;
    const auto add_scan_stage = [&](const VulkanScanStage stage) noexcept {
      std::uint64_t source_bytes = 0u;
      return VulkanScanSourceBytes(active.desc.element, plan.domain, stage,
                                   inclusive, source_bytes) &&
             AddPreparedBackendCacheDependency(
                 manifest,
                 PreparedBackendCacheDependency{
                     .source_recipe = 0x76756c6b2e736e00ull +
                                      static_cast<std::uint64_t>(stage) + 1u,
                     .source_upper_bytes = source_bytes,
                     .pipeline_stage_count = 1u,
                 });
    };
    if (!add_scan_stage(VulkanScanStage::Block) ||
        (stages != 1u && (!add_scan_stage(VulkanScanStage::Prefix) ||
                          !add_scan_stage(VulkanScanStage::Offset)))) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto &active = step.operation.get<operation::SegmentedScan>();
    const std::uint64_t stages = scan_stages(active.plan.pass_count);
    manifest = PreparedBackendManifest{
        .source_build_count = stages,
        .source_library_dependency_count = stages,
        .pipeline_stage_count = stages,
        .descriptor_set_count = stages,
        .descriptor_binding_count = 7u * stages,
        .descriptor_lease_count = stages,
        .descriptor_dependency_count = stages,
    };
    const auto add_stage = [&](const VulkanSegmentedScanStage stage) noexcept {
      std::uint64_t source_bytes = 0u;
      return VulkanSegmentedScanSourceBytes(active.desc.element, plan.domain,
                                            stage, source_bytes) &&
             AddPreparedBackendCacheDependency(
                 manifest,
                 PreparedBackendCacheDependency{
                     .source_recipe = 0x76756c6b2e736700ull +
                                      static_cast<std::uint64_t>(stage) + 1u,
                     .source_upper_bytes = source_bytes,
                     .pipeline_stage_count = 1u,
                 });
    };
    if (!add_stage(VulkanSegmentedScanStage::Block) ||
        (stages != 1u && (!add_stage(VulkanSegmentedScanStage::Prefix) ||
                          !add_stage(VulkanSegmentedScanStage::Offset)))) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    manifest = PreparedBackendManifest{.source_build_count = 4u,
                                       .source_library_dependency_count = 4u,
                                       .pipeline_stage_count = 4u,
                                       .descriptor_set_count = 4u,
                                       .descriptor_binding_count = 24u,
                                       .descriptor_lease_count = 4u,
                                       .descriptor_dependency_count = 4u};
    const auto &active = step.operation.get<operation::SegmentedReduce>();
    for (const VulkanSegmentedReduceStage stage :
         {VulkanSegmentedReduceStage::Classify,
          VulkanSegmentedReduceStage::Prefix,
          VulkanSegmentedReduceStage::Scatter,
          VulkanSegmentedReduceStage::Reduce}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanSegmentedReduceSourceBytes(active.plan, plan.domain, stage,
                                            source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e727300ull +
                                             static_cast<std::uint64_t>(stage),
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Sort: {
    const auto &active = step.operation.get<operation::Sort>();
    const std::uint64_t passes = active.plan.radix_pass_count;
    if (!rund::kernel::checked::mul(4u, passes,
                                    manifest.descriptor_set_count) ||
        !rund::kernel::checked::add(manifest.descriptor_set_count, 1u,
                                    manifest.descriptor_set_count) ||
        !rund::kernel::checked::mul(9u, manifest.descriptor_set_count,
                                    manifest.descriptor_binding_count)) {
      return manifest;
    }
    manifest.source_build_count = 5u;
    manifest.source_library_dependency_count = 5u;
    manifest.pipeline_stage_count = 5u;
    manifest.descriptor_lease_count = manifest.descriptor_set_count;
    manifest.descriptor_dependency_count = 5u;
    for (const SortStage stage :
         {SortStage::Dispatch, SortStage::Classify, SortStage::Prefix,
          SortStage::Base, SortStage::Scatter}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanSortSourceBytes(active.desc.key, stage, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest,
              PreparedBackendCacheDependency{
                  .source_recipe = 0x76756c6b2e736f00ull +
                                   static_cast<std::uint64_t>(stage) + 1u,
                  .source_upper_bytes = source_bytes,
                  .pipeline_stage_count = 1u,
              })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Compact: {
    manifest = PreparedBackendManifest{.source_build_count = 3u,
                                       .source_library_dependency_count = 3u,
                                       .pipeline_stage_count = 3u,
                                       .descriptor_set_count = 3u,
                                       .descriptor_binding_count = 18u,
                                       .descriptor_lease_count = 3u,
                                       .descriptor_dependency_count = 3u};
    for (const CompactStage stage :
         {CompactStage::Classify, CompactStage::Prefix,
          CompactStage::Scatter}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanCompactSourceBytes(stage, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest,
              PreparedBackendCacheDependency{
                  .source_recipe = 0x76756c6b2e636d00ull +
                                   static_cast<std::uint64_t>(stage) + 1u,
                  .source_upper_bytes = source_bytes,
                  .pipeline_stage_count = 1u,
              })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Gather: {
    manifest = PreparedBackendManifest{.source_build_count = 2u,
                                       .source_library_dependency_count = 2u,
                                       .pipeline_stage_count = 2u,
                                       .descriptor_set_count = 2u,
                                       .descriptor_binding_count = 14u,
                                       .descriptor_lease_count = 2u,
                                       .descriptor_dependency_count = 2u};
    const auto element = step.operation.get<operation::Gather>().plan.element;
    for (const bool control : {true, false}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanGatherSourceBytes(element, control, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = control ? 0x76756c6b2e676374ull
                                                     : 0x76756c6b2e676174ull,
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Histogram: {
    manifest = PreparedBackendManifest{.source_build_count = 2u,
                                       .source_library_dependency_count = 2u,
                                       .pipeline_stage_count = 2u,
                                       .descriptor_set_count = 2u,
                                       .descriptor_binding_count = 8u,
                                       .descriptor_lease_count = 2u,
                                       .descriptor_dependency_count = 2u};
    for (const bool clear : {true, false}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanHistogramSourceBytes(clear, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = clear ? 0x76756c6b2e68636cull
                                                   : 0x76756c6b2e686374ull,
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Partition: {
    const auto &active = step.operation.get<operation::Partition>();
    const std::uint64_t scan = scan_stages(
        active.plan.element_count > block::VulkanPartition ? 2u : 1u);
    manifest.source_build_count = 2u + scan;
    manifest.source_library_dependency_count = 2u + scan;
    manifest.pipeline_stage_count = 2u + scan;
    manifest.descriptor_set_count = 2u + scan;
    manifest.descriptor_binding_count = 8u + 6u * scan;
    manifest.descriptor_lease_count = manifest.descriptor_set_count;
    manifest.descriptor_dependency_count = manifest.pipeline_stage_count;
    for (const PartitionStage stage :
         {PartitionStage::Classify, PartitionStage::Scatter}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanPartitionSourceBytes(stage, active.desc.flag_bytes,
                                      active.desc.value_bytes, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest,
              PreparedBackendCacheDependency{
                  .source_recipe = 0x76756c6b2e707400ull +
                                   static_cast<std::uint64_t>(stage) + 1u,
                  .source_upper_bytes = source_bytes,
                  .pipeline_stage_count = 1u,
              })) {
        return manifest;
      }
    }
    const auto add_scan_stage = [&](const VulkanScanStage stage) noexcept {
      std::uint64_t source_bytes = 0u;
      return VulkanScanSourceBytes(rund::kernel::ScanElement::U32,
                                   rund::kernel::ComputeDomain::U32, stage,
                                   false, source_bytes) &&
             AddPreparedBackendCacheDependency(
                 manifest,
                 PreparedBackendCacheDependency{
                     .source_recipe = 0x76756c6b2e707300ull +
                                      static_cast<std::uint64_t>(stage) + 1u,
                     .source_upper_bytes = source_bytes,
                     .pipeline_stage_count = 1u,
                 });
    };
    if (!add_scan_stage(VulkanScanStage::Block) ||
        (scan != 1u && (!add_scan_stage(VulkanScanStage::Prefix) ||
                        !add_scan_stage(VulkanScanStage::Offset)))) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto &active = step.operation.get<operation::Reduce>();
    const std::uint64_t passes = active.plan.pass_count;
    if (!rund::kernel::checked::mul(6u, passes,
                                    manifest.descriptor_binding_count)) {
      return manifest;
    }
    manifest.source_build_count = 1u;
    manifest.source_library_dependency_count = 1u;
    manifest.pipeline_stage_count = 1u;
    manifest.descriptor_set_count = passes;
    manifest.descriptor_lease_count = passes;
    manifest.descriptor_dependency_count = 1u;
    std::uint64_t source_bytes = 0u;
    if (!VulkanReduceSourceBytes(active.desc.op, active.desc.element,
                                 active.desc.block_size, plan.domain,
                                 source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e726564ull,
                          .source_upper_bytes = source_bytes,
                          .pipeline_stage_count = 1u,
                      })) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::Scatter: {
    manifest = PreparedBackendManifest{.source_build_count = 1u,
                                       .source_library_dependency_count = 1u,
                                       .pipeline_stage_count = 1u,
                                       .descriptor_set_count = 1u,
                                       .descriptor_binding_count = 5u,
                                       .descriptor_lease_count = 1u,
                                       .descriptor_dependency_count = 1u};
    std::uint64_t source_bytes = 0u;
    if (!VulkanScatterSourceBytes(
            step.operation.get<operation::Scatter>().plan.element,
            source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e736361ull,
                          .source_upper_bytes = source_bytes,
                          .pipeline_stage_count = 1u,
                      })) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    manifest = PreparedBackendManifest{.source_build_count = 3u,
                                       .source_library_dependency_count = 3u,
                                       .pipeline_stage_count = 3u,
                                       .descriptor_set_count = 3u,
                                       .descriptor_binding_count = 24u,
                                       .descriptor_lease_count = 3u,
                                       .descriptor_dependency_count = 3u};
    const auto &active = step.operation.get<operation::ScatterReduce>();
    for (const VulkanScatterReduceStage stage :
         {VulkanScatterReduceStage::Control, VulkanScatterReduceStage::Init,
          VulkanScatterReduceStage::Fold}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanScatterReduceSourceBytes(active.plan, stage, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e737200ull +
                                             static_cast<std::uint64_t>(stage),
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Stencil: {
    manifest = PreparedBackendManifest{.source_build_count = 1u,
                                       .source_library_dependency_count = 1u,
                                       .pipeline_stage_count = 1u,
                                       .descriptor_set_count = 1u,
                                       .descriptor_binding_count = 3u,
                                       .descriptor_lease_count = 1u,
                                       .descriptor_dependency_count = 1u};
    const auto &active = step.operation.get<operation::Stencil>();
    std::uint64_t source_bytes = 0u;
    if (!VulkanStencilSourceBytes(active.desc.op, active.desc.element,
                                  plan.domain, source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e737465ull,
                          .source_upper_bytes = source_bytes,
                          .pipeline_stage_count = 1u,
                      })) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum: {
    const std::uint64_t bindings =
        step.kind() == rund::kernel::NodeKind::Matrix
            ? 4u
            : (step.kind() == rund::kernel::NodeKind::Factor ||
                       step.kind() == rund::kernel::NodeKind::Spectrum
                   ? 5u
                   : 6u);
    manifest = PreparedBackendManifest{.source_build_count = 1u,
                                       .source_library_dependency_count = 1u,
                                       .pipeline_stage_count = 1u,
                                       .descriptor_set_count = 1u,
                                       .descriptor_binding_count = bindings,
                                       .descriptor_lease_count = 1u,
                                       .descriptor_dependency_count = 1u};
    bool wide = false;
    switch (step.kind()) {
    case rund::kernel::NodeKind::Transform:
      wide = step.operation.get<operation::Transform>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    case rund::kernel::NodeKind::Matrix:
      wide = step.operation.get<operation::Matrix>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    case rund::kernel::NodeKind::Factor:
      wide = step.operation.get<operation::Factor>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    case rund::kernel::NodeKind::Solve:
      wide = step.operation.get<operation::Solve>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    case rund::kernel::NodeKind::Spectrum:
      wide = step.operation.get<operation::Spectrum>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    default:
      return manifest;
    }
    std::uint64_t source_bytes = 0u;
    if (!VulkanNumericSourceBytes(step.kind(), wide, source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest,
            PreparedBackendCacheDependency{
                .source_recipe = 0x76756c6b2e6e7500ull +
                                 static_cast<std::uint64_t>(step.kind()),
                .source_upper_bytes = source_bytes,
                .pipeline_stage_count = 1u,
            })) {
      return manifest;
    }
    break;
  }
  }
  const bool complete = PlanVulkanCaptureManifest(
                            step, plan, bound, max_dispatch_groups, manifest) &&
                        CompleteVulkanBackendManifest(manifest);
  (void)complete;
  return manifest;
}
#else
PreparedBackendManifest
BuildVulkanBackendManifest(const KernelExecutionStep &,
                           const rund::kernel::ComputePlan &, const BoundStep *,
                           std::uint64_t) noexcept {
  return {};
}
#endif

rund::AccelCheck PlanVulkanPipelineRecurrence(
    const MapRecurrencePreparationPlan &plan,
    PreparedMapRecurrenceReservation &reservation) noexcept {
  reservation = {};
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using backend_template_plan::add;
  using backend_template_plan::product;
  if (!plan.ok) {
    return rund::AccelCheck{
        false, plan.reason == nullptr ? "compute_pipeline_recurrence_invalid"
                                      : plan.reason};
  }
  if (plan.group_count == 0u) {
    return plan.history_group_count == 0u
               ? rund::AccelCheck{true, "ok"}
               : rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const std::uint64_t terminal_group_count = plan.terminal_group_count();
  const bool terminal_variant = terminal_group_count != 0u;
  const bool history_variant = plan.history_group_count != 0u;
  const std::uint64_t template_count =
      static_cast<std::uint64_t>(terminal_variant) +
      static_cast<std::uint64_t>(history_variant);
  if (plan.authority == nullptr || plan.canonical_artifact == nullptr ||
      plan.canonical_artifact != &plan.authority->artifact ||
      plan.authority->kind() != rund::kernel::NodeKind::Map ||
      plan.history_group_count > plan.group_count || template_count == 0u ||
      template_count > 2u || !plan.plan.ok ||
      plan.plan.api != rund::kernel::ComputeApi::Vulkan ||
      plan.canonical_artifact->key.api !=
          rund::kernel::ComputeApi::Vulkan ||
      plan.canonical_artifact->key.variant !=
          rund::kernel::LoweringArtifactVariant::Canonical ||
      plan.canonical_artifact->kind !=
          rund::kernel::LoweringArtifactKind::VulkanSource ||
      !plan.canonical_artifact->ok || !plan.canonical_artifact->metadata.ok ||
      !plan.canonical_artifact->metadata.read_routes.empty() ||
      plan.window_count == 0u ||
      plan.window_count != plan.plan.dispatch_count ||
      plan.input_count != plan.plan.input_buffer_count ||
      plan.output_count != plan.plan.output_buffer_count ||
      plan.input_layouts().size() != plan.input_count ||
      plan.output_layouts().size() != plan.output_count ||
      plan.terminal_source.ok != terminal_variant ||
      plan.history_source.ok != history_variant ||
      (terminal_variant
           ? plan.terminal_template_group_capacity < terminal_group_count
           : plan.terminal_template_group_capacity != 0u) ||
      (history_variant
           ? plan.history_template_group_capacity < plan.history_group_count
           : plan.history_template_group_capacity != 0u)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (const PreparedKernelProgramBindingIdentity identity :
       plan.input_layouts()) {
    if (identity.element_bytes == 0u || identity.count == 0u ||
        identity.stride_bytes < identity.element_bytes ||
        identity.usage != rund::kernel::kResidentUsageRead) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
  }
  for (const PreparedKernelProgramBindingIdentity identity :
       plan.output_layouts()) {
    if (identity.element_bytes == 0u || identity.count == 0u ||
        identity.stride_bytes < identity.element_bytes ||
        identity.usage != rund::kernel::kResidentUsageWrite) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
  }

  std::uint64_t per_route_host = sizeof(VulkanMapEncodeResources);
  std::uint64_t window_bytes = 0u;
  std::uint64_t routes_host = 0u;
  std::uint64_t history_host = 0u;
  std::uint64_t per_route_native = std::max<std::uint64_t>(4u,
                                                           plan.plan.param_bytes);
  if (!AddVulkanHostBytes(per_route_host, plan.input_count,
                          sizeof(VulkanResidentBufferResult)) ||
      !AddVulkanHostBytes(per_route_host, plan.output_count,
                          sizeof(VulkanResidentBufferResult)) ||
      !product(plan.window_count,
               sizeof(rund::kernel::ComputeDispatchWindow), window_bytes) ||
      !add(per_route_host, window_bytes) ||
      !product(per_route_host, plan.group_count, routes_host) ||
      !product(plan.history_group_count, sizeof(MapRecurrenceHistory),
               history_host) ||
      !add(routes_host, history_host) ||
      !product(per_route_native, plan.group_count,
               reservation.route_native_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_host_bytes = routes_host;

  std::uint64_t descriptors_per_set = 0u;
  if (!rund::kernel::checked::add(plan.input_count, plan.output_count,
                                  descriptors_per_set) ||
      !add(descriptors_per_set, 1u)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const auto add_template = [&](const MapRecurrenceSourcePlan &source,
                                const bool history,
                                const std::uint64_t group_capacity) noexcept {
    std::uint64_t expected_source_storage = 0u;
    std::uint64_t final_source_upper = 0u;
    std::uint64_t final_source_storage = 0u;
    std::uint64_t descriptor_sets = 0u;
    std::uint64_t descriptor_bindings = 0u;
    std::uint64_t host = sizeof(VulkanMapRecurrenceTemplate);
    std::uint64_t native_objects = 0u;
    if (!source.ok || source.history != history ||
        source.exact_source_bytes == 0u ||
        source.source_upper_bytes < source.exact_source_bytes ||
        !backend_source_recipe::string_external_storage_upper_bytes(
            source.source_upper_bytes, expected_source_storage) ||
        source.source_storage_upper_bytes != expected_source_storage ||
        !MapSpecializedSourceUpperBytes(source.exact_source_bytes,
                                        source.source_upper_bytes, plan.plan,
                                        final_source_upper) ||
        !backend_source_recipe::string_external_storage_upper_bytes(
            final_source_upper, final_source_storage) ||
        group_capacity == 0u ||
        !product(group_capacity, plan.window_count, descriptor_sets) ||
        descriptor_sets == 0u ||
        descriptor_sets > std::numeric_limits<std::uint32_t>::max() ||
        !add(host, sizeof(VulkanMapTemplateResources)) ||
        !add(host, sizeof(VulkanMapDescriptorArena)) ||
        !AddVulkanHostBytes(host, plan.input_count, sizeof(InputWindowPlan)) ||
        !AddVulkanHostBytes(host, plan.input_count,
                            sizeof(VulkanMapBindingLayout)) ||
        !AddVulkanHostBytes(host, plan.output_count,
                            sizeof(VulkanMapBindingLayout)) ||
        !add(host, sizeof(VulkanCachedPipeline)) ||
        !add(host, final_source_storage) ||
        !AddVulkanHostBytes(host, descriptor_sets, sizeof(VkDescriptorSet)) ||
        !add(reservation.template_host_bytes, host) ||
        !add(reservation.template_source_bytes, final_source_upper) ||
        !add(reservation.descriptor_set_count, descriptor_sets) ||
        !product(descriptor_sets, descriptors_per_set, descriptor_bindings) ||
        !add(reservation.descriptor_count, descriptor_bindings) ||
        !rund::kernel::checked::add(descriptor_sets, 4u, native_objects) ||
        !add(reservation.template_native_allocation_count, native_objects)) {
      return false;
    }
    // The final source allocation moves into VulkanCachedPipeline and is
    // retained template storage. Only copied semantic metadata dies after
    // the in-place specialization/cache move.
    reservation.source_transient_bytes =
        std::max(reservation.source_transient_bytes,
                 source.metadata_storage_upper_bytes);
    return true;
  };
  if ((terminal_variant &&
       !add_template(plan.terminal_source, false,
                     plan.terminal_template_group_capacity)) ||
      (history_variant &&
       !add_template(plan.history_source, true,
                     plan.history_template_group_capacity))) {
    reservation = {};
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

  reservation.group_count = plan.group_count;
  reservation.history_group_count = plan.history_group_count;
  reservation.template_count = template_count;
  reservation.terminal_template_group_capacity =
      plan.terminal_template_group_capacity;
  reservation.history_template_group_capacity =
      plan.history_template_group_capacity;
  reservation.route_step_count = plan.group_count;
  reservation.template_step_count = template_count;
  reservation.route_native_allocation_count = plan.group_count;
  return rund::AccelCheck{true, "ok"};
#else
  (void)plan;
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
#endif
}

[[nodiscard]] static rund::AccelCheck PlanVulkanPipelineStructureCandidate(
    const rund::AccelContext &,
    PreparedKernelPipelineReservation &reservation) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using backend_template_plan::add;
  using backend_template_plan::product;
  const bool has_windows = reservation.window_count != 0u;
  if (reservation.backend_terminal_publication_count >
          reservation.backend_publication_count ||
      reservation.backend_window_descriptor_state_count >
          reservation.backend_window_state_count ||
      reservation.backend_indirect_dispatch_count >
          reservation.backend_window_dispatch_count ||
      (has_windows &&
       (reservation.backend_window_dispatch_count == 0u ||
        reservation.backend_window_state_count == 0u ||
        reservation.backend_window_descriptor_state_count == 0u)) ||
      (!has_windows &&
       (reservation.backend_window_dispatch_count != 0u ||
        reservation.backend_indirect_dispatch_count != 0u ||
        reservation.backend_window_state_count != 0u ||
        reservation.backend_window_descriptor_state_count != 0u))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.backend_query_count = 0u;
  if (!product(2u, reservation.window_count,
               reservation.backend_window_control_command_count) ||
      (reservation.backend_profile_command_count != 0u &&
       !product(2u, reservation.backend_profile_command_count,
                reservation.backend_query_count))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t commands = reservation.backend_dispatch_count;
  for (const std::uint64_t count :
       {reservation.backend_reset_dispatch_count,
        reservation.backend_status_command_count,
        reservation.backend_telemetry_command_count,
        reservation.backend_window_control_command_count,
        reservation.backend_indirect_dispatch_count,
        reservation.backend_publication_command_count, std::uint64_t{3u}}) {
    if (!add(commands, count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  std::uint64_t host = sizeof(VulkanPipeline);
  std::uint64_t bytes = 0u;
  std::uint64_t control_lease_count = 0u;
  std::uint64_t publication_descriptor_count = 0u;
  std::uint64_t window_descriptor_count = 0u;
  std::uint64_t control_binding_count = 0u;
  std::uint64_t publication_binding_count = 0u;
  std::uint64_t window_binding_count = 0u;
  std::uint64_t pipeline_descriptor_sets = 0u;
  std::uint64_t pipeline_descriptors = 0u;
  if (!rund::kernel::checked::add(reservation.backend_status_source_count, 1u,
                                  control_lease_count) ||
      !rund::kernel::checked::add(control_lease_count,
                                  reservation.backend_telemetry_count,
                                  control_lease_count) ||
      !rund::kernel::checked::add(
          reservation.backend_publication_count,
          reservation.backend_terminal_publication_count,
          publication_descriptor_count) ||
      !rund::kernel::checked::add(
          reservation.backend_window_descriptor_state_count,
          reservation.backend_indirect_dispatch_count,
          window_descriptor_count) ||
      !product(reservation.backend_status_source_count, 2u,
               control_binding_count) ||
      !rund::kernel::checked::add(control_binding_count, 2u,
                                  control_binding_count) ||
      !product(reservation.backend_telemetry_count,
               reservation.backend_profile_step_count == 0u ? 5u : 6u, bytes) ||
      !rund::kernel::checked::add(control_binding_count, bytes,
                                  control_binding_count) ||
      !product(publication_descriptor_count, 7u, publication_binding_count) ||
      !product(reservation.backend_window_descriptor_state_count, 8u,
               window_binding_count) ||
      !product(reservation.backend_indirect_dispatch_count, 3u, bytes) ||
      !rund::kernel::checked::add(window_binding_count, bytes,
                                  window_binding_count) ||
      !rund::kernel::checked::add(control_lease_count,
                                  publication_descriptor_count,
                                  pipeline_descriptor_sets) ||
      !rund::kernel::checked::add(pipeline_descriptor_sets,
                                  window_descriptor_count,
                                  pipeline_descriptor_sets) ||
      !rund::kernel::checked::add(control_binding_count,
                                  publication_binding_count,
                                  pipeline_descriptors) ||
      !rund::kernel::checked::add(pipeline_descriptors, window_binding_count,
                                  pipeline_descriptors)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t pipeline_source_bytes = 0u;
  std::uint64_t pipeline_source_storage = 0u;
  std::uint64_t pipeline_source_transient = 0u;
  std::uint64_t pipeline_source_dependency_count = 0u;
  const auto add_pipeline_source = [&](const std::uint64_t source_bytes) {
    std::uint64_t storage_bytes = 0u;
    return source_bytes != 0u &&
           backend_source_recipe::string_external_storage_upper_bytes(
               source_bytes, storage_bytes) &&
           add(pipeline_source_bytes, source_bytes) &&
           add(pipeline_source_storage, storage_bytes) &&
           add(pipeline_source_dependency_count, 1u) &&
           (pipeline_source_transient =
                std::max(pipeline_source_transient, storage_bytes),
            true);
  };
  const std::uint64_t telemetry_source_bytes =
      reservation.backend_profile_step_count == 0u
          ? VulkanTelemetrySourceText().size()
          : VulkanProfileSourceBytes();
  if (!add_pipeline_source(VulkanReduceStatusSourceText().size()) ||
      (reservation.backend_status_source_count != 0u &&
       !add_pipeline_source(VulkanCanonicalStatusSourceText().size())) ||
      (reservation.backend_telemetry_count != 0u &&
       !add_pipeline_source(telemetry_source_bytes)) ||
      (has_windows && !add_pipeline_source(VulkanWindowSourceText().size())) ||
      (reservation.backend_indirect_dispatch_count != 0u &&
       !add_pipeline_source(VulkanGateSourceText().size())) ||
      (reservation.backend_publication_count != 0u &&
       !add_pipeline_source(VulkanPublishSourceText().size()))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t pipeline_cache_host = 0u;
  std::uint64_t pipeline_cache_native = 0u;
  if (!product(pipeline_descriptor_sets,
               sizeof(VkDescriptorSet) + sizeof(std::uint8_t),
               pipeline_cache_host) ||
      !product(pipeline_source_dependency_count, sizeof(VkDescriptorPool),
               bytes) ||
      !add(pipeline_cache_host, bytes) ||
      !product(pipeline_source_dependency_count,
               sizeof(VulkanCollectivePipeline), bytes) ||
      !add(pipeline_cache_host, bytes) ||
      !add(pipeline_cache_host, pipeline_source_storage) ||
      !product(pipeline_source_dependency_count, 4u, pipeline_cache_native) ||
      !add(pipeline_cache_native, pipeline_descriptor_sets) ||
      !add(reservation.template_source_bytes, pipeline_source_bytes) ||
      !add(reservation.template_host_bytes, pipeline_cache_host) ||
      !add(host, pipeline_cache_host)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.source_transient_bytes =
      std::max(reservation.source_transient_bytes, pipeline_source_transient);
  if (reservation.backend_profile_step_count != 0u &&
      !add(host, sizeof(VulkanPipelineProfile))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!product(reservation.backend_telemetry_count,
               sizeof(VulkanPipelineTelemetryRecord), bytes) ||
      !add(host, bytes) ||
      !product(reservation.nested_group_count, sizeof(std::shared_ptr<void>),
               bytes) ||
      !add(host, bytes) ||
      !product(control_lease_count, sizeof(VulkanCollectiveDescriptorLease),
               bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_publication_count,
               sizeof(VulkanPipelinePublishRoute), bytes) ||
      !add(host, bytes) ||
      !product(publication_descriptor_count,
               sizeof(VulkanCollectiveDescriptorLease), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_window_dispatch_count,
               sizeof(VkDispatchIndirectCommand), bytes) ||
      !add(host, bytes) ||
      !product(reservation.window_count, sizeof(VulkanWindowRoute), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_indirect_dispatch_count,
               sizeof(VulkanGateRoute), bytes) ||
      !add(host, bytes) ||
      !product(window_descriptor_count, sizeof(VulkanCollectiveDescriptorLease),
               bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_profile_command_count,
               sizeof(std::uint8_t) + sizeof(std::uint32_t), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_query_count, sizeof(std::uint64_t), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_telemetry_command_count,
               reservation.backend_profile_step_count == 0u
                   ? sizeof(VulkanPipelineTelemetryParams)
                   : sizeof(VulkanPipelineProfileTelemetryParams),
               bytes) ||
      !add(reservation.backend_parameter_bytes, bytes) ||
      !product(reservation.backend_window_control_command_count,
               sizeof(VulkanWindowParams), bytes) ||
      !add(reservation.backend_parameter_bytes, bytes) ||
      !product(reservation.backend_indirect_dispatch_count,
               VulkanGateParameterBytes, bytes) ||
      !add(reservation.backend_parameter_bytes, bytes) ||
      !product(reservation.backend_publication_command_count,
               sizeof(VulkanPipelinePublishParams), bytes) ||
      !add(reservation.backend_parameter_bytes, bytes) ||
      !add(reservation.backend_parameter_bytes,
           2u * VulkanPipelineControlParameterBytes) ||
      !add(reservation.host_bytes, host)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.backend_command_count = commands;
  reservation.backend_command_binding_count = pipeline_descriptors;
  if (!add(reservation.descriptor_set_count, pipeline_descriptor_sets) ||
      !add(reservation.descriptor_count, pipeline_descriptors)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Control owns summary unconditionally, arena only with status entries,
  // and profile only when requested. A live window owns four exact buffers:
  // states, mutable arguments, frozen original arguments, and owners.
  reservation.backend_native_buffer_count =
      1u +
      static_cast<std::uint64_t>(reservation.backend_status_entry_count != 0u) +
      static_cast<std::uint64_t>(reservation.backend_profile_step_count != 0u) +
      (has_windows ? 4u : 0u);
  reservation.backend_native_object_count = 3u;
  if (!add(reservation.backend_native_object_count,
           reservation.backend_native_buffer_count) ||
      !add(reservation.backend_native_object_count, pipeline_cache_native) ||
      (reservation.backend_query_count != 0u &&
       !add(reservation.backend_native_object_count, 1u))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t native_payload = sizeof(PreparedPipelineControl);
  if (!product(reservation.backend_status_entry_count, sizeof(std::uint32_t),
               bytes) ||
      !add(native_payload, bytes) ||
      !product(reservation.backend_profile_step_count,
               PreparedPipelineStepControlBytes, bytes) ||
      !add(native_payload, bytes) ||
      !product(reservation.backend_window_state_count, sizeof(ResidentState),
               bytes) ||
      !add(native_payload, bytes) ||
      !product(reservation.backend_window_dispatch_count,
               2u * sizeof(VkDispatchIndirectCommand), bytes) ||
      !add(native_payload, bytes) ||
      !product(reservation.backend_window_dispatch_count, sizeof(std::uint32_t),
               bytes) ||
      !add(native_payload, bytes) ||
      !add(reservation.native_bytes, native_payload) ||
      !add(reservation.native_bytes, reservation.backend_parameter_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t descriptor_owner_bytes = 0u;
  std::uint64_t profile_description_bytes = 0u;
  if (!product(reservation.backend_window_state_count, sizeof(std::size_t),
               descriptor_owner_bytes) ||
      (reservation.backend_profile_step_count != 0u &&
       (!product(reservation.nested_group_count,
                 sizeof(VulkanPipelineWork) + sizeof(std::uint64_t),
                 profile_description_bytes))) ||
      !add(reservation.host_transient_bytes, descriptor_owner_bytes) ||
      !add(reservation.host_transient_bytes, profile_description_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)reservation;
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
#endif
}

rund::AccelCheck PlanVulkanPipelineStructure(
    const rund::AccelContext &context,
    PreparedKernelPipelineReservation &reservation) noexcept {
  PreparedKernelPipelineReservation candidate = reservation;
  const rund::AccelCheck planned =
      PlanVulkanPipelineStructureCandidate(context, candidate);
  if (planned.ok) {
    reservation = candidate;
  }
  return planned;
}

PreparedMemory
ObserveVulkanPipelineTemplate(const void *const prepared) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  std::uint64_t bytes = 0u;
  bool observed = false;
  if (prepared != nullptr) {
    switch (VulkanKernelTemplateKindOf(prepared)) {
    case VulkanKernelTemplateKind::Program:
      observed = ObserveVulkanProgramTemplate(
          *static_cast<const VulkanKernelProgramTemplate *>(prepared), bytes);
      break;
    case VulkanKernelTemplateKind::MapRecurrence:
      observed = ObserveVulkanRecurrenceTemplate(
          *static_cast<const VulkanMapRecurrenceTemplate *>(prepared), bytes);
      break;
    }
  }
  if (!observed || bytes == 0u) {
    constexpr std::uint64_t invalid =
        std::numeric_limits<std::uint64_t>::max();
    return PreparedMemory{.current = invalid,
                          .peak = invalid,
                          .cumulative = invalid,
                          .budget = invalid};
  }
  return PreparedMemory{.current = bytes,
                        .peak = bytes,
                        .cumulative = bytes,
                        .budget = bytes};
#else
  (void)prepared;
  return {};
#endif
}

rund::AccelCheck RunVulkanKernel(const BackendRun &run) {
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::shared_ptr<void> prepared{};
  PreparedMemory memory{};
  const rund::AccelCheck ready = PrepareVulkanResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::Standalone, run.resets, run.views, run.view_binds,
      nullptr, nullptr, nullptr, run.failed_node, prepared, memory);
  if (ready.ok && run.traffic != nullptr) {
    *run.traffic = VulkanKernelTraffic(prepared);
  }
  return ready.ok ? RunVulkanResources(*run.pick, prepared) : ready;
}

rund::AccelCheck PrepareVulkanKernel(const BackendRun &run,
                                     std::shared_ptr<void> &prepared,
                                     PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareVulkanResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::Standalone, run.resets, run.views, run.view_binds,
      nullptr, nullptr, nullptr, run.failed_node, prepared, memory);
}

rund::AccelCheck
PrepareVulkanPipelinePrivateKernel(const BackendRun &run,
                                   std::shared_ptr<void> &prepared,
                                   PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareVulkanResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::PipelinePrivate, run.resets, run.views,
      run.view_binds, run.scratch, &run, run.templates, run.failed_node,
      prepared, memory);
}

rund::AccelCheck PlanVulkanPipelinePrivateKernel(
    const BackendRun &run,
    PreparedKernelRouteReservation &reservation) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter =
      run.pick == nullptr ? nullptr : CheckedVulkanAdapter(*run.pick);
  const std::uint64_t alignment =
      adapter == nullptr ? 0u : run.pick->caps.storage_alignment;
  const rund::AccelCheck planned = backend_template_plan::plan(
      run,
      VulkanBackendShape(
          alignment, adapter == nullptr ? 0u : adapter->max_dispatch_groups),
      reservation);
  return !planned.ok || adapter == nullptr
             ? planned
             : CompleteVulkanRouteCaptureStructure(
                   run.views, run.resets == nullptr ? 0u : run.resets->size(),
                   *adapter, reservation);
#else
  (void)run;
  reservation = {};
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
#endif
}

rund::AccelCheck PlanVulkanPipelineProgram(
    const KernelExecution &execution, const PreparedKernelProgramRoute &route,
    PreparedKernelRouteReservation &reservation) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const rund::AccelDevice *const pick =
      execution.context_admission.pick == nullptr
          ? nullptr
          : &execution.context_admission.pick->raw;
  VulkanAdapter *const adapter =
      pick == nullptr ? nullptr : CheckedVulkanAdapter(*pick);
  const rund::AccelCheck planned = backend_template_plan::plan_program(
      execution, route,
      VulkanBackendShape(execution.admission.frozen_caps.storage_alignment,
                         adapter == nullptr ? 0u
                                            : adapter->max_dispatch_groups),
      reservation);
  return !planned.ok || adapter == nullptr
             ? planned
             : CompleteVulkanRouteCaptureStructure(
                   route.views, execution.resets.size(), *adapter, reservation);
#else
  (void)execution;
  (void)route;
  reservation = {};
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
#endif
}

bool SameVulkanPipelineProgramTemplate(
    const KernelExecution &execution, const PreparedKernelProgramRoute &left,
    const PreparedKernelProgramRoute &right) noexcept {
  const std::uint64_t alignment =
      left.kernel == nullptr ? 0u : left.kernel->frozen_caps.storage_alignment;
  return backend_template_plan::same_program_template(execution, left, right,
                                                      alignment);
}

bool SameVulkanPipelineTemplate(const BackendRun &left,
                                const BackendRun &right) noexcept {
  const std::uint64_t alignment =
      left.pick == nullptr ? 0u : left.pick->caps.storage_alignment;
  const std::size_t left_reset_count =
      left.resets == nullptr ? 0u : left.resets->size();
  const std::size_t right_reset_count =
      right.resets == nullptr ? 0u : right.resets->size();
  return alignment != 0u && left_reset_count == right_reset_count &&
         backend_template_plan::same_template(left, right, alignment);
}

rund::AccelCheck SubmitPreparedVulkanKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    const KernelCompletion completion, void *const user, PreparedMemoryMeter *,
    const std::shared_ptr<void> &) noexcept {
  return run.pick != nullptr && prepared != nullptr && completion != nullptr
             ? SubmitVulkanResources(*run.pick, prepared, completion, user)
             : rund::AccelCheck{false, "accel_kernel_run_invalid"};
}

} // namespace rund::node::accel::detail

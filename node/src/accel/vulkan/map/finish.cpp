#include <accel/check.hpp>

#include "../../kernel/backend/template_plan.hpp"
#include "../../kernel/step/map/stride.hpp"
#include "../collective/finish.hpp"
#include "../kernel/ops/status.hpp"
#include "../status.hpp"
#include "api.hpp"
#include "control.hpp"
#include "encode/window.hpp"
#include "local.hpp"
#include "resources/admission.hpp"
#include <rund/counter.hpp>

#include <rund/compute/reason.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanMapEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanMapEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  if (resources->adapter != nullptr) {
    ReleaseVulkanStatus(*resources->adapter, resources->control_status);
  }
  delete resources;
}
#endif

namespace {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] bool
SameVulkanMapTemplateBindings(const VulkanMapTemplateResources &prepared,
                              const rund::kernel::BindingSet &bindings,
                              const std::uint64_t alignment) noexcept {
  const auto same =
      [alignment](const std::vector<VulkanMapBindingLayout> &layouts,
                  const rund::kernel::ResidentBindingRange &refs) {
        if (alignment == 0u || layouts.size() != refs.count) {
          return false;
        }
        for (std::size_t index = 0u; index < layouts.size(); ++index) {
          const auto *const ref = refs.ref(index);
          if (ref == nullptr || ref->stride_bytes != layouts[index].stride ||
              ref->offset_bytes % alignment != layouts[index].base) {
            return false;
          }
        }
        return true;
      };
  return same(prepared.input_layouts, bindings.resident_inputs) &&
         same(prepared.output_layouts, bindings.resident_outputs);
}

[[nodiscard]] bool PrepareStandaloneVulkanMapDescriptorDemand(
    VulkanAdapter &adapter, const VulkanMapTemplateResources &prepared,
    std::shared_ptr<VulkanMapDescriptorArena> &descriptors) {
  if (prepared.pipeline == nullptr || prepared.plan.dispatch_count == 0u ||
      !PrepareVulkanMapDescriptorArena(adapter, *prepared.pipeline,
                                       prepared.plan.dispatch_count,
                                       descriptors)) {
    return false;
  }
  if (prepared.control_pipeline != nullptr &&
      !ReserveVulkanCollectiveDescriptorDemand(
          adapter, *prepared.control_pipeline,
          prepared.control_pipeline->descriptor_count, 1u)) {
    return false;
  }
  return prepared.check_pipeline == nullptr ||
         ReserveVulkanCollectiveDescriptorDemand(
             adapter, *prepared.check_pipeline,
             prepared.check_pipeline->descriptor_count, 1u);
}

[[nodiscard]] rund::AccelCheck PrepareVulkanMapRouteResources(
    VulkanAdapter &adapter, const rund::AccelDevice &pick,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    const bool history_recurrence,
    std::shared_ptr<const VulkanMapTemplateResources> prepared,
    std::shared_ptr<VulkanMapDescriptorArena> descriptors,
    std::shared_ptr<void> &resources, const rund::kernel::u32 iterations) {
  auto *const raw = new VulkanMapEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanMapEncodeResources};
  raw->adapter = &adapter;
  raw->prepared = std::move(prepared);
  raw->iterations = iterations;
  raw->history_recurrence = history_recurrence;
  raw->bindings = bindings;
  raw->windows.assign(windows, windows + window_count);
  const char *history_reason = "ok";
  if (!ValidateVulkanMapHistoryOutputs(*raw, history_reason)) {
    SetVulkanLastError(adapter, history_reason);
    return rund::AccelCheck{false, history_reason};
  }
  if (!PrepareVulkanResidentBindings(adapter, raw->prepared->plan,
                                     raw->bindings, raw->resident)) {
    SetVulkanLastError(adapter, raw->resident.reason);
    return rund::AccelCheck{false, raw->resident.reason};
  }
  if (!MakeVulkanMapHostBuffer(adapter, bindings.param_data, plan.param_bytes,
                               raw->param)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (!PrepareVulkanMapControl(pick, control, raw->windows, *raw)) {
    const char *const reason = VulkanLastError(&adapter);
    return rund::AccelCheck{false, reason == nullptr || reason[0] == '\0'
                                       ? "compute_plan_invalid"
                                       : reason};
  }
  if (!AcquireVulkanMapDescriptorSets(descriptors, window_count,
                                      raw->descriptor_sets)) {
    return rund::AccelCheck{false, "accel_vulkan_descriptor_unavailable"};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace

bool VulkanMapTemplateMatches(
    const VulkanMapTemplateResources &prepared, const VulkanAdapter &adapter,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return prepared.adapter == &adapter &&
         backend_template_plan::same_plan(prepared.plan, plan) &&
         SameVulkanMapTemplateBindings(prepared, bindings,
                                       adapter.storage_align);
#else
  (void)prepared;
  (void)adapter;
  (void)plan;
  (void)bindings;
  return false;
#endif
}

[[nodiscard]] static rund::AccelCheck PrepareVulkanMapTemplateImpl(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    rund::kernel::LoweringArtifact *const owned_artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> &prepared) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  prepared.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  const rund::AccelCheck valid = ValidateVulkanMapPrepare(
      *adapter, plan, artifact, windows, window_count, bindings);
  if (!valid.ok) {
    return valid;
  }

  auto owned = std::make_shared<VulkanMapTemplateResources>();
  VulkanMapTemplateResources *const raw = owned.get();
  raw->adapter = adapter;
  raw->plan = plan;
  raw->input_plans.resize(static_cast<std::size_t>(plan.input_buffer_count));
  if (!FreezeInputWindowPlans(artifact.metadata, plan.tile_count,
                              raw->input_plans)) {
    return rund::AccelCheck{false, "compute_binding_mismatch"};
  }
  const auto freeze_layout = [&](const rund::kernel::ResidentBindingRange &refs,
                                 std::vector<VulkanMapBindingLayout> &layouts) {
    layouts.reserve(static_cast<std::size_t>(refs.count));
    for (std::uint64_t index = 0u; index < refs.count; ++index) {
      const auto *const ref = refs.ref(index);
      if (ref == nullptr || adapter->storage_align == 0u) {
        return false;
      }
      layouts.push_back(VulkanMapBindingLayout{
          .stride = ref->stride_bytes,
          .base = ref->offset_bytes % adapter->storage_align,
      });
    }
    return true;
  };
  if (!freeze_layout(bindings.resident_inputs, raw->input_layouts) ||
      !freeze_layout(bindings.resident_outputs, raw->output_layouts)) {
    return rund::AccelCheck{false, "compute_binding_mismatch"};
  }
  for (const rund::kernel::ReadRoute route : artifact.metadata.read_routes) {
    const auto *const ref = bindings.resident_inputs.ref(route.index);
    if (ref == nullptr || adapter->storage_align == 0u) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
    const auto found = std::find_if(raw->checks.begin(), raw->checks.end(),
                                    [&](const VulkanMapCheck check) {
                                      return check.binding == route.index;
                                    });
    if (found == raw->checks.end()) {
      raw->checks.push_back(VulkanMapCheck{
          .binding = route.index,
          .limit = route.count,
          .offset = ref->offset_bytes % adapter->storage_align,
          .stride = ref->stride_bytes,
      });
    } else {
      found->limit =
          std::min(found->limit, static_cast<std::uint64_t>(route.count));
    }
  }
  const bool controlled_source = control.active() || !raw->checks.empty();
  std::uint64_t specialized_upper = 0u;
  std::uint64_t final_upper = 0u;
  if (!MapSpecializedSourceUpperBytes(artifact, plan, specialized_upper) ||
      (controlled_source && !VulkanControlledMapSourceUpperBytes(
                                plan, specialized_upper, final_upper))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!controlled_source) {
    final_upper = specialized_upper;
  }
  rund::kernel::LoweringArtifact strided_artifact =
      owned_artifact == nullptr
          ? SpecializeMap(artifact, plan, bindings, adapter->storage_align,
                          final_upper)
          : SpecializeMapInPlace(std::move(*owned_artifact), plan, bindings,
                                 adapter->storage_align, final_upper);
  rund::kernel::LoweringArtifact controlled_artifact =
      controlled_source
          ? VulkanControlledMapArtifact(std::move(strided_artifact), plan)
          : std::move(strided_artifact);
  if (!controlled_artifact.ok) {
    return rund::AccelCheck{false, controlled_artifact.reason};
  }
  rund::kernel::ComputePlan pipeline_plan = plan;
  if (control.active() || !raw->checks.empty()) {
    ++pipeline_plan.input_buffer_count;
  }
  raw->pipeline = AcquireVulkanCachedPipeline(*adapter, pipeline_plan,
                                              std::move(controlled_artifact));
  if (raw->pipeline == nullptr) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  if (controlled_source) {
    const rund::kernel::LoweringArtifact control_artifact =
        VulkanMapControlArtifact(plan);
    raw->control_pipeline = AcquireVulkanCollectivePipeline(
        *adapter, 4u, sizeof(VulkanMapControlPush), plan, control_artifact);
    if (raw->control_pipeline == nullptr) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
  }
  if (!raw->checks.empty()) {
    const std::uint64_t descriptor_count = raw->checks.size() + 3u;
    if (descriptor_count > std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const rund::kernel::LoweringArtifact check_artifact =
        VulkanMapCheckArtifact(*raw);
    raw->check_pipeline = AcquireVulkanCollectivePipeline(
        *adapter, static_cast<std::uint32_t>(descriptor_count),
        sizeof(VulkanMapControlPush), plan, check_artifact);
    if (raw->check_pipeline == nullptr) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
  }
  prepared = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)plan;
  (void)artifact;
  (void)owned_artifact;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
  (void)prepared;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck PrepareVulkanMapTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> &prepared) {
  return PrepareVulkanMapTemplateImpl(pick, plan, artifact, nullptr, windows,
                                      window_count, bindings, control,
                                      prepared);
}

rund::AccelCheck PrepareVulkanMapOwnedTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    rund::kernel::LoweringArtifact &&artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> &prepared) {
  return PrepareVulkanMapTemplateImpl(pick, plan, artifact, &artifact, windows,
                                      window_count, bindings, control,
                                      prepared);
}

rund::AccelCheck PrepareVulkanMapRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> prepared,
    std::shared_ptr<VulkanMapDescriptorArena> descriptors,
    std::shared_ptr<void> &resources, const rund::kernel::u32 iterations) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  const bool history_recurrence =
      artifact.key.variant ==
      rund::kernel::LoweringArtifactVariant::HistoryRecurrence;
  if (iterations == 0u || (history_recurrence && iterations < 2u)) {
    return rund::AccelCheck{false,
                            "compute_pipeline_recurrence_history_invalid"};
  }
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || prepared == nullptr ||
      !VulkanMapTemplateMatches(*prepared, *adapter, plan, bindings)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  const rund::AccelCheck valid = ValidateVulkanMapPrepare(
      *adapter, plan, artifact, windows, window_count, bindings);
  if (!valid.ok) {
    return valid;
  }
  return PrepareVulkanMapRouteResources(
      *adapter, pick, plan, windows, window_count, bindings, control,
      history_recurrence, std::move(prepared), std::move(descriptors),
      resources, iterations);
#else
  (void)pick;
  (void)plan;
  (void)artifact;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
  (void)prepared;
  (void)descriptors;
  (void)resources;
  (void)iterations;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck PrepareVulkanMapProvedRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    const bool history_recurrence,
    std::shared_ptr<const VulkanMapTemplateResources> prepared,
    std::shared_ptr<VulkanMapDescriptorArena> descriptors,
    std::shared_ptr<void> &resources, const rund::kernel::u32 iterations) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  if (iterations == 0u || (history_recurrence && iterations < 2u) ||
      (iterations != 1u && control.active())) {
    return rund::AccelCheck{false,
                            history_recurrence
                                ? "compute_pipeline_recurrence_history_invalid"
                                : "compute_pipeline_invalid"};
  }
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || prepared == nullptr || descriptors == nullptr ||
      descriptors->adapter != adapter ||
      !VulkanMapTemplateMatches(*prepared, *adapter, plan, bindings)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  // Artifact/source admission belongs to common recurrence proof and the
  // immutable template miss path. Only route-varying windows and bindings
  // remain here; rebuilding an artifact would restore the removed layer.
  if (!RuntimeWindowsMatchPlan(plan, windows, window_count, bindings) ||
      !bindings.has_resident_output()) {
    SetVulkanLastError(*adapter, "compute_dispatch_count_mismatch");
    return rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
  }
  return PrepareVulkanMapRouteResources(
      *adapter, pick, plan, windows, window_count, bindings, control,
      history_recurrence, std::move(prepared), std::move(descriptors),
      resources, iterations);
#else
  (void)pick;
  (void)plan;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
  (void)history_recurrence;
  (void)prepared;
  (void)descriptors;
  (void)resources;
  (void)iterations;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck PrepareVulkanMap(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<void> &resources, const rund::kernel::u32 iterations) {
  std::shared_ptr<const VulkanMapTemplateResources> prepared;
  const rund::AccelCheck template_ready = PrepareVulkanMapTemplate(
      pick, plan, artifact, windows, window_count, bindings, control, prepared);
  if (!template_ready.ok) {
    return template_ready;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const adapter = CheckedVulkanAdapter(pick);
  std::shared_ptr<VulkanMapDescriptorArena> descriptors;
  if (adapter == nullptr || prepared == nullptr ||
      !PrepareStandaloneVulkanMapDescriptorDemand(*adapter, *prepared,
                                                  descriptors)) {
    return rund::AccelCheck{false, adapter == nullptr
                                       ? "accel_vulkan_unavailable"
                                       : VulkanLastError(adapter)};
  }
  return PrepareVulkanMapRoute(pick, plan, artifact, windows, window_count,
                               bindings, control, std::move(prepared),
                               std::move(descriptors), resources, iterations);
#else
  (void)resources;
  (void)iterations;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck EncodeVulkanMap(VulkanAdapter &adapter,
                                 const std::shared_ptr<void> &resources,
                                 void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const map = static_cast<VulkanMapEncodeResources *>(resources.get());
  const VkCommandBuffer command_buffer =
      reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (map == nullptr || map->adapter != &adapter || map->prepared == nullptr ||
      command_buffer == VK_NULL_HANDLE || map->prepared->pipeline == nullptr ||
      map->param.buffer.buffer == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  const std::uint32_t descriptor_count =
      static_cast<std::uint32_t>(map->prepared->plan.input_buffer_count +
                                 map->prepared->plan.output_buffer_count + 1u +
                                 (map->controlled() ? 1u : 0u));
  if (map->descriptor_sets.count != map->windows.size()) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_descriptor_unavailable"};
  }
  if (map->controlled() &&
      !ResetVulkanStatus(command_buffer, map->control_status,
                         2u * sizeof(std::uint32_t))) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (!map->prepared->checks.empty()) {
    if (map->check_pipeline == nullptr ||
        map->check_descriptor == VK_NULL_HANDLE) {
      SetVulkanLastError(adapter, "compute_plan_invalid");
      return rund::AccelCheck{false, "compute_plan_invalid"};
    }
    EncodeVulkanMapCheck(command_buffer, *map);
  }
  if (map->controlled()) {
    for (std::size_t index = 0u; index < map->windows.size(); ++index) {
      EncodeVulkanMapControl(command_buffer, *map, map->windows[index],
                             static_cast<std::uint32_t>(index));
    }
  }
  for (std::size_t index = 0u; index < map->windows.size(); ++index) {
    const rund::AccelCheck encoded = EncodeVulkanMapWindow(
        adapter, *map, map->windows[index], map->descriptor_sets.at(index),
        descriptor_count, static_cast<std::uint32_t>(index), command_buffer);
    if (!encoded.ok) {
      return encoded;
    }
  }
  if (map->controlled() &&
      !FinishVulkanStatus(command_buffer, map->control_status, {})) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

[[nodiscard]] rund::AccelCheck
DescribeVulkanMapPipelineStatus(const std::shared_ptr<void> &resources,
                                VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const map = static_cast<VulkanMapEncodeResources *>(resources.get());
  if (map == nullptr) {
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (!map->controlled()) {
    source = {};
    return rund::AccelCheck{true, "ok"};
  }
  constexpr std::array mapping{
      VulkanPipelineStatusMapping{1u, rund::compute::Reason::WorksetOverflow},
      VulkanPipelineStatusMapping{
          2u, rund::compute::Reason::GatherIndexOutOfRange}};
  return DescribeVulkanPipelineStatus(map->control_status, 1u,
                                      VulkanPipelineStatusRule::Exact, 0u,
                                      mapping, source);
#else
  (void)resources;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

bool ObserveVulkanMapFailure(const std::shared_ptr<void> &resources,
                             std::uint64_t &ordinal) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto *const map =
      static_cast<const VulkanMapEncodeResources *>(resources.get());
  const std::uint32_t *const status =
      map == nullptr || map->prepared == nullptr ||
              map->prepared->checks.empty()
          ? nullptr
          : VulkanStatusValue(map->control_status);
  if (status == nullptr ||
      map->control_status.read_bytes < 2u * sizeof(*status) ||
      status[0] != 2u) {
    return false;
  }
  ordinal = status[1];
  return true;
#else
  (void)resources;
  (void)ordinal;
  return false;
#endif
}

rund::AccelCheck FinishVulkanMap(VulkanAdapter &adapter,
                                 const std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const map = static_cast<VulkanMapEncodeResources *>(resources.get());
  if (map == nullptr || map->adapter != &adapter) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (map->controlled()) {
    const rund::kernel::u32 *status = nullptr;
    const rund::AccelCheck read =
        ReadVulkanStatusU32(adapter, map->control_status, status);
    if (!read.ok) {
      return read;
    }
    if (status[0] != 0u) {
      const char *const reason = status[0] == 2u
                                     ? "compute_gather_index_out_of_range"
                                     : "compute_workset_overflow";
      SetVulkanLastError(adapter, reason);
      return rund::AccelCheck{false, reason};
    }
  }
  ::rund::detail::counter::Accumulate(
      adapter.dispatch_count,
      static_cast<rund::kernel::u64>(map->windows.size()));
  SetVulkanLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck
DescribeVulkanMapPipelineTelemetry(const std::shared_ptr<void> &resources,
                                   VulkanPipelineTelemetrySource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  source = {};
  const auto *const map =
      static_cast<const VulkanMapEncodeResources *>(resources.get());
  if (map == nullptr || map->prepared == nullptr) {
    return {false, "compute_plan_invalid"};
  }
  if (!map->controlled()) {
    return {true, "ok"};
  }
  if (map->control_args.buffer.buffer == VK_NULL_HANDLE ||
      map->windows.empty() ||
      map->windows.size() > std::numeric_limits<std::uint32_t>::max() / 4u) {
    return {false, "compute_plan_invalid"};
  }
  source = VulkanPipelineTelemetrySource{
      .kind = map->prepared->checks.empty()
                  ? VulkanPipelineTelemetryKind::ControlledMap
                  : VulkanPipelineTelemetryKind::GatherControl,
      .primary = map->prepared->checks.empty() ? &map->control_args.buffer
                                               : &map->control_status.device,
      .count =
          map->control.has_count() ? map->control_count.device_buffer : nullptr,
      .predicate = map->control.has_predicate()
                       ? map->control_predicate.device_buffer
                       : nullptr,
      .control = map->control,
      .count_offset =
          map->control_count.ref.offset_bytes + map->control.count_byte_offset,
      .predicate_offset = map->control_predicate.ref.offset_bytes +
                          map->control.predicate_byte_offset,
      .capacity = map->control.capacity,
      .work_item_count =
          map->windows.back().begin_sequence + map->windows.back().tile_count,
      .primary_word_count =
          map->prepared->checks.empty()
              ? static_cast<std::uint32_t>(map->windows.size() * 4u)
              : 2u,
      .indirect_dispatch_count = static_cast<std::uint32_t>(
          1u + (!map->prepared->checks.empty() ? 1u : 0u)),
  };
  if ((map->control.has_count() && source.count == nullptr) ||
      (map->control.has_predicate() && source.predicate == nullptr)) {
    return {false, "compute_plan_invalid"};
  }
  return {true, "ok"};
#else
  (void)resources;
  (void)source;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck DescribeVulkanMapPipelineCaptureDemand(
    const std::shared_ptr<void> &resources,
    std::uint64_t &indirect_dispatch_count) noexcept {
  indirect_dispatch_count = 0u;
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto *const map =
      static_cast<const VulkanMapEncodeResources *>(resources.get());
  if (map == nullptr || map->prepared == nullptr || map->windows.empty()) {
    return {false, "compute_plan_invalid"};
  }
  if (map->controlled()) {
    indirect_dispatch_count = map->windows.size();
  }
  return {true, "ok"};
#else
  (void)resources;
  return {false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail

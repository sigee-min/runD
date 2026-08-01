#include <accel/check.hpp>

#include "../../kernel/step/map/stride.hpp"
#include "../collective/finish.hpp"
#include "../kernel/ops/status.hpp"
#include "../status.hpp"
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
  if (resources->adapter != nullptr &&
      resources->descriptor_pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(resources->adapter->device,
                            resources->descriptor_pool, nullptr);
  }
  if (resources->adapter != nullptr) {
    ReleaseVulkanStatus(*resources->adapter, resources->control_status);
  }
  delete resources;
}
#endif

rund::AccelCheck PrepareVulkanMap(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<void> &resources, const rund::kernel::u32 iterations) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  const bool history_recurrence =
      artifact.key.variant ==
      rund::kernel::LoweringArtifactVariant::HistoryRecurrence;
  if (iterations == 0u || (history_recurrence && iterations < 2u)) {
    return rund::AccelCheck{false, "compute_pipeline_recurrence_history_invalid"};
  }
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

  auto *const raw = new VulkanMapEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanMapEncodeResources};
  raw->adapter = adapter;
  raw->iterations = iterations;
  raw->history_recurrence = history_recurrence;
  raw->plan = plan;
  raw->bindings = bindings;
  raw->input_plans.resize(static_cast<std::size_t>(plan.input_buffer_count));
  if (!FreezeInputWindowPlans(artifact.metadata, plan.tile_count,
                              raw->input_plans)) {
    return rund::AccelCheck{false, "compute_binding_mismatch"};
  }
  for (const rund::kernel::ReadRoute route : artifact.metadata.read_routes) {
    const auto found = std::find_if(raw->checks.begin(), raw->checks.end(),
                                    [&](const VulkanMapCheck check) {
                                      return check.binding == route.index;
                                    });
    if (found == raw->checks.end()) {
      raw->checks.push_back(
          VulkanMapCheck{.binding = route.index, .limit = route.count});
    } else {
      found->limit =
          std::min(found->limit, static_cast<std::uint64_t>(route.count));
    }
  }
  raw->windows.assign(windows, windows + window_count);
  const char *history_reason = "ok";
  if (!ValidateVulkanMapHistoryOutputs(*raw, history_reason)) {
    SetVulkanLastError(*adapter, history_reason);
    return rund::AccelCheck{false, history_reason};
  }
  const rund::kernel::LoweringArtifact strided_artifact =
      SpecializeMap(artifact, plan, bindings, adapter->storage_align);
  const rund::kernel::LoweringArtifact controlled_artifact =
      (control.active() || !raw->checks.empty())
          ? VulkanControlledMapArtifact(strided_artifact, plan)
          : strided_artifact;
  if (!controlled_artifact.ok) {
    return rund::AccelCheck{false, controlled_artifact.reason};
  }
  rund::kernel::ComputePlan pipeline_plan = plan;
  if (control.active() || !raw->checks.empty()) {
    ++pipeline_plan.input_buffer_count;
  }
  raw->pipeline =
      AcquireVulkanCachedPipeline(*adapter, pipeline_plan, controlled_artifact);
  if (raw->pipeline == nullptr) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  if (!PrepareVulkanResidentBindings(*adapter, raw->plan, raw->bindings,
                                     raw->resident)) {
    SetVulkanLastError(*adapter, raw->resident.reason);
    return rund::AccelCheck{false, raw->resident.reason};
  }
  if (!MakeVulkanMapHostBuffer(*adapter, bindings.param_data, plan.param_bytes,
                               raw->param)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  if (!AllocateVulkanMapDescriptorSets(
          *adapter, *raw->pipeline, raw->windows.size(), raw->descriptor_pool,
          raw->descriptor_sets)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  if (!PrepareVulkanMapControl(pick, control, plan, raw->windows, *raw)) {
    const char *const reason = VulkanLastError(adapter);
    return rund::AccelCheck{false, reason == nullptr || reason[0] == '\0'
                                       ? "compute_plan_invalid"
                                       : reason};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)plan;
  (void)artifact;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
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
  if (map == nullptr || map->adapter != &adapter ||
      command_buffer == VK_NULL_HANDLE || map->pipeline == nullptr ||
      map->param.buffer.buffer == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  const std::uint32_t descriptor_count = static_cast<std::uint32_t>(
      map->plan.input_buffer_count + map->plan.output_buffer_count + 1u +
      (map->controlled() ? 1u : 0u));
  if (map->descriptor_sets.size() != map->windows.size()) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_descriptor_unavailable"};
  }
  if (map->controlled() &&
      !ResetVulkanStatus(command_buffer, map->control_status,
                         2u * sizeof(std::uint32_t))) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (!map->checks.empty()) {
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
        adapter, *map, map->windows[index], map->descriptor_sets[index],
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
      map == nullptr || map->checks.empty()
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
  if (map == nullptr) {
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
      .kind = map->checks.empty() ? VulkanPipelineTelemetryKind::ControlledMap
                                  : VulkanPipelineTelemetryKind::GatherControl,
      .primary = map->checks.empty() ? &map->control_args.buffer
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
      .primary_word_count = map->checks.empty() ? static_cast<std::uint32_t>(
                                                      map->windows.size() * 4u)
                                                : 2u,
      .indirect_dispatch_count =
          static_cast<std::uint32_t>(1u + (!map->checks.empty() ? 1u : 0u)),
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

} // namespace rund::node::accel::detail

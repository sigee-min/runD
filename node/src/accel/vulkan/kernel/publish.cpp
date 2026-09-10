#include "../adapter/error.hpp"

#include "publish.hpp"

#include "copy.hpp"
#include "lease.hpp"
#include "publish/internal.hpp"
#include "window.hpp"

#include "../../kernel/grid.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../buffer/resident/find.hpp"
#include "../collective/pipeline.hpp"
#include "../descriptor.hpp"
#include "../resident/access.hpp"

#include <array>
#include <limits>
#include <mutex>
#include <utility>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *DescriptorFailure(VulkanAdapter &adapter) noexcept {
  const char *const reason = VulkanLastError(&adapter);
  return reason == nullptr || reason[0] == '\0'
             ? "accel_vulkan_descriptor_unavailable"
             : reason;
}

} // namespace

rund::AccelCheck
PrepareVulkanPipelinePublish(VulkanAdapter &adapter,
                             const std::span<const BackendPublish> publications,
                             const PreparedPipelineStatusLayout &status,
                             const VulkanPipelineControlResources &control,
                             const VulkanWindowResources &window,
                             VulkanPipelinePublishResources &resources) {
  resources = {};
  if (publications.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (control.adapter != &adapter || control.summary.buffer == VK_NULL_HANDLE ||
      window.adapter != &adapter || window.states.buffer == VK_NULL_HANDLE ||
      status.declared_step_count == 0u || adapter.max_dispatch_groups == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  resources.adapter = &adapter;
  if (adapter.dispatch_rows == 0u) {
    DestroyVulkanPipelinePublish(resources);
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::size_t descriptor_set_count = publications.size();
  for (const BackendPublish &publication : publications) {
    if (publication.identity.kind == PreparedKernelPublicationKind::Terminal) {
      if (descriptor_set_count == std::numeric_limits<std::size_t>::max()) {
        DestroyVulkanPipelinePublish(resources);
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      ++descriptor_set_count;
    }
  }
  resources.routes.reserve(publications.size());
  resources.descriptor_leases.reserve(descriptor_set_count);

  VulkanResidentState &resident = VulkanResidents(adapter);
  {
    std::lock_guard lock{resident.mutex};
    for (const BackendPublish &publication : publications) {
      const PreparedKernelPublicationIdentity &identity = publication.identity;
      const bool window_publish =
          identity.kind == PreparedKernelPublicationKind::Window;
      std::array<VulkanResidentBufferResult, 3u> sources{};
      VulkanResidentBufferResult target = ResolveVulkanResidentBuffer(
          resident, publication.target.source, publication.target.handle,
          "compute_resident_id_invalid", true);
      if (!target.check.ok || identity.state >= window.state_count ||
          (!window_publish && identity.final >= publication.sources.size())) {
        DestroyVulkanPipelinePublish(resources);
        return target.check.ok
                   ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                   : target.check;
      }
      target.ref = publication.target.source;
      std::array<VulkanCopyRange, 3u> source_ranges{};
      VulkanCopyRange target_range{};
      if (!PlanVulkanCopyRange(adapter, publication.target.source,
                               target.device_buffer, target_range)) {
        DestroyVulkanPipelinePublish(resources);
        return rund::AccelCheck{false, "compute_resident_stride_invalid"};
      }
      std::array<VulkanStorageBinding, 3u> source_bindings{};
      for (std::size_t bank = 0u; bank < sources.size(); ++bank) {
        const BackendRead &source = publication.sources[bank];
        sources[bank] =
            ResolveVulkanResidentBuffer(resident, source.source, source.handle,
                                        "compute_resident_id_invalid", true);
        const bool valid =
            sources[bank].check.ok &&
            ((!window_publish && bank != identity.final) ||
             sources[bank].device_buffer != target.device_buffer) &&
            source.source.count ==
                (window_publish ? identity.tile
                                : publication.target.source.count) &&
            source.source.element_bytes ==
                publication.target.source.element_bytes &&
            PlanVulkanCopyRange(adapter, source.source,
                                sources[bank].device_buffer,
                                source_ranges[bank]);
        if (!valid) {
          const rund::AccelCheck failed =
              sources[bank].check.ok
                  ? rund::AccelCheck{false, "compute_resident_stride_invalid"}
                  : sources[bank].check;
          DestroyVulkanPipelinePublish(resources);
          return failed;
        }
        sources[bank].ref = source.source;
        source_bindings[bank] = VulkanStorageBinding{
            sources[bank].device_buffer, source_ranges[bank].base,
            source_ranges[bank].bytes};
      }
      const Grid grid = PlanGrid(
          window_publish ? identity.tile : publication.target.source.count,
          kVulkanPublishThreads, adapter.max_dispatch_groups,
          adapter.dispatch_rows);
      if (!grid.valid()) {
        DestroyVulkanPipelinePublish(resources);
        return rund::AccelCheck{false, "compute_resident_stride_invalid"};
      }
      const VulkanStorageBinding target_binding{
          .buffer = target.device_buffer,
          .offset = target_range.base,
          .range = target_range.bytes,
      };
      VulkanResidentBufferResult count{};
      VulkanCopyRange count_range{};
      VulkanStorageBinding count_binding =
          VulkanStorageBindingFor(control.summary);
      if (window_publish) {
        count = ResolveVulkanResidentBuffer(
            resident, publication.count.source, publication.count.handle,
            "compute_resident_id_invalid", true);
        if (!count.check.ok || count.device_buffer == VK_NULL_HANDLE ||
            publication.count.source.count != 1u ||
            publication.count.source.element_bytes != sizeof(std::uint32_t) ||
            !PlanVulkanCopyRange(adapter, publication.count.source,
                                 count.device_buffer, count_range)) {
          const rund::AccelCheck failed =
              count.check.ok
                  ? rund::AccelCheck{false, "compute_resident_stride_invalid"}
                  : count.check;
          DestroyVulkanPipelinePublish(resources);
          return failed;
        }
        count.ref = publication.count.source;
        count_binding = VulkanStorageBinding{count.device_buffer,
                                             count_range.base,
                                             count_range.bytes};
      }
      resources.routes.push_back(VulkanPipelinePublishRoute{
          .sources = std::move(sources),
          .target = std::move(target),
          .count = std::move(count),
          .source_bindings = source_bindings,
          .target_binding = target_binding,
          .count_binding = count_binding,
          .params = VulkanPipelinePublishParams{
              .count = publication.target.source.count,
              .source_offset_words = {source_ranges[0].offset_words,
                                      source_ranges[1].offset_words,
                                      source_ranges[2].offset_words},
              .source_stride_words = {source_ranges[0].stride_words,
                                      source_ranges[1].stride_words,
                                      source_ranges[2].stride_words},
              .target_offset_words = target_range.offset_words,
              .target_stride_words = target_range.stride_words,
              .element_words = static_cast<std::uint32_t>(
                  publication.target.source.element_bytes /
                  sizeof(std::uint32_t)),
              .declared_step_count = status.declared_step_count,
              .state = identity.state,
              .final = identity.final,
              .maximum = identity.maximum,
              .tile = identity.tile,
              .kind = static_cast<std::uint32_t>(identity.kind),
              .count_offset_words =
                  window_publish ? count_range.offset_words : 0u,
          },
          .groups_x = grid.x,
          .groups_y = grid.y,
      });
    }
  }

  bool ready = false;
  {
    VulkanLeaseScope lease_scope{adapter, resources.descriptor_leases};
    resources.pipeline = AcquireVulkanPublishPipeline(adapter);
    ready = resources.pipeline != nullptr &&
            ReserveVulkanCollectiveDescriptorDemand(
                adapter, *resources.pipeline, 7u, descriptor_set_count);
    for (VulkanPipelinePublishRoute &route : resources.routes) {
      const bool terminal =
          route.params.kind ==
          static_cast<std::uint32_t>(PreparedKernelPublicationKind::Terminal);
      ready = ready &&
              AcquireVulkanCollectiveDescriptorSet(adapter, *resources.pipeline,
                                                   7u, route.descriptor) &&
              (!terminal || AcquireVulkanCollectiveDescriptorSet(
                                adapter, *resources.pipeline, 7u,
                                route.canonical_descriptor));
      if (!ready) {
        break;
      }
      const std::array<VulkanStorageBinding, 7u> bindings{
          route.source_bindings[0],
          route.source_bindings[1],
          route.source_bindings[2],
          route.target_binding,
          VulkanStorageBindingFor(control.summary),
          VulkanStorageBindingFor(window.states),
          route.count_binding,
      };
      ready =
          WriteVulkanStorageDescriptorSet(adapter, route.descriptor, bindings);
      if (ready && terminal) {
        const std::array<VulkanStorageBinding, 7u> canonical_bindings{
            route.source_bindings[0],
            route.source_bindings[1],
            route.source_bindings[2],
            route.source_bindings[route.params.final],
            VulkanStorageBindingFor(control.summary),
            VulkanStorageBindingFor(window.states),
            route.count_binding,
        };
        ready = WriteVulkanStorageDescriptorSet(
            adapter, route.canonical_descriptor, canonical_bindings);
      }
      if (!ready) {
        break;
      }
    }
  }
  if (!ready) {
    const char *const reason = DescriptorFailure(adapter);
    DestroyVulkanPipelinePublish(resources);
    return rund::AccelCheck{false, reason};
  }
  return rund::AccelCheck{true, "ok"};
}

void DestroyVulkanPipelinePublish(
    VulkanPipelinePublishResources &resources) noexcept {
  if (resources.adapter != nullptr) {
    ReleaseVulkanLeases(resources.descriptor_leases);
  }
  resources = {};
}

} // namespace rund::node::accel::detail

#endif

#include "publish.hpp"
#include "copy.hpp"
#include "lease.hpp"
#include "pipeline/source_artifact.hpp"
#include "window.hpp"

#include "../../kernel/footprint.hpp"
#include "../../kernel/grid.hpp"
#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../buffer/resident/find.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../resident/access.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>

#include <array>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

namespace rund::node::accel::detail {
namespace {

inline constexpr std::uint64_t kPublishThreads = 256u;

[[nodiscard]] constexpr std::string_view PublishSource() noexcept {
  return R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Source0 {
  uint source0[];
};
layout(set = 0, binding = 1, std430) readonly buffer Source1 {
  uint source1[];
};
layout(set = 0, binding = 2, std430) readonly buffer Source2 {
  uint source2[];
};
layout(set = 0, binding = 3, std430) buffer Target {
  uint target_words[];
};
layout(set = 0, binding = 4, std430) readonly buffer ControlSummary {
  uint control[];
};
layout(set = 0, binding = 5, std430) readonly buffer States {
  uvec2 states[];
};
layout(set = 0, binding = 6, std430) readonly buffer ResidentCount {
  uint resident_count[];
};
layout(push_constant) uniform PublishParams {
  uint64_t count;
  uint64_t source_offset_words[3];
  uint64_t source_stride_words[3];
  uint64_t target_offset_words;
  uint64_t target_stride_words;
  uint element_words;
  uint declared_step_count;
  uint state;
  uint final;
  uint stop;
  uint maximum;
  uint tile;
  uint outer;
  uint kind;
  uint64_t count_offset_words;
} p;
shared uint allowed;
void main() {
  const uint lane = gl_LocalInvocationID.x;
  if (lane == 0u) {
    const uvec2 state = states[p.state];
    if (p.kind == 1u) {
      const uint64_t base = uint64_t(p.outer) * uint64_t(p.tile);
      allowed = control[1] == 0u && state.y == 0u &&
                        base < min(uint64_t(resident_count[uint(
                                            p.count_offset_words)]),
                                   uint64_t(p.maximum))
                    ? 1u
                    : 0u;
    } else {
      allowed = p.stop == 0u
                    ? (control[1] == 0u && control[2] == 0xffffffffu &&
                       control[3] == p.declared_step_count
                   ? 1u
                   : 0u)
                    : (control[1] == 0u && state.x != p.final ? 1u : 0u);
    }
  }
  barrier();
  if (allowed == 0u) { return; }
  const uint64_t group =
      uint64_t(gl_WorkGroupID.x) +
      uint64_t(gl_WorkGroupID.y) * uint64_t(gl_NumWorkGroups.x);
  const uint64_t index = group * 256ul + uint64_t(lane);
  const uint64_t base = uint64_t(p.outer) * uint64_t(p.tile);
  uint64_t active_count = p.count;
  if (p.kind == 1u && allowed != 0u) {
    active_count = min(min(uint64_t(p.tile), uint64_t(p.maximum) - base),
                       uint64_t(resident_count[uint(p.count_offset_words)]) -
                           base);
  }
  if (index >= active_count) { return; }
  const uint current =
      p.kind == 1u ? 0u : (p.stop == 0u ? p.final : states[p.state].x);
  const uint64_t source =
      p.source_offset_words[current] + index * p.source_stride_words[current];
  const uint64_t target =
      p.target_offset_words +
      (p.kind == 1u ? base + index : index) * p.target_stride_words;
  target_words[uint(target)] =
      current == 1u ? source1[uint(source)]
                    : (current == 2u ? source2[uint(source)]
                                     : source0[uint(source)]);
  if (p.element_words == 2u) {
    target_words[uint(target + 1ul)] =
        current == 1u ? source1[uint(source + 1ul)]
                      : (current == 2u ? source2[uint(source + 1ul)]
                                       : source0[uint(source + 1ul)]);
  }
}
)GLSL";
}

[[nodiscard]] VulkanCollectivePipeline *
AcquirePublishPipeline(VulkanAdapter &adapter) {
  const rund::kernel::ComputePlan plan{
      .op_hash_hi = 0x7075626c69736833ull,
      .op_hash_lo = 0x3262697472617738ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact artifact =
      VulkanFixedSourceArtifact(PublishSource());
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(
      adapter, 7u, sizeof(VulkanPipelinePublishParams), plan, artifact);
}

[[nodiscard]] const char *DescriptorFailure(VulkanAdapter &adapter) noexcept {
  const char *const reason = VulkanLastError(&adapter);
  return reason == nullptr || reason[0] == '\0'
             ? "accel_vulkan_descriptor_unavailable"
             : reason;
}

} // namespace

std::string_view VulkanPublishSourceText() noexcept { return PublishSource(); }

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
    if (publication.identity.kind ==
        PreparedKernelPublicationKind::Terminal) {
      if (descriptor_set_count == std::numeric_limits<std::size_t>::max()) {
        DestroyVulkanPipelinePublish(resources);
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      ++descriptor_set_count;
    }
  }
  try {
    resources.routes.reserve(publications.size());
    resources.descriptor_leases.reserve(descriptor_set_count);
  } catch (const std::bad_alloc &) {
    DestroyVulkanPipelinePublish(resources);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    DestroyVulkanPipelinePublish(resources);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

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
          (!window_publish &&
           identity.final >= publication.sources.size())) {
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
            source.source.count == (window_publish
                                        ? identity.tile
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
          kPublishThreads, adapter.max_dispatch_groups, adapter.dispatch_rows);
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
        count_binding = VulkanStorageBinding{
            count.device_buffer, count_range.base, count_range.bytes};
      }
      resources.routes.push_back(VulkanPipelinePublishRoute{
          .sources = std::move(sources),
          .target = std::move(target),
          .count = std::move(count),
          .source_bindings = source_bindings,
          .target_binding = target_binding,
          .count_binding = count_binding,
          .params =
              VulkanPipelinePublishParams{
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
  try {
    VulkanLeaseScope lease_scope{adapter, resources.descriptor_leases};
    resources.pipeline = AcquirePublishPipeline(adapter);
    ready = resources.pipeline != nullptr &&
            ReserveVulkanCollectiveDescriptorDemand(
                adapter, *resources.pipeline, 7u, descriptor_set_count);
    for (VulkanPipelinePublishRoute &route : resources.routes) {
      const bool terminal =
          route.params.kind ==
          static_cast<std::uint32_t>(
              PreparedKernelPublicationKind::Terminal);
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
  } catch (const std::bad_alloc &) {
    DestroyVulkanPipelinePublish(resources);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    DestroyVulkanPipelinePublish(resources);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
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

bool EncodeVulkanPipelinePublish(
    const VkCommandBuffer command,
    const VulkanPipelinePublishResources &resources) noexcept {
  if (resources.routes.empty()) {
    return true;
  }
  if (command == VK_NULL_HANDLE || resources.pipeline == nullptr) {
    return false;
  }
  EncodeVulkanComputeToComputeBarrier(command);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     resources.pipeline->pipeline);
  for (const VulkanPipelinePublishRoute &route : resources.routes) {
    if (route.params.kind !=
        static_cast<std::uint32_t>(
            PreparedKernelPublicationKind::Terminal)) {
      continue;
    }
    if (route.descriptor == VK_NULL_HANDLE || route.groups_x == 0u ||
        route.groups_y == 0u) {
      return false;
    }
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipeline->pipeline_layout, 0u, 1u,
                          &route.descriptor, 0u, nullptr);
    PushVulkanConstants(command, resources.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(route.params),
                        &route.params);
    DispatchVulkan(command, route.groups_x, route.groups_y, 1u);
  }
  VkMemoryBarrier visible{};
  visible.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  visible.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  visible.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                          VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0u, 1u, &visible, 0u, nullptr, 0u, nullptr);
  return true;
}

bool EncodeVulkanPipelineCanonicalize(
    const VkCommandBuffer command,
    const VulkanPipelinePublishResources &resources,
    const std::uint32_t state) noexcept {
  if (resources.routes.empty()) {
    return true;
  }
  if (command == VK_NULL_HANDLE || resources.pipeline == nullptr) {
    return false;
  }
  EncodeVulkanComputeToComputeBarrier(command);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     resources.pipeline->pipeline);
  for (const VulkanPipelinePublishRoute &route : resources.routes) {
    if (route.params.kind !=
            static_cast<std::uint32_t>(
                PreparedKernelPublicationKind::Terminal) ||
        route.params.state != state) {
      continue;
    }
    if (route.canonical_descriptor == VK_NULL_HANDLE || route.groups_x == 0u ||
        route.groups_y == 0u || route.params.final >= 3u) {
      return false;
    }
    VulkanPipelinePublishParams params = route.params;
    params.target_offset_words = params.source_offset_words[params.final];
    params.target_stride_words = params.source_stride_words[params.final];
    params.stop = std::numeric_limits<std::uint32_t>::max();
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipeline->pipeline_layout, 0u, 1u,
                          &route.canonical_descriptor, 0u, nullptr);
    PushVulkanConstants(command, resources.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                        &params);
    ::vkCmdDispatch(command, route.groups_x, route.groups_y, 1u);
  }
  EncodeVulkanComputeToComputeBarrier(command);
  return true;
}

bool EncodeVulkanPipelineWindowPublish(
    const VkCommandBuffer command,
    const VulkanPipelinePublishResources &resources, const std::uint32_t state,
    const std::uint32_t outer) noexcept {
  if (resources.routes.empty()) {
    return true;
  }
  if (command == VK_NULL_HANDLE || resources.pipeline == nullptr) {
    return false;
  }
  EncodeVulkanComputeToComputeBarrier(command);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     resources.pipeline->pipeline);
  for (const VulkanPipelinePublishRoute &route : resources.routes) {
    if (route.params.kind !=
            static_cast<std::uint32_t>(
                PreparedKernelPublicationKind::Window) ||
        route.params.state != state) {
      continue;
    }
    if (route.descriptor == VK_NULL_HANDLE || route.groups_x == 0u ||
        route.groups_y == 0u || route.params.tile == 0u) {
      return false;
    }
    VulkanPipelinePublishParams params = route.params;
    params.outer = outer;
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipeline->pipeline_layout, 0u, 1u,
                          &route.descriptor, 0u, nullptr);
    PushVulkanConstants(command, resources.pipeline->pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                        &params);
    DispatchVulkan(command, route.groups_x, route.groups_y, 1u);
  }
  EncodeVulkanComputeToComputeBarrier(command);
  return true;
}

std::uint64_t VulkanPipelinePublishHostBytes(
    const VulkanPipelinePublishResources &resources) noexcept {
  const std::uint64_t routes = capacity_bytes(resources.routes);
  const std::uint64_t leases = capacity_bytes(resources.descriptor_leases);
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  return routes > maximum - leases ? maximum : routes + leases;
}

} // namespace rund::node::accel::detail

#endif

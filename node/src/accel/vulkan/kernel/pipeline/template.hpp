#pragma once

#include "../../../kernel/backend/manifest.hpp"
#include "../../collective/pipeline.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/graph/schema.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// One immutable executable dependency and its exact descriptor demand for one
// route.  The native pipeline is adapter-cache owned; this Program-template
// value freezes only the collision-safe pointer identity and demand tuple.
struct VulkanKernelImmutablePipelineStage final {
  VulkanCollectivePipeline *pipeline{};
  std::uint32_t descriptor_count{};
  std::uint64_t sets_per_route{};
};

// Range PrefixDifference has at most 24 hierarchy/fix-up/window
// stages for the legal 64-lane width. Route-owned buffers, descriptor leases,
// and mutable dispatch state never enter this Program-level owner.
struct VulkanKernelImmutablePipelines final {
  rund::kernel::NodeKind kind{rund::kernel::NodeKind::Map};
  VulkanKernelImmutablePipelineStage control{};
  std::array<VulkanKernelImmutablePipelineStage, 24u> stages{};
  std::uint32_t count{};
  std::uint64_t capture_direct_dispatch_count{};
  std::uint64_t capture_indirect_dispatch_count{};

  [[nodiscard]] bool
  append_control(VulkanCollectivePipeline *const pipeline,
                 const std::uint32_t descriptor_count,
                 const std::uint64_t sets_per_route) noexcept {
    if (control.pipeline != nullptr || pipeline == nullptr ||
        descriptor_count == 0u || sets_per_route == 0u ||
        pipeline->descriptor_count != descriptor_count) {
      return false;
    }
    control = VulkanKernelImmutablePipelineStage{pipeline, descriptor_count,
                                                 sets_per_route};
    return true;
  }

  [[nodiscard]] bool append(VulkanCollectivePipeline *const pipeline,
                            const std::uint32_t descriptor_count,
                            const std::uint64_t sets_per_route) noexcept {
    if (pipeline == nullptr || descriptor_count == 0u || sets_per_route == 0u ||
        count == stages.size() ||
        pipeline->descriptor_count != descriptor_count) {
      return false;
    }
    stages[count++] = VulkanKernelImmutablePipelineStage{
        pipeline, descriptor_count, sets_per_route};
    return true;
  }

  [[nodiscard]] bool
  ready(const rund::kernel::NodeKind expected_kind,
        const PreparedBackendManifest &manifest) const noexcept {
    const std::uint64_t control_count = control.pipeline == nullptr ? 0u : 1u;
    std::uint64_t total_count = 0u;
    if (kind != expected_kind || expected_kind == rund::kernel::NodeKind::Map ||
        count == 0u || count > stages.size() ||
        !rund::kernel::checked::add(count, control_count, total_count) ||
        total_count != manifest.pipeline_stage_count ||
        total_count != manifest.descriptor_dependency_count || !manifest.ok) {
      return false;
    }
    std::uint64_t sets = 0u;
    std::uint64_t bindings = 0u;
    const auto account = [&](const VulkanKernelImmutablePipelineStage &stage) {
      std::uint64_t stage_bindings = 0u;
      return stage.pipeline != nullptr && stage.descriptor_count != 0u &&
             stage.sets_per_route != 0u &&
             stage.pipeline->descriptor_count == stage.descriptor_count &&
             rund::kernel::checked::add(sets, stage.sets_per_route, sets) &&
             rund::kernel::checked::mul(stage.sets_per_route,
                                        stage.descriptor_count,
                                        stage_bindings) &&
             rund::kernel::checked::add(bindings, stage_bindings, bindings);
    };
    if (control.pipeline != nullptr && !account(control)) {
      return false;
    }
    for (std::size_t index = 0u; index < count; ++index) {
      const VulkanKernelImmutablePipelineStage &stage = stages[index];
      if (!account(stage)) {
        return false;
      }
    }
    return capture_direct_dispatch_count ==
               manifest.capture_direct_dispatch_count &&
           capture_indirect_dispatch_count ==
               manifest.capture_indirect_dispatch_count &&
           sets == manifest.descriptor_set_count &&
           sets == manifest.descriptor_lease_count &&
           bindings == manifest.descriptor_binding_count;
  }

  [[nodiscard]] VulkanCollectivePipeline *
  borrow(const rund::kernel::NodeKind expected_kind,
         const std::uint32_t expected_count, const std::size_t index,
         const std::uint32_t descriptor_count,
         const std::uint64_t sets_per_route) const noexcept {
    if (kind != expected_kind || count != expected_count || index >= count) {
      return nullptr;
    }
    const VulkanKernelImmutablePipelineStage &stage = stages[index];
    return stage.pipeline != nullptr &&
                   stage.descriptor_count == descriptor_count &&
                   stage.sets_per_route == sets_per_route &&
                   stage.pipeline->descriptor_count == descriptor_count
               ? stage.pipeline
               : nullptr;
  }

  [[nodiscard]] VulkanCollectivePipeline *
  borrow_control(const rund::kernel::NodeKind expected_kind,
                 const std::uint32_t descriptor_count,
                 const std::uint64_t sets_per_route) const noexcept {
    return kind == expected_kind && control.pipeline != nullptr &&
                   control.descriptor_count == descriptor_count &&
                   control.sets_per_route == sets_per_route &&
                   control.pipeline->descriptor_count == descriptor_count
               ? control.pipeline
               : nullptr;
  }
};

#endif

} // namespace rund::node::accel::detail

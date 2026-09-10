#include "../../adapter/error.hpp"

#include "local.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <functional>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool AppendVulkanDescriptorDependency(
    VulkanKernelProgramTemplate &program,
    const VulkanKernelDescriptorDependencyKind kind, const void *const identity,
    const std::uint32_t descriptor_count, const std::uint64_t sets_per_route,
    const std::uint32_t source_node, std::uint32_t &order) noexcept {
  if (identity == nullptr || descriptor_count == 0u || sets_per_route == 0u ||
      source_node == NoNode ||
      order == std::numeric_limits<std::uint32_t>::max() ||
      program.descriptor_dependencies.size() ==
          program.descriptor_dependencies.capacity()) {
    return false;
  }
  program.descriptor_dependencies.push_back(VulkanKernelDescriptorDependency{
      .kind = kind,
      .identity = identity,
      .descriptor_count = descriptor_count,
      .sets_per_route = sets_per_route,
      .earliest_order = order++,
      .source_node = source_node,
  });
  return true;
}

[[nodiscard]] bool AppendVulkanMapDescriptorDependencies(
    VulkanKernelProgramTemplate &program,
    const VulkanMapTemplateResources &prepared, const std::uint32_t source_node,
    std::uint32_t &order) noexcept {
  std::uint64_t descriptor_count = 0u;
  if (prepared.pipeline == nullptr ||
      !rund::kernel::checked::add(prepared.pipeline->input_buffer_count,
                                  prepared.pipeline->output_buffer_count,
                                  descriptor_count) ||
      !rund::kernel::checked::add(descriptor_count, 1u, descriptor_count) ||
      descriptor_count > std::numeric_limits<std::uint32_t>::max() ||
      !AppendVulkanDescriptorDependency(
          program, VulkanKernelDescriptorDependencyKind::Map, prepared.pipeline,
          static_cast<std::uint32_t>(descriptor_count),
          prepared.plan.dispatch_count, source_node, order)) {
    return false;
  }
  if (prepared.control_pipeline != nullptr &&
      !AppendVulkanDescriptorDependency(
          program, VulkanKernelDescriptorDependencyKind::Collective,
          prepared.control_pipeline,
          prepared.control_pipeline->descriptor_count, 1u, source_node,
          order)) {
    return false;
  }
  return prepared.check_pipeline == nullptr ||
         AppendVulkanDescriptorDependency(
             program, VulkanKernelDescriptorDependencyKind::Collective,
             prepared.check_pipeline, prepared.check_pipeline->descriptor_count,
             1u, source_node, order);
}

[[nodiscard]] bool AppendVulkanCollectiveDescriptorDependencies(
    VulkanKernelProgramTemplate &program,
    const VulkanKernelImmutablePipelines &pipelines,
    const PreparedBackendManifest &manifest, const std::uint32_t source_node,
    std::uint32_t &order) noexcept {
  if (!pipelines.ready(pipelines.kind, manifest)) {
    return false;
  }
  if (pipelines.control.pipeline != nullptr &&
      !AppendVulkanDescriptorDependency(
          program, VulkanKernelDescriptorDependencyKind::Collective,
          pipelines.control.pipeline, pipelines.control.descriptor_count,
          pipelines.control.sets_per_route, source_node, order)) {
    return false;
  }
  for (std::size_t index = 0u; index < pipelines.count; ++index) {
    const VulkanKernelImmutablePipelineStage &stage = pipelines.stages[index];
    if (!AppendVulkanDescriptorDependency(
            program, VulkanKernelDescriptorDependencyKind::Collective,
            stage.pipeline, stage.descriptor_count, stage.sets_per_route,
            source_node, order)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] static bool SameVulkanDescriptorDependencyKey(
    const VulkanKernelDescriptorDependency &left,
    const VulkanKernelDescriptorDependency &right) noexcept {
  return left.kind == right.kind && left.identity == right.identity &&
         left.descriptor_count == right.descriptor_count;
}

[[nodiscard]] rund::AccelCheck
FinalizeVulkanDescriptorDependencies(VulkanAdapter &adapter,
                                     VulkanKernelProgramTemplate &program,
                                     std::uint32_t *const failed_node) {
  const auto fail = [&](const VulkanKernelDescriptorDependency *dependency,
                        const char *const reason) {
    if (failed_node != nullptr && *failed_node == NoNode &&
        dependency != nullptr && dependency->source_node != NoNode) {
      *failed_node = dependency->source_node;
    }
    return rund::AccelCheck{false, reason};
  };
  if (!program.route_demand.valid() ||
      program.descriptor_dependencies.empty()) {
    return fail(nullptr, "accel_kernel_template_invalid");
  }
  auto &dependencies = program.descriptor_dependencies;
  const auto key_less = [](const VulkanKernelDescriptorDependency &left,
                           const VulkanKernelDescriptorDependency &right) {
    if (left.kind != right.kind) {
      return left.kind < right.kind;
    }
    if (left.identity != right.identity) {
      return std::less<const void *>{}(left.identity, right.identity);
    }
    if (left.descriptor_count != right.descriptor_count) {
      return left.descriptor_count < right.descriptor_count;
    }
    return left.earliest_order < right.earliest_order;
  };
  std::sort(dependencies.begin(), dependencies.end(), key_less);

  std::size_t group_count = 0u;
  for (std::size_t index = 0u; index < dependencies.size(); ++index) {
    const VulkanKernelDescriptorDependency dependency = dependencies[index];
    if (index != 0u && dependencies[index - 1u].kind == dependency.kind &&
        dependencies[index - 1u].identity == dependency.identity &&
        dependencies[index - 1u].descriptor_count !=
            dependency.descriptor_count) {
      const VulkanKernelDescriptorDependency &previous =
          dependencies[index - 1u];
      return fail(previous.earliest_order <= dependency.earliest_order
                      ? &previous
                      : &dependency,
                  "accel_vulkan_descriptor_unavailable");
    }
    if (group_count == 0u || !SameVulkanDescriptorDependencyKey(
                                 dependencies[group_count - 1u], dependency)) {
      if (group_count != index) {
        dependencies[group_count] = dependency;
      }
      ++group_count;
      continue;
    }
    VulkanKernelDescriptorDependency &group = dependencies[group_count - 1u];
    if (!rund::kernel::checked::add(group.sets_per_route,
                                    dependency.sets_per_route,
                                    group.sets_per_route)) {
      return fail(&dependency, "compute_pipeline_capacity");
    }
    if (dependency.earliest_order < group.earliest_order) {
      group.earliest_order = dependency.earliest_order;
      group.source_node = dependency.source_node;
    }
  }
  dependencies.resize(group_count);
  std::sort(dependencies.begin(), dependencies.end(),
            [](const VulkanKernelDescriptorDependency &left,
               const VulkanKernelDescriptorDependency &right) {
              return left.earliest_order < right.earliest_order;
            });

  // Prove every complete group before the first native reservation. A tuple
  // mismatch or overflow can therefore never admit only a canonical prefix.
  for (VulkanKernelDescriptorDependency &dependency : dependencies) {
    std::uint64_t capacity = 0u;
    if (!rund::kernel::checked::mul(dependency.sets_per_route,
                                    program.route_demand.capacity, capacity) ||
        capacity == 0u ||
        capacity > std::numeric_limits<std::uint32_t>::max()) {
      return fail(&dependency, "accel_vulkan_descriptor_unavailable");
    }
    dependency.set_capacity = capacity;
    if (dependency.kind == VulkanKernelDescriptorDependencyKind::Collective) {
      const auto *const pipeline =
          static_cast<const VulkanCollectivePipeline *>(dependency.identity);
      if (pipeline == nullptr ||
          pipeline->descriptor_count != dependency.descriptor_count) {
        return fail(&dependency, "accel_vulkan_descriptor_unavailable");
      }
      continue;
    }
    const auto *const pipeline =
        static_cast<const VulkanCachedPipeline *>(dependency.identity);
    std::uint64_t descriptor_count = 0u;
    if (pipeline == nullptr ||
        !rund::kernel::checked::add(pipeline->input_buffer_count,
                                    pipeline->output_buffer_count,
                                    descriptor_count) ||
        !rund::kernel::checked::add(descriptor_count, 1u, descriptor_count) ||
        descriptor_count != dependency.descriptor_count) {
      return fail(&dependency, "accel_vulkan_descriptor_unavailable");
    }
  }

  for (VulkanKernelDescriptorDependency &dependency : dependencies) {
    if (dependency.kind == VulkanKernelDescriptorDependencyKind::Map) {
      if (!PrepareVulkanMapDescriptorArena(
              adapter,
              *static_cast<const VulkanCachedPipeline *>(dependency.identity),
              dependency.set_capacity, dependency.map_arena)) {
        return fail(&dependency, VulkanLastError(&adapter));
      }
      continue;
    }
    auto *const pipeline = const_cast<VulkanCollectivePipeline *>(
        static_cast<const VulkanCollectivePipeline *>(dependency.identity));
    if (!ReserveVulkanCollectiveDescriptorDemand(adapter, *pipeline,
                                                 dependency.descriptor_count,
                                                 dependency.set_capacity)) {
      return fail(&dependency, VulkanLastError(&adapter));
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail

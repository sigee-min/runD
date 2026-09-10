#include "internal.hpp"

#include "../../../../../kernel/backend/exception.hpp"

#include <limits>

namespace rund::node::accel::detail::metal_pipeline_describe_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
DescribeMetalTemplates(MetalPipelineBuild &build,
                       const MetalPipelineDescribeCapacity &capacity) {
  // Describe immutable status and telemetry ownership exactly once per compact
  // route template. Physical nested occurrences reference these slices during
  // capture instead of manufacturing K*N status arenas or retained owners.
  for (std::size_t template_index = 0u; template_index < build.templates.size();
       ++template_index) {
    const BackendBatchEntry &entry = build.templates[template_index];
    build.failure_context.template_route(
        static_cast<std::uint32_t>(template_index));
    auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<MetalKernelResources *>(entry.prepared->get());
    MetalKernelContext current{};
    if (entry.run == nullptr || entry.run->pick == nullptr ||
        resources == nullptr || resources->size() == 0u ||
        !IsPipelinePrivatePreparation(resources->mode) ||
        !ValidateMetalKernelContext(*entry.run->pick, current).ok ||
        current.adapter != build.context.adapter) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const std::uint32_t declared_step =
        build.status.declared_steps[template_index];
    if (entry.template_index != template_index ||
        declared_step >= build.status.declared_step_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const std::size_t telemetry_begin = build.telemetry_steps.size();
    for (std::size_t step_index = 0u; step_index < resources->size();
         ++step_index) {
      build.failure_context.template_node_route(entry, step_index);
      MetalKernelEntry *const step = resources->entry(step_index);
      if (step == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      if (build.pipeline->telemetry.size() >
          std::numeric_limits<std::uint32_t>::max()) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      PreparedProgramStatusSlice slice{.first = static_cast<std::uint32_t>(
                                           build.pipeline->telemetry.size())};
      MetalPipelineTelemetrySource source{};
      if (step->ops.pipeline_telemetry != nullptr) {
        if (!step->ops.pipeline_telemetry(step->resource, source)) {
          return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
        }
      }
      if (source.kind != MetalPipelineTelemetryKind::None) {
        if (build.pipeline->telemetry.size() >= capacity.telemetry_capacity) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        try {
          build.pipeline->telemetry.push_back(MetalPipelineTelemetryRecord{
              .source = source,
              .owner = step->resource,
          });
        } catch (...) {
          backend_exception::RethrowUnlessCapacityException();
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        slice.count = 1u;
      }
      if (build.telemetry_steps.size() >= capacity.template_step_capacity) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      try {
        build.telemetry_steps.push_back(slice);
      } catch (...) {
        backend_exception::RethrowUnlessCapacityException();
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
    build.failure_context.template_route(
        static_cast<std::uint32_t>(template_index));
    const std::size_t telemetry_count =
        build.telemetry_steps.size() - telemetry_begin;
    if (telemetry_begin > std::numeric_limits<std::uint32_t>::max() ||
        telemetry_count > std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    build.telemetry_ranges[template_index] = PreparedProgramStatusSlice{
        .first = static_cast<std::uint32_t>(telemetry_begin),
        .count = static_cast<std::uint32_t>(telemetry_count),
    };
    const std::size_t binding_begin = build.status_bindings.size();
    const std::uint32_t status_begin = build.status_entry_count;
    bool status_capacity_failed = false;
    try {
      if (!CollectMetalStatus(
              *resources, declared_step, build.status_bindings,
              build.status_sources, build.raw_status_count,
              build.status_entry_count, capacity.status_source_capacity,
              capacity.status_entry_capacity, status_capacity_failed)) {
        return rund::AccelCheck{false,
                                status_capacity_failed
                                    ? "compute_pipeline_capacity"
                                    : "accel_kernel_primitive_unsupported"};
      }
    } catch (...) {
      backend_exception::RethrowUnlessCapacityException();
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const std::size_t binding_size =
        build.status_bindings.size() - binding_begin;
    const std::uint32_t status_size = build.status_entry_count - status_begin;
    if (binding_begin > std::numeric_limits<std::uint32_t>::max() ||
        binding_size > std::numeric_limits<std::uint32_t>::max() ||
        !SetPreparedProgramStatusSlice(
            build.status, static_cast<std::uint32_t>(template_index),
            status_size)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    build.binding_slices[template_index] = PreparedProgramStatusSlice{
        .first = static_cast<std::uint32_t>(binding_begin),
        .count = static_cast<std::uint32_t>(binding_size),
    };
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_describe_internal

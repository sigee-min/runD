#include "internal.hpp"

#include "../../../../../kernel/backend/exception.hpp"

namespace rund::node::accel::detail::metal_pipeline_describe_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck DescribeMetalAggregate(MetalPipelineBuild &build) {
  // Direct aggregate admission already proved and resolved the complete
  // pipeline. Canonical status, telemetry, reset, and occurrence metadata
  // have no execution consumer on this path. Preserve only the public logical
  // status slices by inspecting one representative of each proved-identical
  // Seed/Action/Fold Program; do not retain raw arenas or occurrence records.
  try {
    const NestedAggregate &aggregate = build.aggregates.front();
    const auto status_count = [&](const std::uint32_t template_index,
                                  std::uint32_t &out) {
      build.failure_context.template_route(template_index);
      if (template_index >= build.templates.size()) {
        return false;
      }
      const BackendBatchEntry &entry = build.templates[template_index];
      auto *const resources =
          entry.prepared == nullptr
              ? nullptr
              : static_cast<MetalKernelResources *>(entry.prepared->get());
      if (resources == nullptr) {
        return false;
      }
      out = 0u;
      return CountMetalDirectAggregateStatus(*resources, out);
    };
    std::uint32_t seed_status = 0u;
    std::uint32_t action_status = 0u;
    std::uint32_t fold_status = 0u;
    if (!status_count(aggregate.shape.seed_first(), seed_status) ||
        !status_count(aggregate.shape.action_first(), action_status) ||
        !status_count(aggregate.shape.fold_first(), fold_status)) {
      return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
    }
    for (std::uint32_t index = 0u; index < build.status.active_step_count;
         ++index) {
      const std::uint32_t count =
          index < aggregate.shape.action_first()
              ? seed_status
              : (index < aggregate.shape.fold_first() ? action_status
                                                      : fold_status);
      if (!SetPreparedProgramStatusSlice(build.status, index, count)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
    build.status_entry_count = build.status.status_entry_count;
    return rund::AccelCheck{true, "ok"};
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_describe_internal

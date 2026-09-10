#include "internal.hpp"

#include "../../../../../kernel/backend/exception.hpp"

#include <limits>

namespace rund::node::accel::detail::metal_pipeline_describe_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
DescribeMetalCapacity(MetalPipelineBuild &build,
                      MetalPipelineDescribeCapacity &capacity) {
  std::size_t template_step_capacity = 0u;
  for (const BackendBatchEntry &entry : build.templates) {
    const auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<const MetalKernelResources *>(entry.prepared->get());
    if (resources == nullptr || resources->size() == 0u) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (resources->size() >
        std::numeric_limits<std::size_t>::max() - template_step_capacity) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    template_step_capacity += resources->size();
  }
  if (!build.template_registry.limit.ok ||
      template_step_capacity >
          build.template_registry.limit.backend_step_description_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const PreparedKernelPipelineReservation &limit =
      build.template_registry.limit;
  if (limit.backend_step_description_count >
          std::numeric_limits<std::size_t>::max() ||
      limit.backend_status_source_count >
          std::numeric_limits<std::size_t>::max() ||
      limit.backend_status_entry_count >
          std::numeric_limits<std::uint32_t>::max() ||
      limit.backend_telemetry_count > std::numeric_limits<std::size_t>::max()) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const std::size_t status_source_capacity =
      static_cast<std::size_t>(limit.backend_status_source_count);
  const std::uint32_t status_entry_capacity =
      static_cast<std::uint32_t>(limit.backend_status_entry_count);
  const std::size_t telemetry_capacity =
      static_cast<std::size_t>(limit.backend_telemetry_count);
  try {
    build.status_bindings.reserve(status_source_capacity);
    build.status_sources.reserve(status_source_capacity);
    build.status_resets.reserve(status_source_capacity);
    build.telemetry_steps.reserve(template_step_capacity);
    build.pipeline->telemetry.reserve(telemetry_capacity);
    if (build.status_bindings.capacity() != status_source_capacity ||
        build.status_sources.capacity() != status_source_capacity ||
        build.status_resets.capacity() != status_source_capacity ||
        build.telemetry_steps.capacity() != template_step_capacity ||
        build.pipeline->telemetry.capacity() != telemetry_capacity) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  capacity.template_step_capacity = template_step_capacity;
  capacity.status_source_capacity = status_source_capacity;
  capacity.telemetry_capacity = telemetry_capacity;
  capacity.status_entry_capacity = status_entry_capacity;
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_describe_internal

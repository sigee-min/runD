#include "internal.hpp"

#include "../../../../runtime/map/resources.hpp"

#include "../../../../../kernel/backend/exception.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail::metal_pipeline_admit_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
PrepareMetalPipelineRoutes(MetalPipelineBuild &build,
                           MetalSpatialWindowProof &&spatial_window,
                           const std::uint32_t state_count) {
  try {
    build.pipeline = std::make_shared<MetalSequence>();
    build.pipeline->adapter = build.context.adapter;
    build.pipeline->state_count = state_count;
    build.pipeline->spatial_window = std::move(spatial_window);
    build.pipeline->profile_steps = build.profile_steps;
    build.pipeline->transducers.resize(build.transducers.size());
    if (build.profile_steps) {
      if (build.status.declared_step_count == 0u ||
          build.status.declared_step_count > PreparedPipelineStepCapacity) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      build.pipeline->step_evidence.resize(build.status.declared_step_count);
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  build.failure_context.clear_route();
  if (build.recurrence.ready()) {
    if (!build.transducers.empty()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (build.recurrence.first == nullptr ||
        build.recurrence.canonical_artifact == nullptr ||
        !build.recurrence.source_plan.ok ||
        build.recurrence.windows == nullptr ||
        build.recurrence.window_count == 0u ||
        build.recurrence.iterations < 2u ||
        build.recurrence.iterations >
            std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    rund::AccelCheck ready{};
    try {
      ready = PrepareMetalRecurrenceRoute(
          build.template_registry, *build.context.adapter,
          *build.entries.front().run->pick, *build.entries.front().run,
          build.recurrence, build.recurrence.first->control,
          build.pipeline->recurrence,
          static_cast<std::uint32_t>(build.recurrence.iterations));
    } catch (...) {
      backend_exception::RethrowUnlessCapacityException();
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (!ready.ok) {
      return ready;
    }
    auto *const prepared = static_cast<MetalMapEncodeResources *>(
        build.pipeline->recurrence.get());
    if (prepared == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    prepared->binding_owner = build.recurrence.history;
  }
  for (std::size_t index = 0u; index < build.transducers.size(); ++index) {
    const TileTransducer &transducer = build.transducers[index];
    build.failure_context.template_route(transducer.template_first);
    const MapRecurrence &map = transducer.recurrence;
    if (!map.ready() || map.first == nullptr ||
        map.canonical_artifact == nullptr || !map.source_plan.ok ||
        map.windows == nullptr || map.window_count == 0u ||
        map.iterations < 2u || map.iterations != transducer.template_count ||
        map.iterations > std::numeric_limits<std::uint32_t>::max() ||
        transducer.template_first >= build.templates.size() ||
        transducer.template_count >
            build.templates.size() - transducer.template_first) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const BackendBatchEntry &owner = build.templates[transducer.template_first];
    if (owner.run == nullptr || owner.run->pick == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    rund::AccelCheck ready{};
    try {
      ready = PrepareMetalRecurrenceRoute(
          build.template_registry, *build.context.adapter, *owner.run->pick,
          *owner.run, map, map.first->control,
          build.pipeline->transducers[index],
          static_cast<std::uint32_t>(map.iterations));
    } catch (...) {
      backend_exception::RethrowUnlessCapacityException();
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (!ready.ok) {
      return ready;
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_admit_internal

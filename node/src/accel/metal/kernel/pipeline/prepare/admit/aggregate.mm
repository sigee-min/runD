#include "internal.hpp"

#include "../../aggregate/admit.hpp"

#include "../../../../../kernel/backend/exception.hpp"

#include <string_view>

namespace rund::node::accel::detail::metal_pipeline_admit_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck AdmitMetalAggregate(MetalPipelineBuild &build,
                                     const bool compact_input) {
  if (build.aggregates.size() == 1u) {
    const NestedAggregate &aggregate = build.aggregates.front();
    NestedTemplateShape expected_shape{};
    const bool complete_templates =
        aggregate.shape.valid() && aggregate.shape.first() == 0u &&
        aggregate.shape.end() == build.templates.size() &&
        ProveNestedTemplateShape(aggregate.shape.first(), aggregate.maximum,
                                 aggregate.tile, aggregate.shape.inner_bound(),
                                 expected_shape) &&
        expected_shape == aggregate.shape;
    const bool complete_publication =
        build.publications.size() == 1u && aggregate.publication_index == 0u;
    const bool profile_ready =
        !build.profile_steps || aggregate.profile.aggregate_profile_supported;
    const bool seed_profile_owner =
        aggregate.shape.seed_first() < build.status.active_step_count &&
        build.status.declared_steps[aggregate.shape.seed_first()] <
            build.status.declared_step_count;
    if (complete_templates && complete_publication && profile_ready &&
        seed_profile_owner) {
      const rund::AccelCheck admitted = AdmitMetalNestedAggregate(
          aggregate, build.status, build.profile_steps, build.context,
          build.native_aggregate);
      if (!admitted.ok) {
        if (compact_input || std::string_view{admitted.reason} !=
                                 "accel_kernel_primitive_unsupported") {
          return admitted;
        }
      } else {
        build.aggregate_profile_owner =
            build.status.declared_steps[aggregate.shape.seed_first()];
        build.aggregate_selected = true;
      }
    }
  }
  if (!build.aggregate_selected) {
    return rund::AccelCheck{true, "ok"};
  }
  try {
    build.pipeline = std::make_shared<MetalSequence>();
    build.pipeline->adapter = build.context.adapter;
    build.pipeline->profile_steps = build.profile_steps;
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
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_admit_internal

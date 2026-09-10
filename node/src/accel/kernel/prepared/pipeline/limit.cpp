#include "../interface/api.hpp"

#include "../model.hpp"
#include "limit/internal.hpp"
#include "registry.hpp"
#include "structure.hpp"

#include <cstdint>
#include <limits>
#include <span>

namespace rund::node::accel::detail {

PreparedKernelPipelineReservation PlanPreparedKernelPipelineLimit(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelProgramRoute> routes,
    const PreparedKernelPipelineShape shape,
    PreparedKernelTemplateRegistry &templates) noexcept {
  prepared_pipeline_limit::State state{};
  state.result.template_capacity = std::numeric_limits<std::uint64_t>::max();
  if (!context.check.ok || context.id == 0u || routes.empty() ||
      routes.size() > PreparedPipelineStepCapacity ||
      shape.publication_count > 32u || shape.declared_step_count == 0u ||
      shape.declared_step_count > PreparedPipelineStepCapacity ||
      !valid_publication_shape(shape) ||
      shape.window_descriptor_state_count > shape.window_state_count ||
      ((shape.window_state_count == 0u) !=
       (shape.window_descriptor_state_count == 0u)) ||
      (shape.route_copies != 1u && shape.route_copies != 2u) ||
      (context.api != rund::AccelApi::Metal &&
       context.api != rund::AccelApi::Vulkan)) {
    state.result.reason = "accel_kernel_run_invalid";
    templates.limit = state.result;
    return state.result;
  }
  fingerprint_pipeline_header(state.result.fingerprint_hi,
                              state.result.fingerprint_lo, context, shape,
                              routes.size());
  for (std::size_t route_index = 0u; route_index < routes.size();
       ++route_index) {
    if (!prepared_pipeline_limit::plan_route(context, routes, route_index, shape,
                                             state)) {
      templates.limit = state.result;
      return state.result;
    }
  }
  static_cast<void>(prepared_pipeline_limit::finalize(context, shape, state));
  templates.limit = state.result;
  return state.result;
}

} // namespace rund::node::accel::detail

#include "../../../state/assembly.hpp"
#include "local.hpp"

#include "../../../../../accel/kernel/recurrence.hpp"
#include "../../../../backend.hpp"
#include "../../../../status.hpp"

#include <kernel/core/checked.hpp>

#include <cstdint>
#include <limits>

namespace rund::compute::detail {

Status finalize_pipeline_accel_preparation(
    const PipelineBuildState &build, PipelineMemoryPlan &plan,
    PipelineAccelPreparationDraft &draft) {
  if (draft.routes.empty()) {
    return Status::success();
  }
  auto &routes = draft.routes;
  const std::uint32_t route_copies = draft.route_copies;
  const std::uint64_t window_state_count = draft.window_state_count;
  const std::uint64_t window_descriptor_state_count =
      draft.window_descriptor_state_count;
  const DeviceOps *const ops = draft.ops;

  node::accel::detail::PreparedKernelTemplateRegistry templates{};
  std::uint64_t terminal_publication_count = 0u;
  std::uint64_t publication_command_count = 0u;
  for (const PipelinePublicationPlan &publication : plan.publications) {
    const auto *window =
        std::get_if<PipelineWindowPublicationPlan>(&publication);
    const std::uint32_t state = std::visit(
        [](const auto &typed) { return typed.state; }, publication);
    if (state >= plan.window_controls.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const node::accel::detail::NestedTemplateShape *const nested_shape =
        window != nullptr ? pipeline_build_nested_shape(build, state)
                          : nullptr;
    if (window != nullptr &&
        (nested_shape == nullptr || !nested_shape->valid())) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::uint64_t contribution = 0u;
    if (!node::accel::detail::PreparedKernelPublicationCommandContribution(
            window == nullptr
                ? node::accel::detail::PreparedKernelPublicationKind::Terminal
                : node::accel::detail::PreparedKernelPublicationKind::Window,
            nested_shape == nullptr ? 0u : nested_shape->outer_bound(),
            contribution) ||
        !kernel::checked::add(publication_command_count, contribution,
                              publication_command_count) ||
        (window == nullptr &&
         !kernel::checked::add(terminal_publication_count, 1u,
                               terminal_publication_count))) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  plan.accel_preparation = ops->plan_pipeline_preparation(
      *build.device, routes,
      node::accel::detail::PreparedKernelPipelineShape{
          .publication_count = plan.publications.size(),
          .terminal_publication_count = terminal_publication_count,
          .backend_publication_command_count = publication_command_count,
          .window_state_count = window_state_count,
          .window_descriptor_state_count = window_descriptor_state_count,
          .publication_fingerprint_hi = plan.publication_fingerprint_hi,
          .publication_fingerprint_lo = plan.publication_fingerprint_lo,
          .declared_step_count = build.steps.size(),
          .route_copies = route_copies,
          .profile_steps = build.profile == PipelineProfile::Steps,
      },
      templates);
  if (!plan.accel_preparation.ok) {
    return Status::fail(project_reason(plan.accel_preparation.reason,
                                       Reason::LoweringInvalid));
  }
  if (!templates.limit.ok) {
    return Status::fail(
        project_reason(templates.limit.reason, Reason::LoweringInvalid));
  }
  if (plan.accel_preparation.fingerprint_hi !=
          templates.limit.fingerprint_hi ||
      plan.accel_preparation.fingerprint_lo !=
          templates.limit.fingerprint_lo) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!kernel::checked::add(plan.summary.prepared_host_bytes,
                            plan.accel_preparation.host_bytes,
                            plan.summary.prepared_host_bytes) ||
      !kernel::checked::add(plan.summary.prepared_native_bytes,
                            plan.accel_preparation.native_bytes,
                            plan.summary.prepared_native_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  return Status::success();
}

} // namespace rund::compute::detail

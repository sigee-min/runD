#include "src/accel/kernel/backend/execute.hpp"
#include "src/accel/kernel/backend/pipeline/failure.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

#include "failure.hpp"
#include "hooks.hpp"

namespace node_accel_contract {

using rund::node::accel::detail::BackendRun;
using BackendWindow = rund::node::accel::detail::BackendWindow;

[[nodiscard]] bool PreparedPipelineFailureCoordinatesAreExact() {
  using namespace rund::node::accel::detail;
  PreparedPipelineFailureContext context{};
  const PreparedPipelineFailure unknown = context.failure(nullptr);
  if (unknown.stage != PreparedPipelineFailureStage::Unknown ||
      unknown.template_index != PreparedPipelineUnknownCoordinate ||
      unknown.occurrence_index != PreparedPipelineUnknownCoordinate ||
      unknown.node != PreparedPipelineUnknownCoordinate ||
      unknown.outer_iteration != PreparedPipelineUnknownCoordinate ||
      unknown.inner_iteration != PreparedPipelineUnknownCoordinate ||
      unknown.nested_phase != rund::compute::PipelineNestedPhase::None ||
      std::string_view{unknown.native_reason_key} !=
          "accel_kernel_pipeline_invalid") {
    return false;
  }

  struct RouteCase final {
    BackendWindowPhase backend_phase;
    rund::compute::PipelineNestedPhase public_phase;
    bool outer_known;
    bool inner_known;
  };
  constexpr std::array<RouteCase, 4u> route_cases{{
      {BackendWindowPhase::Ordinary, rund::compute::PipelineNestedPhase::None,
       false, false},
      {BackendWindowPhase::NestedSeed, rund::compute::PipelineNestedPhase::Seed,
       true, false},
      {BackendWindowPhase::NestedAction,
       rund::compute::PipelineNestedPhase::Action, true, true},
      {BackendWindowPhase::NestedFold, rund::compute::PipelineNestedPhase::Fold,
       true, false},
  }};
  for (std::size_t index = 0u; index < route_cases.size(); ++index) {
    const RouteCase route = route_cases[index];
    BackendWindow phase_window{
        .maximum = 8u,
        .tile = 1u,
        .outer_iteration = 5u,
        .outer_bound = 8u,
        .inner_iteration = 7u,
        .inner_bound = 8u,
        .inner_advance =
            route.backend_phase == BackendWindowPhase::NestedAction ? 1u : 0u,
        .phase = route.backend_phase,
    };
    const BackendRecurrence recurrence{.window = &phase_window};
    const std::uint32_t template_index = static_cast<std::uint32_t>(3u + index);
    const std::uint32_t occurrence_index =
        static_cast<std::uint32_t>(9u + index);
    const BackendBatchEntry phase_entry{
        .recurrence = recurrence,
        .template_index = template_index,
        .occurrence_index = occurrence_index,
    };
    const std::uint32_t expected_outer =
        route.outer_known ? 5u : PreparedPipelineUnknownCoordinate;
    const std::uint32_t expected_inner =
        route.inner_known ? 7u : PreparedPipelineUnknownCoordinate;

    BackendWindow compact_window{
        .maximum = 8u,
        .tile = 1u,
        .outer_iteration =
            route.backend_phase == BackendWindowPhase::NestedSeed ? 5u : 0u,
        .outer_bound = 8u,
        .inner_iteration = 7u,
        .inner_bound = 8u,
        .inner_advance =
            route.backend_phase == BackendWindowPhase::NestedAction ? 1u : 0u,
        .phase = route.backend_phase,
    };
    const BackendRecurrence compact_recurrence{.window = &compact_window};
    context.stage(PreparedPipelineFailureStage::BackendCapture);
    context.compact_template_route(template_index, compact_recurrence);
    const PreparedPipelineFailure template_route = context.failure("template");
    const bool seed_template =
        route.backend_phase == BackendWindowPhase::NestedSeed;
    if (template_route.stage != PreparedPipelineFailureStage::BackendCapture ||
        template_route.template_index != template_index ||
        template_route.occurrence_index != PreparedPipelineUnknownCoordinate ||
        template_route.node != PreparedPipelineUnknownCoordinate ||
        template_route.outer_iteration !=
            (seed_template ? 5u : PreparedPipelineUnknownCoordinate) ||
        template_route.inner_iteration != PreparedPipelineUnknownCoordinate ||
        template_route.nested_phase !=
            (seed_template ? rund::compute::PipelineNestedPhase::Seed
                           : rund::compute::PipelineNestedPhase::None)) {
      return false;
    }

    context.occurrence_route(phase_entry);
    const PreparedPipelineFailure occurrence = context.failure("occurrence");
    if (occurrence.stage != PreparedPipelineFailureStage::BackendCapture ||
        occurrence.template_index != template_index ||
        occurrence.occurrence_index != occurrence_index ||
        occurrence.node != PreparedPipelineUnknownCoordinate ||
        occurrence.outer_iteration != expected_outer ||
        occurrence.inner_iteration != expected_inner ||
        occurrence.nested_phase != route.public_phase) {
      return false;
    }
  }

  for (const std::uint32_t invalid_bound : {0u, 4u}) {
    BackendWindow invalid_seed{
        .maximum = 8u,
        .tile = 1u,
        .outer_iteration = 4u,
        .outer_bound = invalid_bound,
        .phase = BackendWindowPhase::NestedSeed,
    };
    context.compact_template_route(13u,
                                   BackendRecurrence{.window = &invalid_seed});
    const PreparedPipelineFailure invalid_route = context.failure("invalid");
    if (invalid_route.outer_iteration != PreparedPipelineUnknownCoordinate ||
        invalid_route.inner_iteration != PreparedPipelineUnknownCoordinate ||
        invalid_route.nested_phase !=
            rund::compute::PipelineNestedPhase::None) {
      return false;
    }
  }

  BackendWindow invalid_action{
      .maximum = 8u,
      .tile = 1u,
      .outer_iteration = 1u,
      .outer_bound = 2u,
      .inner_iteration = 3u,
      .inner_bound = 3u,
      .inner_advance = 1u,
      .phase = BackendWindowPhase::NestedAction,
  };
  context.occurrence_route(BackendBatchEntry{
      .recurrence = BackendRecurrence{.window = &invalid_action},
      .template_index = 14u,
      .occurrence_index = 15u,
  });
  const PreparedPipelineFailure invalid_occurrence =
      context.failure("invalid_occurrence");
  if (invalid_occurrence.template_index != 14u ||
      invalid_occurrence.occurrence_index != 15u ||
      invalid_occurrence.node != PreparedPipelineUnknownCoordinate ||
      invalid_occurrence.outer_iteration != PreparedPipelineUnknownCoordinate ||
      invalid_occurrence.inner_iteration != PreparedPipelineUnknownCoordinate ||
      invalid_occurrence.nested_phase !=
          rund::compute::PipelineNestedPhase::None) {
    return false;
  }

  BackendWindow transduced_seed{
      .maximum = 8u,
      .tile = 1u,
      .outer_iteration = 1u,
      .outer_bound = 2u,
      .phase = BackendWindowPhase::NestedSeed,
  };
  context.occurrence_route(BackendBatchEntry{
      .recurrence = BackendRecurrence{.window = &transduced_seed},
      .transducer = 0u,
      .template_index = 16u,
      .occurrence_index = 17u,
  });
  const PreparedPipelineFailure invalid_transduced =
      context.failure("invalid_transduced");
  if (invalid_transduced.template_index != 16u ||
      invalid_transduced.occurrence_index != 17u ||
      invalid_transduced.outer_iteration != PreparedPipelineUnknownCoordinate ||
      invalid_transduced.inner_iteration != PreparedPipelineUnknownCoordinate ||
      invalid_transduced.nested_phase !=
          rund::compute::PipelineNestedPhase::None) {
    return false;
  }

  struct MaterializedCase final {
    BackendWindowPhase backend_phase;
    rund::compute::PipelineNestedPhase public_phase;
    std::uint32_t inner;
    std::uint32_t route;
    std::uint32_t inner_advance;
    bool transduced;
  };
  constexpr std::array<MaterializedCase, 3u> materialized_cases{{
      {BackendWindowPhase::NestedAction,
       rund::compute::PipelineNestedPhase::Action, 3u, 0u, 1u, false},
      {BackendWindowPhase::NestedAction,
       rund::compute::PipelineNestedPhase::Action, 3u, 0u, 0u, true},
      {BackendWindowPhase::NestedFold, rund::compute::PipelineNestedPhase::Fold,
       5u, 2u, 5u, false},
  }};
  for (const MaterializedCase route : materialized_cases) {
    BackendWindow compact{
        .maximum = 8u,
        .tile = 1u,
        .outer_iteration = 0u,
        .outer_bound = 3u,
        .inner_iteration = route.inner,
        .inner_bound = 5u,
        .inner_advance = route.inner_advance,
        .route = route.route,
        .phase = route.backend_phase,
    };
    PreparedPipelineFailure materialized{};
    if (!MaterializePreparedPipelineOccurrenceForContract(
            compact, 2u, 3u, route.inner, 5u, route.route, route.inner_advance,
            route.transduced, materialized) ||
        materialized.stage != PreparedPipelineFailureStage::CommonExpansion ||
        materialized.template_index != 0u ||
        materialized.occurrence_index != 0u ||
        materialized.node != PreparedPipelineUnknownCoordinate ||
        materialized.outer_iteration != 2u ||
        materialized.inner_iteration !=
            (route.backend_phase == BackendWindowPhase::NestedAction
                 ? route.inner
                 : PreparedPipelineUnknownCoordinate) ||
        materialized.nested_phase != route.public_phase) {
      return false;
    }
  }

  KernelExecutionStep execution_step{};
  execution_step.source.begin.index = 17u;
  BoundStep step{.step = &execution_step};
  BackendRun run{.steps = &step, .step_count = 1u};
  BackendWindow window{
      .maximum = 8u,
      .tile = 1u,
      .outer_iteration = 5u,
      .outer_bound = 8u,
      .inner_iteration = 7u,
      .inner_bound = 8u,
      .inner_advance = 1u,
      .phase = BackendWindowPhase::NestedAction,
  };
  const BackendBatchEntry entry{
      .run = &run,
      .recurrence = BackendRecurrence{.window = &window},
      .template_index = 3u,
      .occurrence_index = 9u,
  };
  context.stage(PreparedPipelineFailureStage::BackendCapture);
  BackendWindow compact_action{
      .maximum = 8u,
      .tile = 1u,
      .outer_iteration = 0u,
      .outer_bound = 8u,
      .inner_iteration = 7u,
      .inner_bound = 8u,
      .inner_advance = 1u,
      .phase = BackendWindowPhase::NestedAction,
  };
  context.compact_template_node_route(
      3u, BackendRecurrence{.window = &compact_action}, run, 0u);
  const PreparedPipelineFailure compact_node = context.failure("capacity");
  if (compact_node.template_index != 3u ||
      compact_node.occurrence_index != PreparedPipelineUnknownCoordinate ||
      compact_node.node != 17u ||
      compact_node.outer_iteration != PreparedPipelineUnknownCoordinate ||
      compact_node.inner_iteration != PreparedPipelineUnknownCoordinate ||
      compact_node.nested_phase != rund::compute::PipelineNestedPhase::None) {
    return false;
  }

  context.node_route(entry, 0u);
  const PreparedPipelineFailure exact = context.failure("native_failure");
  if (exact.stage != PreparedPipelineFailureStage::BackendCapture ||
      exact.template_index != 3u || exact.occurrence_index != 9u ||
      exact.node != 17u || exact.outer_iteration != 5u ||
      exact.inner_iteration != 7u ||
      exact.nested_phase != rund::compute::PipelineNestedPhase::Action ||
      std::string_view{exact.native_reason_key} != "native_failure") {
    return false;
  }

  context.stage(PreparedPipelineFailureStage::BackendDescription);
  context.template_node_route(11u, 23u);
  const PreparedPipelineFailure restored = context.failure("status_source");
  if (restored.stage != PreparedPipelineFailureStage::BackendDescription ||
      restored.template_index != 11u ||
      restored.occurrence_index != PreparedPipelineUnknownCoordinate ||
      restored.node != 23u ||
      restored.outer_iteration != PreparedPipelineUnknownCoordinate ||
      restored.inner_iteration != PreparedPipelineUnknownCoordinate ||
      restored.nested_phase != rund::compute::PipelineNestedPhase::None ||
      std::string_view{restored.native_reason_key} != "status_source") {
    return false;
  }

  context.stage(PreparedPipelineFailureStage::BackendFinalization);
  const PreparedPipelineFailure cleared = context.failure("");
  return cleared.stage == PreparedPipelineFailureStage::BackendFinalization &&
         cleared.template_index == PreparedPipelineUnknownCoordinate &&
         cleared.occurrence_index == PreparedPipelineUnknownCoordinate &&
         cleared.node == PreparedPipelineUnknownCoordinate &&
         cleared.outer_iteration == PreparedPipelineUnknownCoordinate &&
         cleared.inner_iteration == PreparedPipelineUnknownCoordinate &&
         cleared.nested_phase == rund::compute::PipelineNestedPhase::None &&
         std::string_view{cleared.native_reason_key} ==
             "accel_kernel_pipeline_invalid";
}

[[nodiscard]] bool PreparedPipelinePreparePathsAlwaysReportFailure() {
  using namespace rund::node::accel::detail;
  const PreparedKernelPipeline common =
      PrepareKernelPipeline({}, {}, {}, {}, {}, {}, 0u, 1u, false);
  if (common.ok ||
      common.failure.stage != PreparedPipelineFailureStage::CommonValidation ||
      common.failure.template_index != PreparedPipelineUnknownCoordinate ||
      std::string_view{common.failure.native_reason_key} !=
          "accel_kernel_run_invalid") {
    return false;
  }

  const auto rejects_with_failure = [](const auto prepare) {
    PreparedKernelTemplateRegistry registry{};
    PreparedPipelineStatusLayout status{};
    std::shared_ptr<void> prepared{};
    PreparedPipelineMemory memory{};
    rund::AccelRunFacts preparation{};
    PreparedPipelineFailure failure{};
    const rund::AccelCheck check =
        prepare({}, {}, {}, {}, {}, {}, registry, status, false, prepared,
                memory, nullptr, preparation, failure);
    return !check.ok &&
           failure.stage == PreparedPipelineFailureStage::BackendAdmission &&
           failure.template_index == PreparedPipelineUnknownCoordinate &&
           failure.occurrence_index == PreparedPipelineUnknownCoordinate &&
           failure.node == PreparedPipelineUnknownCoordinate &&
           failure.outer_iteration == PreparedPipelineUnknownCoordinate &&
           failure.inner_iteration == PreparedPipelineUnknownCoordinate &&
           failure.nested_phase == rund::compute::PipelineNestedPhase::None &&
           check.reason != nullptr && failure.native_reason_key != nullptr &&
           std::string_view{failure.native_reason_key} == check.reason;
  };
  return rejects_with_failure(&PrepareMetalPipeline) &&
         rejects_with_failure(&PrepareVulkanPipeline);
}

} // namespace node_accel_contract

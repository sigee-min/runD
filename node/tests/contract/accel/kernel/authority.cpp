#include "src/accel/kernel/backend/execute.hpp"
#include "src/accel/kernel/backend/manifest.hpp"
#include "src/accel/kernel/backend/pipeline_failure.hpp"
#include "src/accel/kernel/backend/source_recipe.hpp"
#include "src/accel/kernel/backend/template_plan.hpp"
#include "src/accel/kernel/finish.hpp"
#include "src/accel/kernel/grid.hpp"
#include "src/accel/kernel/prepared.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/kernel/submission.hpp"
#include "src/accel/kernel/telemetry.hpp"
#include "src/accel/metal/kernel/pipeline/identity_index.hpp"
#include "src/accel/metal/kernel/pipeline/icb.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "src/accel/vulkan/kernel/pipeline/state.hpp"
#include "src/accel/vulkan/kernel/reset_source.hpp"
#include "src/accel/vulkan/map/control.hpp"
#endif

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "src/accel/metal/compact/local.hpp"
#include "src/accel/metal/gather/local.hpp"
#include "src/accel/metal/histogram/local.hpp"
#include "src/accel/metal/kernel/manifest.hpp"
#include "src/accel/metal/kernel/pipeline/capture.hpp"
#include "src/accel/metal/kernel/pipeline/state.hpp"
#include "src/accel/metal/numeric/source.hpp"
#include "src/accel/metal/partition/local.hpp"
#include "src/accel/metal/reduce/local.hpp"
#include "src/accel/metal/runtime/map/control.hpp"
#include "src/accel/metal/scan/source.hpp"
#include "src/accel/metal/scatter/local.hpp"
#include "src/accel/metal/scatter/reduce/model.hpp"
#include "src/accel/metal/segmented/local.hpp"
#include "src/accel/metal/segmented/reduce/model.hpp"
#include "src/accel/metal/sort/source.hpp"
#include "src/accel/metal/stencil/local.hpp"
#include "src/accel/metal/stencil/pipeline/name.hpp"
#include "src/accel/sort/block/metal.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>

namespace rund::node::accel::detail {

[[nodiscard]] bool AccumulatePreparedKernelRouteProjectionForContract(
    PreparedKernelPipelineReservation &projection,
    const PreparedKernelRouteReservation &route,
    std::uint64_t compact_entry_count, std::uint64_t occurrence_count,
    std::uint64_t window_count) noexcept;
[[nodiscard]] bool BackendTemplateRouteDemandForContract(
    std::uint64_t owner_count, std::uint64_t route_copies,
    BackendTemplateRouteDemand &demand) noexcept;
[[nodiscard]] bool PlanBackendTemplateRouteDemandsForContract(
    std::span<const PreparedKernelRun *const> runs, std::uint32_t route_copies,
    std::span<BackendTemplateRouteDemand> demands,
    std::uint64_t &unique_route_count, std::uint64_t &template_count) noexcept;
[[nodiscard]] bool BackendPreparationCursorLifecycleForContract(
    BackendRun &run, BackendTemplateRouteDemand demand) noexcept;
[[nodiscard]] bool ScalePreparedMapRecurrenceRoutesForContract(
    const PreparedMapRecurrenceReservation &reservation, std::uint64_t copies,
    PreparedMapRecurrenceReservation &scaled) noexcept;

} // namespace rund::node::accel::detail

namespace node_accel_contract {
namespace {

using rund::node::accel::detail::BackendRun;
using rund::node::accel::detail::KernelResult;

namespace source_recipe = rund::node::accel::detail::backend_source_recipe;

inline constexpr std::array<std::string_view, 3u> kBackendSourceRecipePrefix{
    "kernel void ", "",
    "rund_backend_source_recipe_contract_with_long_name(value="};
inline constexpr std::array<std::uint64_t, 6u> kBackendSourceRecipeDecimals{
    0u, 9u, 10u, 99u, 100u, std::numeric_limits<std::uint64_t>::max()};
inline constexpr std::array<std::string_view, 2u> kBackendSourceRecipeSuffix{
    "", ");"};

struct BackendSourceRecipeEmitter final {
  template <typename Sink>
  [[nodiscard]] bool operator()(Sink &sink) const noexcept(
      noexcept(source_recipe::append_fixed(sink, kBackendSourceRecipePrefix)) &&
      noexcept(source_recipe::append_decimal(sink, std::uint64_t{})) &&
      noexcept(sink.append(std::string_view{}))) {
    if (!source_recipe::append_fixed(sink, kBackendSourceRecipePrefix)) {
      return false;
    }
    for (std::size_t index = 0u; index < kBackendSourceRecipeDecimals.size();
         ++index) {
      if ((index != 0u && !sink.append(",")) ||
          !source_recipe::append_decimal(sink,
                                         kBackendSourceRecipeDecimals[index])) {
        return false;
      }
    }
    return source_recipe::append_fixed(sink, kBackendSourceRecipeSuffix);
  }
};

struct DivergentBackendSourceRecipeEmitter final {
  [[nodiscard]] bool operator()(source_recipe::CountSink &sink) const noexcept {
    return sink.append("x");
  }

  [[nodiscard]] bool operator()(source_recipe::StringSink &sink) const {
    return sink.append("xx");
  }
};

void IgnoreCompletion(void *, KernelResult) noexcept {}

struct SubmissionOwner final {};

[[nodiscard]] bool MatchPreparedInteger(const void *const prepared,
                                        const void *const probe) noexcept {
  return prepared != nullptr && probe != nullptr &&
         *static_cast<const int *>(prepared) ==
             *static_cast<const int *>(probe);
}

[[nodiscard]] rund::node::accel::detail::PreparedMemory
ObservePreparedInteger(const void *const prepared) noexcept {
  using rund::node::accel::detail::PreparedMemory;
  return prepared == nullptr ? PreparedMemory{}
                             : PreparedMemory{.current = sizeof(int),
                                              .peak = sizeof(int),
                                              .cumulative = sizeof(int),
                                              .budget = sizeof(int)};
}

[[nodiscard]] bool MatchBackendTemplateGroup(const BackendRun &left,
                                             const BackendRun &right) noexcept {
  return left.original_dispatch_count == right.original_dispatch_count;
}

[[nodiscard]] bool
AsymmetricBackendTemplateGroup(const BackendRun &left,
                               const BackendRun &right) noexcept {
  return left.original_dispatch_count <= right.original_dispatch_count;
}

[[nodiscard]] bool SubmissionTransitions() {
  using namespace rund::node::accel::detail;
  SubmissionOwner owner{};
  submission::State<SubmissionOwner> state{};
  std::atomic_bool start{};
  std::atomic_uint accepted{};
  const auto claim = [&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (submission::Begin(state, owner, IgnoreCompletion, nullptr)) {
      accepted.fetch_add(1u, std::memory_order_relaxed);
    }
  };
  std::thread first{claim};
  std::thread second{claim};
  start.store(true, std::memory_order_release);
  first.join();
  second.join();
  if (accepted.load(std::memory_order_relaxed) != 1u) {
    return false;
  }
  const submission::Claim<SubmissionOwner> taken = submission::Take(state);
  if (!taken || taken.owner != &owner || taken.completion != IgnoreCompletion ||
      taken.user != nullptr || submission::Take(state)) {
    return false;
  }
  if (submission::Begin(state, owner, nullptr, nullptr) ||
      !submission::Begin(state, owner, IgnoreCompletion, &owner)) {
    return false;
  }
  submission::Cancel(state);
  if (submission::Take(state) ||
      !submission::Begin(state, owner, IgnoreCompletion, nullptr)) {
    return false;
  }
  submission::Cancel(state);
  return true;
}

[[nodiscard]] bool PreparedPipelineClaimHasOneAuthority() {
  using namespace rund::node::accel::detail;
  prepared::PipelineSubmission submission{};
  if (submission.active() || submission.pipeline() != nullptr) {
    return false;
  }
  const auto owner = std::make_shared<prepared::PipelineState>();
  submission.owner = owner;
  if (!submission.active() || submission.pipeline() != owner.get()) {
    return false;
  }
  submission.owner.reset();
  return !submission.active() && submission.pipeline() == nullptr;
}

[[nodiscard]] constexpr bool
PreparedPipelineNestedCoordinateShapesAreCanonical() noexcept {
  using rund::compute::PipelineNestedPhase;
  using rund::compute::valid_pipeline_nested_coordinate;

  constexpr std::array<PipelineNestedPhase, 4u> phases{
      PipelineNestedPhase::None,
      PipelineNestedPhase::Seed,
      PipelineNestedPhase::Action,
      PipelineNestedPhase::Fold,
  };
  constexpr std::array<bool, 2u> known_values{false, true};
  for (const PipelineNestedPhase phase : phases) {
    for (const bool outer_known : known_values) {
      for (const bool inner_known : known_values) {
        const bool expected =
            outer_known ==
                rund::compute::pipeline_nested_phase_has_outer(phase) &&
            inner_known ==
                rund::compute::pipeline_nested_phase_has_inner(phase);
        if (valid_pipeline_nested_coordinate(phase, outer_known, inner_known) !=
            expected) {
          return false;
        }
      }
    }
  }

  constexpr PipelineNestedPhase invalid =
      static_cast<PipelineNestedPhase>(0xffu);
  if (rund::compute::pipeline_nested_phase_valid(invalid)) {
    return false;
  }
  for (const bool outer_known : known_values) {
    for (const bool inner_known : known_values) {
      if (valid_pipeline_nested_coordinate(invalid, outer_known, inner_known)) {
        return false;
      }
    }
  }
  return true;
}

static_assert(PreparedPipelineNestedCoordinateShapesAreCanonical());

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
        .outer_iteration = 5u,
        .inner_iteration = 7u,
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

    context.stage(PreparedPipelineFailureStage::BackendCapture);
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

    context.template_recurrence_route(template_index, recurrence);
    const PreparedPipelineFailure template_route = context.failure("template");
    if (template_route.stage != PreparedPipelineFailureStage::BackendCapture ||
        template_route.template_index != template_index ||
        template_route.occurrence_index != PreparedPipelineUnknownCoordinate ||
        template_route.node != PreparedPipelineUnknownCoordinate ||
        template_route.outer_iteration != expected_outer ||
        template_route.inner_iteration != expected_inner ||
        template_route.nested_phase != route.public_phase) {
      return false;
    }
  }

  KernelExecutionStep execution_step{};
  execution_step.source.begin.index = 17u;
  BoundStep step{.step = &execution_step};
  BackendRun run{.steps = &step, .step_count = 1u};
  BackendWindow window{
      .outer_iteration = 5u,
      .inner_iteration = 7u,
      .phase = BackendWindowPhase::NestedAction,
  };
  const BackendBatchEntry entry{
      .run = &run,
      .recurrence = BackendRecurrence{.window = &window},
      .template_index = 3u,
      .occurrence_index = 9u,
  };
  context.stage(PreparedPipelineFailureStage::BackendCapture);
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
    PreparedPipelineFailure failure{};
    const rund::AccelCheck check =
        prepare({}, {}, {}, {}, {}, {}, registry, status, false, prepared,
                memory, failure);
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

[[nodiscard]] bool PreparedTemplateRegistryIsColdAndCollisionSafe() {
  using namespace rund::node::accel::detail;
  const BackendOps ops{
      .api = rund::AccelApi::Metal,
      .observe_pipeline_template = ObservePreparedInteger,
  };
  PreparedKernelTemplateRegistry registry{};
  const PreparedKernelPipelineReservation invalid =
      PlanPreparedKernelPipelineLimit({}, {}, {}, registry);
  if (invalid.ok || registry.owner != nullptr || registry.limit.ok) {
    return false;
  }
  if (!BindPreparedKernelTemplateRegistry(rund::AccelApi::Metal, 41u, registry)
           .ok ||
      registry.owner == nullptr) {
    return false;
  }

  KernelExecutionStep authority{};
  int first_probe = 7;
  std::shared_ptr<void> first = std::make_shared<int>(first_probe);
  if (!PublishPreparedKernelTemplate(registry, &authority, 11u, 13u, ops,
                                     MatchPreparedInteger, &first_probe, first)
           .ok) {
    return false;
  }
  const std::shared_ptr<void> found = FindPreparedKernelTemplate(
      registry, &authority, 11u, 13u, MatchPreparedInteger, &first_probe);
  if (found != first) {
    return false;
  }

  // Equal hashes are only a partition. Semantic mismatch must publish a
  // second immutable owner instead of aliasing the collision.
  int second_probe = 9;
  std::shared_ptr<void> second = std::make_shared<int>(second_probe);
  if (!PublishPreparedKernelTemplate(registry, &authority, 11u, 13u, ops,
                                     MatchPreparedInteger, &second_probe,
                                     second)
           .ok ||
      second == first ||
      FindPreparedKernelTemplate(registry, &authority, 11u, 13u,
                                 MatchPreparedInteger,
                                 &second_probe) != second) {
    return false;
  }

  std::shared_ptr<void> duplicate = std::make_shared<int>(first_probe);
  return PublishPreparedKernelTemplate(registry, &authority, 11u, 13u, ops,
                                       MatchPreparedInteger, &first_probe,
                                       duplicate)
             .ok &&
         duplicate == first;
}

[[nodiscard]] bool PreparedReservationIsFieldwiseFailClosed() {
  using namespace rund::node::accel::detail;
  PreparedKernelPipelineReservation limit{
      .fingerprint_hi = 0x1122334455667788ull,
      .fingerprint_lo = 0x8877665544332211ull,
      .template_step_capacity = 100u,
      .template_capacity = 100u,
      .descriptor_set_capacity = 100u,
      .descriptor_capacity = 100u,
      .ok = true,
      .reason = "ok",
  };
  constexpr std::array numeric_fields{
      &PreparedKernelPipelineReservation::host_bytes,
      &PreparedKernelPipelineReservation::native_bytes,
      &PreparedKernelPipelineReservation::route_host_bytes,
      &PreparedKernelPipelineReservation::route_native_bytes,
      &PreparedKernelPipelineReservation::template_host_bytes,
      &PreparedKernelPipelineReservation::template_native_bytes,
      &PreparedKernelPipelineReservation::template_source_bytes,
      &PreparedKernelPipelineReservation::source_transient_bytes,
      &PreparedKernelPipelineReservation::host_transient_bytes,
      &PreparedKernelPipelineReservation::route_count,
      &PreparedKernelPipelineReservation::template_count,
      &PreparedKernelPipelineReservation::template_step_count,
      &PreparedKernelPipelineReservation::route_step_count,
      &PreparedKernelPipelineReservation::authored_entry_count,
      &PreparedKernelPipelineReservation::occurrence_count,
      &PreparedKernelPipelineReservation::window_count,
      &PreparedKernelPipelineReservation::nested_group_count,
      &PreparedKernelPipelineReservation::backend_dispatch_count,
      &PreparedKernelPipelineReservation::backend_reset_dispatch_count,
      &PreparedKernelPipelineReservation::backend_window_dispatch_count,
      &PreparedKernelPipelineReservation::backend_indirect_dispatch_count,
      &PreparedKernelPipelineReservation::backend_window_state_count,
      &PreparedKernelPipelineReservation::backend_window_descriptor_state_count,
      &PreparedKernelPipelineReservation::backend_step_occurrence_count,
      &PreparedKernelPipelineReservation::backend_step_description_count,
      &PreparedKernelPipelineReservation::backend_status_source_count,
      &PreparedKernelPipelineReservation::backend_status_entry_count,
      &PreparedKernelPipelineReservation::backend_telemetry_count,
      &PreparedKernelPipelineReservation::backend_status_command_count,
      &PreparedKernelPipelineReservation::backend_telemetry_command_count,
      &PreparedKernelPipelineReservation::backend_publication_count,
      &PreparedKernelPipelineReservation::backend_terminal_publication_count,
      &PreparedKernelPipelineReservation::backend_window_control_command_count,
      &PreparedKernelPipelineReservation::backend_publication_command_count,
      &PreparedKernelPipelineReservation::backend_command_count,
      &PreparedKernelPipelineReservation::backend_command_chunk_count,
      &PreparedKernelPipelineReservation::backend_command_native_bytes,
      &PreparedKernelPipelineReservation::backend_command_binding_slot_upper,
      &PreparedKernelPipelineReservation::backend_command_binding_count,
      &PreparedKernelPipelineReservation::backend_parameter_bytes,
      &PreparedKernelPipelineReservation::backend_profile_step_count,
      &PreparedKernelPipelineReservation::backend_profile_command_count,
      &PreparedKernelPipelineReservation::backend_query_count,
      &PreparedKernelPipelineReservation::backend_native_buffer_count,
      &PreparedKernelPipelineReservation::backend_native_object_count,
      &PreparedKernelPipelineReservation::descriptor_set_count,
      &PreparedKernelPipelineReservation::descriptor_count,
      &PreparedKernelPipelineReservation::native_allocation_count,
  };
  for (const auto field : numeric_fields) {
    limit.*field = 7u;
  }
  constexpr std::array recurrence_fields{
      &PreparedMapRecurrenceReservation::route_host_bytes,
      &PreparedMapRecurrenceReservation::route_native_bytes,
      &PreparedMapRecurrenceReservation::template_host_bytes,
      &PreparedMapRecurrenceReservation::template_native_bytes,
      &PreparedMapRecurrenceReservation::template_source_bytes,
      &PreparedMapRecurrenceReservation::source_transient_bytes,
      &PreparedMapRecurrenceReservation::group_count,
      &PreparedMapRecurrenceReservation::history_group_count,
      &PreparedMapRecurrenceReservation::template_count,
      &PreparedMapRecurrenceReservation::route_step_count,
      &PreparedMapRecurrenceReservation::template_step_count,
      &PreparedMapRecurrenceReservation::descriptor_set_count,
      &PreparedMapRecurrenceReservation::descriptor_count,
      &PreparedMapRecurrenceReservation::route_native_allocation_count,
      &PreparedMapRecurrenceReservation::template_native_allocation_count,
  };
  for (const auto field : recurrence_fields) {
    limit.map_recurrence.*field = 7u;
  }
  if (!PreparedKernelPipelineReservationWithin(limit, limit)) {
    return false;
  }
  for (const auto field : numeric_fields) {
    PreparedKernelPipelineReservation candidate = limit;
    candidate.*field = limit.*field + 1u;
    if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
      return false;
    }
  }
  for (const auto field : recurrence_fields) {
    PreparedKernelPipelineReservation candidate = limit;
    candidate.map_recurrence.*field = limit.map_recurrence.*field + 1u;
    if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
      return false;
    }
  }
  for (const bool high : {true, false}) {
    PreparedKernelPipelineReservation candidate = limit;
    (high ? candidate.fingerprint_hi : candidate.fingerprint_lo) ^= 1u;
    if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
      return false;
    }
  }
  PreparedKernelPipelineReservation capacity = limit;
  capacity.template_step_capacity += 1u;
  return !PreparedKernelPipelineReservationWithin(capacity, limit);
}

[[nodiscard]] bool RecurrenceRouteCopiesDoNotCloneTemplates() {
  using namespace rund::node::accel::detail;
  const PreparedMapRecurrenceReservation one{
      .route_host_bytes = 11u,
      .route_native_bytes = 13u,
      .template_host_bytes = 17u,
      .template_native_bytes = 19u,
      .template_source_bytes = 23u,
      .source_transient_bytes = 29u,
      .group_count = 3u,
      .history_group_count = 1u,
      .template_count = 2u,
      .terminal_template_group_capacity = 6u,
      .history_template_group_capacity = 2u,
      .route_step_count = 3u,
      .template_step_count = 2u,
      .descriptor_set_count = 31u,
      .descriptor_count = 37u,
      .route_native_allocation_count = 41u,
      .template_native_allocation_count = 43u,
  };
  PreparedMapRecurrenceReservation two{};
  return ScalePreparedMapRecurrenceRoutesForContract(one, 2u, two) &&
         two.route_host_bytes == 22u && two.route_native_bytes == 26u &&
         two.group_count == 6u && two.history_group_count == 2u &&
         two.route_step_count == 6u && two.descriptor_set_count == 0u &&
         two.descriptor_count == 0u &&
         two.route_native_allocation_count == 82u &&
         two.template_host_bytes == 0u && two.template_native_bytes == 0u &&
         two.template_source_bytes == 0u && two.source_transient_bytes == 0u &&
         two.template_count == 0u &&
         two.terminal_template_group_capacity == 0u &&
         two.history_template_group_capacity == 0u &&
         two.template_step_count == 0u &&
         two.template_native_allocation_count == 0u &&
         !ScalePreparedMapRecurrenceRoutesForContract(one, 0u, two);
}

[[nodiscard]] bool MetalPointerIdentityIndexIsExactAndOneShot() {
  using namespace rund::node::accel::detail;
  const MetalPointerIdentityIndexLayout empty =
      PlanMetalPointerIdentityIndex(0u);
  const MetalPointerIdentityIndexLayout one = PlanMetalPointerIdentityIndex(1u);
  const MetalPointerIdentityIndexLayout three =
      PlanMetalPointerIdentityIndex(3u);
  const MetalPointerIdentityIndexLayout eight =
      PlanMetalPointerIdentityIndex(8u);
  const MetalPointerIdentityIndexLayout overflow =
      PlanMetalPointerIdentityIndex(
          std::numeric_limits<std::uint64_t>::max() / 2u + 1u);
  if (!empty.ok || empty.slot_count != 0u || empty.byte_count != 0u ||
      !one.ok || one.slot_count != 2u ||
      one.byte_count != 2u * sizeof(const void *) || !three.ok ||
      three.slot_count != 8u || three.byte_count != 8u * sizeof(const void *) ||
      !eight.ok || eight.slot_count != 16u ||
      eight.byte_count != 16u * sizeof(const void *) || overflow.ok) {
    return false;
  }

  int first = 0;
  int second = 0;
  int third = 0;
  MetalPointerIdentityIndex index{3u};
  bool inserted = false;
  if (!index.ready() || !index.insert(&first, inserted) || !inserted ||
      !index.insert(&first, inserted) || inserted ||
      !index.insert(&second, inserted) || !inserted) {
    return false;
  }
  index.clear();
  return index.insert(&first, inserted) && inserted &&
         index.insert(&second, inserted) && inserted &&
         index.insert(&third, inserted) && inserted;
}

[[nodiscard]] bool MetalIcbSizeClassPlanIsExactAndPortable() {
  using namespace rund::node::accel::detail;
  MetalIcbCalibration calibration{};
  for (std::size_t index = 0u; index < calibration.allocated_bytes.size();
       ++index) {
    calibration.allocated_bytes[index] =
        4096u + static_cast<std::uint64_t>(index) * 4096u;
  }
  if (!ValidMetalIcbCalibration(calibration)) {
    return false;
  }
  const MetalIcbChunkPlan empty = PlanMetalIcbChunks(0u, calibration);
  const MetalIcbChunkPlan one = PlanMetalIcbChunks(1u, calibration);
  const MetalIcbChunkPlan three = PlanMetalIcbChunks(3u, calibration);
  const MetalIcbChunkPlan full =
      PlanMetalIcbChunks(MetalPipelineIcbFullCommandCapacity, calibration);
  const MetalIcbChunkPlan next = PlanMetalIcbChunks(
      MetalPipelineIcbFullCommandCapacity + 1u, calibration);
  const MetalIcbChunkPlan multi = PlanMetalIcbChunks(
      2u * MetalPipelineIcbFullCommandCapacity + 32'769u, calibration);
  const MetalIcbChunkPlan product_actual =
      PlanMetalIcbChunks(189'825u, calibration);
  const MetalIcbChunkPlan product_upper =
      PlanMetalIcbChunks(355'766u, calibration);
  if (!empty.ok || empty.chunk_count != 0u || empty.allocated_bytes != 0u ||
      !one.ok || one.chunk_count != 1u || one.tail_command_count != 1u ||
      one.tail_command_capacity != 1u ||
      one.allocated_bytes != calibration.allocated_bytes[0u] ||
      one.retained_chunk_bytes != MetalPipelineIcbChunkHostBytes ||
      !three.ok || three.tail_command_capacity != 4u ||
      three.allocated_bytes != calibration.allocated_bytes[2u] || !full.ok ||
      full.full_chunk_count != 1u || full.tail_command_count != 0u ||
      full.chunk_count != 1u ||
      full.allocated_bytes != calibration.allocated_bytes.back() || !next.ok ||
      next.full_chunk_count != 1u || next.tail_command_count != 1u ||
      next.tail_command_capacity != 1u || next.chunk_count != 2u ||
      next.allocated_bytes != calibration.allocated_bytes.back() +
                                  calibration.allocated_bytes.front() ||
      !multi.ok || multi.full_chunk_count != 2u ||
      multi.tail_command_count != 32'769u ||
      multi.tail_command_capacity != MetalPipelineIcbFullCommandCapacity ||
      multi.chunk_count != 3u ||
      multi.allocated_bytes != 3u * calibration.allocated_bytes.back() ||
      !product_actual.ok || product_actual.chunk_count != 3u ||
      product_actual.full_chunk_count != 2u ||
      product_actual.tail_command_count != 58'753u ||
      product_actual.tail_command_capacity !=
          MetalPipelineIcbFullCommandCapacity ||
      !product_upper.ok || product_upper.chunk_count != 6u ||
      product_upper.full_chunk_count != 5u ||
      product_upper.tail_command_count != 28'086u ||
      product_upper.tail_command_capacity != 32'768u) {
    return false;
  }

  MetalIcbCalibration zero = calibration;
  zero.allocated_bytes[3u] = 0u;
  MetalIcbCalibration descending = calibration;
  descending.allocated_bytes[8u] = descending.allocated_bytes[7u] - 1u;
  MetalIcbCalibration overflow{};
  overflow.allocated_bytes.fill(
      std::numeric_limits<std::uint64_t>::max() / 2u + 1u);
  if (ValidMetalIcbCalibration(zero) ||
      ValidMetalIcbCalibration(descending) ||
      PlanMetalIcbChunks(1u, zero).ok ||
      PlanMetalIcbChunks(3u * MetalPipelineIcbFullCommandCapacity, overflow)
          .ok ||
      MetalIcbClassIndex(0u) != MetalPipelineIcbClassCount ||
      MetalIcbClassIndex(3u) != MetalPipelineIcbClassCount ||
      MetalIcbClassIndex(MetalPipelineIcbFullCommandCapacity) !=
          MetalPipelineIcbClassCount - 1u) {
    return false;
  }

  // Physical calibration is not semantic identity. It is deliberately a
  // field-wise actual<=frozen admission dimension under the same fingerprint.
  PreparedKernelPipelineReservation limit{};
  limit.ok = true;
  limit.reason = "ok";
  limit.fingerprint_hi = 11u;
  limit.fingerprint_lo = 13u;
  limit.backend_command_chunk_count = product_upper.chunk_count;
  limit.backend_command_native_bytes = product_upper.allocated_bytes;
  PreparedKernelPipelineReservation actual = limit;
  actual.backend_command_chunk_count = product_actual.chunk_count;
  actual.backend_command_native_bytes = product_actual.allocated_bytes;
  if (!PreparedKernelPipelineReservationWithin(actual, limit)) {
    return false;
  }
  actual.backend_command_chunk_count = limit.backend_command_chunk_count + 1u;
  if (PreparedKernelPipelineReservationWithin(actual, limit)) {
    return false;
  }
  actual = limit;
  actual.backend_command_native_bytes =
      limit.backend_command_native_bytes + 1u;
  return !PreparedKernelPipelineReservationWithin(actual, limit);
}

[[nodiscard]] bool MetalColdIdentityIndexBudgetIsExact() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  MetalIcbCalibration calibration{};
  for (std::size_t index = 0u; index < calibration.allocated_bytes.size();
       ++index) {
    calibration.allocated_bytes[index] =
        4096u + static_cast<std::uint64_t>(index) * 4096u;
  }
  const auto plan = [&](PreparedKernelPipelineReservation &value) {
    return PlanMetalPipelineStructureForCalibration(calibration, value);
  };
  PreparedKernelPipelineReservation reservation{};
  reservation.backend_dispatch_count = 1u;
  const rund::AccelCheck planned = plan(reservation);
  const MetalPointerIdentityIndexLayout index = PlanMetalPointerIdentityIndex(
      std::max(reservation.backend_command_count,
               reservation.backend_command_binding_count));
  if (!planned.ok || !index.ok || reservation.backend_command_count != 3u ||
      reservation.backend_command_binding_slot_upper != 4u ||
      reservation.backend_command_binding_count != 15u ||
      reservation.backend_parameter_bytes !=
          3u * sizeof(MetalNestedAggregateParams) ||
      reservation.backend_command_chunk_count != 1u ||
      reservation.native_bytes != reservation.backend_parameter_bytes +
                                      reservation.backend_command_native_bytes ||
      reservation.host_transient_bytes !=
          index.byte_count + reservation.backend_parameter_bytes ||
      reservation.host_transient_bytes == 0u) {
    return false;
  }
  PreparedKernelPipelineReservation windows{};
  windows.backend_dispatch_count = 1u;
  windows.backend_window_dispatch_count = 7u;
  windows.window_count = 5u;
  const rund::AccelCheck windows_planned = plan(windows);
  const MetalPointerIdentityIndexLayout windows_index =
      PlanMetalPointerIdentityIndex(
          std::max(windows.backend_command_count,
                   windows.backend_command_binding_count));
  if (!windows_planned.ok || !windows_index.ok ||
      windows.backend_command_count != 8u ||
      windows.backend_window_control_command_count != 5u ||
      windows.backend_command_binding_slot_upper != 8u ||
      windows.backend_command_binding_count !=
          windows.backend_command_count * 9u ||
      windows.backend_command_chunk_count != 1u ||
      windows.native_bytes != windows.backend_parameter_bytes +
                                  windows.backend_command_native_bytes ||
      windows.host_transient_bytes != windows_index.byte_count +
                                          5u * sizeof(MetalWindow) +
                                          windows.backend_parameter_bytes) {
    return false;
  }
  PreparedKernelPipelineReservation telemetry{};
  telemetry.backend_dispatch_count = 1u;
  telemetry.backend_telemetry_command_count = 1u;
  PreparedKernelPipelineReservation status{};
  status.backend_dispatch_count = 1u;
  status.backend_status_source_count = 1u;
  status.backend_status_command_count = 1u;
  status.backend_profile_step_count = 1u;
  PreparedKernelPipelineReservation route{};
  route.backend_dispatch_count = 1u;
  route.backend_command_binding_slot_upper = 11u;
  PreparedKernelPipelineReservation publication{};
  publication.backend_dispatch_count = 1u;
  publication.backend_publication_count = 99u;
  publication.backend_publication_command_count = 3u;
  return plan(telemetry).ok &&
         telemetry.backend_command_binding_slot_upper == 9u &&
         telemetry.backend_command_binding_count ==
             telemetry.backend_command_count * 10u &&
         plan(status).ok &&
         status.backend_command_binding_slot_upper == 7u &&
         status.backend_command_binding_count ==
             status.backend_command_count * 8u &&
         plan(route).ok &&
         route.backend_command_binding_slot_upper == 11u &&
         route.backend_command_binding_count ==
             route.backend_command_count * 12u &&
         plan(publication).ok &&
         publication.backend_command_count == 6u &&
         publication.backend_command_binding_slot_upper == 8u;
#else
  return true;
#endif
}

[[nodiscard]] bool MetalCaptureRowCapacityIsExact() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using rund::node::accel::detail::PlanMetalCaptureRowCapacity;
  using rund::node::accel::detail::PlanMetalParameterCapacity;
  const auto empty = PlanMetalCaptureRowCapacity(0u);
  const auto one = PlanMetalCaptureRowCapacity(1u);
  const auto eight = PlanMetalCaptureRowCapacity(8u);
  const auto nine = PlanMetalCaptureRowCapacity(9u);
  const auto overflow =
      PlanMetalCaptureRowCapacity(std::numeric_limits<std::uint64_t>::max());
  const auto parameter_first = PlanMetalParameterCapacity(0u, 1u, 1000u);
  const auto parameter_second = PlanMetalParameterCapacity(256u, 257u, 1000u);
  const auto parameter_capped = PlanMetalParameterCapacity(512u, 999u, 1000u);
  const auto parameter_stable = PlanMetalParameterCapacity(1000u, 1000u, 1000u);
  const auto parameter_overflow =
      PlanMetalParameterCapacity(1000u, 1001u, 1000u);
  return empty.ok && empty.rows == 0u && one.ok && one.rows == 1u && eight.ok &&
         eight.rows == 8u && nine.ok && nine.rows == 9u && !overflow.ok &&
         parameter_first.ok && parameter_first.rows == 256u &&
         parameter_second.ok && parameter_second.rows == 512u &&
         parameter_capped.ok && parameter_capped.rows == 1000u &&
         parameter_stable.ok && parameter_stable.rows == 1000u &&
         !parameter_overflow.ok;
#else
  return true;
#endif
}

[[nodiscard]] bool PreparedTemplateStepCapacityIsBackendBounded() {
  using namespace rund::node::accel::detail;
  PreparedKernelPipelineReservation limit{
      .fingerprint_hi = 0x74656d706c617465ull,
      .fingerprint_lo = 0x737465702e636170ull,
      // Plan-only inspection may freeze demand above the backend's native
      // materialization ceiling; the runtime candidate must still fit it.
      .template_step_count = 9u,
      .template_step_capacity = 8u,
      .ok = true,
      .reason = "ok",
  };
  PreparedKernelPipelineReservation candidate = limit;
  candidate.template_step_count = 8u;
  if (!PreparedKernelPipelineReservationWithin(candidate, limit)) {
    return false;
  }
  candidate.template_step_count = 9u;
  if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
    return false;
  }
  candidate.template_step_count = 8u;
  candidate.template_step_capacity = 9u;
  if (PreparedKernelPipelineReservationWithin(candidate, limit)) {
    return false;
  }

  PreparedPipelineFailureContext failure{};
  failure.stage(PreparedPipelineFailureStage::CommonAccounting);
  failure.template_node_route(7u, 11u);
  const PreparedPipelineFailure exact =
      failure.failure(PreparedPipelineTemplateStepCapacityReasonKey);
  return exact.stage == PreparedPipelineFailureStage::CommonAccounting &&
         exact.template_index == 7u &&
         exact.occurrence_index == PreparedPipelineUnknownCoordinate &&
         exact.node == 11u &&
         exact.outer_iteration == PreparedPipelineUnknownCoordinate &&
         exact.inner_iteration == PreparedPipelineUnknownCoordinate &&
         exact.nested_phase == rund::compute::PipelineNestedPhase::None &&
         std::string_view{exact.native_reason_key} ==
             PreparedPipelineTemplateStepCapacityReasonKey;
}

[[nodiscard]] bool PreparedBackendProjectionIsRouteWiseAndChecked() {
  using namespace rund::node::accel::detail;
  PreparedKernelPipelineReservation projection{};
  const PreparedKernelRouteReservation first{
      .route_step_count = 1u,
      .host_transient_bytes = 11u,
      .dispatch_count = 2u,
      .capture_direct_dispatch_count = 3u,
      .capture_indirect_dispatch_count = 1u,
      .capture_binding_slot_upper = 6u,
      .reset_dispatch_count = 1u,
      .status_entry_count = 4u,
      .status_source_count = 2u,
      .status_command_count = 3u,
      .status_parameter_bytes = 5u,
      .telemetry_source_count = 1u,
  };
  const PreparedKernelRouteReservation second{
      .route_step_count = 3u,
      .host_transient_bytes = 13u,
      .dispatch_count = 5u,
      .capture_direct_dispatch_count = 4u,
      .capture_indirect_dispatch_count = 2u,
      .capture_binding_slot_upper = 11u,
      .reset_dispatch_count = 4u,
      .status_entry_count = 7u,
      .status_source_count = 3u,
      .status_command_count = 4u,
      .status_parameter_bytes = 7u,
      .telemetry_source_count = 2u,
  };
  if (!AccumulatePreparedKernelRouteProjectionForContract(projection, first, 2u,
                                                          2u, 2u) ||
      !AccumulatePreparedKernelRouteProjectionForContract(projection, second,
                                                          3u, 3u, 1u) ||
      projection.occurrence_count != 5u ||
      projection.host_transient_bytes != 24u ||
      projection.backend_dispatch_count != 19u ||
      projection.backend_reset_dispatch_count != 14u ||
      projection.backend_step_occurrence_count != 11u ||
      projection.backend_step_description_count != 11u ||
      projection.backend_status_entry_count != 29u ||
      projection.backend_window_dispatch_count != 14u ||
      projection.backend_indirect_dispatch_count != 4u ||
      projection.backend_command_binding_slot_upper != 11u ||
      projection.backend_status_source_count != 13u ||
      projection.backend_telemetry_count != 8u ||
      projection.backend_status_command_count != 18u ||
      projection.backend_telemetry_command_count != 8u ||
      projection.backend_parameter_bytes != 31u) {
    return false;
  }

  // The former max(route)*occurrence projection would report 25/20/15/35.
  // Heterogeneous routes instead retain only their checked additive owners;
  // immutable descriptions are compact-entry-scaled, not
  // occurrence-scaled command counts.
  if (projection.backend_dispatch_count == 25u ||
      projection.backend_reset_dispatch_count == 20u ||
      projection.backend_step_occurrence_count == 15u ||
      projection.backend_status_entry_count == 35u) {
    return false;
  }

  const PreparedKernelPipelineReservation before = projection;
  const auto unchanged = [&before](
                             const PreparedKernelPipelineReservation &value) {
    return value.occurrence_count == before.occurrence_count &&
           value.host_transient_bytes == before.host_transient_bytes &&
           value.backend_dispatch_count == before.backend_dispatch_count &&
           value.backend_reset_dispatch_count ==
               before.backend_reset_dispatch_count &&
           value.backend_step_occurrence_count ==
               before.backend_step_occurrence_count &&
           value.backend_step_description_count ==
               before.backend_step_description_count &&
           value.backend_status_entry_count ==
               before.backend_status_entry_count &&
           value.backend_command_binding_slot_upper ==
               before.backend_command_binding_slot_upper &&
           value.backend_status_source_count ==
               before.backend_status_source_count &&
           value.backend_telemetry_count == before.backend_telemetry_count &&
           value.backend_status_command_count ==
               before.backend_status_command_count &&
           value.backend_telemetry_command_count ==
               before.backend_telemetry_command_count &&
           value.backend_parameter_bytes == before.backend_parameter_bytes &&
           value.backend_window_dispatch_count ==
               before.backend_window_dispatch_count &&
           value.backend_indirect_dispatch_count ==
               before.backend_indirect_dispatch_count &&
           value.backend_window_control_command_count ==
               before.backend_window_control_command_count &&
           value.backend_publication_command_count ==
               before.backend_publication_command_count;
  };
  const PreparedKernelRouteReservation overflow{
      .route_step_count = 1u,
      .dispatch_count = std::numeric_limits<std::uint64_t>::max(),
      .status_entry_count = 1u,
  };
  if (AccumulatePreparedKernelRouteProjectionForContract(projection, overflow,
                                                         1u, 2u, 0u) ||
      !unchanged(projection)) {
    return false;
  }
  const std::array overflow_routes{
      PreparedKernelRouteReservation{
          .route_step_count = std::numeric_limits<std::uint64_t>::max(),
          .dispatch_count = 1u,
      },
      PreparedKernelRouteReservation{
          .route_step_count = 1u,
          .dispatch_count = 1u,
          .status_parameter_bytes = std::numeric_limits<std::uint64_t>::max(),
      },
      PreparedKernelRouteReservation{
          .route_step_count = 1u,
          .dispatch_count = 1u,
          .capture_direct_dispatch_count =
              std::numeric_limits<std::uint64_t>::max(),
          .capture_indirect_dispatch_count = 1u,
      },
      PreparedKernelRouteReservation{
          .route_step_count = 1u,
          .dispatch_count = 1u,
          .capture_direct_dispatch_count =
              std::numeric_limits<std::uint64_t>::max(),
      },
  };
  const std::array<std::uint64_t, 4u> compact_counts{2u, 1u, 1u, 1u};
  const std::array<std::uint64_t, 4u> occurrence_counts{1u, 2u, 1u, 1u};
  const std::array<std::uint64_t, 4u> window_counts{0u, 0u, 1u, 2u};
  for (std::size_t index = 0u; index < overflow_routes.size(); ++index) {
    if (AccumulatePreparedKernelRouteProjectionForContract(
            projection, overflow_routes[index], compact_counts[index],
            occurrence_counts[index], window_counts[index]) ||
        !unchanged(projection)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool PreparedPublicationCommandShapeIsCanonical() {
  using namespace rund::node::accel::detail;
  std::uint64_t commands = 0u;
  if (!PreparedKernelPublicationCommandContribution(false, 0u, 0u, commands) ||
      commands != 2u ||
      !PreparedKernelPublicationCommandContribution(true, 516096u, 8192u,
                                                    commands) ||
      commands != 63u ||
      !PreparedKernelPublicationCommandContribution(true, 10u, 3u, commands) ||
      commands != 4u) {
    return false;
  }
  return !PreparedKernelPublicationCommandContribution(false, 1u, 0u,
                                                       commands) &&
         !PreparedKernelPublicationCommandContribution(true, 0u, 1u,
                                                       commands) &&
         !PreparedKernelPublicationCommandContribution(true, 1u, 0u,
                                                       commands) &&
         !PreparedKernelPublicationCommandContribution(true, 1u, 2u, commands);
}

[[nodiscard]] bool VulkanPhysicalCommandShapeIsNonOverlapping() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  PreparedKernelPipelineReservation reservation{
      .window_count = 3u,
      .backend_dispatch_count = 7u,
      .backend_reset_dispatch_count = 3u,
      .backend_window_dispatch_count = 7u,
      .backend_indirect_dispatch_count = 2u,
      .backend_window_state_count = 1u,
      .backend_window_descriptor_state_count = 1u,
      .backend_status_source_count = 1u,
      .backend_status_entry_count = 1u,
      .backend_telemetry_count = 1u,
      .backend_status_command_count = 4u,
      .backend_telemetry_command_count = 2u,
      .backend_publication_count = 2u,
      .backend_terminal_publication_count = 1u,
      .backend_publication_command_count = 5u,
      .backend_parameter_bytes = 13u,
  };
  constexpr std::uint64_t expected_window_control = 6u;
  constexpr std::uint64_t expected_commands =
      7u + 3u + 4u + 2u + expected_window_control + 2u + 5u + 3u;
  constexpr std::uint64_t expected_parameters =
      13u + 2u * sizeof(VulkanPipelineTelemetryParams) +
      expected_window_control * sizeof(VulkanWindowParams) +
      2u * VulkanGateParameterBytes + 5u * sizeof(VulkanPipelinePublishParams) +
      2u * VulkanPipelineControlParameterBytes;
  const rund::AccelCheck planned =
      PlanVulkanPipelineStructure(rund::AccelContext{}, reservation);
  return planned.ok &&
         reservation.backend_window_control_command_count ==
             expected_window_control &&
         reservation.backend_command_count == expected_commands &&
         reservation.backend_parameter_bytes == expected_parameters;
#else
  return true;
#endif
}

[[nodiscard]] bool VulkanPhysicalCapacityFailureIsTransactional() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  constexpr std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  const auto rejected_without_commit = [](auto reservation) {
    const PreparedKernelPipelineReservation before = reservation;
    const rund::AccelCheck planned =
        PlanVulkanPipelineStructure(rund::AccelContext{}, reservation);
    return !planned.ok && planned.reason != nullptr &&
           std::string_view{planned.reason} == "compute_pipeline_capacity" &&
           reservation == before;
  };

  PreparedKernelPipelineReservation window_product{
      .host_bytes = 17u,
      .native_bytes = 19u,
      .window_count = maximum / 2u + 1u,
      .backend_window_dispatch_count = 1u,
      .backend_window_state_count = 1u,
      .backend_window_descriptor_state_count = 1u,
  };
  PreparedKernelPipelineReservation command_sum{
      .host_bytes = 23u,
      .native_bytes = 29u,
      .backend_dispatch_count = maximum,
  };
  PreparedKernelPipelineReservation parameter_product{
      .host_bytes = 37u,
      .native_bytes = 41u,
      .backend_telemetry_command_count =
          maximum / sizeof(VulkanPipelineTelemetryParams) + 1u,
      .backend_parameter_bytes = 31u,
  };
  return rejected_without_commit(window_product) &&
         rejected_without_commit(command_sum) &&
         rejected_without_commit(parameter_product);
#else
  return true;
#endif
}

[[nodiscard]] bool PreparedBackendControlManifestIsDimensionallyClosed() {
  using namespace rund::node::accel::detail;
  PreparedBackendManifest manifest{};
  if (!ValidPreparedBackendControlManifest(manifest)) {
    return false;
  }
  manifest.status_source_count = 1u;
  if (ValidPreparedBackendControlManifest(manifest)) {
    return false;
  }
  manifest.status_entry_count = 4u;
  manifest.status_command_count = 2u;
  manifest.status_parameter_bytes = 16u;
  if (!ValidPreparedBackendControlManifest(manifest)) {
    return false;
  }
  manifest.status_entry_count = 0u;
  return !ValidPreparedBackendControlManifest(manifest);
}

[[nodiscard]] bool BackendSourceRecipeIsCheckedAndCanonical() {
  constexpr std::string_view expected =
      "kernel void "
      "rund_backend_source_recipe_contract_with_long_name(value="
      "0,9,10,99,100,18446744073709551615);";
  std::uint64_t upper = 0u;
  const BackendSourceRecipeEmitter emit{};
  if (!source_recipe::bytes(emit, upper) || upper != expected.size() ||
      source_recipe::materialize(emit) != expected ||
      source_recipe::materialize(emit, upper) != expected ||
      source_recipe::materialize(emit, upper, upper + 37u) != expected ||
      !source_recipe::materialize(emit, upper - 1u).empty() ||
      !source_recipe::materialize(emit, upper, upper - 1u).empty() ||
      !source_recipe::materialize(emit, 0u).empty()) {
    return false;
  }

  // A failed count never overwrites the caller's previously frozen upper.
  const auto reject = [](source_recipe::CountSink &) noexcept { return false; };
  const auto empty = [](source_recipe::CountSink &) noexcept { return true; };
  upper = 41u;
  if (source_recipe::bytes(reject, upper) || upper != 41u ||
      source_recipe::bytes(empty, upper) || upper != 41u) {
    return false;
  }

  // Overflow is checked at the counter boundary. The rejected append leaves
  // the preceding exact count intact instead of wrapping it to zero.
  source_recipe::CountSink full{std::numeric_limits<std::uint64_t>::max()};
  if (full.append("x") ||
      full.bytes() != std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  source_recipe::CountSink fragments{std::numeric_limits<std::uint64_t>::max() -
                                     1u};
  constexpr std::array<std::string_view, 3u> overflow_fragments{"", "x", "yz"};
  if (source_recipe::append_fixed(fragments, overflow_fragments) ||
      fragments.bytes() != std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }

  // Count and materialization are one recipe authority. If an emitter ever
  // diverges by sink, materialization rejects the result rather than retaining
  // an under-reserved or semantically different source.
  if (!source_recipe::materialize(DivergentBackendSourceRecipeEmitter{})
           .empty()) {
    return false;
  }

  std::uint64_t storage_upper = 0u;
  std::uint64_t overflow_storage = 0u;
  const std::string materialized =
      source_recipe::materialize(emit, expected.size());
  if (!source_recipe::string_external_storage_upper_bytes(expected.size(),
                                                          storage_upper) ||
      source_recipe::string_external_storage_upper_bytes(
          std::numeric_limits<std::uint64_t>::max(), overflow_storage) ||
      materialized != expected ||
      !source_recipe::string_external_storage_within(materialized,
                                                     storage_upper)) {
    return false;
  }
  std::array<char, 39u> fixed_storage{};
  source_recipe::FixedBufferSink<39u> fixed{fixed_storage};
  if (!fixed.append("0x") ||
      !source_recipe::append_hex64_digits(
          fixed, std::numeric_limits<std::uint64_t>::max()) ||
      !fixed.append(":") ||
      !source_recipe::append_decimal(
          fixed, std::numeric_limits<std::uint64_t>::max()) ||
      fixed.text() != "0xffffffffffffffff:18446744073709551615") {
    return false;
  }
  std::string over_capacity{expected};
  over_capacity.reserve(static_cast<std::size_t>(storage_upper + 64u));
  return !source_recipe::string_external_storage_within(over_capacity,
                                                        storage_upper);
}

[[nodiscard]] bool VulkanMapAndResetSourceRecipesAreExact() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  VulkanMapTemplateResources prepared{};
  prepared.plan.scalar = rund::kernel::ComputeScalar::Lane32;
  prepared.plan.domain = rund::kernel::ComputeDomain::U32;
  prepared.plan.input_buffer_count = 2u;
  prepared.plan.output_buffer_count = 1u;
  prepared.plan.op_hash_hi = 0x1234u;
  prepared.plan.op_hash_lo = 0x5678u;
  prepared.checks = {
      VulkanMapCheck{
          .binding = 0u, .limit = 17u, .offset = 4096u, .stride = 8u},
      VulkanMapCheck{
          .binding = 1u, .limit = 1009u, .offset = 64u, .stride = 16u},
  };
  const rund::kernel::LoweringArtifact check = VulkanMapCheckArtifact(prepared);
  std::uint64_t check_upper = 0u;
  std::uint64_t check_storage = 0u;
  if (!check.ok ||
      !VulkanMapCheckSourceUpperBytes(2u, 6u, 3u, 6u, check_upper) ||
      check.source_text.size() != check_upper ||
      check.source_text_upper_bytes != check_upper ||
      !source_recipe::string_external_storage_upper_bytes(check_upper,
                                                          check_storage) ||
      !source_recipe::string_external_storage_within(check.source_text,
                                                     check_storage)) {
    return false;
  }

  rund::kernel::LoweringArtifact base{};
  base.key.op_hash_hi = prepared.plan.op_hash_hi;
  base.key.op_hash_lo = prepared.plan.op_hash_lo;
  base.source_text =
      std::string{vulkan_controlled_map_source_detail::CanonicalVariant} +
      "\n" + std::string{vulkan_controlled_map_source_detail::Entry} +
      std::string{vulkan_controlled_map_source_detail::Guard};
  base.source_text_upper_bytes = base.source_text.size();
  base.ok = true;
  base.reason = "ok";
  std::uint64_t controlled_upper = 0u;
  if (!VulkanControlledMapSourceUpperBytes(
          prepared.plan, base.source_text_upper_bytes, controlled_upper)) {
    return false;
  }
  const rund::kernel::LoweringArtifact controlled =
      VulkanControlledMapArtifact(std::move(base), prepared.plan);
  const std::string expected_binding =
      "layout(set = 0, binding = 4, std430) readonly buffer RundControlArgs";
  if (!controlled.ok || controlled.source_text.size() != controlled_upper ||
      controlled.source_text_upper_bytes != controlled_upper ||
      controlled.source_text.find(expected_binding) == std::string::npos ||
      controlled.source_text.find(
          vulkan_controlled_map_source_detail::ControlledGuard) ==
          std::string::npos ||
      controlled.source_text.find(
          vulkan_controlled_map_source_detail::ControlledVariant) ==
          std::string::npos ||
      controlled.source_text.find(
          vulkan_controlled_map_source_detail::CanonicalVariant) !=
          std::string::npos) {
    return false;
  }

  const rund::kernel::LoweringArtifact control =
      VulkanMapControlArtifact(prepared.plan);
  const rund::kernel::LoweringArtifact reset = VulkanResetArtifact();
  return control.ok &&
         control.source_text.size() == VulkanMapControlSourceText().size() &&
         control.source_text_upper_bytes ==
             VulkanMapControlSourceText().size() &&
         reset.ok && reset.source_text == VulkanResetSourceText() &&
         reset.source_text_upper_bytes == VulkanResetSourceText().size();
#else
  return true;
#endif
}

[[nodiscard]] bool MapSourceSpecializationIsSingleOwnerAndExact() {
  using namespace rund::node::accel::detail;
  static_assert(MapSpecializationEditCapacity ==
                2u * rund::kernel::kMaxComputeBindingCount);

  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Metal;
  artifact.kind = rund::kernel::LoweringArtifactKind::MetalSource;
  artifact.source_text = "constant uint RundStride_read_78 = 4u;\n"
                         "constant uint RundBase_read_78 = 0u;\n"
                         "constant uint RundStride_write_79 = 4u;\n"
                         "constant uint RundBase_write_79 = 0u;\n";
  artifact.source_text_upper_bytes = artifact.source_text.size();
  artifact.metadata.input_element_bytes = {4u};
  artifact.metadata.output_element_bytes = {4u};
  artifact.metadata.binding_accesses = {
      rund::kernel::ComputeBindingAccess::Read,
      rund::kernel::ComputeBindingAccess::Write};
  artifact.metadata.binding_names = {"x", "y"};
  artifact.metadata.ok = true;
  artifact.metadata.reason = "ok";
  artifact.ok = true;
  artifact.reason = "ok";

  rund::kernel::ComputePlan plan{
      .api = rund::kernel::ComputeApi::Metal,
      .input_buffer_count = 1u,
      .output_buffer_count = 1u,
  };
  KernelExecutionStep step{};
  step.artifact = artifact;
  std::uint64_t retained = 0u;
  std::uint64_t transient = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t LiteralGrowthPerBinding = 38u;
  if (!backend_template_plan::map_source_upper(step, plan, retained,
                                               transient) ||
      retained != artifact.source_text.size() + 2u * LiteralGrowthPerBinding ||
      transient != 0u) {
    return false;
  }
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const PreparedBackendManifest map_manifest =
      BuildMetalBackendManifest(step, plan, nullptr, 1u);
  if (!map_manifest.ok || map_manifest.capture_binding_slot_upper != 4u ||
      map_manifest.cold_source_transient_bytes != 0u ||
      map_manifest.cold_cache_source_storage_bytes <=
          map_manifest.cold_cache_source_bytes) {
    return false;
  }
  KernelExecutionStep controlled_step = step;
  controlled_step.control.count_source = rund::kernel::GraphControlSource::U32;
  controlled_step.control.count_binding = 0u;
  controlled_step.control.capacity = 1u;
  const PreparedBackendManifest controlled_manifest =
      BuildMetalBackendManifest(controlled_step, plan, nullptr, 1u);
  rund::kernel::ComputePlan binding_overflow = plan;
  binding_overflow.input_buffer_count = kMetalPipelineGuardBinding;
  const PreparedBackendManifest rejected_manifest =
      BuildMetalBackendManifest(step, binding_overflow, nullptr, 1u);
  if (!controlled_manifest.ok ||
      controlled_manifest.capture_binding_slot_upper != 6u ||
      rejected_manifest.ok) {
    return false;
  }
#endif

  const std::array input{rund::kernel::ResidentBufferRef{
      .bytes = 64u,
      .offset_bytes = 3u,
      .element_bytes = 4u,
      .stride_bytes = 8u,
      .count = 8u,
      .usage = rund::kernel::kResidentUsageRead}};
  const std::array output{rund::kernel::ResidentBufferRef{
      .bytes = 64u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 8u,
      .usage = rund::kernel::kResidentUsageWrite}};
  const rund::kernel::BindingSet bindings{
      .resident_inputs =
          rund::kernel::ResidentBindingRange{.refs = input.data(),
                                             .storage_count = input.size(),
                                             .count = input.size()},
      .resident_outputs =
          rund::kernel::ResidentBindingRange{.refs = output.data(),
                                             .storage_count = output.size(),
                                             .count = output.size()},
  };
  const std::string expected = "constant uint RundStride_read_78 = 8u;\n"
                               "constant uint RundBase_read_78 = 3u;\n"
                               "constant uint RundStride_write_79 = 4u;\n"
                               "constant uint RundBase_write_79 = 0u;\n";
  const rund::kernel::LoweringArtifact specialized =
      SpecializeMap(artifact, plan, bindings, 16u);
  std::uint64_t specialized_storage_upper = 0u;
  if (!specialized.ok || specialized.source_text != expected ||
      specialized.source_text.size() != expected.size() ||
      specialized.source_text_upper_bytes != retained ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          retained, specialized_storage_upper) ||
      !backend_source_recipe::string_external_storage_within(
          specialized.source_text, specialized_storage_upper)) {
    return false;
  }

  rund::kernel::ComputePlan oversized = plan;
  oversized.input_buffer_count = rund::kernel::kMaxComputeBindingCount + 1u;
  oversized.output_buffer_count = 0u;
  const rund::kernel::LoweringArtifact rejected =
      SpecializeMap(artifact, oversized, bindings, 16u);
  return !rejected.ok && rejected.reason != nullptr &&
         std::string_view{rejected.reason} == "compute_pipeline_capacity";
}

[[nodiscard]] bool MetalSourceRecipesAreExactAndSemantic() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  const auto exact = [](const std::string &source, const std::uint64_t bytes) {
    return !source.empty() && source.size() == bytes;
  };
  std::uint64_t bytes = 0u;
  if (!MetalScanSourceUpperBytes(bytes) || !exact(MetalScanSource(), bytes) ||
      !MetalSegmentedScanSourceUpperBytes(bytes) ||
      !exact(MetalSegmentedScanSource(), bytes) ||
      !MetalSortSourceUpperBytes(256u, bytes) ||
      !exact(MetalSortSource(256u), bytes) ||
      !MetalCompactSourceUpperBytes(bytes) ||
      !exact(MetalCompactSource(), bytes) ||
      !exact(MetalGatherSource(), MetalGatherSourceUpperBytes()) ||
      !exact(MetalHistogramSource(), MetalHistogramSourceUpperBytes()) ||
      !MetalPartitionSourceUpperBytes(bytes) ||
      !exact(MetalPartitionSource(), bytes) ||
      !exact(MetalScatterSource(), MetalScatterSourceUpperBytes()) ||
      !MetalNumericSourceUpperBytes(bytes) ||
      !exact(MetalNumericSource(), bytes)) {
    return false;
  }

  constexpr std::array<rund::kernel::ReduceOp, 4u> reduce_ops{
      rund::kernel::ReduceOp::Sum,
      rund::kernel::ReduceOp::CountNonzero,
      rund::kernel::ReduceOp::Min,
      rund::kernel::ReduceOp::Max,
  };
  constexpr std::array<rund::kernel::ComputeDomain, 2u> domains{
      rund::kernel::ComputeDomain::U32,
      rund::kernel::ComputeDomain::I32,
  };
  for (const rund::kernel::ReduceOp op : reduce_ops) {
    for (const rund::kernel::ComputeDomain domain : domains) {
      if (!MetalSegmentedReduceSourceUpperBytes(op, domain, bytes) ||
          !exact(MetalSegmentedReduceSource(op, domain), bytes) ||
          !MetalReduceSourceUpperBytes(op, 256u, domain, bytes) ||
          !exact(MetalReduceSource(op, 256u, domain), bytes)) {
        return false;
      }
    }
  }
  constexpr std::array<rund::kernel::StencilOp, 3u> stencil_ops{
      rund::kernel::StencilOp::Sum,
      rund::kernel::StencilOp::Min,
      rund::kernel::StencilOp::Max,
  };
  for (const rund::kernel::StencilOp op : stencil_ops) {
    if (!MetalStencilSourceUpperBytes(op, bytes) ||
        !exact(MetalStencilSource(op), bytes)) {
      return false;
    }
  }

  rund::kernel::ScatterReducePlan scatter{};
  scatter.op = rund::kernel::ScatterReduceOp::Sum;
  scatter.domain = rund::kernel::ComputeDomain::U32;
  scatter.element_bytes = 4u;
  if (!MetalScatterReduceSourceUpperBytes(scatter, bytes) ||
      !exact(MetalScatterReduceSource(scatter), bytes)) {
    return false;
  }
  const std::string modular_sum_key = MetalScatterReduceKey(scatter);
  const std::string modular_sum_source = MetalScatterReduceSource(scatter);
  scatter.domain = rund::kernel::ComputeDomain::I32;
  if (MetalScatterReduceKey(scatter) != modular_sum_key ||
      MetalScatterReduceSource(scatter) != modular_sum_source) {
    return false;
  }
  scatter.domain = rund::kernel::ComputeDomain::Fixed;
  scatter.fixed_format.overflow = rund::kernel::ComputeOverflow::Wrap;
  if (MetalScatterReduceKey(scatter) != modular_sum_key ||
      MetalScatterReduceSource(scatter) != modular_sum_source) {
    return false;
  }
  scatter.fixed_format.overflow = rund::kernel::ComputeOverflow::Saturate;
  if (!MetalScatterReduceSourceUpperBytes(scatter, bytes) ||
      !exact(MetalScatterReduceSource(scatter), bytes) ||
      MetalScatterReduceKey(scatter) == modular_sum_key) {
    return false;
  }

  scatter.op = rund::kernel::ScatterReduceOp::Min;
  scatter.fixed_format.overflow = rund::kernel::ComputeOverflow::Wrap;
  const std::string signed_min_key = MetalScatterReduceKey(scatter);
  scatter.domain = rund::kernel::ComputeDomain::I32;
  if (MetalScatterReduceKey(scatter) != signed_min_key) {
    return false;
  }
  scatter.domain = rund::kernel::ComputeDomain::U32;
  if (MetalScatterReduceKey(scatter) == signed_min_key) {
    return false;
  }

  return StencilPipelineKey(rund::kernel::StencilOp::Sum,
                            rund::kernel::StencilElement::U32,
                            rund::kernel::ComputeDomain::U32) ==
             StencilPipelineKey(rund::kernel::StencilOp::Sum,
                                rund::kernel::StencilElement::U32,
                                rund::kernel::ComputeDomain::I32) &&
         StencilPipelineKey(rund::kernel::StencilOp::Min,
                            rund::kernel::StencilElement::U32,
                            rund::kernel::ComputeDomain::U32) !=
             StencilPipelineKey(rund::kernel::StencilOp::Min,
                                rund::kernel::StencilElement::U32,
                                rund::kernel::ComputeDomain::I32);
#else
  return true;
#endif
}

[[nodiscard]] bool MetalMapCheckSourceHasOneGuardAuthority() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  MetalMapTemplateResources prepared{};
  prepared.plan.scalar = rund::kernel::ComputeScalar::Lane32;
  prepared.plan.domain = rund::kernel::ComputeDomain::U32;
  prepared.plan.op_hash_hi = 0x6d61702e63686563ull;
  prepared.plan.op_hash_lo = 0x6b2e736f75726365ull;
  prepared.checks.push_back(MetalMapCheck{.binding = 0u, .limit = 123u});
  const rund::kernel::ResidentBufferRef input{
      .bytes = 64u,
      .element_bytes = 4u,
      .stride_bytes = 8u,
      .count = 8u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  rund::kernel::BindingSet bindings{};
  bindings.resident_inputs = rund::kernel::ResidentBindingRange{
      .refs = &input,
      .storage_count = 1u,
      .count = 1u,
  };

  const KernelPreparationScope scope{KernelPreparationMode::PipelinePrivate};
  rund::kernel::LoweringArtifact artifact =
      MetalMapCheckArtifact(prepared, bindings);
  std::uint64_t raw_upper = 0u;
  std::uint64_t guarded_upper = 0u;
  const bool upper_ok =
      MetalMapCheckSourceUpperBytes(1u, DecimalDigitCount(input.stride_bytes),
                                    DecimalDigitCount(123u), raw_upper);
  const bool guarded_ok = upper_ok && PipelinePrivateMetalSourceUpperBytes(
                                          raw_upper, 1u, true, guarded_upper);
  if (!artifact.ok || !guarded_ok || artifact.source_text.size() != raw_upper ||
      artifact.source_text_upper_bytes != raw_upper ||
      guarded_upper <= raw_upper) {
    return false;
  }
  const std::size_t check_capacity = artifact.source_text.capacity();
  std::string guarded = PipelinePrivateMetalSource(
      std::move(artifact.source_text), guarded_upper);
  std::uint64_t guarded_storage_upper = 0u;
  if (guarded.size() != guarded_upper || guarded.capacity() != check_capacity ||
      guarded.find("buffer(30)") == std::string::npos ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          guarded_upper, guarded_storage_upper) ||
      !backend_source_recipe::string_external_storage_within(
          guarded, guarded_storage_upper)) {
    return false;
  }

  std::string over_capacity = guarded;
  over_capacity.reserve(static_cast<std::size_t>(guarded_storage_upper + 64u));
  if (!PipelinePrivateMetalSource(std::move(over_capacity), guarded_upper)
           .empty()) {
    return false;
  }

  std::uint64_t control_guarded_upper = 0u;
  if (!PipelinePrivateMetalSourceUpperBytes(MetalMapControlSourceText().size(),
                                            1u, true, control_guarded_upper)) {
    return false;
  }
  std::string control_source = MetalMapControlSource();
  const std::size_t control_capacity = control_source.capacity();
  control_source = PipelinePrivateMetalSource(std::move(control_source),
                                              control_guarded_upper);
  if (control_source.size() != control_guarded_upper ||
      control_source.capacity() != control_capacity) {
    return false;
  }

  rund::kernel::LoweringArtifact controlled_input{};
  controlled_input.key.api = rund::kernel::ComputeApi::Metal;
  controlled_input.key.op_hash_hi = 0x1111111111111111ull;
  controlled_input.key.op_hash_lo = 0x2222222222222222ull;
  controlled_input.kind = rund::kernel::LoweringArtifactKind::MetalSource;
  constexpr std::string_view CanonicalControlledSource =
      "// artifact_variant=canonical\n"
      "kernel void "
      "rund_compute_map_1111111111111111_2222222222222222(\n"
      "    uint gid [[thread_position_in_grid]]) {\n"
      "}\n";
  controlled_input.source_text_upper_bytes = CanonicalControlledSource.size();
  controlled_input.ok = true;
  controlled_input.reason = "ok";
  const rund::kernel::ComputePlan controlled_plan{
      .api = rund::kernel::ComputeApi::Metal,
      .input_buffer_count = 1u,
      .output_buffer_count = 1u,
  };
  std::uint64_t controlled_upper = 0u;
  std::uint64_t controlled_guarded_upper = 0u;
  if (!MetalControlledMapSourceUpperBytes(controlled_plan,
                                          CanonicalControlledSource.size(),
                                          controlled_upper) ||
      !PipelinePrivateMetalSourceUpperBytes(controlled_upper, 1u, true,
                                            controlled_guarded_upper)) {
    return false;
  }
  const auto canonical_recipe =
      [source = CanonicalControlledSource]<typename Sink>(Sink &sink) noexcept(
          noexcept(sink.append(std::string_view{}))) {
        return sink.append(source);
      };
  controlled_input.source_text = backend_source_recipe::materialize(
      canonical_recipe, CanonicalControlledSource.size(),
      controlled_guarded_upper);
  const std::size_t controlled_capacity =
      controlled_input.source_text.capacity();
  rund::kernel::LoweringArtifact controlled =
      MetalControlledMapArtifact(std::move(controlled_input), controlled_plan);
  if (!controlled.ok || controlled.source_text.size() != controlled_upper ||
      controlled.source_text.capacity() != controlled_capacity ||
      controlled.source_text.find("artifact_variant=controlled") ==
          std::string::npos ||
      controlled.source_text.find("_controlled(") == std::string::npos) {
    return false;
  }
  controlled.source_text = PipelinePrivateMetalSource(
      std::move(controlled.source_text), controlled_guarded_upper);
  return controlled.source_text.size() == controlled_guarded_upper &&
         controlled.source_text.capacity() == controlled_capacity;
#else
  return true;
#endif
}

[[nodiscard]] bool MetalColdManifestIsExact() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  rund::kernel::ComputePlan plan{};
  plan.domain = rund::kernel::ComputeDomain::U32;
  const KernelPreparationScope scope{KernelPreparationMode::PipelinePrivate};
  const auto verify =
      [&plan](const KernelExecutionStep &step,
              const std::uint64_t source_builds, const std::uint64_t libraries,
              const std::uint64_t stages, const std::uint64_t binding_slots,
              const std::uint64_t source_bytes) {
        const PreparedBackendManifest manifest =
            BuildMetalBackendManifest(step, plan, nullptr, 1u);
        std::uint64_t dependency_bytes = 0u;
        std::uint64_t dependency_storage_bytes = 0u;
        std::uint64_t dependency_stages = 0u;
        for (std::size_t index = 0u;
             index < manifest.cache_dependency_entry_count; ++index) {
          const PreparedBackendCacheDependency &dependency =
              manifest.source_dependencies[index];
          dependency_bytes += dependency.source_upper_bytes;
          dependency_storage_bytes += dependency.source_storage_upper_bytes;
          dependency_stages += dependency.pipeline_stage_count;
        }
        return manifest.ok && manifest.source_dependencies_complete &&
               manifest.source_build_count == source_builds &&
               manifest.source_library_dependency_count == libraries &&
               manifest.pipeline_stage_count == stages &&
               manifest.capture_binding_slot_upper == binding_slots &&
               manifest.cache_dependency_entry_count == libraries &&
               manifest.cold_cache_source_bytes == source_bytes &&
               dependency_bytes == source_bytes &&
               manifest.cold_cache_source_storage_bytes ==
                   dependency_storage_bytes &&
               dependency_storage_bytes > dependency_bytes &&
               manifest.cold_source_transient_bytes != 0u &&
               dependency_stages == stages &&
               manifest.cold_cache_native_object_count == libraries + stages;
      };
  const auto guarded_size = [](std::string source,
                               const std::uint64_t entry_count) {
    std::uint64_t upper = 0u;
    if (!PipelinePrivateMetalSourceUpperBytes(source.size(), entry_count, true,
                                              upper)) {
      return std::uint64_t{0u};
    }
    source = PipelinePrivateMetalSource(std::move(source), upper);
    return source.size() == upper ? upper : std::uint64_t{0u};
  };

  KernelExecutionStep step{};
  step.operation.set<operation::Scan>();
  if (!verify(step, 1u, 1u, 3u, 11u, guarded_size(MetalScanSource(), 7u))) {
    return false;
  }
  step.operation.set<operation::SegmentedScan>();
  if (!verify(step, 1u, 1u, 3u, 7u,
              guarded_size(MetalSegmentedScanSource(), 3u))) {
    return false;
  }
  auto &segmented_reduce = step.operation.set<operation::SegmentedReduce>();
  segmented_reduce.plan.op = rund::kernel::ReduceOp::Sum;
  if (!verify(step, 1u, 1u, 4u, 6u,
              guarded_size(MetalSegmentedReduceSource(segmented_reduce.plan.op,
                                                      plan.domain),
                           5u))) {
    return false;
  }
  step.operation.set<operation::Sort>();
  if (!verify(step, 1u, 1u, 5u, 8u,
              guarded_size(MetalSortSource(kMetalSortBlockSize), 7u))) {
    return false;
  }
  step.operation.set<operation::Compact>();
  if (!verify(step, 2u, 2u, 5u, 11u,
              guarded_size(MetalCompactSource(), 4u) +
                  guarded_size(MetalScanSource(), 7u))) {
    return false;
  }
  step.operation.set<operation::Gather>();
  if (!verify(step, 1u, 1u, 2u, 5u, guarded_size(MetalGatherSource(), 3u))) {
    return false;
  }
  step.operation.set<operation::Histogram>();
  if (!verify(step, 1u, 1u, 2u, 4u, guarded_size(MetalHistogramSource(), 2u))) {
    return false;
  }
  step.operation.set<operation::Partition>();
  if (!verify(step, 3u, 2u, 5u, 11u,
              guarded_size(MetalPartitionSource(), 6u) +
                  guarded_size(MetalScanSource(), 7u))) {
    return false;
  }
  auto &reduce = step.operation.set<operation::Reduce>();
  reduce.plan.op = rund::kernel::ReduceOp::Sum;
  reduce.plan.block_size = 256u;
  if (!verify(
          step, 1u, 1u, 1u, 6u,
          guarded_size(MetalReduceSource(reduce.plan.op, reduce.plan.block_size,
                                         plan.domain),
                       2u))) {
    return false;
  }
  step.operation.set<operation::Scatter>();
  if (!verify(step, 1u, 1u, 1u, 5u, guarded_size(MetalScatterSource(), 2u))) {
    return false;
  }
  auto &scatter_reduce = step.operation.set<operation::ScatterReduce>();
  scatter_reduce.plan.op = rund::kernel::ScatterReduceOp::Sum;
  scatter_reduce.plan.domain = rund::kernel::ComputeDomain::U32;
  scatter_reduce.plan.element_bytes = 4u;
  if (!verify(
          step, 1u, 1u, 3u, 8u,
          guarded_size(MetalScatterReduceSource(scatter_reduce.plan), 3u))) {
    return false;
  }
  auto &stencil = step.operation.set<operation::Stencil>();
  stencil.plan.op = rund::kernel::StencilOp::Sum;
  if (!verify(step, 1u, 1u, 1u, 3u,
              guarded_size(MetalStencilSource(stencil.plan.op), 4u))) {
    return false;
  }
  const std::uint64_t numeric_source_bytes =
      guarded_size(MetalNumericSource(), 10u);
  step.operation.set<operation::Transform>();
  if (!verify(step, 1u, 1u, 1u, 6u, numeric_source_bytes)) {
    return false;
  }
  step.operation.set<operation::Matrix>();
  if (!verify(step, 1u, 1u, 1u, 4u, numeric_source_bytes)) {
    return false;
  }
  step.operation.set<operation::Factor>();
  if (!verify(step, 1u, 1u, 1u, 5u, numeric_source_bytes)) {
    return false;
  }
  step.operation.set<operation::Solve>();
  if (!verify(step, 1u, 1u, 1u, 6u, numeric_source_bytes)) {
    return false;
  }
  step.operation.set<operation::Spectrum>();
  return verify(step, 1u, 1u, 1u, 5u, numeric_source_bytes);
#else
  return true;
#endif
}

[[nodiscard]] bool BackendTemplateRouteDemandIsExactAndFrozen() {
  using namespace rund::node::accel::detail;
  BackendTemplateRouteDemand scalar{
      .owner_count = 7u, .route_copies = 1u, .capacity = 7u};
  if (!BackendTemplateRouteDemandForContract(2u, 2u, scalar) ||
      scalar.owner_count != 2u || scalar.route_copies != 2u ||
      scalar.capacity != 4u || !scalar.valid()) {
    return false;
  }
  const BackendTemplateRouteDemand frozen = scalar;
  if (BackendTemplateRouteDemandForContract(
          std::numeric_limits<std::uint32_t>::max(), 2u, scalar) ||
      scalar.owner_count != frozen.owner_count ||
      scalar.route_copies != frozen.route_copies ||
      scalar.capacity != frozen.capacity) {
    return false;
  }
  for (const std::uint64_t illegal_copies : {0u, 3u}) {
    if (BackendTemplateRouteDemandForContract(2u, illegal_copies, scalar) ||
        scalar.owner_count != frozen.owner_count ||
        scalar.route_copies != frozen.route_copies ||
        scalar.capacity != frozen.capacity) {
      return false;
    }
  }

  const BackendOps grouped_ops{
      .same_pipeline_template = MatchBackendTemplateGroup,
  };
  const auto route = [&](const std::uint64_t group) {
    auto state = std::make_shared<prepared::RunState>();
    state->mode = KernelPreparationMode::PipelinePrivate;
    state->bound.run.ops = &grouped_ops;
    state->bound.run.original_dispatch_count = group;
    return state;
  };
  const std::shared_ptr<prepared::RunState> first = route(11u);
  const std::shared_ptr<prepared::RunState> second = route(11u);
  const std::shared_ptr<prepared::RunState> other = route(29u);
  // A route already materialized by an earlier attempt remains one semantic
  // owner in the group; it is neither omitted nor counted twice.
  first->backend = std::make_shared<int>(7);
  std::array<PreparedKernelRun, 4u> owners{
      PreparedKernelRun{.owner = first, .ok = true, .reason = "ok"},
      PreparedKernelRun{.owner = second, .ok = true, .reason = "ok"},
      PreparedKernelRun{.owner = first, .ok = true, .reason = "ok"},
      PreparedKernelRun{.owner = other, .ok = true, .reason = "ok"},
  };
  const std::array<const PreparedKernelRun *, 4u> runs{&owners[0], &owners[1],
                                                       &owners[2], &owners[3]};
  std::array<BackendTemplateRouteDemand, 4u> demands{};
  std::uint64_t unique_routes = 0u;
  std::uint64_t templates = 0u;
  if (!PlanBackendTemplateRouteDemandsForContract(runs, 2u, demands,
                                                  unique_routes, templates) ||
      unique_routes != 3u || templates != 2u) {
    return false;
  }
  for (const std::size_t index : {0u, 1u, 2u}) {
    if (demands[index].owner_count != 2u || demands[index].route_copies != 2u ||
        demands[index].capacity != 4u || !demands[index].valid()) {
      return false;
    }
  }
  if (demands[3].owner_count != 1u || demands[3].route_copies != 2u ||
      demands[3].capacity != 2u || !demands[3].valid() ||
      !BackendPreparationCursorLifecycleForContract(first->bound.run,
                                                    demands[0])) {
    return false;
  }

  // The complete prepass is transactional for an illegal generation stride:
  // no demand or cardinality escapes the rejected plan.
  const auto legal_demands = demands;
  unique_routes = 43u;
  templates = 47u;
  if (PlanBackendTemplateRouteDemandsForContract(runs, 3u, demands,
                                                 unique_routes, templates) ||
      demands[0].capacity != legal_demands[0].capacity ||
      demands[1].capacity != legal_demands[1].capacity ||
      demands[2].capacity != legal_demands[2].capacity ||
      demands[3].capacity != legal_demands[3].capacity ||
      unique_routes != 43u || templates != 47u) {
    return false;
  }

  // Backend equality is an equivalence authority. A predicate that groups a
  // pair in only one direction is rejected before any cursor/native reserve.
  const BackendOps asymmetric_ops{
      .same_pipeline_template = AsymmetricBackendTemplateGroup,
  };
  const auto asymmetric_route = [&](const std::uint64_t group) {
    auto state = std::make_shared<prepared::RunState>();
    state->mode = KernelPreparationMode::PipelinePrivate;
    state->bound.run.ops = &asymmetric_ops;
    state->bound.run.original_dispatch_count = group;
    return state;
  };
  const auto asymmetric_first = asymmetric_route(1u);
  const auto asymmetric_second = asymmetric_route(2u);
  std::array<PreparedKernelRun, 2u> asymmetric_owners{
      PreparedKernelRun{.owner = asymmetric_first, .ok = true, .reason = "ok"},
      PreparedKernelRun{.owner = asymmetric_second, .ok = true, .reason = "ok"},
  };
  const std::array<const PreparedKernelRun *, 2u> asymmetric_runs{
      &asymmetric_owners[0], &asymmetric_owners[1]};
  std::array<BackendTemplateRouteDemand, 2u> rejected_demands{
      BackendTemplateRouteDemand{
          .owner_count = 3u, .route_copies = 1u, .capacity = 3u},
      BackendTemplateRouteDemand{
          .owner_count = 5u, .route_copies = 1u, .capacity = 5u},
  };
  const auto rejected_before = rejected_demands;
  unique_routes = 31u;
  templates = 37u;
  return !PlanBackendTemplateRouteDemandsForContract(
             asymmetric_runs, 1u, rejected_demands, unique_routes, templates) &&
         rejected_demands[0].capacity == rejected_before[0].capacity &&
         rejected_demands[1].capacity == rejected_before[1].capacity &&
         unique_routes == 31u && templates == 37u;
}

[[nodiscard]] bool PreparedPublicationFingerprintIsSemantic() {
  using namespace rund::node::accel::detail;
  PreparedKernelPublicationIdentity publication{};
  publication.sources[0] = PreparedKernelPublicationViewIdentity{
      .backing_bytes = 4096u,
      .offset_bytes = 64u,
      .count = 16u,
      .stride_bytes = 8u,
      .element_bytes = 4u,
      .resource_ordinal = 3u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  publication.sources[1] = publication.sources[0];
  publication.sources[2] = publication.sources[0];
  publication.target = PreparedKernelPublicationViewIdentity{
      .backing_bytes = 8192u,
      .offset_bytes = 128u,
      .count = 16u,
      .stride_bytes = 4u,
      .element_bytes = 4u,
      .resource_ordinal = 7u,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  publication.state = 5u;
  publication.final = 2u;

  std::uint64_t expected_hi = 0u;
  std::uint64_t expected_lo = 0u;
  SeedPreparedKernelPublicationFingerprint(expected_hi, expected_lo);
  MixPreparedKernelPublicationFingerprint(expected_hi, expected_lo,
                                          publication);

  PreparedKernelPublicationIdentity tampered = publication;
  ++tampered.sources[1].offset_bytes;
  std::uint64_t tampered_hi = 0u;
  std::uint64_t tampered_lo = 0u;
  SeedPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo);
  MixPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo, tampered);
  if (tampered_hi == expected_hi && tampered_lo == expected_lo) {
    return false;
  }

  tampered = publication;
  ++tampered.target.resource_ordinal;
  tampered_hi = 0u;
  tampered_lo = 0u;
  SeedPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo);
  MixPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo, tampered);
  return tampered_hi != expected_hi || tampered_lo != expected_lo;
}

[[nodiscard]] bool GridBoundaries() {
  using rund::node::accel::detail::Grid;
  using rund::node::accel::detail::PlanGrid;
  constexpr std::uint64_t width = 256u;
  constexpr std::uint64_t x_limit = 4u;
  constexpr std::uint64_t y_limit = 3u;
  const Grid one = PlanGrid(1u, width, x_limit, y_limit);
  const Grid row = PlanGrid(width * x_limit, width, x_limit, y_limit);
  const Grid next = PlanGrid(width * x_limit + 1u, width, x_limit, y_limit);
  const Grid full =
      PlanGrid(width * x_limit * y_limit, width, x_limit, y_limit);
  const Grid excess =
      PlanGrid(width * x_limit * y_limit + 1u, width, x_limit, y_limit);
  const Grid overflow = PlanGrid(std::numeric_limits<std::uint64_t>::max(), 1u,
                                 std::numeric_limits<std::uint32_t>::max(),
                                 std::numeric_limits<std::uint32_t>::max());
  return one.x == 1u && one.y == 1u && row.x == x_limit && row.y == 1u &&
         next.x == x_limit && next.y == 2u && full.x == x_limit &&
         full.y == y_limit && !excess.valid() && !overflow.valid() &&
         !PlanGrid(0u, width, x_limit, y_limit).valid() &&
         !PlanGrid(1u, 0u, x_limit, y_limit).valid();
}

struct FinishEntry final {
  rund::AccelCheck result{};
};

struct FinishResources final {
  std::array<FinishEntry, 3u> entries{};

  [[nodiscard]] std::size_t size() const noexcept { return entries.size(); }
  [[nodiscard]] FinishEntry *entry(const std::size_t index) noexcept {
    return index < entries.size() ? &entries[index] : nullptr;
  }
};

[[nodiscard]] bool FinishPrecedence() {
  using rund::node::accel::detail::finish::Steps;
  FinishResources resources{.entries = {
                                FinishEntry{rund::AccelCheck{true, "ok"}},
                                FinishEntry{rund::AccelCheck{true, "ok"}},
                                FinishEntry{rund::AccelCheck{true, "ok"}},
                            }};
  resources.entries[0].result.failed_batches = 2u;
  resources.entries[0].result.first_failed_batch = 7u;
  resources.entries[0].result.first_status = 11u;
  resources.entries[1].result.failed_batches = 3u;
  resources.entries[1].result.first_failed_batch = 13u;
  resources.entries[1].result.first_status = 17u;
  const rund::AccelCheck folded =
      Steps(resources, [](const FinishEntry &entry) { return entry.result; });
  if (!folded.ok || folded.failed_batches != 5u ||
      folded.first_failed_batch != 7u || folded.first_status != 11u) {
    return false;
  }
  resources.entries[1].result = rund::AccelCheck{false, "step_failed"};
  const rund::AccelCheck failed =
      Steps(resources, [](const FinishEntry &entry) { return entry.result; });
  return !failed.ok && failed.reason != nullptr &&
         std::string_view{failed.reason} == "step_failed";
}

[[nodiscard]] bool TelemetryProjection() {
  using namespace rund::node::accel::detail;
  const PreparedPipelineControl control{
      .generated_item_count = 1u,
      .generated_capacity = 2u,
      .indirect_dispatch_count = 3u,
      .indirect_work_item_count = 4u,
      .iteration_count = 5u,
      .skipped_iteration_count = 6u,
      .conflict_count = 7u,
      .overflow_ordinal = 8u,
  };
  rund::RuntimeStats stats{};
  ProjectTelemetry(control, stats);
  return stats.generated_item_count == 1u && stats.generated_capacity == 2u &&
         stats.indirect_dispatch_count == 3u &&
         stats.indirect_work_item_count == 4u && stats.iteration_count == 5u &&
         stats.skipped_iteration_count == 6u && stats.conflict_count == 7u &&
         stats.overflow_ordinal == 8u;
}

} // namespace

bool AuthorityContract() {
  return SubmissionTransitions() && PreparedPipelineClaimHasOneAuthority() &&
         PreparedPipelineNestedCoordinateShapesAreCanonical() &&
         PreparedPipelineFailureCoordinatesAreExact() &&
         PreparedPipelinePreparePathsAlwaysReportFailure() &&
         PreparedTemplateRegistryIsColdAndCollisionSafe() &&
         PreparedReservationIsFieldwiseFailClosed() &&
         RecurrenceRouteCopiesDoNotCloneTemplates() &&
         MetalPointerIdentityIndexIsExactAndOneShot() &&
         MetalIcbSizeClassPlanIsExactAndPortable() &&
         MetalColdIdentityIndexBudgetIsExact() &&
         MetalCaptureRowCapacityIsExact() &&
         PreparedTemplateStepCapacityIsBackendBounded() &&
         PreparedBackendProjectionIsRouteWiseAndChecked() &&
         PreparedPublicationCommandShapeIsCanonical() &&
         VulkanPhysicalCommandShapeIsNonOverlapping() &&
         VulkanPhysicalCapacityFailureIsTransactional() &&
         PreparedBackendControlManifestIsDimensionallyClosed() &&
         BackendSourceRecipeIsCheckedAndCanonical() &&
         VulkanMapAndResetSourceRecipesAreExact() &&
         MapSourceSpecializationIsSingleOwnerAndExact() &&
         MetalSourceRecipesAreExactAndSemantic() &&
         MetalMapCheckSourceHasOneGuardAuthority() &&
         MetalColdManifestIsExact() &&
         BackendTemplateRouteDemandIsExactAndFrozen() &&
         PreparedPublicationFingerprintIsSemantic() && GridBoundaries() &&
         FinishPrecedence() && TelemetryProjection();
}

} // namespace node_accel_contract

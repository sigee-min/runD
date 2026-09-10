#include "local.hpp"

#include "../../../../compute/allocation.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rund::node::test_contract::telemetry_contract {
namespace {

struct BoundObserver final {
  void operator()(const ::rund::telemetry::Event &) const noexcept {}
};

struct Text final {
  void operator()(const std::string_view chunk) noexcept {
    if (chunk.size() > bytes.size() - size) {
      overflow = true;
      return;
    }
    for (const char value : chunk) {
      bytes[size++] = value;
    }
  }

  [[nodiscard]] std::string_view view() const noexcept {
    return {bytes.data(), size};
  }

  std::array<char, 512u> bytes{};
  std::size_t size = 0u;
  bool overflow = false;
};

using SinkCallback = void (*)(void *, const ::rund::telemetry::Event &);

template <class Observer>
concept CanBind = requires(Observer &&observer) {
  ::rund::telemetry::bind(std::forward<Observer>(observer));
};

static_assert(CanBind<BoundObserver &>);
static_assert(!CanBind<BoundObserver>);
static_assert(!std::is_aggregate_v<::rund::telemetry::Sink>);
static_assert(!std::constructible_from<::rund::telemetry::Sink, void *,
                                       SinkCallback, ::rund::telemetry::Level>);
static_assert(std::same_as<decltype(::rund::telemetry::bind(
                               std::declval<BoundObserver &>())),
                           ::rund::telemetry::Sink>);
static_assert(
    noexcept(::rund::telemetry::bind(std::declval<BoundObserver &>())));
static_assert(noexcept(::rund::telemetry::describe(
    std::declval<const ::rund::telemetry::Event &>(), std::declval<Text &>())));

[[nodiscard]] int CheckFindings() {
  using namespace ::rund::telemetry;
  constexpr Event basic{
      .source = Source::Compute,
      .level = Level::Basic,
      .compute =
          {
              .buffer_allocations = 3u,
              .buffer_reuses = 5u,
              .copied_bytes = 7u,
              .graph_read_bytes = 11u,
          },
      .queue = {.depth = 13u, .capacity = 13u},
  };
  const Findings basic_findings = basic.findings();
  if (basic_findings.size() != Findings::Capacity ||
      basic_findings[0u].cost != Cost::Allocation ||
      basic_findings[0u].observed != 3u || basic_findings[0u].reference != 5u ||
      basic_findings[0u].reference_kind != Reference::ReuseEvents ||
      basic_findings[0u].action != Action::ReuseJob ||
      basic_findings[1u].cost != Cost::Copy ||
      basic_findings[1u].observed != 7u || basic_findings[1u].has_reference() ||
      basic_findings[1u].action != Action::KeepResident ||
      basic_findings[2u].cost != Cost::Scan ||
      basic_findings[2u].observed != 11u ||
      basic_findings[2u].action != Action::ReduceGraphBound ||
      basic_findings[3u].cost != Cost::Queue ||
      basic_findings[3u].observed != 13u ||
      basic_findings[3u].reference != 13u ||
      basic_findings[3u].reference_kind != Reference::QueueCapacity ||
      basic_findings[3u].cause != Cause::QueueAtBound ||
      basic_findings[3u].action != Action::ReduceFanout ||
      basic_findings[4u].cost != Cost::CriticalPath ||
      basic_findings[4u].exact() ||
      basic_findings[4u].accuracy != Accuracy::Unavailable ||
      basic_findings[4u].cause != Cause::TimingUnavailable ||
      basic_findings[4u].action != Action::EnableDetail) {
    return 1;
  }

  constexpr Event detail{
      .source = Source::Compute,
      .level = Level::Detail,
      .compute = basic.compute,
      .queue = basic.queue,
      .detail = {.prepare_ns = 17u, .work_ns = 19u, .finish_ns = 19u},
  };
  const Findings detail_findings = detail.findings();
  static_assert(members(static_cast<Cause>(0x07ffu)).size() == 11u);
  static_assert(members(static_cast<Action>(0x07ffu)).size() == 11u);
  for (std::size_t index = 0u; index != 4u; ++index) {
    if (basic_findings[index] != detail_findings[index]) {
      return 1;
    }
  }
  const Finding &critical = detail_findings[4u];
  if (!critical.exact() || critical.observed != 19u ||
      critical.reference != 55u ||
      critical.reference_kind != Reference::PhaseTotal ||
      !contains(critical.cause, Cause::Work) ||
      !contains(critical.cause, Cause::Finish) ||
      contains(critical.cause, Cause::Prepare) ||
      !contains(critical.action, Action::ReduceGraphBound) ||
      !contains(critical.action, Action::ReadSelectedOutput)) {
    return 2;
  }

  constexpr Event submission{
      .source = Source::Compute,
      .level = Level::Detail,
      .compute = {.command_submits = 1u,
                  .kernel_ns = 7u,
                  .kernel_samples = 1u,
                  .submit_wait_ns = 19u},
      .detail = {.work_ns = 19u},
  };
  const Findings submission_findings = submission.findings();
  const Finding &submission_path =
      submission_findings[submission_findings.size() - 1u];
  if (!contains(submission_path.cause, Cause::Work) ||
      !contains(submission_path.cause, Cause::SubmitOverhead) ||
      !contains(submission_path.action, Action::BatchJobs) ||
      contains(submission_path.action, Action::ReduceGraphBound) ||
      name(Cause::SubmitOverhead) != "submit-overhead" ||
      name(Action::BatchJobs) != "batch-jobs") {
    return 2;
  }

  constexpr Event replay{
      .source = Source::Replay,
      .level = Level::Detail,
      .replay =
          {
              .retained_bytes = 29u,
              .copied_bytes = 23u,
              .storage_growths = 2u,
          },
      .detail = {.prepare_ns = 31u, .work_ns = 31u, .finish_ns = 3u},
  };
  const Findings replay_findings = replay.findings();
  if (replay_findings.size() != 3u ||
      replay_findings[0u].cause != Cause::StorageGrowth ||
      replay_findings[0u].action != Action::ConfigureStorage ||
      replay_findings[0u].has_reference() ||
      replay_findings[1u].cause != Cause::ReplayCopy ||
      replay_findings[1u].reference_kind != Reference::RetainedBytes ||
      !contains(replay_findings[2u].cause, Cause::Prepare) ||
      !contains(replay_findings[2u].cause, Cause::Work) ||
      !contains(replay_findings[2u].action, Action::ReuseReplayPlan) ||
      !contains(replay_findings[2u].action, Action::ReduceReplayEvidence)) {
    return 3;
  }

  constexpr std::string_view expected_description =
      "allocation cause=buffer-allocation action=reuse-job\n"
      "copy cause=boundary-copy action=keep-resident\n"
      "scan cause=graph-read action=reduce-graph-bound\n"
      "queue cause=queue-at-bound action=reduce-fanout\n"
      "critical-path cause=work,finish "
      "action=reduce-graph-bound,read-selected-output";
  constexpr std::string_view expected_empty_masks =
      "allocation cause=none action=none";
  Text description{};
  Text repeated_description{};
  Text empty_masks{};
  Text empty_range{};
  const Findings no_findings{};
  node_compute_allocation::Start();
  describe(detail, description);
  describe(detail, repeated_description);
  describe(Finding{}, empty_masks);
  for (const Finding &finding : no_findings) {
    describe(finding, empty_range);
  }
  const Findings observed = detail.findings();
  const Members<Cause> observed_causes = members(critical.cause);
  const Members<Action> observed_actions = members(critical.action);
  node_compute_allocation::Stop();
  return node_compute_allocation::Count() == 0u && !description.overflow &&
                 !repeated_description.overflow && !empty_masks.overflow &&
                 !empty_range.overflow &&
                 description.view() == expected_description &&
                 repeated_description.view() == expected_description &&
                 empty_masks.view() == expected_empty_masks &&
                 empty_range.view().empty() &&
                 observed.size() == Findings::Capacity &&
                 observed_causes.size() == 2u &&
                 observed_actions.size() == 2u &&
                 name(observed_causes[0u]) == "work" &&
                 name(observed_causes[1u]) == "finish" &&
                 name(observed_actions[0u]) == "reduce-graph-bound" &&
                 name(observed_actions[1u]) == "read-selected-output" &&
                 name(observed[0u].cost) == "allocation" &&
                 name(observed[0u].unit) == "events" &&
                 name(Accuracy::Exact) == "exact" &&
                 name(Accuracy::Unavailable) == "unavailable" &&
                 name(Accuracy::Saturated) == "saturated" &&
                 name(Reference::None) == "none" &&
                 name(Reference::ReuseEvents) == "reuse-events" &&
                 name(Reference::RetainedBytes) == "retained-bytes" &&
                 name(Reference::QueueCapacity) == "queue-capacity" &&
                 name(Reference::PhaseTotal) == "phase-total" &&
                 name(Action::ReuseJob) == "reuse-job"
             ? 0
             : 3;
}

} // namespace

int CheckSurfaceAndFindings() {
  if (const int findings = CheckFindings(); findings != 0) {
    return 100 + findings;
  }
  BoundObserver bound_observer{};
  const ::rund::telemetry::Sink empty_sink{};
  const ::rund::telemetry::Sink bound_sink =
      ::rund::telemetry::bind(bound_observer, ::rund::telemetry::Level::Detail);
  if (empty_sink || empty_sink.level() != ::rund::telemetry::Level::Basic ||
      !bound_sink || bound_sink.level() != ::rund::telemetry::Level::Detail) {
    return 1;
  }
  constexpr ::rund::telemetry::Event compute_failure{
      .source = ::rund::telemetry::Source::Compute,
      .compute = {.code = compute::Code::Binding},
  };
  constexpr ::rund::telemetry::Event invalid_source{
      .source = static_cast<::rund::telemetry::Source>(0xffu),
  };
  constexpr ::rund::telemetry::Event invalid_compute{
      .source = ::rund::telemetry::Source::Compute,
      .compute = {.code = static_cast<compute::Code>(0xffu)},
  };
  if (compute_failure.error() != "compute_binding" ||
      invalid_source.error() != "telemetry_source_invalid" ||
      invalid_compute.error() != "telemetry_compute_code_invalid") {
    return 1;
  }
  const ::rund::telemetry::Event replay_failure{
      .source = ::rund::telemetry::Source::Replay,
      .replay = {.code = ::rund::replay::Code::ScenarioInputDuplicate},
  };
  return replay_failure.error() == "replay_scenario_input_duplicate" ? 0 : 1;
}

} // namespace rund::node::test_contract::telemetry_contract

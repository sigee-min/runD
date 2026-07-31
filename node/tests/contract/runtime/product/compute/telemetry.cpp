#include "../support.hpp"

#include "../../../compute/allocation.hpp"
#include "src/runtime/session/result.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

namespace rund::node::test_contract {
namespace {

struct TelemetryGate final {
  ::rund::Session *session = nullptr;
  std::mutex mutex{};
  std::condition_variable changed{};
  bool entered = false;
  bool drain_returned = false;
  bool release = false;
  bool reentry_forbidden = false;

  void operator()(const ::rund::telemetry::Event &) {
    const ::rund::Trace trace = session->trace();
    std::unique_lock lock{mutex};
    reentry_forbidden =
        trace.code == ::rund::ReasonCode::RuntimeReentryForbidden;
    entered = true;
    changed.notify_all();
    changed.wait(lock, [&] { return release; });
  }
};

struct ThrowingObserver final {
  void operator()(const ::rund::telemetry::Event &) const { throw 1; }
};

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

int CheckTelemetry() {
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
  if (replay_failure.error() != "replay_scenario_input_duplicate") {
    return 1;
  }

  constexpr std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-telemetry", input.size(),
                             [](auto value) { return value * 2 + 1; })
          .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    return 2;
  }

  ::rund::Session session{};
  TelemetryGate gate{.session = &session};
  rund::SessionConfig options = Options();
  options.telemetry = ::rund::telemetry::bind(gate);
  if (!session.open(options)) {
    return 3;
  }
  auto task = session.compute(*job).submit();
  {
    std::unique_lock lock{gate.mutex};
    gate.changed.wait(lock, [&] { return gate.entered; });
  }

  rund::Session::Status drained{};
  rund::Session::Status stopped{};
  std::thread lifecycle{[&] {
    drained = session.drain();
    {
      std::lock_guard lock{gate.mutex};
      gate.drain_returned = true;
    }
    gate.changed.notify_all();
    stopped = session.close();
  }};
  {
    std::unique_lock lock{gate.mutex};
    gate.changed.wait(lock, [&] { return gate.drain_returned; });
    if (session.snapshot().state != ::rund::SessionState::Draining) {
      gate.release = true;
      lock.unlock();
      gate.changed.notify_all();
      lifecycle.join();
      return 4;
    }
    gate.release = true;
  }
  gate.changed.notify_all();
  const compute::Completion result = task.wait();
  lifecycle.join();
  if (!result || !drained || !gate.reentry_forbidden) {
    std::fprintf(stderr,
                 "compute telemetry result=%u error=%.*s drained=%u "
                 "drain_reason=%.*s reentry=%u\n",
                 static_cast<bool>(result),
                 static_cast<int>(result.error().size()), result.error().data(),
                 static_cast<bool>(drained),
                 static_cast<int>(drained.error().size()),
                 drained.error().data(), gate.reentry_forbidden);
    return 5;
  }
  if (!stopped && !session.close()) {
    return 6;
  }

  auto throwing_job = program->resident(input);
  if (!throwing_job) {
    return 7;
  }
  ::rund::Session throwing{};
  rund::SessionConfig throwing_options = Options();
  ThrowingObserver throwing_observer{};
  throwing_options.telemetry = ::rund::telemetry::bind(throwing_observer);
  if (!throwing.open(throwing_options)) {
    return 8;
  }
  const compute::Completion throwing_result =
      throwing.compute(*throwing_job).submit().wait();
  if (!throwing_result ||
      !Saw(throwing.trace(), ::rund::TraceEvent::TelemetrySkipped)) {
    return 9;
  }
  if (!throwing.close()) {
    return 10;
  }

  auto detail_job = program->resident(input);
  if (!detail_job) {
    return 11;
  }
  TelemetryProbe detail_probe{};
  auto detail_observer = [&](const ::rund::telemetry::Event &event) {
    detail_probe(event);
  };
  ::rund::Session detail_session{};
  rund::SessionConfig detail_options = Options();
  detail_options.telemetry = ::rund::telemetry::bind(
      detail_observer, ::rund::telemetry::Level::Detail);
  if (!detail_session.open(detail_options)) {
    return 12;
  }
  const compute::Completion detail_result =
      detail_session.compute(*detail_job).submit().wait();
  if (!detail_result || detail_probe.events != 1u ||
      detail_probe.event.source != ::rund::telemetry::Source::Compute ||
      detail_probe.event.level != ::rund::telemetry::Level::Detail ||
      detail_probe.event.compute.code != compute::Code::Ok ||
      !detail_probe.event.error().empty() ||
      detail_probe.event.replay.code != ::rund::replay::Code::Ok) {
    return 13;
  }
  const compute::Stats &detail_stats = detail_result.stats();
  const std::uint64_t expected_prepare =
      detail_stats.shader_compile_ns + detail_stats.spirv_compile_ns +
      detail_stats.pipeline_create_ns + detail_stats.descriptor_setup_ns;
  const std::uint64_t expected_work = detail_stats.submit_wait_ns != 0u
                                          ? detail_stats.submit_wait_ns
                                          : detail_stats.kernel_ns;
  if (detail_probe.event.detail.prepare_ns != expected_prepare ||
      detail_probe.event.detail.work_ns != expected_work ||
      detail_probe.event.detail.finish_ns != detail_stats.readback_ns ||
      detail_probe.event.compute.graph_read_bytes !=
          detail_stats.graph_read_bytes ||
      detail_probe.event.compute.graph_read_bytes != sizeof(input) ||
      detail_probe.event.replay.mode != ::rund::telemetry::Mode::None ||
      detail_probe.event.replay.plan != ::rund::telemetry::Preparation::None ||
      detail_probe.event.replay.input_rows != 0u ||
      detail_probe.event.replay.input_bytes != 0u ||
      detail_probe.event.replay.choices != 0u ||
      detail_probe.event.replay.evidence_rows != 0u ||
      detail_probe.event.replay.evidence_bytes != 0u ||
      detail_probe.event.replay.retained_bytes != 0u ||
      detail_probe.event.replay.copied_bytes != 0u ||
      detail_probe.event.replay.physical_bytes != 0u ||
      detail_probe.event.replay.allocated_bytes != 0u ||
      detail_probe.event.replay.reserved_bytes != 0u ||
      detail_probe.event.replay.storage_growths != 0u ||
      detail_probe.event.replay.result_hash != 0u) {
    return 14;
  }
  const auto basic_output = job->read();
  const auto detail_output = detail_job->read();
  if (!basic_output || !detail_output || *basic_output != *detail_output) {
    return 15;
  }
  if (!detail_session.close()) {
    return 16;
  }

  BoundObserver basic_observer{};
  ::rund::Session basic_scope{};
  rund::SessionConfig basic_options = Options();
  basic_options.telemetry = ::rund::telemetry::bind(basic_observer);
  if (!basic_scope.open(basic_options)) {
    return 17;
  }
  const ::rund::Session::Result basic_result = basic_scope.scope([] {});
  const ::rund::telemetry::Detail &basic_timing =
      ::rund::detail::session::ResultAccess::timing(basic_result);
  if (!basic_result ||
      ::rund::detail::session::ResultAccess::telemetry_clock_reads(
          basic_result) != 0u ||
      basic_timing.prepare_ns != 0u || basic_timing.work_ns != 0u ||
      basic_timing.finish_ns != 0u || !basic_scope.close()) {
    return 18;
  }

  BoundObserver timing_observer{};
  ::rund::Session detail_scope{};
  rund::SessionConfig timing_options = Options();
  timing_options.telemetry = ::rund::telemetry::bind(
      timing_observer, ::rund::telemetry::Level::Detail);
  if (!detail_scope.open(timing_options)) {
    return 19;
  }
  const ::rund::Session::Result timing_result = detail_scope.scope([] {});
  if (!timing_result ||
      ::rund::detail::session::ResultAccess::telemetry_clock_reads(
          timing_result) != 4u ||
      !detail_scope.close()) {
    return 20;
  }
  return 0;
}

} // namespace rund::node::test_contract

int RunTelemetryDetailContract() {
  return rund::node::test_contract::CheckTelemetry();
}

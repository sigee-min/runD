#include "test/assert.hpp"

#include "src/runtime/session/result.hpp"

#include <rund/replay.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t kSession = 0x7107u;
constexpr rund::replay::Input kInput{.id = 17u, .schema = 71u};
constexpr std::uint64_t kSequence = 5u;
constexpr std::size_t kParityEvents = 6u;

struct Collector final {
  std::array<rund::telemetry::Event, 16u> values{};
  std::size_t count = 0u;

  void operator()(const rund::telemetry::Event &event) {
    TEST_ASSERT(count < values.size());
    values[count++] = event;
  }

  [[nodiscard]] const rund::telemetry::Event &
  one(const std::size_t before) const noexcept {
    TEST_ASSERT(count == before + 1u);
    return values[before];
  }
};

template <class Observer>
[[nodiscard]] rund::SessionConfig Config(Observer &observer,
                                         const rund::telemetry::Level level,
                                         const std::uint64_t id = kSession) {
  rund::SessionConfig config{};
  config.id = id;
  config.workers = 1u;
  config.trace_capacity = 64u;
  config.scheduler.task_workers = 1u;
  config.scheduler.task_capacity = 4u;
  config.scheduler.ready_queue_capacity = 4u;
  config.scheduler.observation_capacity = 16u;
  config.scheduler.host_event_capacity = 16u;
  config.scheduler.host_payload_capacity_bytes = 256u;
  config.replay.input_capacity = 4u;
  config.replay.storage.cached_bytes = 64u;
  config.replay.storage.segment_bytes = 64u;
  config.replay.storage.max_bytes = 1024u;
  config.telemetry = rund::telemetry::bind(observer, level);
  return config;
}

[[nodiscard]] std::uint64_t Count(const std::size_t value) noexcept {
  return static_cast<std::uint64_t>(value);
}

void ExpectBasic(const rund::telemetry::Event &event,
                 const rund::telemetry::Level level) {
  TEST_ASSERT(event.level == level);
  if (level == rund::telemetry::Level::Basic) {
    TEST_ASSERT(event.detail.prepare_ns == 0u);
    TEST_ASSERT(event.detail.work_ns == 0u);
    TEST_ASSERT(event.detail.finish_ns == 0u);
  }
}

void ExpectHeader(const rund::telemetry::Event &event,
                  const rund::telemetry::Level level,
                  const rund::telemetry::Mode mode,
                  const rund::telemetry::Preparation plan,
                  const rund::replay::Code code, const std::uint64_t choices) {
  TEST_ASSERT(event.source == rund::telemetry::Source::Replay);
  TEST_ASSERT(event.session == kSession);
  TEST_ASSERT(event.replay.mode == mode);
  TEST_ASSERT(event.replay.plan == plan);
  TEST_ASSERT(event.replay.code == code);
  TEST_ASSERT(event.replay.choices == choices);
  TEST_ASSERT(event.error() == rund::replay::error(code));
  ExpectBasic(event, level);
}

void ExpectLive(const rund::telemetry::Event &event,
                const rund::telemetry::Level level,
                const rund::replay::Live &result,
                const std::uint64_t input_rows, const std::uint64_t input_bytes,
                const std::uint64_t produced_rows) {
  ExpectHeader(event, level, rund::telemetry::Mode::Live,
               rund::telemetry::Preparation::None, result.code(), 0u);
  TEST_ASSERT(event.scope == result.scope());
  TEST_ASSERT(event.scope != 0u);
  TEST_ASSERT(event.replay.input_rows == input_rows);
  TEST_ASSERT(event.replay.input_bytes == input_bytes);
  TEST_ASSERT(event.replay.produced_rows == produced_rows);
  TEST_ASSERT(event.replay.evidence_rows ==
              Count(result.observations().size()) +
                  Count(result.events().size()) +
                  Count(result.trace().records.size()));
  TEST_ASSERT(event.replay.evidence_bytes == 0u);
  TEST_ASSERT(event.replay.retained_bytes == 0u);
  TEST_ASSERT(event.replay.copied_bytes == 0u);
  TEST_ASSERT(event.replay.physical_bytes == 0u);
  TEST_ASSERT(event.replay.allocated_bytes == 0u);
  TEST_ASSERT(event.replay.reserved_bytes == 0u);
  TEST_ASSERT(event.replay.storage_growths == 0u);
  TEST_ASSERT(event.replay.result_hash == 0u);
}

void ExpectRecord(const rund::telemetry::Event &event,
                  const rund::telemetry::Level level,
                  const rund::telemetry::Mode mode,
                  const rund::telemetry::Preparation plan,
                  const rund::replay::Record &record,
                  const std::uint64_t produced_rows,
                  const std::uint64_t choices) {
  ExpectHeader(event, level, mode, plan, record.code(), choices);
  TEST_ASSERT(event.scope != 0u);
  const rund::replay::StorageReport storage = record.storage_report();
  const rund::replay::DiagnosticReport diagnostic = record.capture_report();
  TEST_ASSERT(event.replay.input_rows == record.input_count());
  TEST_ASSERT(event.replay.input_bytes == storage.logical_bytes);
  TEST_ASSERT(event.replay.produced_rows == produced_rows);
  TEST_ASSERT(
      event.replay.evidence_rows ==
      Count(record.observation_count()) + Count(record.host_event_count()) +
          Count(record.input_count()) + Count(record.trace_record_count()) +
          Count(record.captures().size()));
  TEST_ASSERT(event.replay.evidence_bytes ==
              storage.encoded_bytes + diagnostic.retained_bytes);
  TEST_ASSERT(event.replay.retained_bytes ==
              storage.retained_bytes + diagnostic.retained_bytes);
  TEST_ASSERT(event.replay.copied_bytes ==
              storage.copied_bytes + diagnostic.retained_bytes);
  TEST_ASSERT(event.replay.physical_bytes == storage.physical_bytes);
  TEST_ASSERT(event.replay.allocated_bytes == storage.allocated_bytes);
  TEST_ASSERT(event.replay.reserved_bytes == storage.reserved_bytes);
  TEST_ASSERT(event.replay.storage_growths == storage.growths);
  TEST_ASSERT(event.replay.result_hash == record.hash());
}

void ExpectRejected(const rund::telemetry::Event &event,
                    const rund::telemetry::Level level,
                    const rund::replay::Code code,
                    const std::uint64_t choices) {
  ExpectHeader(event, level, rund::telemetry::Mode::Scenario,
               rund::telemetry::Preparation::Reused, code, choices);
  TEST_ASSERT(event.scope == 0u);
  TEST_ASSERT(event.replay.input_rows == 0u);
  TEST_ASSERT(event.replay.input_bytes == 0u);
  TEST_ASSERT(event.replay.produced_rows == 0u);
  TEST_ASSERT(event.replay.evidence_rows == 0u);
  TEST_ASSERT(event.replay.evidence_bytes == 0u);
  TEST_ASSERT(event.replay.retained_bytes == 0u);
  TEST_ASSERT(event.replay.copied_bytes == 0u);
  TEST_ASSERT(event.replay.physical_bytes == 0u);
  TEST_ASSERT(event.replay.allocated_bytes == 0u);
  TEST_ASSERT(event.replay.reserved_bytes == 0u);
  TEST_ASSERT(event.replay.storage_growths == 0u);
  TEST_ASSERT(event.replay.result_hash == 0u);
}

struct Identity final {
  rund::replay::Code code = rund::replay::Code::CheckpointInvalid;
  std::uint64_t input = 0u;
  std::uint64_t transcript = 0u;
  std::uint64_t result = 0u;

  [[nodiscard]] friend bool operator==(const Identity &,
                                       const Identity &) = default;
};

[[nodiscard]] Identity Of(const rund::replay::Record &record) noexcept {
  return Identity{.code = record.code(),
                  .input = record.input_hash(),
                  .transcript = record.transcript_hash(),
                  .result = record.hash()};
}

struct LevelRun final {
  std::array<rund::telemetry::Event, kParityEvents> events{};
  std::array<Identity, 4u> identities{};
  std::uint64_t producer_calls = 0u;
};

[[nodiscard]] LevelRun RunLevel(const rund::telemetry::Level level) {
  Collector observer{};
  rund::Session session{};
  TEST_ASSERT(session.open(Config(observer, level)));

  const rund::Session::Result clock = session.scope([] {});
  TEST_ASSERT(clock);
  const rund::telemetry::Detail &clock_detail =
      rund::detail::session::ResultAccess::timing(clock);
  if (level == rund::telemetry::Level::Basic) {
    TEST_ASSERT(rund::detail::session::ResultAccess::telemetry_clock_reads(
                    clock) == 0u);
    TEST_ASSERT(clock_detail.prepare_ns == 0u);
    TEST_ASSERT(clock_detail.work_ns == 0u);
    TEST_ASSERT(clock_detail.finish_ns == 0u);
  } else {
    TEST_ASSERT(rund::detail::session::ResultAccess::telemetry_clock_reads(
                    clock) == 4u);
  }
  TEST_ASSERT(observer.count == 0u);

  const std::array payload{std::byte{0x11}, std::byte{0x22}, std::byte{0x33}};
  const std::array replacement{std::byte{0x44}, std::byte{0x55}};
  std::span<const std::byte> expected = payload;
  std::uint64_t producer_calls = 0u;
  auto restore = [](std::span<const std::byte>) {
    return rund::replay::Restore::Restored;
  };
  rund::replay::Binding binding{0x7401u, restore};
  auto source = [&](rund::replay::Writer &writer) {
    ++producer_calls;
    TEST_ASSERT(writer.append(payload));
    return kSequence;
  };
  const auto commands = binding.input(kInput, source);
  auto simulate = [&](rund::replay::Context &context) {
    const rund::replay::Value value = commands.read(context);
    TEST_ASSERT(value);
    TEST_ASSERT(value.size() == expected.size());
    TEST_ASSERT(std::equal(value.bytes().begin(), value.bytes().end(),
                           expected.begin(), expected.end()));
  };

  std::size_t before = observer.count;
  std::uint64_t produced = producer_calls;
  const rund::replay::Live live = rund::replay::live(session, simulate);
  TEST_ASSERT(live);
  TEST_ASSERT(producer_calls - produced == 1u);
  ExpectLive(observer.one(before), level, live, 1u, payload.size(), 1u);

  before = observer.count;
  produced = producer_calls;
  const rund::replay::Record recorded = rund::replay::record(session, simulate);
  TEST_ASSERT(recorded);
  TEST_ASSERT(producer_calls - produced == 1u);
  ExpectRecord(observer.one(before), level, rund::telemetry::Mode::Record,
               rund::telemetry::Preparation::Built, recorded, 1u, 0u);

  std::vector<std::byte> artifact{};
  const rund::replay::Save saved =
      recorded.save([&artifact](const std::span<const std::byte> bytes) {
        artifact.insert(artifact.end(), bytes.begin(), bytes.end());
        return true;
      });
  TEST_ASSERT(saved);
  const rund::replay::Load<rund::replay::Record> decoded =
      rund::replay::Record::load(artifact);
  TEST_ASSERT(decoded);
  const rund::replay::Record expected_record = *decoded;

  before = observer.count;
  produced = producer_calls;
  const rund::replay::Check replayed =
      rund::replay::run(session, expected_record, simulate);
  TEST_ASSERT(replayed);
  TEST_ASSERT(replayed.actual().has_value());
  TEST_ASSERT(producer_calls - produced == 0u);
  ExpectRecord(observer.one(before), level, rund::telemetry::Mode::Replay,
               rund::telemetry::Preparation::Built, *replayed.actual(), 0u, 0u);

  const rund::replay::Load<rund::replay::Record> scenario_decoded =
      rund::replay::Record::load(artifact);
  TEST_ASSERT(scenario_decoded);
  const rund::replay::Record scenario_expected = *scenario_decoded;
  const std::array choices{commands.choice(kSequence, replacement)};
  expected = replacement;
  before = observer.count;
  produced = producer_calls;
  const rund::replay::Scenario scenario_built = rund::replay::scenario(
      session, scenario_expected,
      std::span<const rund::replay::Choice>{choices}, simulate);
  TEST_ASSERT(scenario_built);
  TEST_ASSERT(scenario_built.actual().has_value());
  TEST_ASSERT(producer_calls - produced == 0u);
  ExpectRecord(observer.one(before), level, rund::telemetry::Mode::Scenario,
               rund::telemetry::Preparation::Built, *scenario_built.actual(),
               0u, choices.size());

  before = observer.count;
  produced = producer_calls;
  const rund::replay::Scenario scenario_reused = rund::replay::scenario(
      session, scenario_expected,
      std::span<const rund::replay::Choice>{choices}, simulate);
  TEST_ASSERT(scenario_reused);
  TEST_ASSERT(scenario_reused.actual().has_value());
  TEST_ASSERT(producer_calls - produced == 0u);
  ExpectRecord(observer.one(before), level, rund::telemetry::Mode::Scenario,
               rund::telemetry::Preparation::Reused, *scenario_reused.actual(),
               0u, choices.size());

  bool rejected_callback = false;
  const std::array duplicate_bytes{std::byte{0x66}};
  const rund::replay::Choice duplicate =
      commands.choice(kSequence, duplicate_bytes);
  const std::array duplicates{duplicate, duplicate};
  before = observer.count;
  const rund::replay::Scenario rejected = rund::replay::scenario(
      session, scenario_expected,
      std::span<const rund::replay::Choice>{duplicates},
      [&](rund::replay::Context &) { rejected_callback = true; });
  TEST_ASSERT(!rejected);
  TEST_ASSERT(rejected.code() == rund::replay::Code::ScenarioInputDuplicate);
  TEST_ASSERT(!rejected_callback);
  ExpectRejected(observer.one(before), level, rejected.code(),
                 duplicates.size());

  TEST_ASSERT(observer.count == kParityEvents);
  TEST_ASSERT(producer_calls == 2u);
  TEST_ASSERT(session.close());
  TEST_ASSERT(observer.count == kParityEvents);

  LevelRun run{};
  std::copy_n(observer.values.begin(), run.events.size(), run.events.begin());
  run.identities = {Of(recorded), Of(*replayed.actual()),
                    Of(*scenario_built.actual()),
                    Of(*scenario_reused.actual())};
  run.producer_calls = producer_calls;
  return run;
}

[[nodiscard]] bool SameReplay(const rund::telemetry::Replay &left,
                              const rund::telemetry::Replay &right) noexcept {
  return left.code == right.code && left.mode == right.mode &&
         left.plan == right.plan && left.input_rows == right.input_rows &&
         left.input_bytes == right.input_bytes &&
         left.produced_rows == right.produced_rows &&
         left.choices == right.choices &&
         left.evidence_rows == right.evidence_rows &&
         left.evidence_bytes == right.evidence_bytes &&
         left.retained_bytes == right.retained_bytes &&
         left.copied_bytes == right.copied_bytes &&
         left.physical_bytes == right.physical_bytes &&
         left.allocated_bytes == right.allocated_bytes &&
         left.reserved_bytes == right.reserved_bytes &&
         left.storage_growths == right.storage_growths &&
         left.result_hash == right.result_hash;
}

[[nodiscard]] bool
SameNontimeFindings(const rund::telemetry::Event &left,
                    const rund::telemetry::Event &right) noexcept {
  const rund::telemetry::Findings left_findings = left.findings();
  const rund::telemetry::Findings right_findings = right.findings();
  std::size_t left_index = 0u;
  std::size_t right_index = 0u;
  while (true) {
    while (left_index != left_findings.size() &&
           left_findings[left_index].cost ==
               rund::telemetry::Cost::CriticalPath) {
      ++left_index;
    }
    while (right_index != right_findings.size() &&
           right_findings[right_index].cost ==
               rund::telemetry::Cost::CriticalPath) {
      ++right_index;
    }
    if (left_index == left_findings.size() ||
        right_index == right_findings.size()) {
      return left_index == left_findings.size() &&
             right_index == right_findings.size();
    }
    if (left_findings[left_index++] != right_findings[right_index++]) {
      return false;
    }
  }
}

void ExpectParity(const LevelRun &basic, const LevelRun &detail) {
  TEST_ASSERT(basic.identities == detail.identities);
  TEST_ASSERT(basic.producer_calls == detail.producer_calls);
  for (std::size_t index = 0u; index < basic.events.size(); ++index) {
    const rund::telemetry::Event &left = basic.events[index];
    const rund::telemetry::Event &right = detail.events[index];
    TEST_ASSERT(left.level == rund::telemetry::Level::Basic);
    TEST_ASSERT(right.level == rund::telemetry::Level::Detail);
    TEST_ASSERT(left.source == right.source);
    TEST_ASSERT(left.session == right.session);
    TEST_ASSERT(left.scope == right.scope);
    TEST_ASSERT(SameReplay(left.replay, right.replay));
    TEST_ASSERT(left.queue.depth == right.queue.depth);
    TEST_ASSERT(left.queue.capacity == right.queue.capacity);
    TEST_ASSERT(SameNontimeFindings(left, right));
    const rund::telemetry::Findings basic_findings = left.findings();
    const rund::telemetry::Findings detail_findings = right.findings();
    TEST_ASSERT(basic_findings.size() <= rund::telemetry::Findings::Capacity);
    TEST_ASSERT(detail_findings.size() <= rund::telemetry::Findings::Capacity);
    TEST_ASSERT(basic_findings[basic_findings.size() - 1u].accuracy ==
                rund::telemetry::Accuracy::Unavailable);
    TEST_ASSERT(detail_findings[detail_findings.size() - 1u].cost ==
                rund::telemetry::Cost::CriticalPath);
    TEST_ASSERT(left.error() == right.error());
    TEST_ASSERT(left.detail.prepare_ns == 0u);
    TEST_ASSERT(left.detail.work_ns == 0u);
    TEST_ASSERT(left.detail.finish_ns == 0u);
  }
}

void CheckUserThrow() {
  Collector observer{};
  rund::Session session{};
  TEST_ASSERT(session.open(
      Config(observer, rund::telemetry::Level::Basic, kSession + 1u)));

  std::size_t before = observer.count;
  const rund::replay::Live failed =
      rund::replay::live(session, [](rund::replay::Context &) {
        throw std::runtime_error{"telemetry replay callback"};
      });
  TEST_ASSERT(!failed);
  TEST_ASSERT(failed.code() == rund::replay::Code::RuntimeScopeCallbackFailed);
  const rund::telemetry::Event &failure = observer.one(before);
  TEST_ASSERT(failure.session == kSession + 1u);
  TEST_ASSERT(failure.replay.code == failed.code());
  TEST_ASSERT(failure.replay.mode == rund::telemetry::Mode::Live);
  TEST_ASSERT(failure.error() == failed.error());
  ExpectBasic(failure, rund::telemetry::Level::Basic);

  before = observer.count;
  const rund::replay::Live healthy =
      rund::replay::live(session, [](rund::replay::Context &) noexcept {});
  TEST_ASSERT(healthy);
  TEST_ASSERT(observer.one(before).replay.code == rund::replay::Code::Ok);
  TEST_ASSERT(session.close());
}

struct Reentry final {
  rund::Session *session = nullptr;
  std::uint32_t calls = 0u;
  std::uint32_t rejected = 0u;

  void operator()(const rund::telemetry::Event &) {
    ++calls;
    const rund::Session::Result nested = session->scope([] {});
    if (!nested && nested.code() == rund::ReasonCode::RuntimeReentryForbidden) {
      ++rejected;
    }
  }
};

void CheckReentry() {
  Reentry observer{};
  rund::Session session{};
  observer.session = &session;
  TEST_ASSERT(session.open(
      Config(observer, rund::telemetry::Level::Basic, kSession + 2u)));
  TEST_ASSERT(
      rund::replay::live(session, [](rund::replay::Context &) noexcept {}));
  TEST_ASSERT(
      rund::replay::live(session, [](rund::replay::Context &) noexcept {}));
  TEST_ASSERT(observer.calls == 2u);
  TEST_ASSERT(observer.rejected == observer.calls);
  TEST_ASSERT(session.close());
}

struct Throwing final {
  std::uint32_t calls = 0u;

  void operator()(const rund::telemetry::Event &) {
    ++calls;
    throw std::runtime_error{"telemetry sink"};
  }
};

[[nodiscard]] bool SawSkipped(const rund::Trace &trace) noexcept {
  for (const rund::TraceRecord &record : trace.records) {
    if (record.event == rund::TraceEvent::TelemetrySkipped) {
      return record.code.runtime_code() ==
             rund::ReasonCode::TelemetrySinkFailed;
    }
  }
  return false;
}

void CheckSinkThrow() {
  Throwing observer{};
  rund::Session session{};
  TEST_ASSERT(session.open(
      Config(observer, rund::telemetry::Level::Basic, kSession + 3u)));
  TEST_ASSERT(
      rund::replay::live(session, [](rund::replay::Context &) noexcept {}));
  TEST_ASSERT(
      rund::replay::live(session, [](rund::replay::Context &) noexcept {}));
  TEST_ASSERT(observer.calls == 2u);
  TEST_ASSERT(SawSkipped(session.trace()));
  TEST_ASSERT(session.close());
}

} // namespace

int RunRuntimeTaskReplayTelemetryContract() {
  const LevelRun basic = RunLevel(rund::telemetry::Level::Basic);
  const LevelRun detail = RunLevel(rund::telemetry::Level::Detail);
  ExpectParity(basic, detail);
  CheckUserThrow();
  CheckReentry();
  CheckSinkThrow();
  return 0;
}

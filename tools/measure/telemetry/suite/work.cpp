#include "core.hpp"

namespace rund::measure::telemetry {

[[nodiscard]] rund::SessionConfig Config(Observer &observer,
                                         const Setting setting) {
  rund::SessionConfig config{};
  config.id = kSession;
  config.workers = 1u;
  config.trace_capacity = 64u;
  config.random_seed = 0x9e3779b97f4a7c15ull;
  if (setting == Setting::Basic) {
    config.telemetry = rund::telemetry::bind(observer, Level::Basic);
  } else if (setting == Setting::Detail) {
    config.telemetry = rund::telemetry::bind(observer, Level::Detail);
  }
  config.replay.input_capacity = 4096u;
  config.replay.storage.mode = rund::replay::StorageMode::Memory;
  config.replay.storage.cached_bytes = 4096u;
  config.replay.storage.segment_bytes = 4096u;
  config.replay.storage.max_bytes = 1u << 20u;
  return config;
}

[[nodiscard]] std::array<std::byte, kPayloadBytes> Payload() noexcept {
  std::array<std::byte, kPayloadBytes> bytes{};
  std::uint64_t state = kSequence ^ 0xD6E8FEB86659FD93ull;
  for (std::size_t index = 0u; index < bytes.size(); ++index) {
    state ^= state >> 12u;
    state ^= state << 25u;
    state ^= state >> 27u;
    bytes[index] =
        static_cast<std::byte>((state * 0x2545F4914F6CDD1Dull) >> 56u);
  }
  return bytes;
}

[[nodiscard]] ReplayFields Formula(const Counters &counter,
                                   const Operation operation) noexcept {
  const std::uint64_t retained =
      Add(counter.retained_bytes, counter.capture_bytes);
  return ReplayFields{
      .input_rows = counter.inputs,
      .input_bytes = counter.logical_bytes,
      .produced_rows =
          operation == Operation::Live || operation == Operation::Record
              ? counter.inputs
              : 0u,
      .choices = operation == Operation::Scenario ? 1u : 0u,
      .evidence_rows = Add(Add(Add(counter.observations, counter.host_events),
                               Add(counter.inputs, counter.trace_records)),
                           counter.captures),
      .evidence_bytes = Add(counter.encoded_bytes, counter.capture_bytes),
      .retained_bytes = retained,
      .copied_bytes = Add(counter.copied_bytes, counter.capture_bytes),
      .physical_bytes = counter.physical_bytes,
      .allocated_bytes = counter.allocated_bytes,
      .reserved_bytes = counter.reserved_bytes,
      .storage_growths = counter.growths,
  };
}

[[nodiscard]] Semantics Public(const rund::replay::Record &record,
                               const Operation operation,
                               const rund::replay::Code code) noexcept {
  const rund::replay::StorageReport storage = record.storage_report();
  const rund::replay::DiagnosticReport capture = record.capture_report();
  Counters counters{
      .inputs = Count(record.input_count()),
      .observations = Count(record.observation_count()),
      .host_events = Count(record.host_event_count()),
      .trace_records = Count(record.trace_record_count()),
      .captures = Count(record.captures().size()),
      .capture_hash = record.capture_hash(),
      .logical_bytes = storage.logical_bytes,
      .encoded_bytes = storage.encoded_bytes,
      .retained_bytes = storage.retained_bytes,
      .copied_bytes = storage.copied_bytes,
      .cached_bytes = storage.cached_bytes,
      .physical_bytes = storage.physical_bytes,
      .allocated_bytes = storage.allocated_bytes,
      .reserved_bytes = storage.reserved_bytes,
      .growths = storage.growths,
      .chunks = storage.chunk_count,
      .segments = storage.segment_count,
      .cache_hits = storage.cache_hits,
      .cache_misses = storage.cache_misses,
      .cache_evictions = storage.cache_evictions,
      .capture_bytes = capture.retained_bytes,
      .capture_records = capture.retained_records,
      .capture_evicted = capture.evicted_records,
      .capture_dropped = capture.dropped_records,
  };
  ReplayFields replay = Formula(counters, operation);
  replay.result_hash = record.hash();
  return Semantics{
      .status = rund::replay::raw(code),
      .ordering = record.input_hash(),
      .transcript = record.transcript_hash(),
      .result = record.hash(),
      .counters = counters,
      .replay = replay,
  };
}

[[nodiscard]] Semantics Public(const rund::replay::Live &result) noexcept {
  Counters counters{
      .inputs = 1u,
      .observations = Count(result.observations().size()),
      .host_events = Count(result.events().size()),
      .trace_records = Count(result.trace().records.size()),
  };
  ReplayFields replay = Formula(counters, Operation::Live);
  replay.input_bytes = kPayloadBytes;
  replay.evidence_rows = Add(Add(counters.observations, counters.host_events),
                             counters.trace_records);
  return Semantics{
      .status = rund::replay::raw(result.code()),
      .counters = counters,
      .replay = replay,
  };
}

[[nodiscard]] bool
Observe(Sample &sample, const Observer &observer, const Setting setting,
        const Operation operation,
        const rund::telemetry::Preparation preparation) noexcept {
  const bool enabled = setting != Setting::Disabled;
  if (observer.count != (enabled ? 1u : 0u)) {
    return false;
  }
  if (!enabled) {
    return true;
  }

  const rund::telemetry::Event &event = observer.event;
  const rund::telemetry::Replay &replay = event.replay;
  const Level level = setting == Setting::Detail ? Level::Detail : Level::Basic;
  const ReplayFields observed{
      .input_rows = replay.input_rows,
      .input_bytes = replay.input_bytes,
      .produced_rows = replay.produced_rows,
      .choices = replay.choices,
      .evidence_rows = replay.evidence_rows,
      .evidence_bytes = replay.evidence_bytes,
      .retained_bytes = replay.retained_bytes,
      .copied_bytes = replay.copied_bytes,
      .physical_bytes = replay.physical_bytes,
      .allocated_bytes = replay.allocated_bytes,
      .reserved_bytes = replay.reserved_bytes,
      .storage_growths = replay.storage_growths,
      .result_hash = replay.result_hash,
  };
  if (event.source != rund::telemetry::Source::Replay || event.level != level ||
      event.session != kSession || event.scope == 0u ||
      replay.code != static_cast<rund::replay::Code>(sample.semantics.status) ||
      replay.mode != Mode(operation) || replay.plan != preparation ||
      observed != sample.semantics.replay ||
      (setting == Setting::Basic &&
       (event.detail.prepare_ns != 0u || event.detail.work_ns != 0u ||
        event.detail.finish_ns != 0u))) {
    return false;
  }
  sample.telemetry = {
      .source = event.source,
      .session = event.session,
      .scope = event.scope,
      .code = rund::replay::raw(replay.code),
      .mode = replay.mode,
      .plan = replay.plan,
      .replay = observed,
  };
  sample.phases = event.detail;
  return true;
}

[[nodiscard]] Sample
Scope(rund::Session &session, Observer &observer, const Setting setting,
      const Operation operation, const rund::telemetry::Preparation preparation,
      const rund::replay::Record *const expected,
      const std::span<const rund::replay::Choice> choices) {
  observer.reset();
  const auto payload = Payload();
  std::uint32_t producers = 0u;
  bool producer_ok = true;
  bool simulation_ok = false;
  rund::replay::Binding replay{};
  auto source = [&](rund::replay::Writer &writer) -> std::uint64_t {
    ++producers;
    producer_ok = writer.append(payload);
    return kSequence;
  };
  const auto commands = replay.input(kInput, source);
  auto simulate = [&](rund::replay::Context &context) {
    const auto value = commands.read(context);
    simulation_ok =
        producer_ok && value && value.size() == payload.size() &&
        std::equal(value.bytes().begin(), value.bytes().end(), payload.begin());
  };

  Sample sample{};
  switch (operation) {
  case Operation::Live: {
    const rund::replay::Live result = rund::replay::live(session, simulate);
    if (!result) {
      return {};
    }
    sample.semantics = Public(result);
    break;
  }
  case Operation::Record: {
    const rund::replay::Record result = rund::replay::record(session, simulate);
    if (!result) {
      return {};
    }
    sample.semantics = Public(result, operation, result.code());
    break;
  }
  case Operation::Replay: {
    if (expected == nullptr) {
      return {};
    }
    const rund::replay::Check result =
        rund::replay::run(session, *expected, simulate);
    if (!result || !result.actual()) {
      return {};
    }
    sample.semantics = Public(*result.actual(), operation, result.code());
    break;
  }
  case Operation::Scenario: {
    if (expected == nullptr) {
      return {};
    }
    const rund::replay::Scenario result =
        rund::replay::scenario(session, *expected, choices, simulate);
    if (!result || !result.matches() || !result.actual()) {
      return {};
    }
    sample.semantics = Public(*result.actual(), operation, result.code());
    break;
  }
  }

  const std::uint32_t expected_producers =
      operation == Operation::Live || operation == Operation::Record ? 1u : 0u;
  sample.ok = simulation_ok && producers == expected_producers &&
              Observe(sample, observer, setting, operation, preparation);
  return sample;
}

template <class Work> [[nodiscard]] Sample Measure(Work &&work) {
  std::uint64_t cpu_begin = 0u;
  std::uint64_t cpu_end = 0u;
  if (!CpuTime(cpu_begin)) {
    return {};
  }
  allocation::Start();
  const auto wall_begin = Clock::now();
  Sample sample = work();
  const auto wall_end = Clock::now();
  sample.allocations = allocation::Stop();
  if (!CpuTime(cpu_end) || cpu_end < cpu_begin || wall_end < wall_begin) {
    return {};
  }
  sample.wall = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end -
                                                           wall_begin)
          .count());
  sample.cpu = cpu_end - cpu_begin;
  const std::uint64_t phase =
      Add(Add(sample.phases.prepare_ns, sample.phases.work_ns),
          sample.phases.finish_ns);
  if (phase > sample.wall) {
    sample.ok = false;
  }
  return sample;
}

[[nodiscard]] std::optional<rund::replay::Record>
Load(const std::string_view encoded) {
  const auto decoded = rund::replay::Record::load(
      std::as_bytes(std::span{encoded.data(), encoded.size()}));
  if (!decoded) {
    return std::nullopt;
  }
  return *decoded;
}

[[nodiscard]] Sample Cold(const Setting setting, const Operation operation,
                          const std::string_view encoded,
                          const std::span<const rund::replay::Choice> choices,
                          const bool measured) {
  Observer observer{};
  std::optional<rund::replay::Record> expected{};
  if (NeedsExpected(operation)) {
    expected = Load(encoded);
    if (!expected) {
      return {};
    }
  }
  const auto work = [&]() {
    rund::Session session{};
    if (!session.open(Config(observer, setting))) {
      return Sample{};
    }
    Sample sample = Scope(session, observer, setting, operation,
                          ExpectedPreparation(operation, false),
                          expected ? &*expected : nullptr, choices);
    const auto closed = session.close();
    if (!closed) {
      sample.ok = false;
    }
    return sample;
  };
  return measured ? Measure(work) : work();
}

[[nodiscard]] Sample Warm(Lane &lane, const Setting setting,
                          const Operation operation,
                          const std::span<const rund::replay::Choice> choices,
                          const bool expected_prepared, const bool measured) {
  const auto work = [&]() {
    return Scope(lane.session, lane.observer, setting, operation,
                 ExpectedPreparation(operation, expected_prepared),
                 lane.expected ? &*lane.expected : nullptr, choices);
  };
  return measured ? Measure(work) : work();
}

[[nodiscard]] std::optional<std::string> Seed() {
  Observer observer{};
  rund::Session session{};
  if (!session.open(Config(observer, Setting::Disabled))) {
    return std::nullopt;
  }
  const auto payload = Payload();
  bool simulation_ok = false;
  rund::replay::Binding replay{};
  auto source = [&](rund::replay::Writer &writer) -> std::uint64_t {
    simulation_ok = writer.append(payload);
    return kSequence;
  };
  const auto commands = replay.input(kInput, source);
  auto simulate = [&](rund::replay::Context &context) {
    const auto value = commands.read(context);
    simulation_ok = simulation_ok && value && value.size() == payload.size();
  };
  const rund::replay::Record record = rund::replay::record(session, simulate);
  const auto closed = session.close();
  if (!record || !simulation_ok || !closed) {
    return std::nullopt;
  }
  std::string artifact{};
  const rund::replay::Save saved =
      record.save([&artifact](const std::span<const std::byte> bytes) noexcept {
        try {
          artifact.append(reinterpret_cast<const char *>(bytes.data()),
                          bytes.size());
          return true;
        } catch (...) {
          return false;
        }
      });
  return saved ? std::optional<std::string>{std::move(artifact)} : std::nullopt;
}


} // namespace rund::measure::telemetry

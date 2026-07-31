#include <node/runtime/replay/record.hpp>

#include <node/runtime/replay/hash.hpp>
#include <node/runtime/replay/host/payload.hpp>

#include "factory.hpp"

#include <cstddef>
#include <utility>

namespace rund::node {

RuntimeReplayRecord::RuntimeReplayRecord(RuntimeReplayRecord &&other) noexcept
    : code(other.code), tasks(other.tasks),
      observations(std::move(other.observations)), host(std::move(other.host)),
      trace(std::move(other.trace)), start_hash(other.start_hash),
      semantic_hash(other.semantic_hash), operation_hash(other.operation_hash),
      observation_hash(other.observation_hash),
      host_event_hash(other.host_event_hash), input_count(other.input_count),
      input_hash(other.input_hash), transcript_hash(other.transcript_hash),
      trace_hash(other.trace_hash), replay_hash(other.replay_hash) {
  reset_moved_from(other);
}

RuntimeReplayRecord::RuntimeReplayRecord(RuntimeReplayRecordDesc desc,
                                         const std::uint64_t trusted_trace_hash)
    : code(desc.code), tasks(desc.tasks),
      observations(std::move(desc.observations)),
      host{.events = std::move(desc.host_events),
           .payload_archive = std::move(desc.host_payload_archive)},
      trace(std::move(desc.trace)), start_hash(desc.start_hash),
      trace_hash(trusted_trace_hash) {
  host.event_hash = ::rund::host::hash_events(host.events).value;
  observation_hash = replay_detail::HashObservations(observations);
  host_event_hash = host.event_hash;
  const replay_detail::InputTranscriptIdentity inputs =
      replay_detail::HashInputs(host.payload_archive);
  input_count = inputs.count;
  input_hash = inputs.hash;
  transcript_hash = host.payload_archive.payload_hash;
  operation_hash = host.events.empty()
                       ? tasks.trace_hash()
                       : replay_detail::HashReplaySubstitutableOperation(
                             tasks, observation_hash);
  semantic_hash = replay_detail::HashSemantic(code, tasks, observation_hash);
  replay_hash = replay_detail::HashReplay(*this);
}

RuntimeReplayRecord &
RuntimeReplayRecord::operator=(RuntimeReplayRecord &&other) noexcept {
  if (this != &other) {
    code = other.code;
    tasks = other.tasks;
    observations = std::move(other.observations);
    host = std::move(other.host);
    trace = std::move(other.trace);
    start_hash = other.start_hash;
    semantic_hash = other.semantic_hash;
    operation_hash = other.operation_hash;
    observation_hash = other.observation_hash;
    host_event_hash = other.host_event_hash;
    input_count = other.input_count;
    input_hash = other.input_hash;
    transcript_hash = other.transcript_hash;
    trace_hash = other.trace_hash;
    replay_hash = other.replay_hash;
    reset_moved_from(other);
  }
  return *this;
}

void RuntimeReplayRecord::reset_moved_from(
    RuntimeReplayRecord &record) noexcept {
  record.code = ::rund::replay::Code::SessionResultMissing;
  record.tasks = task::Stats{};
  record.observations.clear();
  record.host = HostReplayEvidence{};
  record.trace =
      ::rund::Trace{.code = ::rund::ReasonCode::SessionResultMissing};
  record.start_hash = 0u;
  record.semantic_hash = 0u;
  record.operation_hash = 0u;
  record.observation_hash = 0u;
  record.host_event_hash = 0u;
  record.input_count = 0u;
  record.input_hash = 0u;
  record.transcript_hash = 0u;
  record.trace_hash = 0u;
  record.replay_hash = 0u;
}

namespace replay_detail {

RuntimeReplayRecord
make_runtime_replay_record(RuntimeReplayRecordDesc desc,
                           const std::uint64_t trusted_trace_hash) {
  return RuntimeReplayRecord{std::move(desc), trusted_trace_hash};
}

} // namespace replay_detail

RuntimeReplayRecord make_runtime_replay_record(RuntimeReplayRecordDesc desc) {
  const std::uint64_t trace_hash = replay_detail::HashTrace(desc.trace);
  return RuntimeReplayRecord{std::move(desc), trace_hash};
}

namespace {

[[nodiscard]] bool ValidTrace(const ::rund::Trace &trace) noexcept {
  if (!::rund::ValidReasonCode(trace.code)) {
    return false;
  }
  for (const ::rund::TraceRecord &record : trace.records) {
    if (!record.code.valid() ||
        !::rund::ValidReasonCode(record.snapshot.code)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool valid_runtime_replay_record(const RuntimeReplayRecord &record) {
  const std::uint64_t observation_hash =
      replay_detail::HashObservations(record.observations);
  const std::uint64_t event_hash =
      ::rund::host::hash_events(record.host.events).value;
  const std::uint64_t operation_hash =
      record.host.events.empty()
          ? record.tasks.trace_hash()
          : replay_detail::HashReplaySubstitutableOperation(record.tasks,
                                                            observation_hash);
  const replay_detail::InputTranscriptIdentity inputs =
      replay_detail::HashInputs(record.host.payload_archive);
  return record.start_hash != 0u && ValidTrace(record.trace) &&
         observation_hash == record.observation_hash &&
         event_hash == record.host.event_hash &&
         event_hash == record.host_event_hash &&
         inputs.count == record.input_count &&
         inputs.hash == record.input_hash &&
         record.host.payload_archive.payload_hash == record.transcript_hash &&
         replay_detail::ValidArchive(record.host.payload_archive) &&
         replay_detail::BindPayloads(record.host.events,
                                     record.host.payload_archive) ==
             ::rund::replay::Code::Ok &&
         ::rund::replay::valid(record.code) &&
         replay_detail::HashSemantic(record.code, record.tasks,
                                     observation_hash) ==
             record.semantic_hash &&
         operation_hash == record.operation_hash &&
         replay_detail::HashTrace(record.trace) == record.trace_hash &&
         replay_detail::HashReplay(record) == record.replay_hash;
}

RuntimeReplayCheck
check_runtime_replay(const RuntimeReplayRecord &expected,
                     const RuntimeReplayRecord &actual) noexcept {
  if (expected.start_hash != actual.start_hash) {
    return RuntimeReplayCheck{.code = ::rund::replay::Code::RecordStartMismatch,
                              .expected_hash = expected.start_hash,
                              .actual_hash = actual.start_hash};
  }
  if (expected.code != actual.code) {
    return RuntimeReplayCheck{.code = ::rund::replay::Code::OutcomeMismatch,
                              .expected_hash = expected.semantic_hash,
                              .actual_hash = actual.semantic_hash};
  }
  if (expected.operation_hash != actual.operation_hash) {
    return RuntimeReplayCheck{.code =
                                  ::rund::replay::Code::OperationHashMismatch,
                              .expected_hash = expected.operation_hash,
                              .actual_hash = actual.operation_hash};
  }
  if (expected.observation_hash != actual.observation_hash) {
    return RuntimeReplayCheck{.code =
                                  ::rund::replay::Code::ObservationHashMismatch,
                              .expected_hash = expected.observation_hash,
                              .actual_hash = actual.observation_hash};
  }
  if (expected.host_event_hash != actual.host_event_hash) {
    return RuntimeReplayCheck{.code =
                                  ::rund::replay::Code::HostEventHashMismatch,
                              .expected_hash = expected.host_event_hash,
                              .actual_hash = actual.host_event_hash};
  }
  if (expected.input_count != actual.input_count ||
      expected.input_hash != actual.input_hash) {
    return RuntimeReplayCheck{.code = ::rund::replay::Code::InputHashMismatch,
                              .expected_hash = expected.input_hash,
                              .actual_hash = actual.input_hash};
  }
  if (expected.transcript_hash != actual.transcript_hash) {
    return RuntimeReplayCheck{.code =
                                  ::rund::replay::Code::TranscriptHashMismatch,
                              .expected_hash = expected.transcript_hash,
                              .actual_hash = actual.transcript_hash};
  }
  if (expected.trace_hash != actual.trace_hash) {
    return RuntimeReplayCheck{.code = ::rund::replay::Code::TraceHashMismatch,
                              .expected_hash = expected.trace_hash,
                              .actual_hash = actual.trace_hash};
  }
  if (expected.replay_hash != actual.replay_hash) {
    return RuntimeReplayCheck{.code = ::rund::replay::Code::HashMismatch,
                              .expected_hash = expected.replay_hash,
                              .actual_hash = actual.replay_hash};
  }
  return RuntimeReplayCheck{.code = ::rund::replay::Code::Ok,
                            .expected_hash = expected.replay_hash,
                            .actual_hash = actual.replay_hash};
}

} // namespace rund::node

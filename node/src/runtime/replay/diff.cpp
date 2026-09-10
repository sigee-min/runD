#include <node/runtime/replay/diff.hpp>

#include "diff/local.hpp"

#include <node/runtime/replay/hash.hpp>

namespace rund::node {

RuntimeReplayDiff DiffRuntimeReplayRecords(const RuntimeReplayRecord &expected,
                                           const RuntimeReplayRecord &actual) {
  using replay_diff_detail::add_mismatch;
  RuntimeReplayDiff diff{};
  if (expected.code != actual.code) {
    add_mismatch(diff.mismatches, "code", ::rund::replay::raw(expected.code),
                 ::rund::replay::raw(actual.code));
  }
  if (expected.semantic_hash != actual.semantic_hash) {
    add_mismatch(diff.mismatches, "semantic_hash", expected.semantic_hash,
                 actual.semantic_hash, ::rund::replay::Code::HashMismatch);
  }
  if (expected.operation_hash != actual.operation_hash) {
    add_mismatch(diff.mismatches, "operation_hash", expected.operation_hash,
                 actual.operation_hash,
                 ::rund::replay::Code::OperationHashMismatch);
  }
  if (expected.observation_hash != actual.observation_hash) {
    add_mismatch(diff.mismatches, "observation_hash", expected.observation_hash,
                 actual.observation_hash,
                 ::rund::replay::Code::ObservationHashMismatch);
  }
  if (expected.host_event_hash != actual.host_event_hash) {
    add_mismatch(diff.mismatches, "host_event_hash", expected.host_event_hash,
                 actual.host_event_hash,
                 ::rund::replay::Code::HostEventHashMismatch);
  }
  if (expected.input_count != actual.input_count) {
    add_mismatch(diff.mismatches, "input_count", expected.input_count,
                 actual.input_count);
  }
  if (expected.input_hash != actual.input_hash) {
    add_mismatch(diff.mismatches, "input_hash", expected.input_hash,
                 actual.input_hash, ::rund::replay::Code::InputHashMismatch);
  }
  if (expected.transcript_hash != actual.transcript_hash) {
    add_mismatch(diff.mismatches, "transcript_hash", expected.transcript_hash,
                 actual.transcript_hash,
                 ::rund::replay::Code::TranscriptHashMismatch);
  }
  if (expected.trace_hash != actual.trace_hash) {
    add_mismatch(diff.mismatches, "trace_hash", expected.trace_hash,
                 actual.trace_hash, ::rund::replay::Code::TraceHashMismatch);
  }
  if (expected.replay_hash != actual.replay_hash) {
    add_mismatch(diff.mismatches, "replay_hash", expected.replay_hash,
                 actual.replay_hash, ::rund::replay::Code::HashMismatch);
  }
  if (expected.observations.size() != actual.observations.size()) {
    add_mismatch(diff.mismatches, "observation.count",
                 static_cast<std::uint64_t>(expected.observations.size()),
                 static_cast<std::uint64_t>(actual.observations.size()));
  } else {
    std::size_t observation_index = 0u;
    if (FindFirstObservationMismatch(expected.observations, actual.observations,
                                     observation_index)) {
      add_mismatch(diff.mismatches, "observation.detail",
                   replay_detail::HashObservation(
                       expected.observations[observation_index]),
                   replay_detail::HashObservation(
                       actual.observations[observation_index]),
                   ::rund::replay::Code::CodecObservationDetailMismatch);
    }
  }
  const replay_detail::HostReplayFieldDiff host_diff =
      replay_detail::DiffHostReplayEvidence(expected.host, actual.host);
  if (host_diff.mismatch) {
    add_mismatch(diff.mismatches, host_diff.field, host_diff.expected,
                 host_diff.actual,
                 ::rund::replay::Code::CodecHostDetailMismatch);
  }
  replay_diff_detail::add_host_payload_diff(diff.mismatches,
                                            expected.host.payload_archive,
                                            actual.host.payload_archive);
  if (expected.trace.records.size() != actual.trace.records.size()) {
    add_mismatch(diff.mismatches, "trace.record.count",
                 static_cast<std::uint64_t>(expected.trace.records.size()),
                 static_cast<std::uint64_t>(actual.trace.records.size()));
  } else {
    std::size_t trace_record_index = 0u;
    if (FindFirstTraceRecordMismatch(
            expected.trace.records, actual.trace.records, trace_record_index)) {
      add_mismatch(diff.mismatches, "trace.record.detail",
                   replay_detail::HashTraceRecord(
                       expected.trace.records[trace_record_index]),
                   replay_detail::HashTraceRecord(
                       actual.trace.records[trace_record_index]),
                   ::rund::replay::Code::CodecTraceDetailMismatch);
    }
  }
  if (diff.mismatches.empty()) {
    diff.code = ::rund::replay::Code::Ok;
    return diff;
  }
  diff.code = diff.mismatches.front().code;
  return diff;
}

} // namespace rund::node

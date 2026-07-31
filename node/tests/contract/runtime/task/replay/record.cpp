#include "test/assert.hpp"
#include <rund/task/api.hpp>

#include "local.hpp"
#include "src/runtime/session/result.hpp"

#include <node/runtime/replay/code.hpp>
#include <rund/replay.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using InternalReplayRecord = rund::node::RuntimeReplayRecord;
using ReplayRecordDesc = rund::node::RuntimeReplayRecordDesc;

static_assert(std::is_same_v<decltype(std::declval<ReplayRecordDesc>().tasks),
                             const rund::task::Stats &>);
static_assert(sizeof(rund::TraceCode) == 4u);
static_assert(std::is_trivially_copyable_v<rund::TraceCode>);
static_assert(!std::is_constructible_v<rund::TraceCode, rund::TraceDomain,
                                       std::uint16_t>);

void ExpectMovedFrom(const InternalReplayRecord &record) {
  TEST_ASSERT(!record.ok());
  TEST_ASSERT(record.code == rund::replay::Code::SessionResultMissing);
  TEST_ASSERT(record.observations.empty());
  TEST_ASSERT(record.host.events.empty());
  TEST_ASSERT(record.host.event_hash == 0u);
  TEST_ASSERT(record.host.payload_archive.records.empty());
  TEST_ASSERT(record.host.payload_archive.chunks.empty());
  TEST_ASSERT(record.host.payload_archive.payload_hash == 0u);
  TEST_ASSERT(record.trace.records.empty());
  TEST_ASSERT(record.trace.dropped == 0u);
  TEST_ASSERT(record.trace.code == rund::ReasonCode::SessionResultMissing);
  TEST_ASSERT(record.start_hash == 0u);
  TEST_ASSERT(record.semantic_hash == 0u);
  TEST_ASSERT(record.operation_hash == 0u);
  TEST_ASSERT(record.observation_hash == 0u);
  TEST_ASSERT(record.host_event_hash == 0u);
  TEST_ASSERT(record.transcript_hash == 0u);
  TEST_ASSERT(record.trace_hash == 0u);
  TEST_ASSERT(record.replay_hash == 0u);
}

void AddPayloadTransferEvidence(InternalReplayRecord &record) {
  rund::node::replay_detail::payload::ArchiveRecord payload_record{};
  payload_record.metadata.event_sequence = 91u;
  payload_record.pieces.push_back(
      rund::node::replay_detail::payload::ArchivePiece{
          .chunk_id = 7u, .offset = 2u, .size = 3u});
  record.host.payload_archive.records.push_back(std::move(payload_record));

  rund::node::replay_detail::payload::ArchiveChunk chunk{};
  chunk.chunk_id = 7u;
  chunk.encoded =
      rund::node::replay_detail::payload::Bytes::freeze(std::vector<std::byte>{
          std::byte{0x11}, std::byte{0x22}, std::byte{0x33}});
  record.host.payload_archive.chunks.push_back(std::move(chunk));
}

[[nodiscard]] InternalReplayRecord WithCode(const InternalReplayRecord &record,
                                            const rund::replay::Code code) {
  InternalReplayRecord source = CloneReplayRecord(record);
  return rund::node::make_runtime_replay_record(ReplayRecordDesc{
      .start_hash = source.start_hash,
      .code = code,
      .tasks = source.tasks,
      .observations = std::move(source.observations),
      .host_events = std::move(source.host.events),
      .host_payload_archive = std::move(source.host.payload_archive),
      .trace = std::move(source.trace),
  });
}

} // namespace

int MakeRuntimeReplayFixture(RuntimeReplayFixture &fixture) {
  fixture.value = 0u;
  fixture.spec = rund::SessionConfig{
      .workers = 2u,
      .scheduler =
          {
              .task_workers = 2u,
              .task_capacity = 8u,
              .ready_queue_capacity = 8u,
          },
  };
  rund::Session::Result report = rund::run(fixture.spec, [&] {
    const rund::task::Handle task = rund::task::spawn(
        "replay-record", ReplaySleepThen(std::chrono::nanoseconds{1},
                                         [&] { fixture.value = 17u; }));
    const rund::task::Status joined = rund::task::join(task);
    if (!joined) {
      fixture.value = 0u;
    }
  });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(fixture.value == 17u);

  fixture.record = rund::node::make_runtime_replay_record(
      rund::node::RuntimeReplayRecordDesc{
          .code = rund::node::replay_detail::code(report.code()),
          .tasks = report.tasks(),
          .observations =
              rund::detail::session::ResultAccess::take_observations(report),
          .host_events =
              rund::detail::session::ResultAccess::take_events(report),
          .host_payload_archive =
              rund::detail::session::ResultAccess::take_payloads(report),
          .trace = rund::detail::session::ResultAccess::take_trace(report),
      });
  TEST_ASSERT(fixture.record.ok());
  fixture.encoded = SaveReplayRecord(fixture.record);
  return 0;
}

int CheckReplayRecordContract(const RuntimeReplayFixture &fixture) {
  static_assert(!std::is_copy_constructible_v<InternalReplayRecord>);
  static_assert(!std::is_copy_assignable_v<InternalReplayRecord>);
  static_assert(std::is_nothrow_move_constructible_v<InternalReplayRecord>);
  static_assert(std::is_nothrow_move_assignable_v<InternalReplayRecord>);

  const rund::TraceCode runtime_code =
      rund::TraceCode::runtime(rund::ReasonCode::RuntimeScopeBusy);
  TEST_ASSERT(runtime_code.valid());
  TEST_ASSERT(runtime_code.runtime_code() ==
              rund::ReasonCode::RuntimeScopeBusy);
  TEST_ASSERT(!runtime_code.compute_reason().has_value());
  TEST_ASSERT(runtime_code.error() == "runtime_scope_busy");
  const rund::TraceCode compute_code =
      rund::TraceCode::compute(rund::compute::Reason::ScanSumOverflow);
  TEST_ASSERT(compute_code.valid());
  TEST_ASSERT(compute_code.compute_reason() ==
              rund::compute::Reason::ScanSumOverflow);
  TEST_ASSERT(!compute_code.runtime_code().has_value());
  TEST_ASSERT(compute_code.error() == "compute_scan_sum_overflow");

  TEST_ASSERT(fixture.record.operation_hash ==
              rund::node::replay_detail::HashReplaySubstitutableOperation(
                  fixture.record.tasks, fixture.record.observation_hash));
  TEST_ASSERT(fixture.record.start_hash != 0u);
  TEST_ASSERT(fixture.record.host_event_hash ==
              rund::host::hash_events(fixture.record.host.events).value);
  TEST_ASSERT(fixture.record.semantic_hash != 0u);
  TEST_ASSERT(fixture.record.replay_hash != 0u);
  TEST_ASSERT(fixture.record.ok());
  TEST_ASSERT(fixture.record.code == rund::replay::Code::Ok);
  TEST_ASSERT(!fixture.encoded.empty());

  const rund::node::RuntimeReplayRecord failed_outcome =
      WithCode(fixture.record, rund::replay::Code::RuntimeScopeCallbackFailed);
  TEST_ASSERT(!failed_outcome.ok());
  TEST_ASSERT(failed_outcome.code ==
              rund::replay::Code::RuntimeScopeCallbackFailed);
  TEST_ASSERT(failed_outcome.semantic_hash != fixture.record.semantic_hash);
  const std::vector<std::byte> failed_encoded =
      SaveReplayRecord(failed_outcome);
  const rund::node::RuntimeReplayDecodeResult failed_decoded =
      rund::node::DecodeRuntimeReplayRecord(failed_encoded);
  TEST_ASSERT(failed_decoded.ok());
  TEST_ASSERT(failed_decoded.record.code ==
              rund::replay::Code::RuntimeScopeCallbackFailed);
  TEST_ASSERT(
      !rund::node::check_runtime_replay(fixture.record, failed_decoded.record));

  const rund::node::RuntimeReplayCheck replay =
      rund::node::check_runtime_replay(fixture.record,
                                       CloneReplayRecord(fixture.record));
  TEST_ASSERT(replay.ok());

  TEST_ASSERT(fixture.encoded.size() >= 8u);
  TEST_ASSERT(fixture.encoded[0] == std::byte{'r'});
  TEST_ASSERT(fixture.encoded[1] == std::byte{'u'});
  TEST_ASSERT(fixture.encoded[2] == std::byte{'n'});
  TEST_ASSERT(fixture.encoded[3] == std::byte{'D'});
  TEST_ASSERT(fixture.encoded[4] == std::byte{0x1a});
  TEST_ASSERT(fixture.encoded[5] == std::byte{0x0a});
  TEST_ASSERT(fixture.encoded[6] == std::byte{1u});
  TEST_ASSERT(fixture.encoded[7] == std::byte{1u});
  const std::vector<std::byte> host_encoded =
      rund::node::EncodeHostReplayEvents(fixture.record.host.events);
  std::vector<rund::host::Event> host_decoded{};
  TEST_ASSERT(rund::node::DecodeHostReplayEvents(host_encoded, host_decoded));
  TEST_ASSERT(rund::host::hash_events(host_decoded).value ==
              fixture.record.host_event_hash);
  rund::node::RuntimeReplayDecodeResult decoded =
      rund::node::DecodeRuntimeReplayRecord(fixture.encoded);
  TEST_ASSERT(decoded.ok());
  TEST_ASSERT(decoded.record.observations.size() ==
              fixture.record.observations.size());
  TEST_ASSERT(decoded.record.trace.records.size() ==
              fixture.record.trace.records.size());
  TEST_ASSERT(decoded.record.tasks.spawned() == fixture.record.tasks.spawned());
  TEST_ASSERT(decoded.record.tasks.observations() ==
              fixture.record.tasks.observations());
  TEST_ASSERT(decoded.record.tasks.trace_hash() ==
              fixture.record.tasks.trace_hash());
  TEST_ASSERT(decoded.record.trace.code == fixture.record.trace.code);
  for (const rund::TraceRecord &record : decoded.record.trace.records) {
    TEST_ASSERT(record.code.valid());
    TEST_ASSERT(rund::ValidReasonCode(record.snapshot.code));
  }
  const rund::node::RuntimeReplayCheck decoded_replay =
      rund::node::check_runtime_replay(fixture.record, decoded.record);
  TEST_ASSERT(decoded_replay.ok());
  const auto *const moved_observations = decoded.record.observations.data();
  const auto *const moved_host_events = decoded.record.host.events.data();
  const auto *const moved_trace_records = decoded.record.trace.records.data();
  const std::size_t moved_observation_capacity =
      decoded.record.observations.capacity();
  const std::size_t moved_host_event_capacity =
      decoded.record.host.events.capacity();
  const std::size_t moved_trace_capacity =
      decoded.record.trace.records.capacity();
  rund::node::RuntimeReplayRecord moved = std::move(decoded.record);
  TEST_ASSERT(moved.observations.data() == moved_observations);
  TEST_ASSERT(moved.host.events.data() == moved_host_events);
  TEST_ASSERT(moved.trace.records.data() == moved_trace_records);
  TEST_ASSERT(moved.observations.capacity() == moved_observation_capacity);
  TEST_ASSERT(moved.host.events.capacity() == moved_host_event_capacity);
  TEST_ASSERT(moved.trace.records.capacity() == moved_trace_capacity);
  ExpectMovedFrom(decoded.record);
  const rund::node::RuntimeReplayCheck moved_replay =
      rund::node::check_runtime_replay(fixture.record, moved);
  TEST_ASSERT(moved_replay.ok());
  TEST_ASSERT(SaveReplayRecord(moved) == fixture.encoded);
  rund::node::RuntimeReplayDecodeResult assignment_decoded =
      rund::node::DecodeRuntimeReplayRecord(fixture.encoded);
  TEST_ASSERT(assignment_decoded.ok());
  InternalReplayRecord assignment_source = std::move(assignment_decoded.record);
  AddPayloadTransferEvidence(assignment_source);
  const auto *const assigned_observations =
      assignment_source.observations.data();
  const auto *const assigned_host_events = assignment_source.host.events.data();
  const auto *const assigned_payload_records =
      assignment_source.host.payload_archive.records.data();
  const auto *const assigned_payload_pieces =
      assignment_source.host.payload_archive.records.back().pieces.data();
  const auto *const assigned_payload_chunks =
      assignment_source.host.payload_archive.chunks.data();
  const auto *const assigned_payload_bytes =
      assignment_source.host.payload_archive.chunks.back().encoded.data();
  const auto *const assigned_trace_records =
      assignment_source.trace.records.data();

  InternalReplayRecord assigned{};
  assigned = std::move(assignment_source);
  TEST_ASSERT(assigned.observations.data() == assigned_observations);
  TEST_ASSERT(assigned.host.events.data() == assigned_host_events);
  TEST_ASSERT(assigned.host.payload_archive.records.data() ==
              assigned_payload_records);
  TEST_ASSERT(assigned.host.payload_archive.records.back().pieces.data() ==
              assigned_payload_pieces);
  TEST_ASSERT(assigned.host.payload_archive.chunks.data() ==
              assigned_payload_chunks);
  TEST_ASSERT(assigned.host.payload_archive.chunks.back().encoded.data() ==
              assigned_payload_bytes);
  TEST_ASSERT(assigned.trace.records.data() == assigned_trace_records);
  ExpectMovedFrom(assignment_source);
  return 0;
}

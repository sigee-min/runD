#include "test/assert.hpp"

#include "local.hpp"
#include "src/runtime/replay/host/payload/materialize.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(
    std::is_same_v<decltype(rund::replay::Mismatch{}.field), std::string_view>);
static_assert(noexcept(
    std::declval<const rund::replay::Diff &>().mismatch(std::size_t{0u})));
static_assert(
    std::is_same_v<decltype(rund::replay::Trace{}.code), rund::TraceCode>);
static_assert(std::is_same_v<decltype(rund::replay::Trace{}.snapshot_code),
                             rund::ReasonCode>);
static_assert(
    noexcept(std::declval<const rund::replay::Window &>().expected_trace()));

int CheckReplayDiffContract(const RuntimeReplayFixture &fixture) {
  std::vector<rund::task::Observation> expected_observations{
      rund::task::Observation{.sequence = 1u,
                              .kind = rund::task::ObservationKind::TimerDue,
                              .task_id = 7u,
                              .wait_id = 9u,
                              .fd = -1,
                              .interest = 0,
                              .revents = 0,
                              .deadline_ns = 11,
                              .reason_code = rund::ReasonCode::Ok}};
  std::vector<rund::task::Observation> actual_observations =
      expected_observations;
  actual_observations[0].kind = rund::task::ObservationKind::IoReady;
  const rund::node::RuntimeReplayRecord expected_detail_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .observations = expected_observations,
          });
  const rund::node::RuntimeReplayRecord actual_detail_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .observations = actual_observations,
          });
  const std::vector<std::byte> expected_detail_encoded =
      SaveReplayRecord(expected_detail_record);
  const rund::node::RuntimeReplayDecodeResult expected_detail_decoded =
      rund::node::DecodeRuntimeReplayRecord(expected_detail_encoded);
  TEST_ASSERT(expected_detail_decoded.ok());
  TEST_ASSERT(expected_detail_decoded.record.observations.size() == 1u);
  TEST_ASSERT(expected_detail_decoded.record.observations[0].kind ==
              rund::task::ObservationKind::TimerDue);
  const rund::node::RuntimeReplayDiff detail_diff =
      rund::node::DiffRuntimeReplayRecords(expected_detail_record,
                                           actual_detail_record);
  TEST_ASSERT(!detail_diff.ok());
  bool saw_observation_detail_mismatch = false;
  for (const rund::node::RuntimeReplayFieldMismatch &mismatch :
       detail_diff.mismatches) {
    if (std::string_view{mismatch.field} == "observation.detail") {
      saw_observation_detail_mismatch = true;
    }
  }
  TEST_ASSERT(saw_observation_detail_mismatch);
  const rund::node::RuntimeReplayMismatchWindow minimized =
      rund::node::MinimizeRuntimeReplayMismatch(expected_detail_record,
                                                actual_detail_record, 1u);
  TEST_ASSERT(!minimized.ok());
  TEST_ASSERT(minimized.code ==
              rund::replay::Code::CodecObservationDetailMismatch);
  TEST_ASSERT(minimized.has_observation_mismatch);
  TEST_ASSERT(minimized.observation_index == 0u);
  TEST_ASSERT(minimized.expected_observations.size() == 1u);
  TEST_ASSERT(minimized.actual_observations.size() == 1u);
  const rund::node::RuntimeReplayMismatchWindow minimized_huge_observation =
      rund::node::MinimizeRuntimeReplayMismatch(
          expected_detail_record, actual_detail_record,
          std::numeric_limits<std::size_t>::max());
  TEST_ASSERT(!minimized_huge_observation.ok());
  TEST_ASSERT(minimized_huge_observation.has_observation_mismatch);
  TEST_ASSERT(minimized_huge_observation.expected_observations.size() ==
              expected_observations.size());
  TEST_ASSERT(minimized_huge_observation.actual_observations.size() ==
              actual_observations.size());

  std::vector<rund::host::Event> expected_host_events{
      rund::host::Event{.sequence = 1u,
                        .kind = rund::host::EventKind::TimerSleep,
                        .status = rund::host::Status::Ok,
                        .task_id = 7u,
                        .logical_time_ns = 11u},
      rund::host::Event{.sequence = 2u,
                        .kind = rund::host::EventKind::IoRead,
                        .status = rund::host::Status::Ok,
                        .task_id = 8u,
                        .logical_time_ns = 12u},
      rund::host::Event{.sequence = 3u,
                        .kind = rund::host::EventKind::IoClose,
                        .status = rund::host::Status::SyscallFailed,
                        .task_id = 9u,
                        .logical_time_ns = 13u}};
  std::vector<rund::host::Event> actual_host_events = expected_host_events;
  actual_host_events[1].logical_time_ns = 99u;
  const rund::node::RuntimeReplayRecord expected_host_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .host_events = expected_host_events,
          });
  const rund::node::RuntimeReplayRecord actual_host_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .host_events = actual_host_events,
          });
  const rund::node::RuntimeReplayMismatchWindow minimized_host =
      rund::node::MinimizeRuntimeReplayMismatch(expected_host_record,
                                                actual_host_record, 1u);
  TEST_ASSERT(!minimized_host.ok());
  TEST_ASSERT(minimized_host.code ==
              rund::replay::Code::CodecHostDetailMismatch);
  TEST_ASSERT(minimized_host.has_host_mismatch);
  TEST_ASSERT(minimized_host.host_event_index == 1u);
  TEST_ASSERT(minimized_host.expected_host_events.size() == 3u);
  TEST_ASSERT(minimized_host.actual_host_events.size() == 3u);
  const rund::node::RuntimeReplayMismatchWindow minimized_huge_host =
      rund::node::MinimizeRuntimeReplayMismatch(
          expected_host_record, actual_host_record,
          std::numeric_limits<std::size_t>::max());
  TEST_ASSERT(!minimized_huge_host.ok());
  TEST_ASSERT(minimized_huge_host.has_host_mismatch);
  TEST_ASSERT(minimized_huge_host.host_event_index == 1u);
  TEST_ASSERT(minimized_huge_host.expected_host_events.size() ==
              expected_host_events.size());
  TEST_ASSERT(minimized_huge_host.actual_host_events.size() ==
              actual_host_events.size());

  rund::Trace expected_trace{};
  expected_trace.records.push_back(rund::TraceRecord{
      .event = rund::TraceEvent::ComputeCompleted,
      .code = rund::TraceCode::compute(rund::compute::Reason::ScanSumOverflow),
      .snapshot =
          rund::TraceRecord::State{.code = rund::ReasonCode::RuntimeScopeBusy},
      .sequence = 1u});
  rund::Trace actual_trace = expected_trace;
  actual_trace.records[0].sequence = 2u;
  const rund::node::RuntimeReplayRecord expected_trace_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .trace = expected_trace,
          });
  const rund::node::RuntimeReplayRecord actual_trace_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .trace = actual_trace,
          });
  const rund::node::RuntimeReplayDiff trace_diff =
      rund::node::DiffRuntimeReplayRecords(expected_trace_record,
                                           actual_trace_record);
  TEST_ASSERT(!trace_diff.ok());
  bool saw_trace_detail_mismatch = false;
  for (const rund::node::RuntimeReplayFieldMismatch &mismatch :
       trace_diff.mismatches) {
    if (std::string_view{mismatch.field} == "trace.record.detail") {
      saw_trace_detail_mismatch = true;
    }
  }
  TEST_ASSERT(saw_trace_detail_mismatch);
  const rund::node::RuntimeReplayMismatchWindow minimized_trace =
      rund::node::MinimizeRuntimeReplayMismatch(expected_trace_record,
                                                actual_trace_record, 1u);
  TEST_ASSERT(!minimized_trace.ok());
  TEST_ASSERT(minimized_trace.code ==
              rund::replay::Code::CodecTraceDetailMismatch);
  TEST_ASSERT(minimized_trace.has_trace_mismatch);
  TEST_ASSERT(minimized_trace.trace_record_index == 0u);
  TEST_ASSERT(minimized_trace.expected_trace_records.size() == 1u);
  TEST_ASSERT(minimized_trace.actual_trace_records.size() == 1u);
  const rund::node::RuntimeReplayMismatchWindow minimized_huge_trace =
      rund::node::MinimizeRuntimeReplayMismatch(
          expected_trace_record, actual_trace_record,
          std::numeric_limits<std::size_t>::max());
  TEST_ASSERT(!minimized_huge_trace.ok());
  TEST_ASSERT(minimized_huge_trace.has_trace_mismatch);
  TEST_ASSERT(minimized_huge_trace.expected_trace_records.size() ==
              expected_trace.records.size());
  TEST_ASSERT(minimized_huge_trace.actual_trace_records.size() ==
              actual_trace.records.size());
  const auto expected_trace_public = rund::replay::Record::load(
      ReplayArtifact(SaveReplayRecord(expected_trace_record)));
  const auto actual_trace_public = rund::replay::Record::load(
      ReplayArtifact(SaveReplayRecord(actual_trace_record)));
  TEST_ASSERT(expected_trace_public);
  TEST_ASSERT(actual_trace_public);
  const rund::replay::Window trace_window =
      rund::replay::window(*expected_trace_public, *actual_trace_public, 1u);
  TEST_ASSERT(trace_window.trace_record_index().has_value());
  TEST_ASSERT(trace_window.expected_trace().size() == 1u);
  const rund::replay::Trace &trace_view = trace_window.expected_trace().front();
  TEST_ASSERT(trace_view.code.compute_reason() ==
              rund::compute::Reason::ScanSumOverflow);
  TEST_ASSERT(trace_view.error() == "compute_scan_sum_overflow");
  TEST_ASSERT(trace_view.snapshot_code == rund::ReasonCode::RuntimeScopeBusy);
  TEST_ASSERT(trace_view.snapshot_error() == "runtime_scope_busy");

  const auto make_input_record = [](const std::span<const std::byte> bytes) {
    using rund::node::replay_detail::payload::Role;
    using rund::host::hash_bytes;
    using rund::node::replay_detail::payload::MakeArchive;
    using rund::node::replay_detail::payload::Materialize;
    using rund::node::replay_detail::payload::MaterializedRecord;
    using rund::node::replay_detail::payload::Store;
    const Store source{};
    const auto source_hash = source.SourceRangeHash(0u, {}, 0u, 0u);
    TEST_ASSERT(source_hash.has_value());
    const auto payload_hash = hash_bytes(bytes.data(), bytes.size());
    const auto materialized = Materialize(std::vector<MaterializedRecord>{
        MaterializedRecord{
            .metadata =
                {
                    .role = ::rund::node::replay_detail::payload::Role::Input,
                    .input_source = 17u,
                    .input_schema = 23u,
                    .input_sequence = 5u,
                    .source_hash = *source_hash,
                    .completed_bytes = bytes.size(),
                    .payload_hash = payload_hash,
                },
            .bytes = {bytes.begin(), bytes.end()}},
    });
    return rund::node::make_runtime_replay_record(
        rund::node::RuntimeReplayRecordDesc{
            .code = rund::replay::Code::Ok,
            .host_payload_archive = MakeArchive(materialized),
        });
  };
  const std::array expected_input{std::byte{0x11}, std::byte{0x22},
                                  std::byte{0x33}};
  const std::array actual_input{std::byte{0x11}, std::byte{0x44}};
  const rund::node::RuntimeReplayRecord expected_input_record =
      make_input_record(expected_input);
  const rund::node::RuntimeReplayRecord actual_input_record =
      make_input_record(actual_input);
  const rund::node::RuntimeReplayMismatchWindow input_window =
      rund::node::MinimizeRuntimeReplayMismatch(expected_input_record,
                                                actual_input_record, 1u);
  TEST_ASSERT(!input_window.ok());
  TEST_ASSERT(input_window.code == rund::replay::Code::InputHashMismatch);
  TEST_ASSERT(input_window.has_input_mismatch);
  TEST_ASSERT(input_window.input_index == 0u);
  TEST_ASSERT(input_window.expected_inputs.size() == 1u);
  TEST_ASSERT(input_window.actual_inputs.size() == 1u);
  TEST_ASSERT(input_window.expected_inputs[0].index == 0u);
  TEST_ASSERT(input_window.actual_inputs[0].index == 0u);
  TEST_ASSERT(input_window.expected_inputs[0].source == 17u);
  TEST_ASSERT(input_window.expected_inputs[0].schema == 23u);
  TEST_ASSERT(input_window.expected_inputs[0].sequence == 5u);
  TEST_ASSERT(input_window.expected_inputs[0].size == expected_input.size());
  TEST_ASSERT(input_window.expected_inputs[0].hash !=
              input_window.actual_inputs[0].hash);

  const rund::node::RuntimeReplayRecord downstream_mismatch =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .observations =
                  std::vector<rund::task::Observation>{rund::task::Observation{
                      .sequence = 1u,
                      .kind = rund::task::ObservationKind::TimerDue}},
              .host_payload_archive = actual_input_record.host.payload_archive,
          });
  const rund::node::RuntimeReplayMismatchWindow causal_window =
      rund::node::MinimizeRuntimeReplayMismatch(expected_input_record,
                                                downstream_mismatch, 1u);
  TEST_ASSERT(causal_window.has_input_mismatch);
  TEST_ASSERT(!causal_window.has_observation_mismatch);

  const rund::node::RuntimeReplayRecord missing_input_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{.code = rund::replay::Code::Ok});
  const rund::node::RuntimeReplayMismatchWindow missing_input_window =
      rund::node::MinimizeRuntimeReplayMismatch(expected_input_record,
                                                missing_input_record, 1u);
  TEST_ASSERT(!missing_input_window.ok());
  TEST_ASSERT(missing_input_window.has_input_mismatch);
  TEST_ASSERT(missing_input_window.input_index == 0u);
  TEST_ASSERT(missing_input_window.expected_inputs.size() == 1u);
  TEST_ASSERT(missing_input_window.expected_inputs[0].index == 0u);
  TEST_ASSERT(missing_input_window.actual_inputs.empty());

  const std::vector<std::byte> expected_artifact =
      SaveReplayRecord(expected_input_record);
  const std::vector<std::byte> actual_artifact =
      SaveReplayRecord(actual_input_record);
  const auto expected_public =
      rund::replay::Record::load(ReplayArtifact(expected_artifact));
  const auto actual_public =
      rund::replay::Record::load(ReplayArtifact(actual_artifact));
  TEST_ASSERT(expected_public);
  TEST_ASSERT(actual_public);
  const rund::replay::Window public_window =
      rund::replay::window(*expected_public, *actual_public, 1u);
  TEST_ASSERT(!public_window);
  TEST_ASSERT(public_window.code() == rund::replay::Code::InputHashMismatch);
  TEST_ASSERT(public_window.input_index() == 0u);
  TEST_ASSERT(public_window.expected_inputs().size() == 1u);
  TEST_ASSERT(public_window.actual_inputs().size() == 1u);
  const rund::replay::Window shared_window = public_window;
  TEST_ASSERT(public_window.expected_inputs().data() ==
              shared_window.expected_inputs().data());
  TEST_ASSERT(public_window.actual_inputs().data() ==
              shared_window.actual_inputs().data());
  const rund::replay::InputPoint &expected_point =
      public_window.expected_inputs().front();
  const rund::replay::InputPoint &actual_point =
      public_window.actual_inputs().front();
  TEST_ASSERT(expected_point.index == 0u);
  TEST_ASSERT(actual_point.index == 0u);
  TEST_ASSERT(expected_point.input.id == 17u);
  TEST_ASSERT(expected_point.input.schema == 23u);
  TEST_ASSERT(expected_point.sequence == 5u);
  TEST_ASSERT(expected_point.size == expected_input.size());
  TEST_ASSERT(expected_point.hash != actual_point.hash);

  rund::node::RuntimeReplayRecord damaged = CloneReplayRecord(fixture.record);
  damaged.operation_hash ^= 1u;
  damaged.replay_hash ^= 1u;
  const rund::node::RuntimeReplayCheck operation_mismatch =
      rund::node::check_runtime_replay(fixture.record, damaged);
  TEST_ASSERT(!operation_mismatch.ok());
  TEST_ASSERT(operation_mismatch.code ==
              rund::replay::Code::OperationHashMismatch);
  const rund::node::RuntimeReplayDiff diff =
      rund::node::DiffRuntimeReplayRecords(fixture.record, damaged);
  TEST_ASSERT(!diff.ok());
  TEST_ASSERT(!diff.mismatches.empty());
  TEST_ASSERT(std::string_view{diff.mismatches[0].field} == "operation_hash");
  return 0;
}

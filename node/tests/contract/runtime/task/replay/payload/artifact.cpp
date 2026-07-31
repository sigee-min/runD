#include "test/assert.hpp"

#include "src/runtime/replay/artifact/bytes.hpp"
#include "src/runtime/replay/artifact/format.hpp"
#include "src/runtime/replay/host/payload/hash.hpp"
#include "src/runtime/replay/host/payload/materialize.hpp"

#include <node/runtime/replay/record.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace {

using rund::node::replay_detail::artifact::Kind;
using rund::node::replay_detail::artifact::Result;
using rund::node::replay_detail::artifact::Sink;
using rund::node::replay_detail::artifact::Writer;

struct Gate final {
  std::vector<std::byte> bytes{};
  std::uint64_t calls = 0u;
  std::uint64_t accepts = 0u;
};

bool WriteGate(void *const state,
               const std::span<const std::byte> bytes) noexcept {
  auto &gate = *static_cast<Gate *>(state);
  ++gate.calls;
  if (gate.calls > gate.accepts) {
    return false;
  }
  try {
    gate.bytes.insert(gate.bytes.end(), bytes.begin(), bytes.end());
    return true;
  } catch (...) {
    return false;
  }
}

Result SaveRecord(const rund::node::RuntimeReplayRecord &record,
                  std::vector<std::byte> &bytes) {
  return rund::node::replay_detail::artifact::save(
      record, nullptr,
      Sink{.state = &bytes,
           .write = rund::node::replay_detail::artifact::append});
}

void CheckCapacityFailure() {
  Gate gate{.accepts = 2u};
  Writer out{Sink{.state = &gate, .write = WriteGate, .max_bytes = 7u}};
  TEST_ASSERT(out.header(Kind::Record));
  const Result result = out.finish();
  TEST_ASSERT(!result.ok());
  TEST_ASSERT(result.code == rund::replay::Code::ArtifactCapacityExceeded);
  TEST_ASSERT(result.bytes ==
              rund::node::replay_detail::artifact::kMagic.size());
  TEST_ASSERT(result.writes == 1u);
  TEST_ASSERT(gate.calls == 1u);
  TEST_ASSERT(gate.bytes.size() == result.bytes);
}

void CheckInvalidSink() {
  Writer out{Sink{}};
  TEST_ASSERT(!out.header(Kind::Record));
  const Result result = out.finish();
  TEST_ASSERT(result.code == rund::replay::Code::ArtifactOutputInvalid);
  TEST_ASSERT(result.bytes == 0u);
  TEST_ASSERT(result.writes == 0u);
}

void CheckSinkFailure() {
  Gate gate{.accepts = 1u};
  Writer out{Sink{.state = &gate, .write = WriteGate}};
  TEST_ASSERT(out.header(Kind::Record));
  const Result result = out.finish();
  TEST_ASSERT(!result.ok());
  TEST_ASSERT(result.code == rund::replay::Code::ArtifactWriteFailed);
  TEST_ASSERT(result.bytes ==
              rund::node::replay_detail::artifact::kMagic.size());
  TEST_ASSERT(result.writes == 1u);
  TEST_ASSERT(gate.calls == 2u);
  TEST_ASSERT(gate.bytes.size() == result.bytes);
}

void CheckFirstFailureWins() {
  Gate gate{.accepts = 2u};
  Writer out{Sink{.state = &gate, .write = WriteGate}};
  TEST_ASSERT(!out.reject(rund::replay::Code::CodecInvariantInvalid));
  TEST_ASSERT(!out.reject(rund::replay::Code::ArtifactWriteFailed));
  TEST_ASSERT(!out.u8(7u));
  const Result result = out.finish();
  TEST_ASSERT(result.code == rund::replay::Code::CodecInvariantInvalid);
  TEST_ASSERT(result.bytes == 0u);
  TEST_ASSERT(result.writes == 0u);
  TEST_ASSERT(gate.calls == 0u);
}

void CheckMalformedObservations() {
  std::vector<rund::task::Observation> observations{
      rund::task::Observation{
          .sequence = 2u,
          .kind = rund::task::ObservationKind::TimerDue,
          .deadline_ns = 2,
      },
      rund::task::Observation{
          .sequence = 1u,
          .kind = rund::task::ObservationKind::TimerDue,
          .deadline_ns = 3,
      },
  };
  const rund::node::RuntimeReplayRecord record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .observations = std::move(observations),
          });
  std::vector<std::byte> bytes{};
  const Result result = SaveRecord(record, bytes);
  TEST_ASSERT(!result.ok());
  TEST_ASSERT(result.code == rund::replay::Code::CodecInvariantInvalid);
  TEST_ASSERT(result.bytes == bytes.size());
  TEST_ASSERT(result.bytes ==
              rund::node::replay_detail::artifact::kMagic.size());
}

void CheckMalformedHost() {
  std::vector<rund::host::Event> events{
      rund::host::Event{
          .sequence = 2u,
          .kind = rund::host::EventKind::LogicalClockRead,
          .status = rund::host::Status::Ok,
          .logical_time_ns = 2u,
      },
      rund::host::Event{
          .sequence = 1u,
          .kind = rund::host::EventKind::LogicalClockRead,
          .status = rund::host::Status::Ok,
          .logical_time_ns = 3u,
      },
  };
  const rund::node::RuntimeReplayRecord record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .host_events = std::move(events),
          });
  std::vector<std::byte> bytes{};
  const Result result = SaveRecord(record, bytes);
  TEST_ASSERT(result.code == rund::replay::Code::CodecInvariantInvalid);
  TEST_ASSERT(result.bytes == bytes.size());
}

void CheckMalformedPayload() {
  using namespace rund::node::replay_detail::payload;
  const std::vector<std::byte> payload{std::byte{0x31}};
  const Materialization materialized =
      Materialize(std::vector<MaterializedRecord>{MaterializedRecord{
          .metadata =
              {
                  .role = Role::Input,
                  .input_source = 7u,
                  .input_schema = 11u,
                  .source_hash = SourceRangeHasher(0u, 0u, 0u, 0u).Finish(),
                  .completed_bytes = payload.size(),
                  .payload_hash =
                      rund::host::hash_bytes(payload.data(), payload.size()),
              },
          .bytes = payload,
      }});
  rund::node::RuntimeReplayRecord record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .host_payload_archive = MakeArchive(materialized),
          });
  record.host.payload_archive.records.front().pieces.front().size = 0u;
  std::vector<std::byte> bytes{};
  const Result result = SaveRecord(record, bytes);
  TEST_ASSERT(result.code == rund::replay::Code::CodecInvariantInvalid);
  TEST_ASSERT(result.bytes == bytes.size());
}

void CheckMalformedTrace() {
  rund::Trace trace{};
  trace.records = {
      rund::TraceRecord{.sequence = 2u},
      rund::TraceRecord{.sequence = 1u},
  };
  const rund::node::RuntimeReplayRecord record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .trace = std::move(trace),
          });
  std::vector<std::byte> bytes{};
  const Result result = SaveRecord(record, bytes);
  TEST_ASSERT(result.code == rund::replay::Code::CodecInvariantInvalid);
  TEST_ASSERT(result.bytes == bytes.size());
}

} // namespace

int CheckReplayArtifactContract() {
  CheckCapacityFailure();
  CheckInvalidSink();
  CheckSinkFailure();
  CheckFirstFailureWins();
  CheckMalformedObservations();
  CheckMalformedHost();
  CheckMalformedPayload();
  CheckMalformedTrace();
  return 0;
}

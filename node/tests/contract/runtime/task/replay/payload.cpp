#include "test/assert.hpp"

#include "local.hpp"
#include "src/runtime/replay/host/payload/backend.hpp"
#include "src/runtime/replay/host/payload/chunk.hpp"
#include "src/runtime/replay/host/payload/materialize.hpp"
#include "src/runtime/replay/host/payload/store.hpp"

#include <node/runtime/replay.hpp>
#include <node/runtime/replay/hash.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string_view>
#include <vector>

namespace {

void CheckLoadedPayloadUsesOneCompactOwner() {
  using namespace rund::node::replay_detail::payload;
  std::vector<std::byte> bytes(kChunkBytes + 17u);
  for (std::size_t index = 0u; index < bytes.size(); ++index) {
    bytes[index] =
        static_cast<std::byte>(((index * 131u) ^ (index >> 7u)) & 0xffu);
  }
  const rund::StableHash hash =
      rund::host::hash_bytes(bytes.data(), bytes.size());
  const Store source{};
  const std::optional<std::uint64_t> source_hash =
      source.SourceRangeHash(0u, {}, 0u, 0u);
  TEST_ASSERT(source_hash.has_value());
  const Materialization materialized =
      Materialize(std::vector<MaterializedRecord>{MaterializedRecord{
          .metadata =
              {
                  .role = Role::Input,
                  .input_source = 31u,
                  .input_schema = 47u,
                  .source_hash = *source_hash,
                  .completed_bytes = bytes.size(),
                  .payload_hash = hash,
              },
          .bytes = std::move(bytes),
      }});
  const rund::node::RuntimeReplayRecord record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .host_payload_archive = MakeArchive(materialized),
          });
  const std::vector<std::byte> artifact = SaveReplayRecord(record);
  const auto loaded = rund::node::DecodeRuntimeReplayRecord(artifact);
  if (!loaded.ok()) {
    const std::string_view error = rund::replay::error(loaded.code);
    std::fprintf(stderr, "compact payload load failed: %.*s\n",
                 static_cast<int>(error.size()), error.data());
  }
  TEST_ASSERT(loaded.ok());
  const Archive &archive = loaded.record.host.payload_archive;
  TEST_ASSERT(archive.chunks.size() == 2u);

  std::size_t encoded_bytes = 0u;
  for (const ArchiveChunk &chunk : archive.chunks) {
    TEST_ASSERT(chunk.segment_offset == 0u);
    TEST_ASSERT(chunk.encoded.data() ==
                archive.chunks.front().encoded.data() + encoded_bytes);
    encoded_bytes += chunk.encoded.size();
  }
  TEST_ASSERT(encoded_bytes == archive.storage.encoded_bytes);
  TEST_ASSERT(archive.storage.retained_bytes == encoded_bytes);
  TEST_ASSERT(archive.storage.copied_bytes == 0u);
  for (const ArchiveChunk &chunk : archive.chunks) {
    TEST_ASSERT(chunk.encoded.retained_bytes() == encoded_bytes);
  }

  const std::byte *const owner = archive.chunks.front().encoded.data();
  const BuildResult prepared = Build(archive, {});
  TEST_ASSERT(prepared.ok());
  TEST_ASSERT(prepared.store.blobs().size() == archive.chunks.size());
  TEST_ASSERT(prepared.store.blobs().front().encoded.data() == owner);
  const Archive republished = prepared.store.Archive();
  TEST_ASSERT(republished.storage.copied_bytes == 0u);
  TEST_ASSERT(republished.chunks.front().encoded.data() == owner);
  TEST_ASSERT(SaveReplayRecord(loaded.record) == artifact);
}

void CheckModeledThirtyMinuteArtifactBudget() {
  constexpr std::size_t kSeconds = 30u * 60u;
  constexpr std::size_t kEventsPerSecond = 10u;
  constexpr std::size_t kEvents = kSeconds * kEventsPerSecond;
  constexpr std::size_t kBudgetBytes = 100u * 1024u;

  const rund::node::replay_detail::payload::Store source{};
  const std::optional<std::uint64_t> source_hash =
      source.SourceRangeHash(0u, {}, 0u, 0u);
  TEST_ASSERT(source_hash.has_value());

  std::vector<rund::node::replay_detail::payload::MaterializedRecord> records{};
  records.reserve(kEvents);
  for (std::size_t index = 0u; index < kEvents; ++index) {
    const std::byte command = static_cast<std::byte>(index & 0x0fu);
    const std::array payload{command};
    records.push_back(rund::node::replay_detail::payload::MaterializedRecord{
        .metadata =
            {
                .role = rund::node::replay_detail::payload::Role::Input,
                .input_source = 17u,
                .input_schema = 23u,
                .input_sequence = static_cast<std::uint64_t>(index),
                .source_hash = *source_hash,
                .completed_bytes = payload.size(),
                .payload_hash =
                    rund::host::hash_bytes(payload.data(), payload.size()),
            },
        .bytes = {payload.begin(), payload.end()}});
  }
  const auto materialized =
      rund::node::replay_detail::payload::Materialize(std::move(records));
  TEST_ASSERT(materialized.payload_hash != 0u);
  const rund::node::RuntimeReplayRecord record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .host_payload_archive =
                  rund::node::replay_detail::payload::MakeArchive(materialized),
          });
  TEST_ASSERT(record.input_count == kEvents);
  const std::vector<std::byte> artifact = SaveReplayRecord(record);
  TEST_ASSERT(artifact.size() == 64'394u);
  TEST_ASSERT(artifact.size() < kBudgetBytes);
  const auto loaded = rund::node::DecodeRuntimeReplayRecord(artifact);
  TEST_ASSERT(loaded.ok());
  TEST_ASSERT(loaded.record.input_count == kEvents);
  TEST_ASSERT(SaveReplayRecord(loaded.record) == artifact);
  std::printf("replay.compact-model events=%zu payload_alphabet=16 bytes=%zu "
              "budget=%zu\n",
              kEvents, artifact.size(), kBudgetBytes);
}

} // namespace

int RunRuntimeTaskReplayPayloadContract() {
  const std::array<std::byte, 1u> host_write_payload{std::byte{'q'}};
  const rund::StableHash host_write_payload_hash = rund::host::hash_bytes(
      host_write_payload.data(), host_write_payload.size());
  const std::vector<rund::host::Event> payload_host_events{
      rund::host::Event{
          .sequence = 1u,
          .kind = rund::host::EventKind::IoWrite,
          .status = rund::host::Status::Ok,
          .host_handle_id = 77u,
          .requested_bytes = host_write_payload.size(),
          .completed_bytes = host_write_payload.size(),
          .native_errno = 0,
          .payload_hash = host_write_payload_hash,
      },
  };
  const rund::node::replay_detail::payload::Materialization payload_evidence =
      rund::node::replay_detail::payload::Materialize(
          std::vector<rund::node::replay_detail::payload::MaterializedRecord>{
              rund::node::replay_detail::payload::MaterializedRecord{
                  .metadata =
                      {
                          .event_sequence = 1u,
                          .kind = rund::host::EventKind::IoWrite,
                          .completed_bytes = host_write_payload.size(),
                          .payload_hash = host_write_payload_hash,
                      },
                  .bytes = std::vector<std::byte>{std::byte{'q'}},
              },
          });
  const rund::node::RuntimeReplayRecord payload_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .host_events = payload_host_events,
              .host_payload_archive =
                  rund::node::replay_detail::payload::MakeArchive(
                      payload_evidence),
          });
  TEST_ASSERT(payload_record.transcript_hash == payload_evidence.payload_hash);
  TEST_ASSERT(payload_record.transcript_hash != 0u);

  rund::node::RuntimeReplayRecord payload_hash_mismatch =
      CloneReplayRecord(payload_record);
  payload_hash_mismatch.transcript_hash ^= 1u;
  payload_hash_mismatch.replay_hash =
      rund::node::replay_detail::HashReplay(payload_hash_mismatch);
  const rund::node::RuntimeReplayCheck payload_hash_check =
      rund::node::check_runtime_replay(payload_record, payload_hash_mismatch);
  TEST_ASSERT(!payload_hash_check.ok());
  TEST_ASSERT(payload_hash_check.code ==
              rund::replay::Code::TranscriptHashMismatch);
  TEST_ASSERT(payload_hash_check.expected_hash ==
              payload_record.transcript_hash);
  TEST_ASSERT(payload_hash_check.actual_hash ==
              payload_hash_mismatch.transcript_hash);
  const rund::node::RuntimeReplayDiff payload_hash_diff =
      rund::node::DiffRuntimeReplayRecords(payload_record,
                                           payload_hash_mismatch);
  TEST_ASSERT(!payload_hash_diff.ok());
  bool found_payload_hash_diff = false;
  for (const rund::node::RuntimeReplayFieldMismatch &mismatch :
       payload_hash_diff.mismatches) {
    if (std::string_view{mismatch.field} == "transcript_hash") {
      found_payload_hash_diff = true;
      TEST_ASSERT(mismatch.expected == payload_record.transcript_hash);
      TEST_ASSERT(mismatch.actual == payload_hash_mismatch.transcript_hash);
    }
  }
  TEST_ASSERT(found_payload_hash_diff);

  std::vector<std::byte> payload_encoded = SaveReplayRecord(payload_record);
  const rund::node::RuntimeReplayDecodeResult payload_decode =
      rund::node::DecodeRuntimeReplayRecord(payload_encoded);
  TEST_ASSERT(payload_decode.ok());
  TEST_ASSERT(payload_decode.record.transcript_hash ==
              payload_record.transcript_hash);
  TEST_ASSERT(payload_decode.record.host.payload_archive.records.size() == 1u);
  TEST_ASSERT(payload_decode.record.host.payload_archive.chunks.size() == 1u);

  const auto payload_byte =
      std::find(payload_encoded.begin(), payload_encoded.end(), std::byte{'q'});
  TEST_ASSERT(payload_byte != payload_encoded.end());
  *payload_byte ^= std::byte{1u};
  const rund::node::RuntimeReplayDecodeResult corrupt_payload =
      rund::node::DecodeRuntimeReplayRecord(payload_encoded);
  TEST_ASSERT(!corrupt_payload.ok());

  rund::replay::Limits no_payload{};
  no_payload.max_payload_bytes = 0u;
  const auto capacity = rund::node::DecodeRuntimeReplayRecord(
      SaveReplayRecord(payload_record), no_payload);
  TEST_ASSERT(!capacity);
  TEST_ASSERT(capacity.code ==
              rund::replay::Code::CodecPayloadCapacityExceeded);

  const std::vector<rund::host::Event> failed_io_events{
      rund::host::Event{
          .sequence = 2u,
          .kind = rund::host::EventKind::IoRead,
          .status = rund::host::Status::SyscallFailed,
          .host_handle_id = 78u,
          .requested_bytes = 16u,
          .completed_bytes = 0u,
          .native_errno = EIO,
      },
  };
  const rund::node::RuntimeReplayRecord failed_io_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::IoSyscallFailed,
              .host_events = failed_io_events,
          });
  const rund::node::RuntimeReplayDecodeResult failed_io_decode =
      rund::node::DecodeRuntimeReplayRecord(SaveReplayRecord(failed_io_record));
  TEST_ASSERT(failed_io_decode.ok());

  const std::vector<rund::host::Event> missing_payload_events{
      rund::host::Event{
          .sequence = 3u,
          .kind = rund::host::EventKind::IoRead,
          .status = rund::host::Status::Ok,
          .host_handle_id = 79u,
          .requested_bytes = 0u,
          .completed_bytes = 0u,
          .native_errno = 0,
          .payload_hash = rund::host::hash_bytes(nullptr, 0u),
      },
  };
  const rund::node::RuntimeReplayRecord missing_payload_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .host_events = missing_payload_events,
          });
  const rund::node::RuntimeReplayDecodeResult missing_payload_decode =
      rund::node::DecodeRuntimeReplayRecord(
          SaveReplayRecord(missing_payload_record));
  TEST_ASSERT(!missing_payload_decode.ok());
  TEST_ASSERT(missing_payload_decode.code ==
              rund::replay::Code::HostPayloadMissing);

  CheckLoadedPayloadUsesOneCompactOwner();
  CheckModeledThirtyMinuteArtifactBudget();
  return 0;
}

#include "test/assert.hpp"

#include "../../local.hpp"
#include "local/model.hpp"

#include <node/runtime/replay.hpp>
#include <rund/replay.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace replay_payload_store {
namespace {

using rund::node::replay_detail::payload::RawByteSource;
using rund::node::replay_detail::payload::ValidDiagnosticArchive;

[[nodiscard]] std::span<const std::byte> RawSlice(const void *const context,
                                                  const std::size_t) noexcept {
  return *static_cast<const std::span<const std::byte> *>(context);
}

struct RawSlices final {
  std::span<const std::span<const std::byte>> values{};
  mutable std::size_t calls = 0u;
};

[[nodiscard]] std::span<const std::byte>
CountedRawSlice(const void *const context, const std::size_t index) noexcept {
  const auto &slices = *static_cast<const RawSlices *>(context);
  ++slices.calls;
  return index < slices.values.size() ? slices.values[index]
                                      : std::span<const std::byte>{};
}

void AppendRaw(Store &store, const std::uint64_t sequence,
               const std::span<const std::byte> bytes) {
  const StableHash hash =
      store.CaptureIngress(sequence, EventKind::NetRecv,
                           RawByteSource{.context = &bytes,
                                         .slice_count = 1u,
                                         .admitted_bytes = bytes.size(),
                                         .byte_count = bytes.size(),
                                         .slice = RawSlice});
  TEST_ASSERT(hash.value == hash_bytes(bytes.data(), bytes.size()).value);
}

int RawDiagnosticRingEvictsCompleteRecordsWithoutSemanticAuthority() {
  Store store = Prepared(
      4096u, 1024u, 4u * 1024u * 1024u, {},
      ::rund::replay::Diagnostic{.window_bytes = 5u, .window_records = 2u});
  const std::vector<std::byte> first = Payload("ab");
  const std::vector<std::byte> second = Payload("cd");
  const std::vector<std::byte> third = Payload("efg");
  const std::vector<std::byte> oversized = Payload("123456");
  AppendRaw(store, 1u, first);
  AppendRaw(store, 2u, second);
  AppendRaw(store, 3u, third);
  AppendRaw(store, 4u, oversized);

  const Archive archive = store.Archive();
  TEST_ASSERT(store.payload_hash() == 0u);
  TEST_ASSERT(archive.payload_hash == 0u);
  TEST_ASSERT(archive.records.empty());
  TEST_ASSERT(archive.chunks.empty());
  TEST_ASSERT(archive.diagnostic.records.size() == 2u);
  TEST_ASSERT(archive.diagnostic.records[0].event_sequence == 2u);
  TEST_ASSERT(archive.diagnostic.records[1].event_sequence == 3u);
  TEST_ASSERT(archive.diagnostic.bytes.size() == 5u);
  TEST_ASSERT(archive.diagnostic.bytes.span()[0] == std::byte{'c'});
  TEST_ASSERT(archive.diagnostic.bytes.span()[4] == std::byte{'g'});
  TEST_ASSERT(archive.diagnostic.hash != 0u);
  TEST_ASSERT(archive.diagnostic.report.retained_bytes == 5u);
  TEST_ASSERT(archive.diagnostic.report.retained_records == 2u);
  TEST_ASSERT(archive.diagnostic.report.evicted_records == 1u);
  TEST_ASSERT(archive.diagnostic.report.dropped_records == 1u);
  TEST_ASSERT(ValidDiagnosticArchive(archive.diagnostic));

  const std::vector<Event> events{
      Event{.sequence = 2u,
            .kind = EventKind::NetRecv,
            .status = Status::Ok,
            .requested_bytes = second.size(),
            .completed_bytes = second.size(),
            .payload_hash = hash_bytes(second.data(), second.size())},
      Event{.sequence = 3u,
            .kind = EventKind::NetRecv,
            .status = Status::Ok,
            .requested_bytes = third.size(),
            .completed_bytes = third.size(),
            .payload_hash = hash_bytes(third.data(), third.size())},
  };
  const rund::node::RuntimeReplayRecord raw_record =
      rund::node::make_runtime_replay_record(
          rund::node::RuntimeReplayRecordDesc{
              .code = rund::replay::Code::Ok,
              .host_events = events,
              .host_payload_archive = archive,
          });
  const std::vector<std::byte> artifact = SaveReplayRecord(raw_record);
  const auto decoded = rund::replay::Record::load(ReplayArtifact(artifact));
  TEST_ASSERT(decoded);
  const rund::replay::Record &record = *decoded;
  const std::span<const rund::replay::Capture> captures = record.captures();
  TEST_ASSERT(captures.size() == 2u);
  TEST_ASSERT(record.capture_hash() == archive.diagnostic.hash);
  TEST_ASSERT(record.capture_report().retained_bytes == 5u);
  const rund::replay::Capture &first_capture = captures.front();
  const rund::replay::Capture &last_capture = captures.back();
  TEST_ASSERT(first_capture.sequence == 2u);
  TEST_ASSERT(first_capture.kind == EventKind::NetRecv);
  TEST_ASSERT(first_capture.bytes.size() == second.size());
  TEST_ASSERT(std::equal(first_capture.bytes.begin(), first_capture.bytes.end(),
                         second.begin()));
  TEST_ASSERT(first_capture.hash ==
              hash_bytes(second.data(), second.size()).value);
  TEST_ASSERT(last_capture.sequence == 3u);
  TEST_ASSERT(last_capture.kind == EventKind::NetRecv);
  TEST_ASSERT(last_capture.bytes.size() == third.size());
  TEST_ASSERT(std::equal(last_capture.bytes.begin(), last_capture.bytes.end(),
                         third.begin()));
  return 0;
}

int RawDiagnosticRingHashesAndCopiesInOneCanonicalSliceSweep() {
  Store store = Prepared(
      4096u, 1024u, 4u * 1024u * 1024u, {},
      ::rund::replay::Diagnostic{.window_bytes = 8u, .window_records = 2u});
  const std::array<std::byte, 2u> first{std::byte{'a'}, std::byte{'b'}};
  const std::array<std::byte, 3u> last{std::byte{'c'}, std::byte{'d'},
                                       std::byte{'e'}};
  const std::array<std::byte, 2u> tail{std::byte{'f'}, std::byte{'g'}};
  const std::array<std::span<const std::byte>, 4u> values{
      std::span<const std::byte>{first}, std::span<const std::byte>{},
      std::span<const std::byte>{last}, std::span<const std::byte>{tail}};
  RawSlices slices{.values = values};
  constexpr std::uint64_t kCompletedBytes = 4u;
  constexpr std::size_t kCompletedSlices = 3u;

  const StableHash retained_hash = store.CaptureIngress(
      1u, EventKind::NetRecvVectored,
      RawByteSource{.context = &slices,
                    .slice_count = values.size(),
                    .admitted_bytes = first.size() + last.size() + tail.size(),
                    .byte_count = kCompletedBytes,
                    .slice = CountedRawSlice});

  const Archive archive = store.Archive();
  TEST_ASSERT(slices.calls == kCompletedSlices);
  const std::array<std::byte, kCompletedBytes> expected{
      std::byte{'a'}, std::byte{'b'}, std::byte{'c'}, std::byte{'d'}};
  TEST_ASSERT(retained_hash.value ==
              hash_bytes(expected.data(), expected.size()).value);
  TEST_ASSERT(archive.diagnostic.records[0].payload_hash.value ==
              retained_hash.value);
  TEST_ASSERT(archive.diagnostic.report.retained_bytes == kCompletedBytes);
  TEST_ASSERT(archive.diagnostic.report.retained_records == 1u);
  TEST_ASSERT(archive.diagnostic.bytes.size() == kCompletedBytes);
  TEST_ASSERT(archive.diagnostic.bytes.span()[0] == std::byte{'a'});
  TEST_ASSERT(archive.diagnostic.bytes.span()[3] == std::byte{'d'});

  Store rejected_store = Prepared(
      4096u, 1024u, 4u * 1024u * 1024u, {},
      ::rund::replay::Diagnostic{.window_bytes = 8u, .window_records = 2u});
  RawSlices rejected{.values = values};
  const StableHash rejected_hash = rejected_store.CaptureIngress(
      2u, EventKind::NetRecvVectored,
      RawByteSource{.context = &rejected,
                    .slice_count = values.size(),
                    .admitted_bytes = kCompletedBytes - 1u,
                    .byte_count = kCompletedBytes,
                    .slice = CountedRawSlice});
  const Archive rejected_archive = rejected_store.Archive();
  TEST_ASSERT(rejected.calls == 0u);
  TEST_ASSERT(rejected_hash.value ==
              hash_bytes(nullptr, kCompletedBytes).value);
  TEST_ASSERT(rejected_archive.diagnostic.records.empty());
  TEST_ASSERT(rejected_archive.diagnostic.bytes.empty());
  TEST_ASSERT(rejected_archive.diagnostic.report.dropped_records == 1u);

  RawSlices disabled{.values = values};
  const StableHash disabled_hash =
      rund::node::replay_detail::payload::HashIngress(RawByteSource{
          .context = &disabled,
          .slice_count = values.size(),
          .admitted_bytes = first.size() + last.size() + tail.size(),
          .byte_count = kCompletedBytes,
          .slice = CountedRawSlice});
  const Store disabled_store = Prepared();
  const Archive disabled_archive = disabled_store.Archive();
  TEST_ASSERT(disabled.calls == kCompletedSlices);
  TEST_ASSERT(disabled_hash.value == retained_hash.value);
  TEST_ASSERT(disabled_archive.diagnostic.records.empty());
  TEST_ASSERT(disabled_archive.diagnostic.bytes.empty());
  TEST_ASSERT(disabled_archive.diagnostic.report.dropped_records == 0u);
  return 0;
}

} // namespace

int DiagnosticContract() {
  return Run(std::array<Contract, 2u>{
      RawDiagnosticRingEvictsCompleteRecordsWithoutSemanticAuthority,
      RawDiagnosticRingHashesAndCopiesInOneCanonicalSliceSweep,
  });
}

} // namespace replay_payload_store

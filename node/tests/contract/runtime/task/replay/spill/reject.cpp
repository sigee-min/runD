#include "test/assert.hpp"

#include "local/model.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <string_view>
#include <utility>

#include <sys/statvfs.h>

namespace replay_spill {
namespace {

using rund::host::hash_bytes;
using rund::node::replay_detail::payload::Capture;
using rund::node::replay_detail::payload::kChunkBytes;

[[nodiscard]] std::uint64_t AllocationUnit(const std::filesystem::path &root) {
  struct statvfs status{};
  TEST_ASSERT(::statvfs(root.c_str(), &status) == 0);
  return status.f_frsize != 0u ? static_cast<std::uint64_t>(status.f_frsize)
                               : static_cast<std::uint64_t>(status.f_bsize);
}

int SpillMissingSegmentFailsClosed() {
  const std::filesystem::path dir = TempDir("missing");
  std::filesystem::remove_all(dir);
  const std::vector<std::byte> payload = Bytes("abcdefghijklmnopqrstuvwx");
  const StableHash hash = hash_bytes(payload.data(), payload.size());

  Store store = Prepared(Storage(dir));
  TEST_ASSERT(
      store.Append(3u, EventKind::IoRead, Capture::verify(payload, hash)));
  TEST_ASSERT(std::filesystem::remove(SegmentPath(dir, 0u)));
  const ResolveResult resolved = store.Resolve(0u);
  TEST_ASSERT(!resolved.ok());
  TEST_ASSERT(resolved.code == rund::replay::Code::HostPayloadMissing);
  TEST_ASSERT(std::filesystem::exists(dir));
  std::filesystem::remove_all(dir);
  return 0;
}

int SpillCorruptedSegmentFailsClosed() {
  const std::filesystem::path dir = TempDir("corrupt");
  std::filesystem::remove_all(dir);
  const std::vector<std::byte> payload = Bytes("abcdefghijklmnopqrstuvwx");
  const StableHash hash = hash_bytes(payload.data(), payload.size());

  Store store = Prepared(Storage(dir));
  TEST_ASSERT(
      store.Append(4u, EventKind::IoRead, Capture::verify(payload, hash)));
  TEST_ASSERT(CorruptLastByte(SegmentPath(dir, 0u)));
  const ResolveResult resolved = store.Resolve(0u);
  TEST_ASSERT(!resolved.ok());
  TEST_ASSERT(resolved.code == rund::replay::Code::HostPayloadHashInvalid);
  TEST_ASSERT(std::filesystem::exists(dir));
  std::filesystem::remove_all(dir);
  return 0;
}

int SpillWriteMatchRejectsHashInvalidExactBytes() {
  const std::filesystem::path dir = TempDir("corrupt-write-match");
  std::filesystem::remove_all(dir);
  const std::vector<std::byte> payload = Bytes("abcdefghijklmnopqrstuvwx");
  const StableHash hash = hash_bytes(payload.data(), payload.size());

  Store store = Prepared(Storage(dir));
  TEST_ASSERT(
      store.Append(5u, EventKind::IoWrite, Capture::verify(payload, hash)));
  TEST_ASSERT(CorruptLastByte(SegmentPath(dir, 0u)));
  std::vector<std::byte> corrupted = payload;
  corrupted.back() ^= std::byte{0x01};
  const auto &record = store.records()[0];
  const Binding binding{.event_sequence = record.metadata.event_sequence,
                        .kind = record.metadata.kind,
                        .completed_bytes = record.metadata.completed_bytes,
                        .payload_hash = record.metadata.payload_hash};
  const auto matched = store.Matches(0u, binding, corrupted);
  TEST_ASSERT(!matched.ok());
  TEST_ASSERT(matched.code == rund::replay::Code::HostPayloadHashInvalid);
  std::filesystem::remove_all(dir);
  return 0;
}

int SpillRejectsRecordHashMismatchAtLoad() {
  const std::filesystem::path dir = TempDir("record-hash");
  std::filesystem::remove_all(dir);
  const ::rund::replay::Storage storage = Storage(dir);
  const std::vector<std::byte> payload = Bytes("abcdefghijklmnopqrstuvwx");
  const StableHash hash = hash_bytes(payload.data(), payload.size());

  Store recorded = Prepared(storage);
  TEST_ASSERT(
      recorded.Append(10u, EventKind::IoWrite, Capture::verify(payload, hash)));
  TEST_ASSERT(
      recorded.Append(11u, EventKind::IoRead, Capture::verify(payload, hash)));
  rund::node::replay_detail::payload::Archive archive = recorded.Archive();
  TEST_ASSERT(archive.records.size() == 2u);
  archive.records[0].metadata.payload_hash.value ^= 0x55u;
  archive.records[1].metadata.payload_hash.value ^= 0xaau;

  Store replay = Prepared(storage);
  TEST_ASSERT(!replay.LoadArchive(std::move(archive)));
  TEST_ASSERT(replay.records().empty());
  TEST_ASSERT(replay.blobs().empty());
  std::filesystem::remove_all(dir);
  return 0;
}

int SpillRejectsChunksLargerThanSegmentBeforeAcceptingRecord() {
  const std::filesystem::path dir = TempDir("quota");
  std::filesystem::remove_all(dir);
  const std::vector<std::byte> payload = Bytes("abcdefghijklmnopqrstuvwx");
  const StableHash hash = hash_bytes(payload.data(), payload.size());

  Store store = Prepared(Storage(dir, 8u));
  TEST_ASSERT(
      !store.Append(6u, EventKind::IoRead, Capture::verify(payload, hash)));
  TEST_ASSERT(store.records().empty());
  TEST_ASSERT(store.blobs().empty());
  TEST_ASSERT(store.logical_bytes() == 0u);
  TEST_ASSERT(store.encoded_bytes() == 0u);
  TEST_ASSERT(!std::filesystem::exists(SegmentPath(dir, 0u)));
  std::filesystem::remove_all(dir);
  return 0;
}

int SpillRejectsMultiChunkFailureBeforeMutatingBackend() {
  const std::filesystem::path dir = TempDir("atomic");
  std::filesystem::remove_all(dir);
  ::rund::replay::Storage storage = Storage(dir, kChunkBytes + 128u);
  storage.budget = ::rund::storage::Budget{storage.max_allocated_bytes};
  Store store = Prepared(storage);
  const std::vector<std::byte> seed = Bytes("seed");
  const StableHash seed_hash = hash_bytes(seed.data(), seed.size());
  TEST_ASSERT(
      store.Append(1u, EventKind::IoRead, Capture::verify(seed, seed_hash)));
  const std::uint64_t prior_hash = store.payload_hash();
  const std::uintmax_t prior_file_bytes =
      std::filesystem::file_size(SegmentPath(dir, 0u));
  const ::rund::storage::Report prior_usage = storage.budget.report();

  std::vector<std::byte> payload(kChunkBytes * 2u);
  for (std::size_t index = 0u; index < payload.size(); ++index) {
    payload[index] = static_cast<std::byte>(
        ((index * 131u) ^ (index >> 3u) ^ 0x5au) & 0xffu);
  }
  payload[kChunkBytes] ^= std::byte{0x01};
  const StableHash hash = hash_bytes(payload.data(), payload.size());
  TEST_ASSERT(std::filesystem::create_directory(SegmentPath(dir, 1u)));

  TEST_ASSERT(
      !store.Append(7u, EventKind::IoRead, Capture::verify(payload, hash)));
  TEST_ASSERT(store.records().size() == 1u);
  TEST_ASSERT(store.blobs().size() == 1u);
  TEST_ASSERT(store.logical_bytes() == seed.size());
  TEST_ASSERT(store.payload_hash() == prior_hash);
  TEST_ASSERT(std::filesystem::file_size(SegmentPath(dir, 0u)) ==
              prior_file_bytes);
  TEST_ASSERT(!std::filesystem::exists(SegmentPath(dir, 1u)));
  const ::rund::storage::Report rolled_back_usage = storage.budget.report();
  TEST_ASSERT(rolled_back_usage.physical_bytes == prior_usage.physical_bytes);
  TEST_ASSERT(rolled_back_usage.allocated_bytes == prior_usage.allocated_bytes);
  TEST_ASSERT(rolled_back_usage.reserved_bytes == 0u);
  const ResolveResult resolved = store.Resolve(0u);
  TEST_ASSERT(resolved.ok());
  TEST_ASSERT(std::equal(resolved.bytes.span().begin(),
                         resolved.bytes.span().end(), seed.begin(),
                         seed.end()));

  const std::vector<std::byte> later = Bytes("later");
  const StableHash later_hash = hash_bytes(later.data(), later.size());
  TEST_ASSERT(
      store.Append(8u, EventKind::IoRead, Capture::verify(later, later_hash)));
  TEST_ASSERT(store.records().size() == 2u);
  TEST_ASSERT(store.blobs().size() == 2u);
  const ResolveResult appended = store.Resolve(1u);
  TEST_ASSERT(appended.ok());
  TEST_ASSERT(std::equal(appended.bytes.span().begin(),
                         appended.bytes.span().end(), later.begin(),
                         later.end()));
  std::filesystem::remove_all(dir);
  return 0;
}

int SpillSharedBudgetRejectsThenRefundsAtLastOwner() {
  const std::filesystem::path dir = TempDir("shared-budget");
  std::filesystem::remove_all(dir);
  TEST_ASSERT(std::filesystem::create_directories(dir));
  const std::uint64_t unit = AllocationUnit(dir);
  TEST_ASSERT(unit != 0u);
  ::rund::replay::Storage storage = Storage(dir, 1024u);
  storage.max_allocated_bytes = unit;
  storage.budget = ::rund::storage::Budget{unit};
  const std::vector<std::byte> first = Bytes("first");
  const std::vector<std::byte> second = Bytes("second");
  const StableHash first_hash = hash_bytes(first.data(), first.size());
  const StableHash second_hash = hash_bytes(second.data(), second.size());

  Store owner = Prepared(storage);
  TEST_ASSERT(
      owner.Append(1u, EventKind::IoRead, Capture::verify(first, first_hash)));
  rund::node::replay_detail::payload::Archive archive = owner.Archive();
  TEST_ASSERT(archive.storage.allocated_bytes == unit);
  owner.Clear();
  TEST_ASSERT(storage.budget.report().allocated_bytes == unit);

  Store waiting = Prepared(storage);
  const std::uint64_t rejections = storage.budget.report().rejection_count;
  TEST_ASSERT(!waiting.Append(2u, EventKind::IoRead,
                              Capture::verify(second, second_hash)));
  TEST_ASSERT(waiting.records().empty());
  TEST_ASSERT(waiting.blobs().empty());
  TEST_ASSERT(storage.budget.report().allocated_bytes == unit);
  TEST_ASSERT(storage.budget.report().reserved_bytes == 0u);
  TEST_ASSERT(storage.budget.report().rejection_count == rejections + 1u);

  archive = {};
  TEST_ASSERT(storage.budget.report().allocated_bytes == 0u);
  TEST_ASSERT(storage.budget.report().physical_bytes == 0u);
  TEST_ASSERT(waiting.Append(3u, EventKind::IoRead,
                             Capture::verify(second, second_hash)));
  TEST_ASSERT(storage.budget.report().allocated_bytes == unit);
  waiting.Clear();
  TEST_ASSERT(storage.budget.report().allocated_bytes == 0u);
  TEST_ASSERT(storage.budget.report().physical_bytes == 0u);
  TEST_ASSERT(GenerationDirectories(dir).empty());
  std::filesystem::remove_all(dir);
  return 0;
}

int SpillMinimumFreeOverflowFailsBeforeWriting() {
  const std::filesystem::path dir = TempDir("minimum-free-overflow");
  std::filesystem::remove_all(dir);
  TEST_ASSERT(std::filesystem::create_directories(dir));
  ::rund::replay::Storage storage = Storage(dir, 1024u);
  storage.minimum_free_bytes = std::numeric_limits<std::uint64_t>::max();
  storage.budget = ::rund::storage::Budget{storage.max_allocated_bytes};
  const std::vector<std::byte> payload = Bytes("headroom");
  const StableHash hash = hash_bytes(payload.data(), payload.size());

  Store store = Prepared(storage);
  TEST_ASSERT(
      !store.Append(1u, EventKind::IoRead, Capture::verify(payload, hash)));
  TEST_ASSERT(store.records().empty());
  TEST_ASSERT(store.blobs().empty());
  TEST_ASSERT(!std::filesystem::exists(SegmentPath(dir, 0u)));
  TEST_ASSERT(storage.budget.report().physical_bytes == 0u);
  TEST_ASSERT(storage.budget.report().allocated_bytes == 0u);
  TEST_ASSERT(storage.budget.report().reserved_bytes == 0u);
  store.Clear();
  TEST_ASSERT(GenerationDirectories(dir).empty());
  std::filesystem::remove_all(dir);
  return 0;
}

int SpillPartialReplayReadCannotReportSuccess() {
  const std::filesystem::path dir = TempDir("partial-read");
  std::filesystem::remove_all(dir);
  std::vector<std::byte> payload(kChunkBytes * 2u);
  for (std::size_t index = 0u; index < payload.size(); ++index) {
    payload[index] = static_cast<std::byte>(index & 0xffu);
  }
  payload[kChunkBytes] = std::byte{0xa5};
  const StableHash hash = hash_bytes(payload.data(), payload.size());

  Store store = Prepared(Storage(dir, kChunkBytes + 128u));
  TEST_ASSERT(
      store.Append(8u, EventKind::IoRead, Capture::verify(payload, hash)));
  TEST_ASSERT(store.records().size() == 1u);
  TEST_ASSERT(store.pieces(store.records()[0]).size() == 2u);
  TEST_ASSERT(std::filesystem::exists(SegmentPath(dir, 0u)));
  TEST_ASSERT(std::filesystem::exists(SegmentPath(dir, 1u)));
  TEST_ASSERT(CorruptLastByte(SegmentPath(dir, 1u)));

  const auto &record = store.records()[0];
  const Binding binding{.event_sequence = record.metadata.event_sequence,
                        .kind = record.metadata.kind,
                        .completed_bytes = record.metadata.completed_bytes,
                        .payload_hash = record.metadata.payload_hash};
  std::vector<std::byte> output(payload.size(), std::byte{0xee});
  const auto read = store.ReadInto(0u, binding, output);
  TEST_ASSERT(!read.ok());
  TEST_ASSERT(read.code == rund::replay::Code::HostPayloadHashInvalid);
  TEST_ASSERT(std::equal(payload.begin(), payload.begin() + kChunkBytes,
                         output.begin()));
  TEST_ASSERT(std::all_of(
      output.begin() + kChunkBytes, output.end(),
      [](const std::byte value) { return value == std::byte{0xee}; }));
  std::filesystem::remove_all(dir);
  return 0;
}

int SpillRejectsArchiveSegmentRecordLargerThanConfiguredSegment() {
  const std::filesystem::path dir = TempDir("bad-archive-segment");
  std::filesystem::remove_all(dir);
  const ::rund::replay::Storage storage = Storage(dir);
  const std::vector<std::byte> payload = Bytes("abcdefghijklmnopqrstuvwx");
  const StableHash hash = hash_bytes(payload.data(), payload.size());

  Store record_store = Prepared(storage);
  TEST_ASSERT(record_store.Append(9u, EventKind::IoRead,
                                  Capture::verify(payload, hash)));
  rund::node::replay_detail::payload::Archive archive = record_store.Archive();
  TEST_ASSERT(!archive.chunks.empty());
  archive.chunks[0].segment_record_bytes = storage.segment_bytes + 1u;

  Store replay_store = Prepared(storage);
  TEST_ASSERT(!replay_store.LoadArchive(archive));
  std::filesystem::remove_all(dir);
  return 0;
}

} // namespace

int RunRejectContract() {
  if (const int rc = SpillMissingSegmentFailsClosed(); rc != 0) {
    return rc;
  }
  if (const int rc = SpillCorruptedSegmentFailsClosed(); rc != 0) {
    return rc;
  }
  if (const int rc = SpillWriteMatchRejectsHashInvalidExactBytes(); rc != 0) {
    return rc;
  }
  if (const int rc = SpillRejectsRecordHashMismatchAtLoad(); rc != 0) {
    return rc;
  }
  if (const int rc = SpillRejectsChunksLargerThanSegmentBeforeAcceptingRecord();
      rc != 0) {
    return rc;
  }
  if (const int rc = SpillRejectsMultiChunkFailureBeforeMutatingBackend();
      rc != 0) {
    return rc;
  }
  if (const int rc = SpillSharedBudgetRejectsThenRefundsAtLastOwner();
      rc != 0) {
    return rc;
  }
  if (const int rc = SpillMinimumFreeOverflowFailsBeforeWriting(); rc != 0) {
    return rc;
  }
  if (const int rc = SpillPartialReplayReadCannotReportSuccess(); rc != 0) {
    return rc;
  }
  return SpillRejectsArchiveSegmentRecordLargerThanConfiguredSegment();
}

} // namespace replay_spill

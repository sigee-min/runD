#include "test/assert.hpp"

#include "../local/model.hpp"

#include <rund/replay.hpp>

#include <algorithm>
#include <filesystem>
#include <vector>

namespace replay_spill {
namespace {

using rund::host::hash_bytes;
using rund::node::replay_detail::payload::Capture;
using rund::node::replay_detail::payload::Codec;

int SpillMatchesRawAndRleWithoutDecodedPayload() {
  const std::filesystem::path dir = TempDir("matches");
  std::filesystem::remove_all(dir);
  std::vector<std::byte> raw(257u);
  for (std::size_t index = 0u; index < raw.size(); ++index) {
    raw[index] = static_cast<std::byte>((index * 73u + 19u) & 0xffu);
  }
  const std::vector<std::byte> rle(257u, std::byte{0x4a});
  const StableHash raw_hash = hash_bytes(raw.data(), raw.size());
  const StableHash rle_hash = hash_bytes(rle.data(), rle.size());

  Store store = Prepared(Storage(dir, 1024u));
  TEST_ASSERT(
      store.Append(3u, EventKind::IoWrite, Capture::verify(raw, raw_hash)));
  TEST_ASSERT(
      store.Append(4u, EventKind::IoWrite, Capture::verify(rle, rle_hash)));
  TEST_ASSERT(store.blobs().size() == 2u);
  TEST_ASSERT(store.blobs()[0].codec == Codec::Raw);
  TEST_ASSERT(store.blobs()[1].codec == Codec::Rle);
  const auto &raw_record = store.records()[0];
  const auto &rle_record = store.records()[1];
  const Binding raw_binding{
      .event_sequence = raw_record.metadata.event_sequence,
      .kind = raw_record.metadata.kind,
      .completed_bytes = raw_record.metadata.completed_bytes,
      .payload_hash = raw_record.metadata.payload_hash};
  const Binding rle_binding{
      .event_sequence = rle_record.metadata.event_sequence,
      .kind = rle_record.metadata.kind,
      .completed_bytes = rle_record.metadata.completed_bytes,
      .payload_hash = rle_record.metadata.payload_hash};
  TEST_ASSERT(store.Matches(0u, raw_binding, raw).ok());
  TEST_ASSERT(store.Matches(1u, rle_binding, rle).ok());

  std::vector<std::byte> wrong_raw = raw;
  std::vector<std::byte> wrong_rle = rle;
  wrong_raw.back() ^= std::byte{0x01};
  wrong_rle.back() ^= std::byte{0x01};
  const auto rejected_raw = store.Matches(0u, raw_binding, wrong_raw);
  const auto rejected_rle = store.Matches(1u, rle_binding, wrong_rle);
  TEST_ASSERT(!rejected_raw.ok());
  TEST_ASSERT(!rejected_rle.ok());
  TEST_ASSERT(rejected_raw.code == rund::replay::Code::HostPayloadHashInvalid);
  TEST_ASSERT(rejected_rle.code == rund::replay::Code::HostPayloadHashInvalid);
  std::filesystem::remove_all(dir);
  return 0;
}

} // namespace

int LayoutContract() {
  const std::filesystem::path dir = TempDir("segments");
  std::filesystem::remove_all(dir);
  const std::vector<std::byte> first = Bytes("abcdefghijklmnopqrstuvwx");
  const std::vector<std::byte> second = Bytes("ABCDEFGHIJKLMNOPQRSTUVWX");
  const StableHash first_hash = hash_bytes(first.data(), first.size());
  const StableHash second_hash = hash_bytes(second.data(), second.size());

  const ::rund::replay::Storage storage = Storage(dir);
  Store store = Prepared(storage);
  TEST_ASSERT(
      store.Append(1u, EventKind::IoRead, Capture::verify(first, first_hash)));
  TEST_ASSERT(store.Append(2u, EventKind::IoWrite,
                           Capture::verify(second, second_hash)));
  TEST_ASSERT(store.records().size() == 2u);
  TEST_ASSERT(store.blobs().size() == 2u);
  TEST_ASSERT(store.retained_bytes() <= storage.cached_bytes);
  TEST_ASSERT(store.encoded_bytes() == first.size() + second.size());
  TEST_ASSERT(store.segment_count() == 2u);
  TEST_ASSERT(std::filesystem::exists(SegmentPath(dir, 0u)));
  TEST_ASSERT(std::filesystem::exists(SegmentPath(dir, 1u)));
  TEST_ASSERT(!std::filesystem::exists(SegmentPath(dir, 2u)));
  TEST_ASSERT(store.blobs()[0].codec == Codec::Raw);
  if (const int result =
          CanonicalSegment(SegmentPath(dir, 0u), first, first_hash);
      result != 0) {
    return result;
  }

  const auto &first_record = store.records()[0];
  const Binding first_binding{
      .event_sequence = first_record.metadata.event_sequence,
      .kind = first_record.metadata.kind,
      .completed_bytes = first_record.metadata.completed_bytes,
      .payload_hash = first_record.metadata.payload_hash,
  };
  std::vector<std::byte> replayed_first(first.size());
  TEST_ASSERT(store.ReadInto(0u, first_binding, replayed_first).ok());
  const ResolveResult resolved_second = store.Resolve(1u);
  TEST_ASSERT(replayed_first == first);
  TEST_ASSERT(resolved_second.ok());
  TEST_ASSERT(std::equal(resolved_second.bytes.span().begin(),
                         resolved_second.bytes.span().end(), second.begin(),
                         second.end()));

  store.Clear();
  TEST_ASSERT(std::filesystem::exists(dir));
  TEST_ASSERT(GenerationDirectories(dir).empty());
  std::filesystem::remove_all(dir);
  return SpillMatchesRawAndRleWithoutDecodedPayload();
}

} // namespace replay_spill

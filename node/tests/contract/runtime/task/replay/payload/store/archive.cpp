#include "test/assert.hpp"

#include "local/model.hpp"

#include <node/runtime/replay/host/payload.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace replay_payload_store {
namespace {

using rund::node::replay_detail::payload::Backend;
using rund::node::replay_detail::payload::Binding;
using rund::node::replay_detail::payload::Encode;
using rund::node::replay_detail::payload::Equal;
using rund::node::replay_detail::payload::kChunkBytes;
using rund::node::replay_detail::payload::Materialization;
using rund::node::replay_detail::payload::Materialize;
using rund::node::replay_detail::payload::MaterializedRecord;

int StoreDeduplicatesRepeatedPayloadChunks() {
  Store store = Prepared();
  const std::vector<std::byte> payload = Payload("repeat-payload");
  const StableHash payload_hash = hash_bytes(payload.data(), payload.size());

  TEST_ASSERT(store.Append(1u, EventKind::IoRead,
                           Capture::verify(payload, payload_hash)));
  TEST_ASSERT(store.Append(2u, EventKind::IoWrite,
                           Capture::verify(payload, payload_hash)));
  TEST_ASSERT(store.records().size() == 2u);
  TEST_ASSERT(store.blobs().size() == 1u);
  TEST_ASSERT(store.logical_bytes() == payload.size() * 2u);
  TEST_ASSERT(store.retained_bytes() == payload.size());
  TEST_ASSERT(store.pieces(store.records()[0])[0].blob_index ==
              store.pieces(store.records()[1])[0].blob_index);
  TEST_ASSERT(Build(store.Archive(), {}).ok());
  const auto &first_record = store.records()[0];
  const Binding first_binding{
      .event_sequence = first_record.metadata.event_sequence,
      .kind = first_record.metadata.kind,
      .completed_bytes = first_record.metadata.completed_bytes,
      .payload_hash = first_record.metadata.payload_hash,
  };
  std::vector<std::byte> replayed(payload.size());
  TEST_ASSERT(store.ReadInto(0u, first_binding, replayed).ok());
  TEST_ASSERT(replayed == payload);

  const Materialization expected = Materialize(std::vector<MaterializedRecord>{
      MaterializedRecord{.metadata =
                             {
                                 .event_sequence = 1u,
                                 .kind = EventKind::IoRead,
                                 .completed_bytes = payload.size(),
                                 .payload_hash = payload_hash,
                             },
                         .bytes = payload},
      MaterializedRecord{.metadata =
                             {
                                 .event_sequence = 2u,
                                 .kind = EventKind::IoWrite,
                                 .completed_bytes = payload.size(),
                                 .payload_hash = payload_hash,
                             },
                         .bytes = payload},
  });
  TEST_ASSERT(Equal(expected, Materialize(store)));
  return 0;
}

int ArchiveValidatesDeduplicatedMultiPiecePayloads() {
  Store store = Prepared();
  std::vector<std::byte> shared(kChunkBytes, std::byte{0x4a});
  std::vector<std::byte> combined = shared;
  combined.insert(combined.end(), 17u, std::byte{0x39});
  const StableHash shared_hash = hash_bytes(shared.data(), shared.size());
  const StableHash combined_hash = hash_bytes(combined.data(), combined.size());

  TEST_ASSERT(store.Append(1u, EventKind::IoRead,
                           Capture::verify(shared, shared_hash)));
  TEST_ASSERT(store.Append(2u, EventKind::IoWrite,
                           Capture::verify(combined, combined_hash)));
  const Archive valid = store.Archive();
  TEST_ASSERT(valid.records.size() == 2u);
  TEST_ASSERT(valid.chunks.size() == 2u);
  TEST_ASSERT(valid.records[0].pieces.size() == 1u);
  TEST_ASSERT(valid.records[1].pieces.size() == 2u);
  TEST_ASSERT(valid.records[0].pieces[0].chunk_id ==
              valid.records[1].pieces[0].chunk_id);
  TEST_ASSERT(valid.chunks[0].codec == Codec::Rle);
  TEST_ASSERT(Build(valid, {}).ok());

  Archive forged = valid;
  forged.records[1].metadata.payload_hash.value ^= 1u;
  forged.payload_hash = Identity(forged);
  TEST_ASSERT(!Build(std::move(forged), {}).ok());
  return 0;
}

int BackendHashIndexKeepsCollisionCandidatesInInsertionOrder() {
  const std::vector<std::byte> first = Payload("first");
  const std::vector<std::byte> second = Payload("second");
  const StableHash collision{.value = 91u};
  Backend backend{{}, 3u};
  std::vector<rund::node::replay_detail::payload::Blob> blobs{};
  blobs.reserve(3u);
  blobs.push_back(Encode(first, collision));
  blobs.push_back(Encode(second, collision));
  blobs.push_back(Encode(first, collision));

  TEST_ASSERT(backend.Append(blobs).ok());

  const auto first_match = backend.Find(first, collision);
  const auto second_match = backend.Find(second, collision);
  TEST_ASSERT(first_match.has_value());
  TEST_ASSERT(second_match.has_value());
  TEST_ASSERT(*first_match == 0u);
  TEST_ASSERT(*second_match == 1u);
  return 0;
}

[[nodiscard]] std::array<std::byte, 8u>
SequentialPayload(const std::size_t index) {
  std::array<std::byte, 8u> bytes{};
  for (std::size_t byte = 0u; byte < bytes.size(); ++byte) {
    bytes[byte] = static_cast<std::byte>(
        ((index >> ((byte % sizeof(index)) * 8u)) ^ (byte * 37u)) & 0xffu);
  }
  return bytes;
}

int StoreResolvesCanonicalSequenceByIndex() {
  constexpr std::size_t kRecordCount = 2048u;
  Store store = Prepared(kRecordCount, 0u);
  for (std::size_t index = 0u; index < kRecordCount; ++index) {
    auto bytes = SequentialPayload(index);
    const StableHash hash = hash_bytes(bytes.data(), bytes.size());
    const EventKind kind =
        (index & 1u) == 0u ? EventKind::IoRead : EventKind::IoWrite;
    TEST_ASSERT(store.Append(static_cast<std::uint64_t>(index + 100u), kind,
                             Capture::verify(bytes, hash)));
  }

  for (std::size_t index = 0u; index < kRecordCount; ++index) {
    auto bytes = SequentialPayload(index);
    const auto &record = store.records()[index];
    const Binding binding{.event_sequence = record.metadata.event_sequence,
                          .kind = record.metadata.kind,
                          .completed_bytes = record.metadata.completed_bytes,
                          .payload_hash = record.metadata.payload_hash};
    if (record.metadata.kind == EventKind::IoRead) {
      bytes.fill(std::byte{0});
      TEST_ASSERT(store.ReadInto(index, binding, bytes).ok());
      TEST_ASSERT(bytes == SequentialPayload(index));
    } else {
      TEST_ASSERT(store.Matches(index, binding, bytes).ok());
    }
  }

  const auto &first = store.records().front();
  const Binding wrong{.event_sequence = first.metadata.event_sequence + 1u,
                      .kind = first.metadata.kind,
                      .completed_bytes = first.metadata.completed_bytes,
                      .payload_hash = first.metadata.payload_hash};
  std::array<std::byte, 8u> rejected_bytes{};
  rejected_bytes.fill(std::byte{0x7d});
  const auto rejected = store.ReadInto(0u, wrong, rejected_bytes);
  TEST_ASSERT(!rejected.ok());
  TEST_ASSERT(rejected.code == rund::replay::Code::HostPayloadMismatch);
  TEST_ASSERT(std::all_of(
      rejected_bytes.begin(), rejected_bytes.end(),
      [](const std::byte value) { return value == std::byte{0x7d}; }));
  return 0;
}

int StoreSplitsLargePayloadIntoFixedChunks() {
  Store store = Prepared();
  std::vector<std::byte> payload(kChunkBytes * 2u + 7u);
  for (std::size_t index = 0u; index < payload.size(); ++index) {
    payload[index] = static_cast<std::byte>(index & 0xffu);
  }
  payload[kChunkBytes] = std::byte{0x9a};
  const StableHash payload_hash = hash_bytes(payload.data(), payload.size());

  TEST_ASSERT(store.Append(3u, EventKind::IoRead,
                           Capture::verify(payload, payload_hash)));
  TEST_ASSERT(store.records().size() == 1u);
  TEST_ASSERT(store.pieces(store.records()[0]).size() == 3u);
  TEST_ASSERT(store.blobs().size() == 3u);
  TEST_ASSERT(store.blobs()[0].uncompressed_bytes == kChunkBytes);
  TEST_ASSERT(store.blobs()[1].uncompressed_bytes == kChunkBytes);
  TEST_ASSERT(store.blobs()[2].uncompressed_bytes == 7u);
  TEST_ASSERT(store.logical_bytes() == payload.size());
  TEST_ASSERT(store.retained_bytes() == payload.size());
  TEST_ASSERT(store.encoded_bytes() == payload.size());
  return 0;
}

int StoreKeepsEncodedChunksAsAuthoritativeBytes() {
  Store store = Prepared();
  std::vector<std::byte> payload(kChunkBytes, std::byte{0x44});
  const StableHash payload_hash = hash_bytes(payload.data(), payload.size());

  TEST_ASSERT(store.Append(8u, EventKind::IoRead,
                           Capture::verify(payload, payload_hash)));
  TEST_ASSERT(store.records().size() == 1u);
  TEST_ASSERT(store.blobs().size() == 1u);
  TEST_ASSERT(store.blobs()[0].codec == Codec::Rle);
  TEST_ASSERT(store.blobs()[0].encoded.size() < payload.size());
  TEST_ASSERT(store.encoded_bytes() == store.blobs()[0].encoded.size());
  TEST_ASSERT(store.retained_bytes() == store.encoded_bytes());
  const auto &record = store.records()[0];
  const Binding binding{.event_sequence = record.metadata.event_sequence,
                        .kind = record.metadata.kind,
                        .completed_bytes = record.metadata.completed_bytes,
                        .payload_hash = record.metadata.payload_hash};
  std::vector<std::byte> replayed(payload.size());
  TEST_ASSERT(store.ReadInto(0u, binding, replayed).ok());
  TEST_ASSERT(replayed == payload);
  return 0;
}

int StoreRejectsPayloadsBeyondStorageBudgetBeforeAcceptingRecord() {
  ::rund::replay::Storage storage{};
  storage.max_bytes = 3u;
  Store store = Prepared(1u, 0u, storage.max_bytes, storage);
  const std::vector<std::byte> payload = Payload("abcd");
  const StableHash payload_hash = hash_bytes(payload.data(), payload.size());

  TEST_ASSERT(!store.Append(9u, EventKind::IoRead,
                            Capture::verify(payload, payload_hash)));
  TEST_ASSERT(store.records().empty());
  TEST_ASSERT(store.blobs().empty());
  TEST_ASSERT(store.logical_bytes() == 0u);
  TEST_ASSERT(store.encoded_bytes() == 0u);
  TEST_ASSERT(store.retained_bytes() == 0u);
  return 0;
}

} // namespace

int ArchiveContract() {
  return Run(std::array<Contract, 7u>{
      StoreDeduplicatesRepeatedPayloadChunks,
      ArchiveValidatesDeduplicatedMultiPiecePayloads,
      BackendHashIndexKeepsCollisionCandidatesInInsertionOrder,
      StoreResolvesCanonicalSequenceByIndex,
      StoreSplitsLargePayloadIntoFixedChunks,
      StoreKeepsEncodedChunksAsAuthoritativeBytes,
      StoreRejectsPayloadsBeyondStorageBudgetBeforeAcceptingRecord,
  });
}

} // namespace replay_payload_store

#include "test/assert.hpp"

#include "local/model.hpp"

#include <node/runtime/replay/host/payload.hpp>
#include <rund/host/event.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

namespace replay_payload_store {
namespace {

using rund::StableHash;
using rund::host::EventKind;
using rund::host::hash_bytes;
using rund::node::replay_detail::payload::ArchiveRecord;
using rund::node::replay_detail::payload::Build;
using rund::node::replay_detail::payload::Capture;
using rund::node::replay_detail::payload::Equal;
using rund::node::replay_detail::payload::kChunkBytes;
using rund::node::replay_detail::payload::Materialization;
using rund::node::replay_detail::payload::Materialize;
using rund::node::replay_detail::payload::MaterializedRecord;
using rund::node::replay_detail::payload::Record;
using rund::node::replay_detail::payload::Store;
using rund::node::replay_detail::payload::StoredRecord;

static_assert(std::is_same_v<decltype(ArchiveRecord{}.metadata), Record>);
static_assert(std::is_same_v<decltype(StoredRecord{}.metadata), Record>);
static_assert(std::is_same_v<decltype(MaterializedRecord{}.metadata), Record>);

int StoreRejectsWrongRecordPayloadHash() {
  Store store = Prepared();
  const std::vector<std::byte> payload = Payload("payload");
  const StableHash wrong_hash{
      .value = hash_bytes(payload.data(), payload.size()).value ^ 1u};

  TEST_ASSERT(!store.Append(4u, EventKind::IoRead,
                            Capture::verify(payload, wrong_hash)));
  TEST_ASSERT(store.records().empty());
  TEST_ASSERT(store.blobs().empty());
  TEST_ASSERT(store.logical_bytes() == 0u);
  TEST_ASSERT(store.retained_bytes() == 0u);

  const Materialization invalid = Materialize(std::vector<MaterializedRecord>{
      MaterializedRecord{.metadata =
                             {
                                 .event_sequence = 4u,
                                 .kind = EventKind::IoRead,
                                 .completed_bytes = payload.size(),
                                 .payload_hash = wrong_hash,
                             },
                         .bytes = payload},
  });
  const auto built = Build(invalid);
  TEST_ASSERT(!built.ok());
  TEST_ASSERT(built.code == rund::replay::Code::HostPayloadHashInvalid);
  return 0;
}

int StoreKeepsZeroBytePayloadRecordsLogical() {
  Store store = Prepared();
  const std::vector<std::byte> payload{};
  const StableHash payload_hash = hash_bytes(nullptr, 0u);

  TEST_ASSERT(store.Append(7u, EventKind::IoRead,
                           Capture::verify(payload, payload_hash)));
  TEST_ASSERT(store.records().size() == 1u);
  TEST_ASSERT(store.pieces(store.records()[0]).empty());
  TEST_ASSERT(store.blobs().empty());
  TEST_ASSERT(store.logical_bytes() == 0u);
  TEST_ASSERT(store.retained_bytes() == 0u);

  const Materialization expected = Materialize(std::vector<MaterializedRecord>{
      MaterializedRecord{.metadata =
                             {
                                 .event_sequence = 7u,
                                 .kind = EventKind::IoRead,
                                 .completed_bytes = 0u,
                                 .payload_hash = payload_hash,
                             },
                         .bytes = payload},
  });
  TEST_ASSERT(store.payload_hash() == expected.payload_hash);
  TEST_ASSERT(Equal(expected, Materialize(store)));

  const auto built = Build(expected);
  TEST_ASSERT(built.ok());
  TEST_ASSERT(Equal(expected, Materialize(built.store)));
  return 0;
}

int StoreMaterializedHashMatchesPublicEvidenceHash() {
  Store store = Prepared();
  const std::vector<std::byte> first = Payload("abc");
  std::vector<std::byte> second(kChunkBytes + 3u, std::byte{0x42});
  second[kChunkBytes] = std::byte{0x43};

  const Materialization canonical =
      Materialize(std::vector<MaterializedRecord>{MaterializedRecord{
          .metadata =
              {
                  .event_sequence = 5u,
                  .kind = EventKind::IoRead,
                  .completed_bytes = first.size(),
                  .payload_hash = hash_bytes(first.data(), first.size()),
              },
          .bytes = first}});
  // Fixes metadata plus ordered (chunk hash, chunk bytes) framing
  // independently from the three production paths that share it.
  TEST_ASSERT(canonical.payload_hash == 0x127fb5bc7da22141ull);

  TEST_ASSERT(store.Append(5u, EventKind::IoRead, Capture::read(first)));
  TEST_ASSERT(store.Append(6u, EventKind::IoWrite, Capture::read(second)));

  const Materialization expected = Materialize(std::vector<MaterializedRecord>{
      MaterializedRecord{
          .metadata =
              {
                  .event_sequence = 5u,
                  .kind = EventKind::IoRead,
                  .completed_bytes = first.size(),
                  .payload_hash = hash_bytes(first.data(), first.size()),
              },
          .bytes = first},
      MaterializedRecord{
          .metadata =
              {
                  .event_sequence = 6u,
                  .kind = EventKind::IoWrite,
                  .completed_bytes = second.size(),
                  .payload_hash = hash_bytes(second.data(), second.size()),
              },
          .bytes = second},
  });
  const Materialization materialized = Materialize(store);
  TEST_ASSERT(store.payload_hash() == expected.payload_hash);
  TEST_ASSERT(Equal(expected, materialized));

  const auto built = Build(expected);
  TEST_ASSERT(built.ok());
  TEST_ASSERT(built.code == rund::replay::Code::Ok);
  TEST_ASSERT(Equal(expected, Materialize(built.store)));
  return 0;
}

int StoreBuildAcceptsDefaultConstructedEmptyEvidence() {
  const auto built = Build(Materialization{});

  TEST_ASSERT(built.ok());
  TEST_ASSERT(built.code == rund::replay::Code::Ok);
  TEST_ASSERT(built.store.records().empty());
  TEST_ASSERT(built.store.blobs().empty());
  TEST_ASSERT(built.store.logical_bytes() == 0u);
  return 0;
}

int MaterializeTransfersOwnedRecordBuffers() {
  std::vector<std::byte> bytes = Payload("owned-materialization");
  const StableHash hash = hash_bytes(bytes.data(), bytes.size());
  std::vector<MaterializedRecord> records{};
  records.push_back(MaterializedRecord{.metadata =
                                           {
                                               .event_sequence = 19u,
                                               .kind = EventKind::IoRead,
                                               .completed_bytes = bytes.size(),
                                               .payload_hash = hash,
                                           },
                                       .bytes = std::move(bytes)});
  const MaterializedRecord *const record_data = records.data();
  const std::byte *const payload_data = records[0].bytes.data();
  const std::size_t payload_capacity = records[0].bytes.capacity();

  const Materialization materialized = Materialize(std::move(records));

  TEST_ASSERT(materialized.records.data() == record_data);
  TEST_ASSERT(materialized.records[0].bytes.data() == payload_data);
  TEST_ASSERT(materialized.records[0].bytes.capacity() == payload_capacity);
  return 0;
}

} // namespace

int MaterializationContract() {
  return Run(std::array<Contract, 5u>{
      StoreRejectsWrongRecordPayloadHash,
      StoreKeepsZeroBytePayloadRecordsLogical,
      StoreMaterializedHashMatchesPublicEvidenceHash,
      StoreBuildAcceptsDefaultConstructedEmptyEvidence,
      MaterializeTransfersOwnedRecordBuffers,
  });
}

} // namespace replay_payload_store

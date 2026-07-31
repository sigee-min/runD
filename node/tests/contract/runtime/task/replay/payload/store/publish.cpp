#include "test/assert.hpp"

#include "local/model.hpp"

#include <node/runtime/replay/host/payload.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace replay_payload_store {
namespace {

using rund::node::replay_detail::payload::ArchiveChunk;
using rund::node::replay_detail::payload::ArchivePiece;
using rund::node::replay_detail::payload::ArchiveRecord;
using rund::node::replay_detail::payload::BuildResult;
using rund::node::replay_detail::payload::Bytes;
using rund::node::replay_detail::payload::InputSourceRange;
using rund::node::replay_detail::payload::ResolveResult;

int StorePublishesOneCompactOwner() {
  Store recorded = Prepared();
  const std::vector<std::byte> first = Payload("first-payload");
  const std::vector<std::byte> second = Payload("second-payload");
  const StableHash first_hash = hash_bytes(first.data(), first.size());
  const StableHash second_hash = hash_bytes(second.data(), second.size());
  TEST_ASSERT(recorded.Append(10u, EventKind::IoRead,
                              Capture::verify(first, first_hash)));
  TEST_ASSERT(recorded.Append(11u, EventKind::IoWrite,
                              Capture::verify(second, second_hash)));

  const std::byte *const store_bytes = recorded.blobs()[0].encoded.data();
  const std::size_t encoded_bytes = static_cast<std::size_t>(
      recorded.blobs()[0].encoded.size() + recorded.blobs()[1].encoded.size());
  Archive retained = recorded.Archive();
  TEST_ASSERT(retained.chunks.size() == 2u);
  TEST_ASSERT(!retained.chunks[0].encoded.empty());
  TEST_ASSERT(retained.chunks[0].encoded.data() != store_bytes);
  TEST_ASSERT(retained.chunks[1].encoded.data() ==
              retained.chunks[0].encoded.data() +
                  retained.chunks[0].encoded.size());
  TEST_ASSERT(retained.chunks[0].encoded.retained_bytes() == encoded_bytes);
  TEST_ASSERT(retained.chunks[1].encoded.retained_bytes() == encoded_bytes);
  TEST_ASSERT(retained.storage.retained_bytes == encoded_bytes);
  TEST_ASSERT(retained.storage.copied_bytes == encoded_bytes);
  TEST_ASSERT(retained.storage.growths == 0u);
  const Archive copied = retained;
  TEST_ASSERT(copied.chunks[0].encoded.data() ==
              retained.chunks[0].encoded.data());

  const Archive repeated = recorded.Archive();
  TEST_ASSERT(repeated.chunks[0].encoded.data() ==
              retained.chunks[0].encoded.data());
  TEST_ASSERT(repeated.storage.retained_bytes == encoded_bytes);
  TEST_ASSERT(repeated.storage.copied_bytes == 0u);

  BuildResult moved{};
  {
    Archive archive = retained;
    moved = Build(std::move(archive), {});
  }
  TEST_ASSERT(moved.ok());
  TEST_ASSERT(moved.store.blobs().size() == 2u);
  TEST_ASSERT(moved.store.blobs()[0].encoded.data() ==
              retained.chunks[0].encoded.data());
  const Archive replayed = moved.store.Archive();
  TEST_ASSERT(replayed.chunks.size() == 2u);
  TEST_ASSERT(replayed.chunks[0].encoded.data() ==
              retained.chunks[0].encoded.data());
  TEST_ASSERT(replayed.storage.copied_bytes == 0u);
  const ResolveResult resolved = moved.store.Resolve(0u);
  TEST_ASSERT(resolved.ok());
  TEST_ASSERT(std::equal(resolved.bytes.span().begin(),
                         resolved.bytes.span().end(), first.begin(),
                         first.end()));
  const std::vector<std::byte> later = Payload("later-scope-bytes");
  const StableHash later_hash = hash_bytes(later.data(), later.size());
  TEST_ASSERT(recorded.Append(12u, EventKind::IoWrite,
                              Capture::verify(later, later_hash)));
  TEST_ASSERT(retained.records.size() == 2u);
  TEST_ASSERT(retained.chunks.size() == 2u);
  recorded.Clear();
  TEST_ASSERT(recorded.records().empty());
  TEST_ASSERT(recorded.blobs().empty());
  TEST_ASSERT(retained.chunks[0].encoded.data() ==
              copied.chunks[0].encoded.data());
  return 0;
}

int StoreAdoptsCompactOwnerAcrossEmptyChunks() {
  Store recorded = Prepared();
  const std::vector<std::byte> payload = Payload("nonempty");
  const StableHash payload_hash = hash_bytes(payload.data(), payload.size());
  TEST_ASSERT(recorded.Append(2u, EventKind::IoRead,
                              Capture::verify(payload, payload_hash)));
  Archive archive = recorded.Archive();
  TEST_ASSERT(archive.chunks.size() == 1u);
  const std::byte *const owner = archive.chunks.front().encoded.data();

  archive.chunks.front().chunk_id = 1u;
  archive.records.front().pieces.front().chunk_id = 1u;
  archive.chunks.insert(
      archive.chunks.begin(),
      ArchiveChunk{.chunk_id = 0u,
                   .codec = Codec::Raw,
                   .uncompressed_hash = hash_bytes(nullptr, 0u)});
  ArchiveRecord empty{
      .metadata =
          {
              .event_sequence = 1u,
              .kind = EventKind::IoRead,
              .payload_hash = hash_bytes(nullptr, 0u),
          },
  };
  empty.pieces.push_back(ArchivePiece{.chunk_id = 0u});
  archive.records.insert(archive.records.begin(), std::move(empty));
  archive.storage.chunk_count = archive.chunks.size();
  archive.payload_hash = Identity(archive);

  const BuildResult built = Build(archive, {});
  TEST_ASSERT(built.ok());
  TEST_ASSERT(built.store.blobs().size() == 2u);
  TEST_ASSERT(built.store.blobs()[1].encoded.data() == owner);
  const Archive republished = built.store.Archive();
  TEST_ASSERT(republished.storage.copied_bytes == 0u);
  TEST_ASSERT(republished.chunks[1].encoded.data() == owner);
  return 0;
}

int TinyInputsReleaseThePreparedOwner() {
  constexpr std::size_t kPreparedBytes = 1024u * 1024u;
  constexpr std::size_t kInputBytes = 1u;
  constexpr std::size_t kRecords = 100u;
  auto prepared =
      std::make_shared<std::vector<std::byte>>(kPreparedBytes, std::byte{0});
  TEST_ASSERT(prepared->capacity() == kPreparedBytes);

  Store store = Prepared(0u, 1u, kPreparedBytes);
  std::vector<Archive> retained{};
  retained.reserve(kRecords);
  for (std::size_t index = 0u; index < kRecords; ++index) {
    (*prepared)[0] = static_cast<std::byte>(index & 0xffu);
    {
      Bytes input = Bytes::share(prepared, 0u, kInputBytes);
      TEST_ASSERT(input.retained_bytes() == kPreparedBytes);
      const auto source_hash = store.SourceRangeHash(0u, {}, 0u, 0u);
      TEST_ASSERT(source_hash.has_value());
      TEST_ASSERT(store.AppendInput(
          41u, 7u, index, InputSourceRange{.hash = *source_hash}, input,
          Capture::verify(input.span(),
                          hash_bytes(input.data(), input.size()))));

      Archive archive = store.Archive();
      TEST_ASSERT(archive.chunks.size() == 1u);
      TEST_ASSERT(archive.chunks[0].encoded.size() == kInputBytes);
      TEST_ASSERT(archive.chunks[0].encoded.retained_bytes() == kInputBytes);
      TEST_ASSERT(archive.chunks[0].encoded.data() != prepared->data());
      TEST_ASSERT(archive.storage.retained_bytes == kInputBytes);
      TEST_ASSERT(archive.storage.copied_bytes == kInputBytes);
      TEST_ASSERT(archive.chunks[0].encoded.span()[0] == (*prepared)[0]);

      const Archive repeated = store.Archive();
      TEST_ASSERT(repeated.chunks[0].encoded.data() ==
                  archive.chunks[0].encoded.data());
      TEST_ASSERT(repeated.storage.retained_bytes == kInputBytes);
      TEST_ASSERT(repeated.storage.copied_bytes == 0u);
      retained.push_back(std::move(archive));
      store.Clear();
    }
    TEST_ASSERT(prepared.use_count() == 1u);
  }

  std::uint64_t total = 0u;
  for (const Archive &archive : retained) {
    total += archive.storage.retained_bytes;
    TEST_ASSERT(archive.chunks[0].encoded.retained_bytes() == kInputBytes);
  }
  TEST_ASSERT(total == kRecords * kInputBytes);
  TEST_ASSERT(total != kRecords * kPreparedBytes);
  return 0;
}

} // namespace

int PublishContract() {
  return Run(std::array<Contract, 3u>{
      StorePublishesOneCompactOwner,
      StoreAdoptsCompactOwnerAcrossEmptyChunks,
      TinyInputsReleaseThePreparedOwner,
  });
}

} // namespace replay_payload_store

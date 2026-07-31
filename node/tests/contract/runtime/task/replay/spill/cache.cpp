#include "test/assert.hpp"

#include "local/model.hpp"

#include <filesystem>
#include <optional>
#include <vector>

namespace replay_spill {

using rund::host::hash_bytes;
using rund::node::replay_detail::payload::Cache;
using rund::node::replay_detail::payload::Capture;
using rund::node::replay_detail::payload::kChunkBytes;

int RunMemoryCacheContract() {
  Cache cache{8u};
  const std::vector<std::byte> one(4u, std::byte{0x01});
  const std::vector<std::byte> two(4u, std::byte{0x02});
  const std::vector<std::byte> three(4u, std::byte{0x03});

  cache.Put(1u, one);
  cache.Put(2u, two);

  std::vector<std::byte> out(4u);
  TEST_ASSERT(cache.Read(1u, out) == std::optional<bool>{true});
  TEST_ASSERT(out == one);

  cache.Put(3u, three);

  TEST_ASSERT(cache.Read(1u, out) == std::optional<bool>{true});
  TEST_ASSERT(out == one);
  TEST_ASSERT(!cache.Read(2u, out).has_value());
  TEST_ASSERT(cache.Read(3u, out) == std::optional<bool>{true});
  TEST_ASSERT(out == three);

  const ::rund::replay::StorageReport report = cache.Report();
  TEST_ASSERT(report.cached_bytes == 8u);
  TEST_ASSERT(report.cache_hits == 3u);
  TEST_ASSERT(report.cache_misses == 1u);
  TEST_ASSERT(report.cache_evictions == 1u);
  TEST_ASSERT(report.growths > 0u);
  TEST_ASSERT(report.chunk_count == 0u);
  TEST_ASSERT(report.logical_bytes == 0u);
  TEST_ASSERT(report.encoded_bytes == 0u);
  return 0;
}

int RunCacheContract() {
  if (const int rc = RunMemoryCacheContract(); rc != 0) {
    return rc;
  }
  constexpr std::size_t kRecordCount = 8u;
  const std::filesystem::path dir = TempDir("cache");
  std::filesystem::remove_all(dir);
  Store record_store = Prepared(CacheStorage(dir));
  std::vector<std::byte> repeated(kChunkBytes);
  for (std::size_t index = 0u; index < repeated.size(); ++index) {
    repeated[index] = static_cast<std::byte>((index * 17u + 23u) & 0xffu);
  }

  for (std::size_t record = 0u; record < kRecordCount; ++record) {
    std::vector<std::byte> payload = repeated;
    if ((record % 4u) != 0u) {
      for (std::size_t index = 0u; index < payload.size(); ++index) {
        payload[index] =
            static_cast<std::byte>((index * 31u + record * 13u) & 0xffu);
      }
    }
    const StableHash hash = hash_bytes(payload.data(), payload.size());
    TEST_ASSERT(record_store.Append(static_cast<std::uint64_t>(record + 1u),
                                    EventKind::IoRead,
                                    Capture::verify(payload, hash)));
  }

  const rund::node::replay_detail::payload::Archive archive =
      record_store.Archive();
  TEST_ASSERT(archive.records.size() == kRecordCount);
  TEST_ASSERT(archive.storage.segment_count > 1u);
  TEST_ASSERT(archive.payload_hash == record_store.payload_hash());

  Store replay_store = Prepared(CacheStorage(dir));
  TEST_ASSERT(replay_store.LoadArchive(archive));
  for (std::size_t index = 0u; index < archive.records.size(); ++index) {
    const rund::node::replay_detail::payload::ArchiveRecord &record =
        archive.records[index];
    const ResolveResult resolved = replay_store.Resolve(index);
    TEST_ASSERT(resolved.ok());
    TEST_ASSERT(resolved.bytes.size() ==
                static_cast<std::size_t>(record.metadata.completed_bytes));
    TEST_ASSERT(
        hash_bytes(resolved.bytes.data(), resolved.bytes.size()).value ==
        record.metadata.payload_hash.value);
  }
  const ResolveResult cached =
      replay_store.Resolve(archive.records.size() - 1u);
  TEST_ASSERT(cached.ok());
  TEST_ASSERT(cached.bytes.size() ==
              static_cast<std::size_t>(
                  archive.records.back().metadata.completed_bytes));
  TEST_ASSERT(hash_bytes(cached.bytes.data(), cached.bytes.size()).value ==
              archive.records.back().metadata.payload_hash.value);
  const rund::node::replay_detail::payload::Archive replay_archive =
      replay_store.Archive();
  TEST_ASSERT(replay_archive.payload_hash == archive.payload_hash);
  TEST_ASSERT(replay_archive.storage.cache_misses > 0u);
  TEST_ASSERT(replay_archive.storage.cache_hits > 0u);
  TEST_ASSERT(replay_archive.storage.cache_evictions > 0u);
  TEST_ASSERT(replay_archive.storage.growths > 0u);
  TEST_ASSERT(replay_archive.storage.cached_bytes <=
              CacheStorage(dir).cached_bytes);
  TEST_ASSERT(replay_store.retained_bytes() > 0u);
  TEST_ASSERT(replay_store.retained_bytes() <= CacheStorage(dir).cached_bytes);

  std::filesystem::remove_all(dir);
  return 0;
}

} // namespace replay_spill

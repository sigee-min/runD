#include "test/assert.hpp"

#include "../local/model.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace replay_spill {

using rund::host::hash_bytes;

int AppendContract() {
  const std::filesystem::path root = TempDir("load-last-segment");
  std::filesystem::remove_all(root);
  ::rund::replay::Storage storage = Storage(root, 100u);
  storage.max_allocated_bytes = 64u * 1024u;
  storage.budget = ::rund::storage::Budget{storage.max_allocated_bytes};
  const std::vector<std::byte> first =
      Bytes("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN");
  const std::vector<std::byte> second = Bytes("small");
  const std::vector<std::byte> third = Bytes("later!");
  const StableHash first_hash = hash_bytes(first.data(), first.size());
  const StableHash second_hash = hash_bytes(second.data(), second.size());
  const StableHash third_hash = hash_bytes(third.data(), third.size());

  rund::node::replay_detail::payload::Spill recorded{storage, 8u};
  TEST_ASSERT(
      recorded
          .Append(rund::node::replay_detail::payload::Encode(first, first_hash))
          .ok());
  TEST_ASSERT(recorded
                  .Append(rund::node::replay_detail::payload::Encode(
                      second, second_hash))
                  .ok());
  TEST_ASSERT(recorded.segment_count() == 2u);
  TEST_ASSERT(std::filesystem::file_size(SegmentPath(root, 0u)) ==
              first.size() + 41u);
  TEST_ASSERT(std::filesystem::file_size(SegmentPath(root, 1u)) ==
              second.size() + 41u);

  std::vector<rund::node::replay_detail::payload::Blob> blobs =
      recorded.blobs();
  std::vector<rund::node::replay_detail::payload::SpillRef> refs =
      recorded.refs();
  std::shared_ptr<const rund::node::replay_detail::payload::SpillGeneration>
      generation = recorded.generation();
  rund::node::replay_detail::payload::Spill loaded{storage, 8u};
  loaded.LoadArchive(std::move(blobs), std::move(refs),
                     recorded.encoded_bytes(), generation);
  TEST_ASSERT(
      loaded
          .Append(rund::node::replay_detail::payload::Encode(third, third_hash))
          .ok());
  TEST_ASSERT(loaded.segment_count() == 2u);
  TEST_ASSERT(std::filesystem::file_size(SegmentPath(root, 1u)) ==
              second.size() + third.size() + 82u);
  TEST_ASSERT(!std::filesystem::exists(SegmentPath(root, 2u)));
  const rund::node::replay_detail::payload::ReadResult resolved =
      loaded.Read(2u);
  TEST_ASSERT(resolved.ok());
  TEST_ASSERT(resolved.bytes == third);

  loaded.Clear();
  recorded.Clear();
  TEST_ASSERT(!GenerationDirectories(root).empty());
  generation.reset();
  TEST_ASSERT(GenerationDirectories(root).empty());
  TEST_ASSERT(storage.budget.report().allocated_bytes == 0u);
  TEST_ASSERT(storage.budget.report().physical_bytes == 0u);
  std::filesystem::remove_all(root);
  return 0;
}

} // namespace replay_spill

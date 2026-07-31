#include "test/assert.hpp"

#include "../local/model.hpp"

#include <rund/replay.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

namespace replay_spill {

using rund::host::hash_bytes;
using rund::node::replay_detail::payload::Capture;

int GenerationContract() {
  const std::filesystem::path root = TempDir("generation-owner");
  std::filesystem::remove_all(root);
  TEST_ASSERT(std::filesystem::create_directories(root));
  const std::filesystem::path foreign = root / "caller-owned.txt";
  {
    std::ofstream out{foreign};
    out << "preserve";
    TEST_ASSERT(out.good());
  }
  const std::filesystem::path unmarked =
      root / ".rund-replay-spill-v1-caller-owned";
  TEST_ASSERT(std::filesystem::create_directory(unmarked));
  const std::filesystem::path stale = root / ".rund-replay-spill-v1-stale";
  TEST_ASSERT(std::filesystem::create_directory(stale));
  {
    std::ofstream marker{stale / ".rund-owner", std::ios::binary};
    marker << "runD replay spill generation v1\n";
    TEST_ASSERT(marker.good());
  }
  {
    std::ofstream lease{stale / ".rund-lease", std::ios::binary};
    TEST_ASSERT(lease.good());
  }

  ::rund::replay::Storage storage = Storage(root, 1024u);
  storage.max_allocated_bytes = 16u * 1024u;
  storage.budget = ::rund::storage::Budget{storage.max_allocated_bytes};
  const std::vector<std::byte> first_bytes =
      Bytes("first immutable spill generation");
  const std::vector<std::byte> second_bytes =
      Bytes("second distinct spill generation");
  const StableHash first_hash =
      hash_bytes(first_bytes.data(), first_bytes.size());
  const StableHash second_hash =
      hash_bytes(second_bytes.data(), second_bytes.size());

  Store first = Prepared(storage);
  TEST_ASSERT(first.Append(1u, EventKind::IoRead,
                           Capture::verify(first_bytes, first_hash)));
  TEST_ASSERT(!std::filesystem::exists(stale));
  TEST_ASSERT(std::filesystem::exists(unmarked));
  rund::node::replay_detail::payload::Archive first_archive = first.Archive();
  TEST_ASSERT(first_archive.storage.physical_bytes ==
              first_archive.storage.encoded_bytes + 41u);
  TEST_ASSERT(first_archive.storage.allocated_bytes >=
              first_archive.storage.physical_bytes);
  TEST_ASSERT(first_archive.storage.reserved_bytes == 0u);
  std::vector<std::filesystem::path> generations = GenerationDirectories(root);
  TEST_ASSERT(generations.size() == 1u);
  const std::filesystem::path first_generation = generations.front();
  TEST_ASSERT(SegmentBytes(first_generation) ==
              first_archive.storage.physical_bytes);
  TEST_ASSERT(AllocatedBytes(first_generation) ==
              first_archive.storage.allocated_bytes);

  first.Clear();
  TEST_ASSERT(std::filesystem::exists(first_generation));

  Store second = Prepared(storage);
  TEST_ASSERT(second.Append(2u, EventKind::IoRead,
                            Capture::verify(second_bytes, second_hash)));
  rund::node::replay_detail::payload::Archive second_archive = second.Archive();
  generations = GenerationDirectories(root);
  TEST_ASSERT(generations.size() == 2u);
  TEST_ASSERT(std::find(generations.begin(), generations.end(),
                        first_generation) != generations.end());
  const ::rund::storage::Report both_usage = storage.budget.report();
  TEST_ASSERT(both_usage.physical_bytes ==
              first_archive.storage.physical_bytes +
                  second_archive.storage.physical_bytes);
  TEST_ASSERT(both_usage.allocated_bytes ==
              first_archive.storage.allocated_bytes +
                  second_archive.storage.allocated_bytes);
  TEST_ASSERT(both_usage.reserved_bytes == 0u);

  Store first_reader = Prepared(storage);
  TEST_ASSERT(first_reader.LoadArchive(first_archive));
  const ResolveResult resolved = first_reader.Resolve(0u);
  TEST_ASSERT(resolved.ok());
  TEST_ASSERT(std::equal(resolved.bytes.span().begin(),
                         resolved.bytes.span().end(), first_bytes.begin(),
                         first_bytes.end()));

  first_archive = {};
  TEST_ASSERT(std::filesystem::exists(first_generation));
  first_reader.Clear();
  TEST_ASSERT(!std::filesystem::exists(first_generation));
  TEST_ASSERT(storage.budget.report().physical_bytes ==
              second_archive.storage.physical_bytes);
  TEST_ASSERT(storage.budget.report().allocated_bytes ==
              second_archive.storage.allocated_bytes);

  const std::filesystem::path second_generation =
      second_archive.spill_generation->directory();
  second.Clear();
  TEST_ASSERT(std::filesystem::exists(second_generation));
  second_archive = {};
  TEST_ASSERT(!std::filesystem::exists(second_generation));
  TEST_ASSERT(storage.budget.report().physical_bytes == 0u);
  TEST_ASSERT(storage.budget.report().allocated_bytes == 0u);
  TEST_ASSERT(storage.budget.report().reserved_bytes == 0u);
  TEST_ASSERT(std::filesystem::exists(foreign));
  TEST_ASSERT(std::filesystem::exists(unmarked));
  std::filesystem::remove_all(root);
  return 0;
}

} // namespace replay_spill

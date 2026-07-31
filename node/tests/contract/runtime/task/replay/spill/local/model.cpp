#include "model.hpp"

#include "test/assert.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>

#include <sys/statvfs.h>
#include <unistd.h>

namespace replay_spill {
namespace {

constexpr std::string_view kGenerationMarkerContents =
    "runD replay spill generation v1\n";

[[nodiscard]] bool IsOwnedGeneration(const std::filesystem::path &path) {
  std::ifstream marker{path / ".rund-owner", std::ios::binary};
  if (!marker) {
    return false;
  }
  std::string contents{std::istreambuf_iterator<char>{marker},
                       std::istreambuf_iterator<char>{}};
  return contents == kGenerationMarkerContents;
}

} // namespace

using rund::node::replay_detail::payload::kChunkBytes;
using ::rund::replay::StorageMode;

[[nodiscard]] std::vector<std::byte> Bytes(const std::string_view text) {
  std::vector<std::byte> bytes{};
  bytes.reserve(text.size());
  for (const char value : text) {
    bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(value)));
  }
  return bytes;
}

[[nodiscard]] std::filesystem::path TempDir(const std::string_view name) {
  return std::filesystem::path{".cache"} /
         (std::string{"runD-host-replay-spill-"} + std::string{name} + "-" +
          std::to_string(static_cast<long long>(::getpid())));
}

[[nodiscard]] std::filesystem::path
SegmentPath(const std::filesystem::path &dir, const std::uint32_t index) {
  std::array<char, 64u> name{};
  static_cast<void>(std::snprintf(name.data(), name.size(),
                                  "host-replay-payload-%06u.segment", index));
  const std::vector<std::filesystem::path> generations =
      GenerationDirectories(dir);
  return (generations.size() == 1u ? generations.front() : dir) / name.data();
}

std::vector<std::filesystem::path>
GenerationDirectories(const std::filesystem::path &root) {
  std::vector<std::filesystem::path> generations{};
  std::error_code error{};
  std::filesystem::directory_iterator entries{root, error};
  if (error) {
    return generations;
  }
  for (const std::filesystem::directory_entry &entry : entries) {
    const std::string name = entry.path().filename().string();
    if (entry.is_directory(error) && !error &&
        name.starts_with(".rund-replay-spill-v1-") &&
        IsOwnedGeneration(entry.path())) {
      generations.push_back(entry.path());
    }
    error.clear();
  }
  std::sort(generations.begin(), generations.end());
  return generations;
}

[[nodiscard]] ::rund::replay::Storage
Storage(const std::filesystem::path &dir, const std::uint64_t segment_bytes) {
  ::rund::replay::Storage storage{};
  storage.mode = ::rund::replay::StorageMode::Spill;
  storage.directory = dir.string();
  storage.cached_bytes = 64u;
  storage.segment_bytes = segment_bytes;
  storage.max_bytes = 1024u * 1024u;
  return storage;
}

[[nodiscard]] ::rund::replay::Storage
CacheStorage(const std::filesystem::path &dir) {
  ::rund::replay::Storage storage{};
  storage.mode = ::rund::replay::StorageMode::Spill;
  storage.directory = dir.string();
  storage.cached_bytes = kChunkBytes;
  storage.segment_bytes = kChunkBytes * 3u;
  storage.max_bytes = kChunkBytes * 256u;
  return storage;
}

Store Prepared(::rund::replay::Storage storage, const std::uint32_t hosts,
               const std::uint32_t inputs) {
  if (!storage.budget) {
    storage.budget = ::rund::storage::Budget{storage.max_allocated_bytes};
  }
  const auto limits = rund::node::replay_detail::payload::Limits::runtime(
      hosts, inputs, storage.max_bytes);
  return Store{std::move(storage), limits.value()};
}

[[nodiscard]] bool CorruptLastByte(const std::filesystem::path &path) {
  std::fstream file{path, std::ios::binary | std::ios::in | std::ios::out};
  if (!file) {
    return false;
  }
  file.seekg(-1, std::ios::end);
  char value = 0;
  file.get(value);
  if (!file) {
    return false;
  }
  file.clear();
  file.seekp(-1, std::ios::end);
  file.put(static_cast<char>(static_cast<unsigned char>(value) ^
                             static_cast<unsigned char>(0x01u)));
  return static_cast<bool>(file);
}

[[nodiscard]] std::uint64_t
SegmentBytes(const std::filesystem::path &generation) {
  std::uint64_t total = 0u;
  std::error_code error{};
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator{generation}) {
    if (entry.path().extension() == ".segment") {
      total += entry.file_size(error);
      if (error) {
        return std::numeric_limits<std::uint64_t>::max();
      }
    }
  }
  return total;
}

[[nodiscard]] std::uint64_t
AllocatedBytes(const std::filesystem::path &generation) {
  struct statvfs status{};
  if (::statvfs(generation.c_str(), &status) != 0) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  const std::uint64_t unit = status.f_frsize != 0u
                                 ? static_cast<std::uint64_t>(status.f_frsize)
                                 : static_cast<std::uint64_t>(status.f_bsize);
  if (unit == 0u) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  std::uint64_t total = 0u;
  std::error_code error{};
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator{generation}) {
    if (entry.path().extension() != ".segment") {
      continue;
    }
    const std::uint64_t bytes = entry.file_size(error);
    if (error ||
        bytes > std::numeric_limits<std::uint64_t>::max() - (unit - 1u)) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    total += ((bytes + unit - 1u) / unit) * unit;
  }
  return total;
}

namespace {

void PutU64(std::span<std::byte, sizeof(std::uint64_t)> output,
            const std::uint64_t value) {
  for (std::uint32_t byte = 0u; byte < sizeof(value); ++byte) {
    output[byte] = static_cast<std::byte>((value >> (byte * 8u)) & 0xffu);
  }
}

} // namespace

int CanonicalSegment(const std::filesystem::path &segment,
                     const std::span<const std::byte> payload,
                     const StableHash hash) {
  constexpr std::uint64_t kMagic = 0x72644853504c3031ull;
  constexpr std::size_t kHeaderBytes = 41u;
  std::ifstream input{segment, std::ios::binary};
  std::array<std::byte, kHeaderBytes> actual{};
  input.read(reinterpret_cast<char *>(actual.data()),
             static_cast<std::streamsize>(actual.size()));
  TEST_ASSERT(input.gcount() == static_cast<std::streamsize>(actual.size()));

  std::array<std::byte, kHeaderBytes> expected{};
  PutU64(std::span<std::byte, sizeof(std::uint64_t)>{expected.data(), 8u},
         kMagic);
  PutU64(std::span<std::byte, sizeof(std::uint64_t)>{expected.data() + 8u, 8u},
         0u);
  expected[16u] =
      static_cast<std::byte>(rund::node::replay_detail::payload::Codec::Raw);
  PutU64(std::span<std::byte, sizeof(std::uint64_t)>{expected.data() + 17u, 8u},
         payload.size());
  PutU64(std::span<std::byte, sizeof(std::uint64_t)>{expected.data() + 25u, 8u},
         payload.size());
  PutU64(std::span<std::byte, sizeof(std::uint64_t)>{expected.data() + 33u, 8u},
         hash.value);
  TEST_ASSERT(actual == expected);

  std::vector<std::byte> encoded(payload.size());
  input.read(reinterpret_cast<char *>(encoded.data()),
             static_cast<std::streamsize>(encoded.size()));
  TEST_ASSERT(input.gcount() == static_cast<std::streamsize>(encoded.size()));
  TEST_ASSERT(std::equal(encoded.begin(), encoded.end(), payload.begin(),
                         payload.end()));
  return 0;
}

int RunSegmentsContract() {
  if (const int result = GenerationContract(); result != 0) {
    return result;
  }
  if (const int result = LifetimeContract(); result != 0) {
    return result;
  }
  if (const int result = BudgetContract(); result != 0) {
    return result;
  }
  if (const int result = AppendContract(); result != 0) {
    return result;
  }
  if (const int result = ArtifactContract(); result != 0) {
    return result;
  }
  return LayoutContract();
}

} // namespace replay_spill

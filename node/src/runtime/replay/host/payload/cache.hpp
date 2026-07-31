#pragma once

#include <rund/replay/storage.hpp>

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace rund::node::replay_detail::payload {

class ByteHash;

class Cache {
public:
  explicit Cache(std::uint64_t capacity_bytes) noexcept;
  Cache(const Cache &) = delete;
  Cache &operator=(const Cache &) = delete;
  Cache(Cache &&) noexcept = default;
  Cache &operator=(Cache &&) noexcept = default;

  [[nodiscard]] bool Get(std::uint64_t chunk_id, std::vector<std::byte> &out);
  [[nodiscard]] std::optional<bool> Read(std::uint64_t chunk_id,
                                         std::span<std::byte> output,
                                         ByteHash *record_hash = nullptr);
  [[nodiscard]] std::optional<bool> Matches(std::uint64_t chunk_id,
                                            std::span<const std::byte> expected,
                                            ByteHash *record_hash = nullptr);
  void Put(std::uint64_t chunk_id, std::vector<std::byte> bytes);
  [[nodiscard]] ::rund::replay::StorageReport Report() const noexcept;
  void Clear() noexcept;

private:
  struct Entry {
    std::uint64_t chunk_id = 0u;
    std::vector<std::byte> bytes{};
  };

  using Order = std::list<Entry>;
  using OrderIterator = Order::iterator;

  [[nodiscard]] Entry *Touch(std::uint64_t chunk_id) noexcept;
  void EvictOverBudget() noexcept;

  std::uint64_t capacity_bytes_ = 0u;
  std::uint64_t cached_bytes_ = 0u;
  std::uint64_t hits_ = 0u;
  std::uint64_t misses_ = 0u;
  std::uint64_t evictions_ = 0u;
  std::uint64_t growths_ = 0u;
  Order order_{};
  std::unordered_map<std::uint64_t, OrderIterator> by_id_{};
};

} // namespace rund::node::replay_detail::payload

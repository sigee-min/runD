#include "cache.hpp"
#include "hash.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <utility>

namespace rund::node::replay_detail::payload {

Cache::Cache(const std::uint64_t capacity_bytes) noexcept
    : capacity_bytes_{capacity_bytes} {}

Cache::Entry *Cache::Touch(const std::uint64_t chunk_id) noexcept {
  const auto found = by_id_.find(chunk_id);
  if (found == by_id_.end()) {
    misses_ = ::rund::detail::counter::SaturatingAdd(misses_, 1u);
    return nullptr;
  }
  order_.splice(order_.begin(), order_, found->second);
  found->second = order_.begin();
  hits_ = ::rund::detail::counter::SaturatingAdd(hits_, 1u);
  return &*found->second;
}

bool Cache::Get(const std::uint64_t chunk_id, std::vector<std::byte> &out) {
  const Entry *const entry = Touch(chunk_id);
  if (entry == nullptr) {
    return false;
  }
  out = entry->bytes;
  return true;
}

std::optional<bool> Cache::Read(const std::uint64_t chunk_id,
                                const std::span<std::byte> output,
                                ByteHash *const record_hash) {
  const Entry *const entry = Touch(chunk_id);
  if (entry == nullptr) {
    return std::nullopt;
  }
  if (entry->bytes.size() != output.size()) {
    return false;
  }
  std::copy(entry->bytes.begin(), entry->bytes.end(), output.begin());
  if (record_hash != nullptr) {
    record_hash->Append(output);
  }
  return true;
}

std::optional<bool> Cache::Matches(const std::uint64_t chunk_id,
                                   const std::span<const std::byte> expected,
                                   ByteHash *const record_hash) {
  const Entry *const entry = Touch(chunk_id);
  if (entry == nullptr) {
    return std::nullopt;
  }
  const std::vector<std::byte> &bytes = entry->bytes;
  if (bytes.size() != expected.size()) {
    return false;
  }
  if (!std::equal(bytes.begin(), bytes.end(), expected.begin())) {
    return false;
  }
  if (record_hash != nullptr) {
    record_hash->Append(bytes);
  }
  return true;
}

void Cache::Put(const std::uint64_t chunk_id, std::vector<std::byte> bytes) {
  const std::uint64_t byte_count = static_cast<std::uint64_t>(bytes.size());
  const auto found = by_id_.find(chunk_id);
  if (found != by_id_.end()) {
    cached_bytes_ -= static_cast<std::uint64_t>(found->second->bytes.size());
    found->second->bytes = std::move(bytes);
    cached_bytes_ += byte_count;
    order_.splice(order_.begin(), order_, found->second);
    found->second = order_.begin();
    EvictOverBudget();
    return;
  }
  if (byte_count > capacity_bytes_) {
    EvictOverBudget();
    return;
  }
  const std::size_t buckets = by_id_.bucket_count();
  order_.push_front(Entry{.chunk_id = chunk_id, .bytes = std::move(bytes)});
  by_id_[chunk_id] = order_.begin();
  if (by_id_.bucket_count() != buckets) {
    growths_ = ::rund::detail::counter::SaturatingAdd(growths_, 1u);
  }
  cached_bytes_ += byte_count;
  EvictOverBudget();
}

::rund::replay::StorageReport Cache::Report() const noexcept {
  ::rund::replay::StorageReport report{};
  report.cached_bytes = cached_bytes_;
  report.cache_hits = hits_;
  report.cache_misses = misses_;
  report.cache_evictions = evictions_;
  report.growths = growths_;
  return report;
}

void Cache::Clear() noexcept {
  cached_bytes_ = 0u;
  hits_ = 0u;
  misses_ = 0u;
  evictions_ = 0u;
  growths_ = 0u;
  order_.clear();
  by_id_.clear();
}

void Cache::EvictOverBudget() noexcept {
  while (cached_bytes_ > capacity_bytes_ && !order_.empty()) {
    const Entry &entry = order_.back();
    cached_bytes_ -= static_cast<std::uint64_t>(entry.bytes.size());
    by_id_.erase(entry.chunk_id);
    order_.pop_back();
    evictions_ = ::rund::detail::counter::SaturatingAdd(evictions_, 1u);
  }
}

} // namespace rund::node::replay_detail::payload

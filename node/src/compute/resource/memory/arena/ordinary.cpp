#include "local.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace rund::compute::detail::resource_detail::memory_detail {
namespace {

struct Active final {
  std::uint32_t last{};
  std::uint32_t id{};
  std::uint64_t offset{};
  std::uint64_t bytes{};

  [[nodiscard]] bool operator>(const Active &other) const noexcept {
    return std::tie(last, id) > std::tie(other.last, other.id);
  }
};

struct Segment final {
  std::uint64_t offset{};
  std::uint64_t bytes{};
};

struct Fit final {
  std::uint64_t usable{};
  std::uint64_t aligned{};
  std::uint64_t offset{};

  [[nodiscard]] bool operator<(const Fit &other) const noexcept {
    return std::tie(usable, aligned, offset) <
           std::tie(other.usable, other.aligned, other.offset);
  }
};

enum class Take : std::uint8_t {
  Missing,
  Ready,
  Invalid,
};

class Free final {
public:
  explicit Free(const std::uint64_t page_bytes) noexcept
      : page_bytes_(page_bytes) {}

  [[nodiscard]] bool release(Segment value) {
    if (value.bytes == 0u || !kernel::checked::add(value.offset, value.bytes)) {
      return value.bytes == 0u;
    }
    auto next = segments_.lower_bound(value.offset);
    if (next != segments_.begin()) {
      const auto prior = std::prev(next);
      if (prior->first / page_bytes_ == value.offset / page_bytes_ &&
          prior->first + prior->second == value.offset) {
        value.offset = prior->first;
        value.bytes += prior->second;
        erase(prior);
      }
    }
    next = segments_.lower_bound(value.offset);
    if (next != segments_.end() &&
        value.offset / page_bytes_ == next->first / page_bytes_ &&
        value.offset + value.bytes == next->first) {
      if (!kernel::checked::add(value.bytes, next->second, value.bytes)) {
        return false;
      }
      erase(next);
    }
    return insert(value);
  }

  [[nodiscard]] Take take(const std::uint64_t bytes, std::uint64_t &offset) {
    const auto choice = fits_.lower_bound(Fit{.usable = bytes});
    if (choice == fits_.end()) {
      return Take::Missing;
    }
    const Fit fit = *choice;
    const auto segment = segments_.find(fit.offset);
    if (segment == segments_.end()) {
      return Take::Invalid;
    }
    const Segment slot{.offset = segment->first, .bytes = segment->second};
    erase(segment);
    offset = fit.aligned;
    const std::uint64_t prefix = offset - slot.offset;
    const std::uint64_t suffix = slot.bytes - prefix - bytes;
    return insert(Segment{.offset = slot.offset, .bytes = prefix}) &&
                   insert(Segment{.offset = offset + bytes, .bytes = suffix})
               ? Take::Ready
               : Take::Invalid;
  }

private:
  using Iterator = std::map<std::uint64_t, std::uint64_t>::iterator;

  [[nodiscard]] static bool fit(const Segment value, Fit &result) noexcept {
    std::uint64_t start = 0u;
    if (!aligned(value.offset, Alignment, start) || start < value.offset ||
        start - value.offset > value.bytes) {
      return false;
    }
    result = Fit{.usable = value.bytes - (start - value.offset),
                 .aligned = start,
                 .offset = value.offset};
    return true;
  }

  [[nodiscard]] bool insert(const Segment value) {
    if (value.bytes == 0u) {
      return true;
    }
    const auto [position, added] = segments_.emplace(value.offset, value.bytes);
    if (!added) {
      return false;
    }
    Fit entry{};
    if (fit(value, entry)) {
      fits_.insert(entry);
    }
    return true;
  }

  void erase(const Iterator position) {
    Fit entry{};
    if (fit(Segment{.offset = position->first, .bytes = position->second},
            entry)) {
      fits_.erase(entry);
    }
    segments_.erase(position);
  }

  std::map<std::uint64_t, std::uint64_t> segments_;
  std::set<Fit> fits_;
  std::uint64_t page_bytes_{};
};

} // namespace

[[nodiscard]] bool
measure_ordinary(const Placement &placement,
                 const std::span<const graph::Resource> resources,
                 const std::span<const std::uint32_t> ids,
                 const std::span<const std::uint64_t> offsets,
                 const std::uint64_t page_bytes, Arena &result) {
  if (page_bytes < Alignment || page_bytes % Alignment != 0u ||
      offsets.size() != resources.size() ||
      (ids.empty() != (placement.bytes == 0u))) {
    return false;
  }
  result = {};
  const std::uint64_t chunk_count =
      placement.bytes == 0u ? 0u : (placement.bytes - 1u) / page_bytes + 1u;
  if (chunk_count > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  result.owners.resize(static_cast<std::size_t>(chunk_count));
  std::vector<std::uint64_t> extents(static_cast<std::size_t>(chunk_count));
  for (const std::uint32_t id : ids) {
    const graph::Resource &value = resources[id - 1u];
    const std::uint64_t offset = offsets[id - 1u];
    const std::uint64_t chunk = offset / page_bytes;
    const std::uint64_t local = offset % page_bytes;
    if (value.bytes > page_bytes || value.bytes > page_bytes - local ||
        chunk >= chunk_count) {
      return false;
    }
    const std::size_t index = static_cast<std::size_t>(chunk);
    std::uint32_t &owner = result.owners[index];
    owner = owner == 0u ? id : std::min(owner, id);
    extents[index] = std::max(extents[index], local + value.bytes);
  }
  for (std::size_t chunk = 0u; chunk < result.owners.size(); ++chunk) {
    if (result.owners[chunk] == 0u) {
      continue;
    }
    ::rund::detail::counter::Accumulate(result.bytes, extents[chunk]);
    ++result.count;
  }
  return true;
}

[[nodiscard]] bool
place_ordinary(const std::span<const graph::Resource> resources,
               const std::span<const Lifetime> lifetimes,
               const std::span<const std::uint32_t> ids,
               const std::uint64_t page_bytes, const bool destructive,
               const std::span<std::uint64_t> offsets, Placement &result) {
  if (page_bytes < Alignment || page_bytes % Alignment != 0u ||
      offsets.size() != resources.size()) {
    return false;
  }
  result = {};
  std::fill(offsets.begin(), offsets.end(), 0u);
  std::vector<Active> active;
  active.reserve(ids.size());
  Free free{page_bytes};
  std::vector<bool> consumed(resources.size());
  for (const std::uint32_t id : ids) {
    const graph::Resource &value = resources[id - 1u];
    const Lifetime lifetime = lifetimes[id - 1u];
    while (!active.empty() && active.front().last < lifetime.first) {
      const Active expired = active.front();
      std::pop_heap(active.begin(), active.end(), std::greater<>{});
      active.pop_back();
      if (consumed[expired.id - 1u]) {
        continue;
      }
      if (!free.release(
              Segment{.offset = expired.offset, .bytes = expired.bytes})) {
        return false;
      }
    }
    std::uint64_t offset = 0u;
    if (destructive && value.source != 0u) {
      offset = offsets[value.source - 1u];
      consumed[value.source - 1u] = true;
    }
    const bool needs_space = !destructive || value.source == 0u;
    const Take taken =
        needs_space ? free.take(value.bytes, offset) : Take::Ready;
    if (taken == Take::Invalid) {
      return false;
    }
    if (taken == Take::Missing) {
      if (!aligned(result.bytes, Alignment, offset) ||
          value.bytes > page_bytes ||
          value.bytes > page_bytes - (offset % page_bytes)) {
        const std::uint64_t remainder = result.bytes % page_bytes;
        if (remainder == 0u) {
          offset = result.bytes;
        } else if (!kernel::checked::add(result.bytes, page_bytes - remainder,
                                         offset)) {
          return false;
        }
      }
      if (!kernel::checked::add(offset, value.bytes, result.bytes)) {
        return false;
      }
    }
    offsets[id - 1u] = offset;
    active.push_back(Active{.last = lifetime.last,
                            .id = id,
                            .offset = offset,
                            .bytes = value.bytes});
    std::push_heap(active.begin(), active.end(), std::greater<>{});
  }
  return true;
}

} // namespace rund::compute::detail::resource_detail::memory_detail

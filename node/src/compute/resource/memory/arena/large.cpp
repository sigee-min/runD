#include "local.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace rund::compute::detail::resource_detail::memory_detail {
namespace {

struct LargeActive final {
  std::uint32_t last{};
  std::uint32_t id{};
  std::uint32_t owner{};
  std::uint64_t capacity{};

  [[nodiscard]] bool operator>(const LargeActive &other) const noexcept {
    return std::tie(last, id) > std::tie(other.last, other.id);
  }
};

} // namespace

[[nodiscard]] bool place_large(const std::span<const graph::Resource> resources,
                               const std::span<const Lifetime> lifetimes,
                               const std::span<const std::uint32_t> ids,
                               const bool destructive, Layout &result) {
  std::vector<LargeActive> active;
  active.reserve(ids.size());
  std::set<std::pair<std::uint64_t, std::uint32_t>> free;
  std::vector<bool> consumed(resources.size());
  std::vector<std::uint64_t> capacities(resources.size());
  for (const std::uint32_t id : ids) {
    const graph::Resource &value = resources[id - 1u];
    const Lifetime lifetime = lifetimes[id - 1u];
    while (!active.empty() && active.front().last < lifetime.first) {
      std::pop_heap(active.begin(), active.end(), std::greater<>{});
      const LargeActive expired = active.back();
      active.pop_back();
      if (consumed[expired.id - 1u]) {
        continue;
      }
      if (!free.emplace(expired.capacity, expired.owner).second) {
        return false;
      }
    }

    std::uint32_t owner = 0u;
    std::uint64_t capacity = 0u;
    if (destructive && value.source != 0u) {
      if (value.source > resources.size() ||
          result.owners[value.source - 1u] == 0u) {
        return false;
      }
      owner = result.owners[value.source - 1u];
      capacity = capacities[owner - 1u];
      if (capacity < value.bytes) {
        return false;
      }
      consumed[value.source - 1u] = true;
    } else {
      const auto reused = free.lower_bound({value.bytes, 0u});
      if (reused != free.end()) {
        capacity = reused->first;
        owner = reused->second;
        free.erase(reused);
      } else if (!free.empty()) {
        const auto grown = std::prev(free.end());
        const std::uint64_t previous = grown->first;
        owner = grown->second;
        free.erase(grown);
        capacity = value.bytes;
        if (previous >= capacity ||
            !kernel::checked::add(result.bytes, capacity - previous,
                                  result.bytes)) {
          return false;
        }
        capacities[owner - 1u] = capacity;
      } else {
        owner = id;
        capacity = value.bytes;
        if (!kernel::checked::add(result.bytes, capacity, result.bytes) ||
            result.count == std::numeric_limits<std::uint64_t>::max()) {
          return false;
        }
        ++result.count;
        capacities[owner - 1u] = capacity;
      }
    }
    if (owner == 0u || capacity == 0u || capacities[owner - 1u] != capacity) {
      return false;
    }
    result.owners[id - 1u] = owner;
    result.offsets[id - 1u] = 0u;
    active.push_back(LargeActive{
        .last = lifetime.last, .id = id, .owner = owner, .capacity = capacity});
    std::push_heap(active.begin(), active.end(), std::greater<>{});
  }
  return true;
}

} // namespace rund::compute::detail::resource_detail::memory_detail

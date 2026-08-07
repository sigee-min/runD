#include "overlap.hpp"

#include "../backend/exception.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace rund::node::accel::detail::reset {
namespace {

struct Interval final {
  const void *owner{};
  std::uint64_t resident{};
  std::uint64_t begin{};
  std::uint64_t end{};
  std::size_t reset{};
};

struct Active final {
  std::uint64_t end{};
  std::size_t reset{};

  [[nodiscard]] bool operator>(const Active other) const noexcept {
    return std::tie(end, reset) > std::tie(other.end, other.reset);
  }
};

class Timeline final {
public:
  void reset(const std::size_t count) {
    nodes_.assign(count == 0u ? 0u : 4u * count, Node{});
    count_ = count;
  }

  void add(const std::size_t begin, const std::size_t end,
           const std::int64_t value) noexcept {
    add(1u, 0u, count_ - 1u, begin, end, value);
  }

  [[nodiscard]] std::int64_t maximum(const std::size_t begin,
                                     const std::size_t end) noexcept {
    return maximum(1u, 0u, count_ - 1u, begin, end);
  }

private:
  struct Node final {
    std::int64_t maximum{};
    std::int64_t lazy{};
  };

  void add(const std::size_t node, const std::size_t left,
           const std::size_t right, const std::size_t begin,
           const std::size_t end, const std::int64_t value) noexcept {
    if (begin <= left && right <= end) {
      nodes_[node].maximum += value;
      nodes_[node].lazy += value;
      return;
    }
    push(node);
    const std::size_t middle = left + (right - left) / 2u;
    if (begin <= middle) {
      add(node * 2u, left, middle, begin, end, value);
    }
    if (end > middle) {
      add(node * 2u + 1u, middle + 1u, right, begin, end, value);
    }
    nodes_[node].maximum =
        std::max(nodes_[node * 2u].maximum, nodes_[node * 2u + 1u].maximum);
  }

  [[nodiscard]] std::int64_t maximum(const std::size_t node,
                                     const std::size_t left,
                                     const std::size_t right,
                                     const std::size_t begin,
                                     const std::size_t end) noexcept {
    if (begin <= left && right <= end) {
      return nodes_[node].maximum;
    }
    push(node);
    const std::size_t middle = left + (right - left) / 2u;
    std::int64_t result = 0;
    if (begin <= middle) {
      result = maximum(node * 2u, left, middle, begin, end);
    }
    if (end > middle) {
      result = std::max(
          result, maximum(node * 2u + 1u, middle + 1u, right, begin, end));
    }
    return result;
  }

  void push(const std::size_t node) noexcept {
    if (nodes_[node].lazy == 0) {
      return;
    }
    nodes_[node * 2u].maximum += nodes_[node].lazy;
    nodes_[node * 2u + 1u].maximum += nodes_[node].lazy;
    nodes_[node * 2u].lazy += nodes_[node].lazy;
    nodes_[node * 2u + 1u].lazy += nodes_[node].lazy;
    nodes_[node].lazy = 0;
  }

  std::vector<Node> nodes_;
  std::size_t count_{};
};

} // namespace

bool Compatible(const std::span<const BoundReset> resets) noexcept try {
  std::vector<Interval> intervals;
  intervals.reserve(resets.size());
  for (std::size_t index = 0u; index < resets.size(); ++index) {
    const BoundReset &item = resets[index];
    const rund::kernel::ResidentBufferRef ref = item.ref();
    const Range range = item.range();
    if (ref.id == 0u || item.handle() == nullptr || !range.valid()) {
      return false;
    }
    intervals.push_back(Interval{
        .owner = item.handle().get(),
        .resident = ref.id,
        .begin = range.offset(),
        .end = range.end(),
        .reset = index,
    });
  }

  const std::less<const void *> pointer_order;
  std::sort(intervals.begin(), intervals.end(),
            [&](const Interval left, const Interval right) {
              if (left.owner != right.owner) {
                return pointer_order(left.owner, right.owner);
              }
              return std::tuple{left.resident, left.begin, left.end,
                                resets[left.reset].step.index,
                                resets[left.reset].binding} <
                     std::tuple{right.resident, right.begin, right.end,
                                resets[right.reset].step.index,
                                resets[right.reset].binding};
            });

  std::vector<std::uint32_t> points;
  std::vector<Active> active;
  Timeline timeline;
  for (std::size_t group = 0u; group < intervals.size();) {
    std::size_t end = group + 1u;
    while (end < intervals.size() &&
           intervals[end].owner == intervals[group].owner &&
           intervals[end].resident == intervals[group].resident) {
      ++end;
    }
    if (end - group > std::numeric_limits<std::size_t>::max() / 2u) {
      return false;
    }
    points.clear();
    points.reserve(2u * (end - group));
    for (std::size_t index = group; index < end; ++index) {
      const BoundReset &item = resets[intervals[index].reset];
      points.push_back(item.step.index);
      points.push_back(item.last.index);
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.empty() ||
        points.size() > std::numeric_limits<std::size_t>::max() / 4u) {
      return false;
    }

    timeline.reset(points.size());
    active.clear();
    active.reserve(end - group);
    std::size_t external = 0u;
    const auto range = [&](const BoundReset &item) {
      return std::pair{
          static_cast<std::size_t>(
              std::lower_bound(points.begin(), points.end(), item.step.index) -
              points.begin()),
          static_cast<std::size_t>(
              std::lower_bound(points.begin(), points.end(), item.last.index) -
              points.begin()),
      };
    };
    for (std::size_t index = group; index < end; ++index) {
      const Interval current = intervals[index];
      // active contains earlier byte ranges intersecting current. Timeline
      // stores their closed internal lifetimes; external ranges never alias.
      while (!active.empty() && active.front().end <= current.begin) {
        std::pop_heap(active.begin(), active.end(), std::greater<>{});
        const BoundReset &expired = resets[active.back().reset];
        if (expired.external) {
          --external;
        } else {
          const auto [first, last] = range(expired);
          timeline.add(first, last, -1);
        }
        active.pop_back();
      }
      const BoundReset &item = resets[current.reset];
      const auto [first, last] = range(item);
      if ((item.external && !active.empty()) ||
          (!item.external &&
           (external != 0u || timeline.maximum(first, last) != 0))) {
        return false;
      }
      if (item.external) {
        ++external;
      } else {
        timeline.add(first, last, 1);
      }
      active.push_back(Active{.end = current.end, .reset = current.reset});
      std::push_heap(active.begin(), active.end(), std::greater<>{});
    }
    group = end;
  }
  return true;
} catch (...) {
  backend_exception::RethrowUnlessCapacityException();
  return false;
}

} // namespace rund::node::accel::detail::reset

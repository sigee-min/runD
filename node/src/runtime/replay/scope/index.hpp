#pragma once

#include "../host/payload/store.hpp"

#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::replay::detail::scope {

struct Key final {
  std::uint64_t source = 0u;
  std::uint64_t schema = 0u;
  std::uint64_t sequence = 0u;

  [[nodiscard]] auto operator<=>(const Key &) const noexcept = default;
};

struct Match final {
  std::size_t count = 0u;
  std::size_t ordinal = 0u;
  std::uint64_t bytes = 0u;

  [[nodiscard]] bool unique() const noexcept { return count == 1u; }
};

class Index final {
public:
  explicit Index(const ::rund::node::replay_detail::payload::Store &store) {
    rows_.reserve(store.input_record_count());
    for (std::size_t ordinal = 0u; ordinal < store.input_record_count();
         ++ordinal) {
      const std::size_t record = store.input_record_index(ordinal);
      if (record >= store.records().size()) {
        continue;
      }
      const ::rund::node::replay_detail::payload::StoredRecord &value =
          store.records()[record];
      rows_.push_back(Row{
          .key = Key{value.metadata.input_source, value.metadata.input_schema,
                     value.metadata.input_sequence},
          .ordinal = ordinal,
          .bytes = value.metadata.completed_bytes,
      });
    }
    std::sort(
        rows_.begin(), rows_.end(),
        [](const Row &left, const Row &right) { return left.key < right.key; });
  }

  [[nodiscard]] Match find(const Key key) const noexcept {
    const auto first =
        std::lower_bound(rows_.begin(), rows_.end(), key,
                         [](const Row &row, const Key candidate) {
                           return row.key < candidate;
                         });
    const auto last = std::upper_bound(first, rows_.end(), key,
                                       [](const Key candidate, const Row &row) {
                                         return candidate < row.key;
                                       });
    if (first == last) {
      return {};
    }
    return Match{
        .count = static_cast<std::size_t>(last - first),
        .ordinal = first->ordinal,
        .bytes = first->bytes,
    };
  }

  [[nodiscard]] std::size_t size() const noexcept { return rows_.size(); }

private:
  struct Row final {
    Key key{};
    std::size_t ordinal = 0u;
    std::uint64_t bytes = 0u;
  };

  std::vector<Row> rows_{};
};

} // namespace rund::replay::detail::scope

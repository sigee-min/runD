#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rund::node::accel::detail::backend_source_recipe {

struct SourceEdit final {
  std::size_t begin{};
  std::size_t end{};
  std::string replacement{};

  [[nodiscard]] std::string_view text() const noexcept { return replacement; }
};

template <typename Edit>
[[nodiscard]] inline bool
canonicalize_edits(const std::span<Edit> edits,
                   const std::size_t source_bytes) noexcept {
  std::sort(edits.begin(), edits.end(),
            [](const Edit &left, const Edit &right) noexcept {
              return left.begin < right.begin ||
                     (left.begin == right.begin && left.end < right.end);
            });
  std::size_t consumed = 0u;
  bool first = true;
  std::size_t previous_begin = 0u;
  for (const Edit &edit : edits) {
    if (edit.begin < consumed || edit.end < edit.begin ||
        edit.end > source_bytes || (!first && edit.begin == previous_begin)) {
      return false;
    }
    consumed = edit.end;
    previous_begin = edit.begin;
    first = false;
  }
  return !edits.empty();
}

template <typename Edit>
[[nodiscard]] inline bool
canonicalize_edits(std::vector<Edit> &edits,
                   const std::size_t source_bytes) noexcept {
  return canonicalize_edits(std::span<Edit>{edits}, source_bytes);
}

template <typename Edit> class BasicSourceEditRecipe final {
public:
  BasicSourceEditRecipe(const std::string_view source,
                        const std::span<const Edit> edits) noexcept
      : source_{source}, edits_{edits} {}

  template <typename Sink>
  [[nodiscard]] bool operator()(Sink &sink) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
    std::size_t cursor = 0u;
    for (const Edit &edit : edits_) {
      if (edit.begin < cursor || edit.end < edit.begin ||
          edit.end > source_.size() ||
          !sink.append(source_.substr(cursor, edit.begin - cursor)) ||
          !sink.append(edit.text())) {
        return false;
      }
      cursor = edit.end;
    }
    return sink.append(source_.substr(cursor));
  }

private:
  std::string_view source_{};
  std::span<const Edit> edits_{};
};

using SourceEditRecipe = BasicSourceEditRecipe<SourceEdit>;

} // namespace rund::node::accel::detail::backend_source_recipe

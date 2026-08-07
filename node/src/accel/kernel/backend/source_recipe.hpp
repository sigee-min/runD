#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/retention.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
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

[[nodiscard]] inline std::string_view
decimal_characters(const std::uint64_t value,
                   std::array<char, 20u> &storage) noexcept {
  const auto converted =
      std::to_chars(storage.data(), storage.data() + storage.size(), value);
  return converted.ec == std::errc{}
             ? std::string_view{storage.data(),
                                static_cast<std::size_t>(converted.ptr -
                                                         storage.data())}
             : std::string_view{};
}

template <typename Sink>
[[nodiscard]] bool
append_decimal(Sink &sink, const std::uint64_t value) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  std::array<char, 20u> digits{};
  const std::string_view text = decimal_characters(value, digits);
  return !text.empty() && sink.append(text);
}

template <typename Sink, std::size_t N>
[[nodiscard]] bool append_fixed(
    Sink &sink,
    const std::array<std::string_view, N>
        &fragments) noexcept(noexcept(sink.append(std::string_view{}))) {
  for (const std::string_view fragment : fragments) {
    if (!sink.append(fragment)) {
      return false;
    }
  }
  return true;
}

class CountSink final {
public:
  explicit CountSink(const std::uint64_t initial = 0u) noexcept
      : bytes_{initial} {}

  [[nodiscard]] bool append(const std::string_view fragment) noexcept {
    if (!ok_) {
      return false;
    }
    ok_ = rund::kernel::checked::add(bytes_, fragment.size(), bytes_);
    return ok_;
  }

  CountSink &operator+=(const std::string_view fragment) noexcept {
    (void)append(fragment);
    return *this;
  }

  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool valid() const noexcept { return ok_; }

private:
  std::uint64_t bytes_{};
  bool ok_{true};
};

class StringSink final {
public:
  explicit StringSink(std::string &text) noexcept : text_{text} {}

  [[nodiscard]] bool append(const std::string_view fragment) {
    if (!fragment.empty()) {
      text_.append(fragment.data(), fragment.size());
    }
    return true;
  }

  StringSink &operator+=(const std::string_view fragment) {
    (void)append(fragment);
    return *this;
  }

  [[nodiscard]] constexpr bool valid() const noexcept { return true; }

private:
  std::string &text_;
};

template <std::size_t Capacity> class FixedBufferSink final {
public:
  explicit FixedBufferSink(std::array<char, Capacity> &storage) noexcept
      : storage_{storage} {}

  [[nodiscard]] bool append(const std::string_view fragment) noexcept {
    if (!ok_ || size_ > storage_.size() ||
        fragment.size() > storage_.size() - size_) {
      ok_ = false;
      return false;
    }
    std::copy(fragment.begin(), fragment.end(), storage_.begin() + size_);
    size_ += fragment.size();
    return true;
  }

  [[nodiscard]] bool valid() const noexcept { return ok_; }
  [[nodiscard]] std::string_view text() const noexcept {
    return std::string_view{storage_.data(), size_};
  }

private:
  std::array<char, Capacity> &storage_;
  std::size_t size_{};
  bool ok_{true};
};

template <std::size_t Capacity> struct FixedSource final {
  std::array<char, Capacity> storage{};
  std::size_t size{};

  [[nodiscard]] std::string_view text() const noexcept {
    return std::string_view{storage.data(), size};
  }
};

template <std::size_t Capacity, typename Emit>
[[nodiscard]] FixedSource<Capacity> materialize_fixed(Emit &&emit) noexcept {
  FixedSource<Capacity> source{};
  FixedBufferSink<Capacity> sink{source.storage};
  if (emit(sink) && sink.valid()) {
    source.size = sink.text().size();
  }
  return source;
}

template <typename Sink>
[[nodiscard]] bool
append_hex64_digits(Sink &sink, const std::uint64_t value) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  std::array<char, 16u> digits{};
  constexpr std::string_view Hex = "0123456789abcdef";
  for (std::size_t index = 0u; index < digits.size(); ++index) {
    const std::size_t shift = (digits.size() - index - 1u) * 4u;
    digits[index] = Hex[(value >> shift) & 0x0fu];
  }
  return sink.append(std::string_view{digits.data(), digits.size()});
}

// std::string may round reserve() above the requested character count. The
// planner therefore keeps text bytes and external storage bytes as separate
// dimensions. One cache-line of slack covers the supported libc++, libstdc++,
// and MSVC string-capacity rounding; runtime verifies the actual capacity and
// fails closed before publication on any implementation that exceeds it.
inline constexpr std::uint64_t StringExternalStorageSlackBytes = 64u;

[[nodiscard]] inline bool string_external_storage_upper_bytes(
    const std::uint64_t text_upper_bytes,
    std::uint64_t &storage_upper_bytes) noexcept {
  if (text_upper_bytes == 0u) {
    storage_upper_bytes = 0u;
    return true;
  }
  return rund::kernel::checked::add(
      text_upper_bytes, StringExternalStorageSlackBytes, storage_upper_bytes);
}

[[nodiscard]] inline bool string_external_storage_within(
    const std::string &text, const std::uint64_t storage_upper_bytes) noexcept {
  return rund::kernel::compute_retained_detail::StringExternalStorageBytes(
             text) <= storage_upper_bytes;
}

// Grow by constructing one empty final owner rather than relying on geometric
// growth from the raw string. When growth is needed, the caller's planned raw
// source transient is the only allocation that coexists with this retained
// owner. A successful return guarantees both frozen character capacity and
// the external-storage envelope.
[[nodiscard]] inline bool
reserve_string(std::string &text, const std::uint64_t reserve_bytes) noexcept {
  if (text.size() > reserve_bytes ||
      reserve_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  std::uint64_t storage_upper_bytes = 0u;
  if (!string_external_storage_upper_bytes(reserve_bytes,
                                           storage_upper_bytes)) {
    return false;
  }
  try {
    if (text.capacity() < reserve_bytes) {
      std::string expanded;
      expanded.reserve(static_cast<std::size_t>(reserve_bytes));
      const std::size_t frozen_capacity = expanded.capacity();
      if (!string_external_storage_within(expanded, storage_upper_bytes)) {
        return false;
      }
      expanded.append(text.data(), text.size());
      if (expanded.capacity() != frozen_capacity) {
        return false;
      }
      text = std::move(expanded);
    }
    return text.capacity() >= reserve_bytes &&
           string_external_storage_within(text, storage_upper_bytes);
  } catch (const std::bad_alloc &) {
    return false;
  } catch (const std::length_error &) {
    return false;
  }
}

template <typename Emit>
[[nodiscard]] bool bytes(Emit &&emit, std::uint64_t &upper) noexcept {
  // Counting emitters are a strict no-throw contract. String materialization
  // may still throw from std::string and is contained by materialize().
  static_assert(std::is_nothrow_invocable_r_v<bool, Emit &, CountSink &>);
  CountSink sink{};
  if (!emit(sink) || sink.bytes() == 0u) {
    return false;
  }
  const std::uint64_t candidate = sink.bytes();
  upper = candidate;
  return true;
}

// Materialize a previously frozen exact count with one final reserve. `Emit`
// must be allocation-free apart from the StringSink append operations;
// CountSink and StringSink consume the exact same branch/fragment recipe.
template <typename Emit>
[[nodiscard]] std::string materialize(Emit &&emit,
                                      const std::uint64_t exact_bytes,
                                      const std::uint64_t reserve_bytes) {
  if (exact_bytes == 0u || exact_bytes > reserve_bytes ||
      reserve_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return {};
  }
  try {
    std::string text;
    if (!reserve_string(text, reserve_bytes)) {
      return {};
    }
    const std::size_t frozen_capacity = text.capacity();
    std::uint64_t storage_upper_bytes = 0u;
    if (!string_external_storage_upper_bytes(reserve_bytes,
                                             storage_upper_bytes) ||
        !string_external_storage_within(text, storage_upper_bytes)) {
      return {};
    }
    StringSink sink{text};
    if (!emit(sink) || text.size() != exact_bytes ||
        text.capacity() != frozen_capacity ||
        !string_external_storage_within(text, storage_upper_bytes)) {
      return {};
    }
    return text;
  } catch (const std::bad_alloc &) {
    return {};
  } catch (const std::length_error &) {
    return {};
  }
}

template <typename Emit>
[[nodiscard]] std::string materialize(Emit &&emit,
                                      const std::uint64_t exact_bytes) {
  return materialize(std::forward<Emit>(emit), exact_bytes, exact_bytes);
}

// Convenience path for callers that have not frozen a count yet.
template <typename Emit> [[nodiscard]] std::string materialize(Emit &&emit) {
  std::uint64_t upper = 0u;
  const auto count_emit = [&](CountSink &sink) noexcept { return emit(sink); };
  if (!bytes(count_emit, upper)) {
    return {};
  }
  return materialize(std::forward<Emit>(emit), upper);
}

} // namespace rund::node::accel::detail::backend_source_recipe

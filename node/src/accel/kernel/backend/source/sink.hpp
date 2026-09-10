#pragma once

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>

namespace rund::node::accel::detail::backend_source_recipe {

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

// One failure-latching builder for source recipes that mix fixed fragments
// and decimal constants. The wrapped CountSink and StringSink therefore
// traverse the same calls instead of relying on a backend-local size formula.
template <typename Sink> class SourceBuilder final {
public:
  explicit SourceBuilder(Sink &sink) noexcept : sink_{sink} {}

  SourceBuilder &operator+=(const std::string_view fragment) noexcept(
      noexcept(sink_.append(fragment))) {
    (void)append(fragment);
    return *this;
  }

  bool append(const std::string_view fragment) noexcept(
      noexcept(sink_.append(fragment))) {
    if (valid_) {
      valid_ = sink_.append(fragment);
    }
    return valid_;
  }

  bool decimal(const std::uint64_t value) noexcept(
      noexcept(sink_.append(std::string_view{}))) {
    if (valid_) {
      valid_ = append_decimal(sink_, value);
    }
    return valid_;
  }

  [[nodiscard]] bool valid() const noexcept { return valid_; }

private:
  Sink &sink_;
  bool valid_{true};
};

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

} // namespace rund::node::accel::detail::backend_source_recipe

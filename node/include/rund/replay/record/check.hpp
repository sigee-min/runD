#pragma once

#include <rund/replay/record/base.hpp>

namespace rund::replay {

class Check final {
public:
  Check(const Check &) noexcept = default;
  Check &operator=(const Check &) noexcept = default;
  Check(Check &&other) noexcept
      : code_(other.code_), expected_hash_(other.expected_hash_),
        actual_(std::move(other.actual_)) {
    other.reset_moved_from();
  }
  Check &operator=(Check &&other) noexcept {
    if (this != &other) {
      code_ = other.code_;
      expected_hash_ = other.expected_hash_;
      actual_ = std::move(other.actual_);
      other.reset_moved_from();
    }
    return *this;
  }

  [[nodiscard]] constexpr bool ok() const noexcept { return code_ == Code::Ok; }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] constexpr Code code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code_);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::replay::exit_code(code());
  }
  [[nodiscard]] constexpr std::uint64_t expected_hash() const noexcept {
    return expected_hash_;
  }
  [[nodiscard]] std::uint64_t actual_hash() const noexcept {
    return actual_.has_value() ? actual_->hash() : 0u;
  }
  [[nodiscard]] const std::optional<Record> &actual() const noexcept {
    return actual_;
  }

private:
  Check(const Code code, const std::uint64_t expected_hash,
        std::optional<Record> actual) noexcept
      : code_(code), expected_hash_(expected_hash), actual_(std::move(actual)) {
  }

  void reset_moved_from() noexcept {
    code_ = Code::CheckMovedFrom;
    expected_hash_ = 0u;
    actual_.reset();
  }

  Code code_ = Code::NotChecked;
  std::uint64_t expected_hash_ = 0u;
  std::optional<Record> actual_{};

  friend Check detail::check_result(const Record &, Session &,
                                    Session::Result &&, std::uint64_t) noexcept;
  friend Check detail::fail_check(const Record &, Code) noexcept;
  friend Check check(const Record &, const Record &) noexcept;
};

} // namespace rund::replay

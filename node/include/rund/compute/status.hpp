#pragma once
#include <rund/compute/reason.hpp>
#include <rund/outcome/result.hpp>

#include <string_view>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
namespace rund::compute {
template <class T> class Result;

struct Location final {
  static constexpr std::uint32_t none =
      std::numeric_limits<std::uint32_t>::max();

  std::uint32_t step{none};
  std::uint32_t iteration{none};
  std::uint32_t node{none};

  [[nodiscard]] constexpr bool known() const noexcept {
    return step != none || iteration != none || node != none;
  }

  [[nodiscard]] constexpr bool
  operator==(const Location &) const noexcept = default;
};

class Status final {
public:
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] constexpr bool ok() const noexcept {
    return reason_ == Reason::Ok;
  }
  [[nodiscard]] constexpr Reason reason() const noexcept { return reason_; }
  [[nodiscard]] constexpr Code code() const noexcept {
    return detail::category(reason_);
  }
  [[nodiscard]] std::string_view error() const noexcept;
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ok() ? 0 : 1;
  }
  [[nodiscard]] static constexpr Status success() noexcept { return {}; }
  [[nodiscard]] static constexpr Status fail(const Reason reason) noexcept {
    return Status{detail::valid(reason) ? reason : Reason::ReasonInvalid};
  }

private:
  constexpr Status() noexcept = default;
  constexpr explicit Status(const Reason reason) noexcept : reason_(reason) {}
  Reason reason_{Reason::Ok};
};

class Failure final {
public:
  [[nodiscard]] constexpr Reason reason() const noexcept {
    return status_.reason();
  }
  [[nodiscard]] constexpr Code code() const noexcept { return status_.code(); }
  [[nodiscard]] std::string_view error() const noexcept {
    return status_.error();
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return status_.exit_code();
  }
  [[nodiscard]] constexpr Location location() const noexcept {
    return location_;
  }

  [[nodiscard]] static constexpr Failure
  fail(const Reason reason, const Location location = {}) noexcept {
    return Failure{Status::fail(reason), location};
  }

private:
  constexpr Failure(const Status status, const Location location) noexcept
      : status_(status), location_(location) {}

  Status status_{Status::fail(Reason::ReasonInvalid)};
  Location location_{};
};

namespace detail {

struct ResultPolicy final {
  using Status = compute::Failure;
  using Reason = compute::Reason;
  using Code = compute::Code;

  static constexpr Reason success = Reason::Ok;
  static constexpr Reason invalidated = Reason::ValueInvalid;
  static constexpr bool unchecked_noexcept = false;

  [[nodiscard]] static constexpr Status status(const Reason reason) noexcept {
    return Failure::fail(reason);
  }

  [[nodiscard]] static constexpr Reason reason(const Status status) noexcept {
    return status.reason();
  }

  [[nodiscard]] static constexpr Code code(const Reason reason) noexcept {
    return detail::category(reason);
  }

  [[nodiscard]] static std::string_view error(const Reason reason) noexcept {
    return Status::fail(reason).error();
  }
};

} // namespace detail

template <class T>
class [[nodiscard]] Result final
    : private ::rund::outcome::Result<::rund::compute::Result, T,
                                      detail::ResultPolicy> {
  using Core =
      ::rund::outcome::Result<::rund::compute::Result, T, detail::ResultPolicy>;
  friend Core;

public:
  using Core::and_then;
  using Core::code;
  using Core::error;
  using Core::exit_code;
  using Core::fail;
  using Core::ok;
  using Core::operator bool;
  using Core::operator*;
  using Core::operator->;
  using Core::reason;
  using Core::success;
  using Core::transform;
  using Core::value;
  using Core::value_or;

  [[nodiscard]] Location location() const noexcept {
    return this->ok() ? Location{} : this->failure_status().location();
  }

  [[nodiscard]] static Result fail(const Reason reason,
                                   const Location location) noexcept {
    return Core::fail(Failure::fail(reason, location));
  }

private:
  template <class U>
  explicit Result(typename Core::ValueTag tag,
                  U &&value) noexcept(std::is_nothrow_constructible_v<T, U &&>)
      : Core(tag, std::forward<U>(value)) {}

  explicit Result(typename Core::FailureTag tag,
                  const Failure failure) noexcept
      : Core(tag, failure) {}
};
} // namespace rund::compute

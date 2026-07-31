#pragma once

#include <rund/outcome/result.hpp>
#include <rund/task/status.hpp>

#include <type_traits>
#include <utility>

namespace rund::detail::task {

struct ResultPolicy final {
  using Status = ::rund::task::Status;
  using Reason = ::rund::ReasonCode;
  using Code = ::rund::ReasonCode;

  static constexpr Reason success = ::rund::ReasonCode::Ok;
  static constexpr Reason invalidated = ::rund::ReasonCode::TaskFailed;
  static constexpr bool unchecked_noexcept = true;

  [[nodiscard]] static constexpr Status status(const Reason reason) noexcept {
    return Status::fail(reason);
  }

  [[nodiscard]] static constexpr Reason reason(const Status status) noexcept {
    return status.code();
  }

  [[nodiscard]] static constexpr Code code(const Reason reason) noexcept {
    return reason;
  }

  [[nodiscard]] static std::string_view error(const Reason reason) noexcept {
    return ::rund::ReasonString(reason);
  }
};

} // namespace rund::detail::task

namespace rund::task {

template <typename T> class Result;

template <typename T>
class Result final
    : private ::rund::outcome::Result<::rund::task::Result, T,
                                      ::rund::detail::task::ResultPolicy> {
  using Core = ::rund::outcome::Result<::rund::task::Result, T,
                                       ::rund::detail::task::ResultPolicy>;
  friend Core;

public:
  using Core::code;
  using Core::error;
  using Core::exit_code;
  using Core::fail;
  using Core::ok;
  using Core::operator bool;
  using Core::operator*;
  using Core::operator->;
  using Core::success;

private:
  template <typename U>
  explicit Result(typename Core::ValueTag tag,
                  U &&value) noexcept(std::is_nothrow_constructible_v<T, U &&>)
      : Core(tag, std::forward<U>(value)) {}

  explicit Result(typename Core::FailureTag tag, const Status status) noexcept
      : Core(tag, status) {}
};

template <>
class Result<void> final
    : private ::rund::outcome::Result<::rund::task::Result, void,
                                      ::rund::detail::task::ResultPolicy> {
  using Core = ::rund::outcome::Result<::rund::task::Result, void,
                                       ::rund::detail::task::ResultPolicy>;
  friend Core;

public:
  using Core::code;
  using Core::error;
  using Core::exit_code;
  using Core::fail;
  using Core::ok;
  using Core::operator bool;
  using Core::success;

private:
  explicit constexpr Result(typename Core::StatusTag tag,
                            const Status status) noexcept
      : Core(tag, status) {}
};

} // namespace rund::task

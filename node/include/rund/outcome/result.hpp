#pragma once

#include <cassert>
#include <concepts>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace rund::outcome {

template <template <class> class Family, class Candidate>
inline constexpr bool is_result = false;

template <template <class> class Family, class T>
inline constexpr bool is_result<Family, Family<T>> = true;

template <template <class> class Family, class T, class Policy> class Result {
  static_assert(!std::is_void_v<T>);

  struct Failure final {
    typename Policy::Status status;
  };

protected:
  struct ValueTag final {};
  struct FailureTag final {};

  template <class U = T>
    requires std::is_constructible_v<T, U &&>
  [[nodiscard]] static Family<T>
  success(U &&value) noexcept(std::is_nothrow_constructible_v<T, U &&>) {
    return Family<T>{ValueTag{}, std::forward<U>(value)};
  }

  [[nodiscard]] static Family<T>
  fail(const typename Policy::Reason reason) noexcept {
    return Family<T>{FailureTag{}, Policy::status(reason)};
  }

  [[nodiscard]] static Family<T>
  fail(const typename Policy::Status status) noexcept {
    return Family<T>{FailureTag{}, status};
  }

  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }

  [[nodiscard]] bool ok() const noexcept {
    return std::holds_alternative<T>(data_);
  }

  [[nodiscard]] typename Policy::Code code() const noexcept {
    return Policy::code(reason());
  }

  [[nodiscard]] typename Policy::Reason reason() const noexcept {
    if (ok()) {
      return Policy::success;
    }
    return Policy::reason(failure_status());
  }

  [[nodiscard]] std::string_view error() const noexcept {
    const auto current = reason();
    return current == Policy::success ? std::string_view{}
                                      : Policy::error(current);
  }

  [[nodiscard]] int exit_code() const noexcept { return ok() ? 0 : 1; }

  [[nodiscard]] T &value() & noexcept(Policy::unchecked_noexcept) {
    assert(ok());
    return std::get<T>(data_);
  }

  [[nodiscard]] const T &value() const & noexcept(Policy::unchecked_noexcept) {
    assert(ok());
    return std::get<T>(data_);
  }

  [[nodiscard]] T &&value() && noexcept(Policy::unchecked_noexcept) {
    assert(ok());
    return std::get<T>(std::move(data_));
  }

  [[nodiscard]] T *operator->() noexcept { return std::get_if<T>(&data_); }

  [[nodiscard]] const T *operator->() const noexcept {
    return std::get_if<T>(&data_);
  }

  [[nodiscard]] T &operator*() & noexcept(Policy::unchecked_noexcept) {
    return value();
  }

  [[nodiscard]] const T &
  operator*() const & noexcept(Policy::unchecked_noexcept) {
    return value();
  }

  [[nodiscard]] T &&operator*() && noexcept(Policy::unchecked_noexcept) {
    return std::move(*this).value();
  }

  template <class Fn>
    requires std::invocable<Fn, T &> &&
             (!std::is_void_v<std::invoke_result_t<Fn, T &>>)
  [[nodiscard]] auto transform(Fn &&function) & {
    using U = std::remove_cvref_t<std::invoke_result_t<Fn, T &>>;
    if (!ok()) {
      return Family<U>::fail(failure_status());
    }
    return Family<U>::success(std::forward<Fn>(function)(value()));
  }

  template <class Fn>
    requires std::invocable<Fn, const T &> &&
             (!std::is_void_v<std::invoke_result_t<Fn, const T &>>)
  [[nodiscard]] auto transform(Fn &&function) const & {
    using U = std::remove_cvref_t<std::invoke_result_t<Fn, const T &>>;
    if (!ok()) {
      return Family<U>::fail(failure_status());
    }
    return Family<U>::success(std::forward<Fn>(function)(value()));
  }

  template <class Fn>
    requires std::invocable<Fn, T &&> &&
             (!std::is_void_v<std::invoke_result_t<Fn, T &&>>)
  [[nodiscard]] auto transform(Fn &&function) && {
    using U = std::remove_cvref_t<std::invoke_result_t<Fn, T &&>>;
    if (!ok()) {
      return Family<U>::fail(failure_status());
    }
    return Family<U>::success(
        std::forward<Fn>(function)(std::move(*this).value()));
  }

  template <class Fn>
    requires std::invocable<Fn, T &> &&
             is_result<Family,
                       std::remove_cvref_t<std::invoke_result_t<Fn, T &>>>
  [[nodiscard]] auto and_then(Fn &&function) & {
    using U = std::remove_cvref_t<std::invoke_result_t<Fn, T &>>;
    if (!ok()) {
      return U::fail(failure_status());
    }
    return std::forward<Fn>(function)(value());
  }

  template <class Fn>
    requires std::invocable<Fn, const T &> &&
             is_result<Family,
                       std::remove_cvref_t<std::invoke_result_t<Fn, const T &>>>
  [[nodiscard]] auto and_then(Fn &&function) const & {
    using U = std::remove_cvref_t<std::invoke_result_t<Fn, const T &>>;
    if (!ok()) {
      return U::fail(failure_status());
    }
    return std::forward<Fn>(function)(value());
  }

  template <class Fn>
    requires std::invocable<Fn, T &&> &&
             is_result<Family,
                       std::remove_cvref_t<std::invoke_result_t<Fn, T &&>>>
  [[nodiscard]] auto and_then(Fn &&function) && {
    using U = std::remove_cvref_t<std::invoke_result_t<Fn, T &&>>;
    if (!ok()) {
      return U::fail(failure_status());
    }
    return std::forward<Fn>(function)(std::move(*this).value());
  }

  template <class U>
    requires std::copy_constructible<T> && std::convertible_to<U, T>
  [[nodiscard]] T value_or(U &&otherwise) const & {
    return ok() ? value() : static_cast<T>(std::forward<U>(otherwise));
  }

  template <class U>
    requires std::move_constructible<T> && std::convertible_to<U, T>
  [[nodiscard]] T value_or(U &&otherwise) && {
    return ok() ? std::move(*this).value()
                : static_cast<T>(std::forward<U>(otherwise));
  }

  template <class U>
  explicit Result(ValueTag,
                  U &&value) noexcept(std::is_nothrow_constructible_v<T, U &&>)
      : data_(std::in_place_type<T>, std::forward<U>(value)) {}

  explicit Result(FailureTag, const typename Policy::Status status) noexcept
      : data_(Failure{status}) {}

  [[nodiscard]] typename Policy::Status failure_status() const noexcept {
    return data_.valueless_by_exception()
               ? Policy::status(Policy::invalidated)
               : std::get<Failure>(data_).status;
  }

private:
  std::variant<T, Failure> data_;
};

template <template <class> class Family, class Policy>
class Result<Family, void, Policy> {
protected:
  struct StatusTag final {};

  [[nodiscard]] static constexpr Family<void> success() noexcept {
    return Family<void>{StatusTag{}, Policy::Status::success()};
  }

  [[nodiscard]] static constexpr Family<void>
  fail(const typename Policy::Reason reason) noexcept {
    return Family<void>{StatusTag{}, Policy::status(reason)};
  }

  [[nodiscard]] static constexpr Family<void>
  fail(const typename Policy::Status status) noexcept {
    return Family<void>{StatusTag{}, status};
  }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }

  [[nodiscard]] constexpr bool ok() const noexcept { return status_.ok(); }

  [[nodiscard]] constexpr typename Policy::Code code() const noexcept {
    return Policy::code(Policy::reason(status_));
  }

  [[nodiscard]] std::string_view error() const noexcept {
    return status_.error();
  }

  [[nodiscard]] constexpr int exit_code() const noexcept {
    return status_.exit_code();
  }

  explicit constexpr Result(StatusTag,
                            const typename Policy::Status status) noexcept
      : status_(status) {}

private:
  typename Policy::Status status_;
};

} // namespace rund::outcome

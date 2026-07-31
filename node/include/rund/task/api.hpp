#pragma once

#include <rund/task/api/access.hpp>
#include <rund/task/coroutine.hpp>
#include <rund/task/handle.hpp>

#include <span>
#include <type_traits>
#include <utility>

namespace rund::task {

[[nodiscard]] inline Handle spawn(const char *const name,
                                  Task<void> &&task) noexcept {
  return ::rund::detail::task::ApiAccess::SpawnCoroutine(name, std::move(task));
}

template <typename Callable>
[[nodiscard]] Handle spawn(const char *const name, Callable &&callable) {
  using StoredCallable = std::decay_t<Callable>;
  static_assert(std::is_void_v<std::invoke_result_t<StoredCallable &>>,
                "task::spawn callables must return void");
  try {
    return ::rund::detail::task::ApiAccess::SpawnRunnable(
        name, ::rund::detail::task::Callable{std::forward<Callable>(callable)});
  } catch (...) {
    return Handle{0u, 0u, 0u, ReasonCode::TaskCallableAllocationFailed};
  }
}

template <typename Callable> [[nodiscard]] Handle spawn(Callable &&callable) {
  return spawn("task", std::forward<Callable>(callable));
}

[[nodiscard]] inline Status
join_all(const std::span<const Handle> handles) noexcept {
  if (handles.empty()) {
    return Status::success();
  }
  return ::rund::detail::task::ApiAccess::JoinMany(handles.data(),
                                                   handles.size());
}

inline Status join(const Handle handle) noexcept {
  return ::rund::detail::task::ApiAccess::JoinMany(&handle, 1u);
}

template <typename... Handles>
[[nodiscard]] Status join(Handle first, Handles... rest) noexcept {
  const Handle handles[] = {first, rest...};
  return ::rund::detail::task::ApiAccess::JoinMany(handles,
                                                   sizeof...(rest) + 1u);
}

template <typename Callable> [[nodiscard]] Status scope(Callable &&callable) {
  using StoredCallable = std::decay_t<Callable>;
  static_assert(std::is_void_v<std::invoke_result_t<StoredCallable &>>,
                "task::scope callables must return void");
  try {
    return ::rund::detail::task::ApiAccess::EnterScope(
        ::rund::detail::task::Callable{std::forward<Callable>(callable)});
  } catch (...) {
    return Status::fail(ReasonCode::TaskCallableAllocationFailed);
  }
}

} // namespace rund::task

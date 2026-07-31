#pragma once

#include <rund/task/callable.hpp>
#include <rund/task/coroutine.hpp>
#include <rund/task/handle.hpp>
#include <rund/task/handle/spawn.hpp>

#include <cstddef>
#include <span>

namespace rund::task {

template <typename T> class TaskAwaiter;
[[nodiscard]] Status join(Handle handle) noexcept;
[[nodiscard]] Status join_all(std::span<const Handle> handles) noexcept;
template <typename... Handles>
[[nodiscard]] Status join(Handle first, Handles... rest) noexcept;
[[nodiscard]] Handle spawn(const char *name, Task<void> &&task) noexcept;
template <typename Callable>
[[nodiscard]] Handle spawn(const char *name, Callable &&callable);
template <typename Callable> [[nodiscard]] Status scope(Callable &&callable);

} // namespace rund::task

namespace rund::detail::task {

class ApiAccess final {
public:
  ApiAccess() = delete;

  [[nodiscard]] static ::rund::task::Handle
  SpawnRunnable(const char *name, Callable callable) noexcept;
  [[nodiscard]] static ::rund::task::Handle
  SpawnCoroutine(const char *name, ::rund::task::Task<void> &&task) noexcept;
  [[nodiscard]] static Spawned SpawnAwaited(CoroutineStart start) noexcept;
  [[nodiscard]] static ::rund::task::Status
  JoinMany(const ::rund::task::Handle *handles, std::size_t count) noexcept;
  [[nodiscard]] static ::rund::task::Status
  EnterScope(Callable callable) noexcept;
};

} // namespace rund::detail::task

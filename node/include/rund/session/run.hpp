#pragma once

#include <rund/session.hpp>

#include <concepts>
#include <functional>
#include <utility>

namespace rund {

namespace detail::run {

struct Access final {
  using Callback = void (*)(void *, Session &);

  [[nodiscard]] static Session::Result execute(SessionConfig config,
                                               void *callback, Callback invoke);
};

template <typename Callback> struct CallbackRef final {
  Callback &value;
};

template <typename Callback> void invoke(void *const raw, Session &session) {
  auto &callable = static_cast<CallbackRef<Callback> *>(raw)->value;
  if constexpr (std::invocable<Callback &, Session &>) {
    std::invoke(callable, session);
  } else {
    static_assert(std::invocable<Callback &>,
                  "rund::run callback must accept Session& or no arguments");
    std::invoke(callable);
  }
}

} // namespace detail::run

template <typename Callback>
[[nodiscard]] Session::Result run(SessionConfig config, Callback &&callback) {
  detail::run::CallbackRef<Callback> callback_ref{callback};
  return detail::run::Access::execute(std::move(config), &callback_ref,
                                      detail::run::invoke<Callback>);
}

template <typename Callback>
[[nodiscard]] Session::Result run(Callback &&callback) {
  return run(SessionConfig{}, std::forward<Callback>(callback));
}

} // namespace rund

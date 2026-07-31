#pragma once

#include <rund/task/await.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/ready.hpp>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace rund::net::accept {

struct Budget {
  std::uint32_t max_accepts = 64u;
};

struct Drain : net::Status {
  using net::Status::Status;

  std::uint32_t accepts = 0u;
  bool would_block = false;
  bool budget_exhausted = false;
  bool handler_stopped = false;
};

[[nodiscard]] inline Drain fail_drain(const ::rund::ReasonCode code) noexcept {
  return Drain{code};
}

namespace detail {

using Handler = bool (*)(const void *, Result &&) noexcept;

[[nodiscard]] Drain drain(ready::Ticket &&ticket, Budget budget,
                          const void *state, Handler handler) noexcept;

} // namespace detail

template <typename Callback>
[[nodiscard]] Drain drain(ready::Ticket &&ticket, const Budget budget,
                          Callback &&callback) noexcept {
  static_assert(std::is_invocable_r_v<bool, Callback &, Result &&>,
                "accept::drain callback must accept Result&& and return bool");
  const auto invoke = [](const void *const state, Result &&accepted) noexcept {
    using State = std::remove_reference_t<Callback>;
    return std::invoke(*const_cast<State *>(static_cast<const State *>(state)),
                       std::move(accepted));
  };
  return detail::drain(std::move(ticket), budget, std::addressof(callback),
                       invoke);
}

template <typename Callback>
[[nodiscard]] task::Task<Drain> drain(const SocketView listener,
                                      const Budget budget,
                                      Callback callback) noexcept {
  static_assert(std::is_invocable_r_v<bool, Callback &, Result &&>,
                "accept::drain callback must accept Result&& and return bool");
  if (budget.max_accepts == 0u) {
    Drain result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    co_return result;
  }
  ready::Ticket ticket = co_await ready::read(listener);
  co_return drain(std::move(ticket), budget, callback);
}

template <typename Callback>
[[nodiscard]] task::Task<Drain> drain(const SocketView listener,
                                      Callback &&callback) noexcept {
  return drain(listener, Budget{}, std::forward<Callback>(callback));
}

} // namespace rund::net::accept

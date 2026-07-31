#pragma once

#include "callback.hpp"

#include <mutex>

namespace rund::node::accel::detail::submission {

template <class Owner> struct State final {
  std::mutex mutex{};
  Owner *owner{};
  KernelCompletion completion{};
  void *user{};

  [[nodiscard]] bool active() const noexcept { return owner != nullptr; }
};

template <class Owner> struct Claim final {
  Owner *owner{};
  KernelCompletion completion{};
  void *user{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner != nullptr && completion != nullptr;
  }
};

template <class Owner>
[[nodiscard]] bool Begin(State<Owner> &state, Owner &owner,
                         const KernelCompletion completion,
                         void *const user) noexcept {
  std::lock_guard lock{state.mutex};
  if (state.active() || completion == nullptr) {
    return false;
  }
  state.owner = &owner;
  state.completion = completion;
  state.user = user;
  return true;
}

template <class Owner>
[[nodiscard]] Claim<Owner> Take(State<Owner> &state) noexcept {
  std::lock_guard lock{state.mutex};
  if (!state.active() || state.completion == nullptr) {
    return {};
  }
  const Claim<Owner> claim{
      .owner = state.owner,
      .completion = state.completion,
      .user = state.user,
  };
  state.owner = nullptr;
  state.completion = nullptr;
  state.user = nullptr;
  return claim;
}

template <class Owner> void Cancel(State<Owner> &state) noexcept {
  std::lock_guard lock{state.mutex};
  state.owner = nullptr;
  state.completion = nullptr;
  state.user = nullptr;
}

} // namespace rund::node::accel::detail::submission

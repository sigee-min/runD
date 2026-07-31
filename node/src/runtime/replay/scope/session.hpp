#pragma once

#include "plan.hpp"

#include <rund/session.hpp>

#include <atomic>
#include <cstdint>

namespace rund::replay::detail::scope {

class Timing;

struct Lease final {
  const std::atomic<std::uint64_t> *generation = nullptr;
  std::uint64_t value = 0u;

  [[nodiscard]] bool valid() const noexcept {
    return generation != nullptr && value != 0u &&
           generation->load(std::memory_order_acquire) == value;
  }
};

struct Generation final {
  std::atomic<std::uint64_t> next{1u};
  std::atomic<std::uint64_t> active{0u};
};

struct Prepared final {
  ExpectedOwner owner{};
  ::rund::replay::Code code = ::rund::replay::Code::ScopePrepareFailed;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok && owner != nullptr;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct Access final {
  using Callback = void (*)(void *, ::rund::Session &, Lease);

  [[nodiscard]] static ::rund::Session::Result
  run(::rund::Session &session, const Plan &plan, void *callback,
      Callback invoke, Timing &timing);

  [[nodiscard]] static bool detail(::rund::Session &session) noexcept;

  [[nodiscard]] static Prepared
  prepare(::rund::Session &session, std::vector<::rund::host::Event> events,
          ::rund::node::replay_detail::payload::Archive payloads) noexcept;

  [[nodiscard]] static std::uint64_t
  capacity(::rund::Session &session) noexcept;

  [[nodiscard]] static std::uint32_t inputs(::rund::Session &session) noexcept;
};

} // namespace rund::replay::detail::scope

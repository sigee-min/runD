#pragma once

#include <node/runtime/runtime.hpp>
#include <rund/session.hpp>

#include "../replay/scope/session.hpp"

#include <memory>

namespace rund {

struct Session::State final {
  std::unique_ptr<node::Runtime> runtime{};
  ::rund::replay::detail::scope::Generation generation{};
  ::rund::replay::Storage replay_storage{};
  ::rund::replay::Diagnostic replay_diagnostic{};
  std::uint64_t replay_capacity = 0u;
  std::uint32_t replay_inputs = 0u;
};

} // namespace rund

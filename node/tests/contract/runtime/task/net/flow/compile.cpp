#include "local.hpp"

#include <rund/net/flow.hpp>

namespace {

constexpr bool CompileTimeFlowAccounting() {
  constexpr rund::net::flow::Limit limit{.max_inflight_bytes = 8u,
                                         .max_total_bytes = 16u};
  constexpr rund::net::flow::Result reserved =
      rund::net::flow::reserve(rund::net::flow::State{}, limit, 4u);
  static_assert(reserved.ok());
  static_assert(reserved.state.inflight_bytes == 4u);
  static_assert(reserved.state.total_bytes == 4u);

  constexpr rund::net::flow::Result released =
      rund::net::flow::release(reserved.state, 2u);
  static_assert(released.ok());
  static_assert(released.state.inflight_bytes == 2u);
  static_assert(released.state.total_bytes == 4u);

  constexpr rund::net::flow::Result zero =
      rund::net::flow::reserve(rund::net::flow::State{}, limit, 0u);
  static_assert(!zero.ok());
  static_assert(zero.code() == rund::ReasonCode::NetFlowZeroBytes);
  constexpr rund::net::flow::Result zero_release =
      rund::net::flow::release(reserved.state, 0u);
  static_assert(!zero_release.ok());
  static_assert(zero_release.code() == rund::ReasonCode::NetFlowZeroBytes);
  return reserved.ok() && released.ok() && !zero.ok();
}

static_assert(CompileTimeFlowAccounting());

} // namespace

int RunNetFlowCompileCase() { return 0; }

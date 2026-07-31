#pragma once

#include <rund/net/ready.hpp>

#include "../../runtime/reactor/readiness/state.hpp"

namespace rund::net {

[[nodiscard]] node::ReactorInterest
ReactorInterestFor(ready::Interest interest) noexcept;
[[nodiscard]] bool InterestFromReactor(node::ReactorInterest interest,
                                       ready::Interest *out) noexcept;

} // namespace rund::net

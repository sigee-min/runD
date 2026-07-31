#pragma once

#include <rund/reason.hpp>
#include <rund/replay/code.hpp>

#include <optional>

namespace rund::node::replay_detail {

[[nodiscard]] ::rund::replay::Code code(ReasonCode reason) noexcept;
[[nodiscard]] std::optional<ReasonCode>
reason(::rund::replay::Code code) noexcept;

} // namespace rund::node::replay_detail

#pragma once

#include <rund/host/event.hpp>

namespace rund::node::host_detail {

[[nodiscard]] bool known_replay_kind(::rund::host::EventKind kind) noexcept;

[[nodiscard]] const char *event_name(::rund::host::EventKind kind) noexcept;

} // namespace rund::node::host_detail

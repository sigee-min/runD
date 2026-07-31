#pragma once

#include <rund/host/event.hpp>
#include <rund/reason.hpp>

#include <cstdint>

namespace rund::node {

[[nodiscard]] ::rund::host::Event
MakeReactorHostEvent(ReasonCode code, std::uint64_t task_id,
                     std::uint64_t host_handle_id) noexcept;

} // namespace rund::node

#pragma once

#include "../local.hpp"

#include <rund/host/event.hpp>
#include <rund/net/server/result.hpp>

#include <cstdint>

namespace server_inline_detail {

[[nodiscard]] bool HasCounts(const rund::net::server::Result &result,
                             std::uint32_t completed, std::uint32_t failed,
                             std::uint32_t stopped) noexcept;
[[nodiscard]] std::uint64_t CountEvents(const rund::Session::Result &run,
                                        rund::host::EventKind kind) noexcept;

[[nodiscard]] int RunServerInlineOutcomeCase();
[[nodiscard]] int RunServerTaskFailureCase();
[[nodiscard]] int RunServerInlineWouldBlockCase();
[[nodiscard]] int RunServerInlineArrivalCase();
[[nodiscard]] int RunServerPublicIoCase();

} // namespace server_inline_detail

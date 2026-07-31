#pragma once

#include <cstdint>
#include <rund/net/ready/set.hpp>
#include <vector>

#include "model.hpp"

namespace rund::node {

[[nodiscard]] ReactorReadySet* ReactorReadySetFind(
    std::vector<ReactorReadySet>& sets, ::rund::net::ready::Set handle) noexcept;

[[nodiscard]] const ReactorReadySet* ReactorReadySetFind(
    const std::vector<ReactorReadySet>& sets, ::rund::net::ready::Set handle) noexcept;

[[nodiscard]] std::uint32_t ReactorReadySetLiveCount(
    const std::vector<ReactorReadySet>& sets) noexcept;

[[nodiscard]] std::uint32_t ReactorReadySetMemberCount(
    const std::vector<ReactorReadySet>& sets) noexcept;

[[nodiscard]] bool ReactorReadySetHasDuplicate(
    const ReactorReadySet& set, ::rund::net::SocketView socket,
    ReactorInterest interest) noexcept;

[[nodiscard]] std::uint32_t ReactorReadySetClearMembers(
    ReactorReadySet& set) noexcept;

}  // namespace rund::node

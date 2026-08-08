#pragma once

#include "../many.hpp"

#include <rund/net/ready/many.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace rund::node {

void ReactorManyEventSlotsReset(std::vector<ReactorManyEventSlot> &slots,
                                std::uint32_t request_count) noexcept;

void ReactorManyEventSlotsAppend(
    ReactorManyGroup &group, const ReactorManyRequest &request,
    ReactorEvent events, ReasonCode code,
    std::vector<ReactorManyEventSlot> &slots) noexcept;

[[nodiscard]] bool ReactorManyEventSlotsCopy(
    const ReactorManyGroup &group, std::span<const ReactorManyEventSlot> slots,
    std::span<::rund::net::ready::Event> out, std::uint32_t *copied) noexcept;

[[nodiscard]] bool
ReactorManyEventSlotsInsertGroup(std::vector<ReactorManyEventSlot> &slots,
                                 std::uint32_t first_request,
                                 std::uint32_t request_count) noexcept;

void ReactorManyEventSlotsEraseGroup(std::vector<ReactorManyEventSlot> &slots,
                                     std::uint32_t first_request,
                                     std::uint32_t request_count) noexcept;

} // namespace rund::node

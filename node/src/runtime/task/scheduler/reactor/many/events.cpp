#include "events.hpp"

#include "../../../../../host/net/interest.hpp"
#include "../../../../../host/net/ready/ticket.hpp"
#include "../../../../../host/net/socket/access.hpp"
#include "../../../../reactor/readiness/handle.hpp"
#include "../../../../reactor/readiness/mask.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace rund::node {

void ReactorManyEventSlotsReset(std::vector<ReactorManyEventSlot>& slots,
                                const std::uint32_t request_count) noexcept {
  try {
    slots.assign(request_count, ReactorManyEventSlot{});
  } catch (...) {
    slots.clear();
  }
}

bool ReactorManyEventSlotsAppend(
    ReactorManyGroup& group,
    const ReactorManyRequest& request,
    const ReactorEvent events,
    const ReasonCode code,
    std::vector<ReactorManyEventSlot>& slots) noexcept {
  if (request.group_id != group.group_id || request.slot >= group.request_count) {
    return true;
  }
  const std::size_t slot_index =
      static_cast<std::size_t>(group.first_request) + request.slot;
  if (slot_index >= slots.size()) {
    return true;
  }

  ReactorManyEventSlot& slot = slots[slot_index];
  if (slot.occupied) {
    return true;
  }
  if (group.stored_event_count >= group.max_events) {
    group.budget_exhausted = true;
    return true;
  }

  slot.occupied = true;
  slot.event = ReactorManyEvent{
      .group_id = group.group_id,
      .socket = request.socket,
      .fd = request.fd,
      .slot = request.slot,
      .event_index = request.event_index,
      .interest = request.interest,
      .events = events,
      .code = code,
  };
  ++group.stored_event_count;
  return true;
}

bool ReactorManyEventSlotsCopy(
    const ReactorManyGroup& group,
    const std::span<const ReactorManyEventSlot> slots,
    const std::span<::rund::net::ready::Event> out,
    std::uint32_t* const copied) noexcept {
  if (copied == nullptr) {
    return false;
  }
  *copied = 0u;
  const std::size_t first = group.first_request;
  if (first >= slots.size()) {
    return group.request_count == 0u;
  }
  const std::size_t count =
      std::min<std::size_t>(group.request_count, slots.size() - first);
  for (std::size_t offset = 0u; offset < count && *copied < out.size();
       ++offset) {
    const ReactorManyEventSlot& slot = slots[first + offset];
    if (!slot.occupied || slot.event.group_id != group.group_id) {
      continue;
    }
    ::rund::net::ready::Interest interest{};
    if (!::rund::net::InterestFromReactor(slot.event.interest, &interest)) {
      return false;
    }
    ::rund::net::ready::Event event{};
    event.index = slot.event.event_index;
    event.ticket = ::rund::net::ready::detail::Access::make(
        slot.event.code, slot.event.socket, interest,
        ReactorEventBits(slot.event.events));
    out[*copied] = std::move(event);
    ++(*copied);
  }
  return true;
}

bool ReactorManyEventSlotsInsertGroup(
    std::vector<ReactorManyEventSlot>& slots,
    const std::uint32_t first_request,
    const std::uint32_t request_count) noexcept {
  if (first_request > slots.size()) {
    return false;
  }
  try {
    slots.insert(slots.begin() + static_cast<std::ptrdiff_t>(first_request),
                 request_count,
                 ReactorManyEventSlot{});
  } catch (...) {
    return false;
  }
  return true;
}

void ReactorManyEventSlotsEraseGroup(
    std::vector<ReactorManyEventSlot>& slots,
    const std::uint32_t first_request,
    const std::uint32_t request_count) noexcept {
  const std::size_t first = first_request;
  if (first >= slots.size()) {
    return;
  }
  const std::size_t count =
      std::min<std::size_t>(request_count, slots.size() - first);
  slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(first),
              slots.begin() + static_cast<std::ptrdiff_t>(first + count));
}

}  // namespace rund::node

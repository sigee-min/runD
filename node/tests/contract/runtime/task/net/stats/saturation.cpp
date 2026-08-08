#include "local.hpp"

#include "src/runtime/task/scheduler/core/record/network.hpp"

#include "test/assert.hpp"

#include <rund/host/event.hpp>
#include <rund/task/stats/slots.hpp>

#include <array>
#include <cstdint>
#include <limits>

namespace {

using ::rund::detail::task::Stat;
using ::rund::detail::task::StatSlot;
using ::rund::detail::task::StatStorage;
using ::rund::host::EventKind;

struct NetworkCounterCase final {
  EventKind kind;
  StatSlot call;
  StatSlot bytes{StatSlot::Count};
};

constexpr std::array kNetworkCounterCases{
    NetworkCounterCase{EventKind::NetSocket, StatSlot::NetworkSocketsOpened},
    NetworkCounterCase{EventKind::NetBind, StatSlot::NetworkSocketsBound},
    NetworkCounterCase{EventKind::NetListen, StatSlot::NetworkSocketsListened},
    NetworkCounterCase{EventKind::NetShutdown,
                       StatSlot::NetworkSocketsShutdown},
    NetworkCounterCase{EventKind::NetLocalAddress,
                       StatSlot::NetworkLocalAddressReads},
    NetworkCounterCase{EventKind::NetAccept, StatSlot::NetworkAccepts},
    NetworkCounterCase{EventKind::NetConnect, StatSlot::NetworkConnects},
    NetworkCounterCase{EventKind::NetRecv, StatSlot::NetworkRecvCalls,
                       StatSlot::NetworkBytesReceived},
    NetworkCounterCase{EventKind::NetSend, StatSlot::NetworkSendCalls,
                       StatSlot::NetworkBytesSent},
    NetworkCounterCase{EventKind::NetRecvDatagram,
                       StatSlot::NetworkDatagramRecvCalls,
                       StatSlot::NetworkBytesReceived},
    NetworkCounterCase{EventKind::NetSendDatagram,
                       StatSlot::NetworkDatagramSendCalls,
                       StatSlot::NetworkBytesSent},
    NetworkCounterCase{EventKind::NetSetSocketOption,
                       StatSlot::NetworkSocketOptionsSet},
    NetworkCounterCase{EventKind::NetGetSocketOption,
                       StatSlot::NetworkSocketOptionsRead},
    NetworkCounterCase{EventKind::NetRecvVectored,
                       StatSlot::NetworkVectoredRecvCalls,
                       StatSlot::NetworkBytesReceived},
    NetworkCounterCase{EventKind::NetSendVectored,
                       StatSlot::NetworkVectoredSendCalls,
                       StatSlot::NetworkBytesSent},
};

static_assert(kNetworkCounterCases.size() == 15u);
static_assert([] {
  for (std::size_t left = 0u; left < kNetworkCounterCases.size(); ++left) {
    for (std::size_t right = left + 1u; right < kNetworkCounterCases.size();
         ++right) {
      if (kNetworkCounterCases[left].kind == kNetworkCounterCases[right].kind ||
          kNetworkCounterCases[left].call == kNetworkCounterCases[right].call) {
        return false;
      }
    }
  }
  return true;
}());

[[nodiscard]] bool
NetworkSlotsMatch(const StatStorage &stats,
                  const StatSlot expected_call = StatSlot::Count,
                  const std::uint64_t call_value = 0u,
                  const StatSlot expected_bytes = StatSlot::Count,
                  const std::uint64_t byte_value = 0u,
                  const std::uint64_t would_block_value = 0u,
                  const std::uint64_t admission_rejection_value = 0u) noexcept {
  for (const NetworkCounterCase entry : kNetworkCounterCases) {
    if (Stat(stats, entry.call) !=
        (entry.call == expected_call ? call_value : 0u)) {
      return false;
    }
  }
  return Stat(stats, StatSlot::NetworkBytesReceived) ==
             (expected_bytes == StatSlot::NetworkBytesReceived ? byte_value
                                                               : 0u) &&
         Stat(stats, StatSlot::NetworkBytesSent) ==
             (expected_bytes == StatSlot::NetworkBytesSent ? byte_value : 0u) &&
         Stat(stats, StatSlot::NetworkWouldBlock) == would_block_value &&
         Stat(stats, StatSlot::NetworkAdmissionRejections) ==
             admission_rejection_value;
}

} // namespace

int NetStatsSaturatingCounters() {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  for (const NetworkCounterCase entry : kNetworkCounterCases) {
    StatStorage stats{};
    Stat(stats, entry.call) = maximum - 1u;
    if (entry.bytes != StatSlot::Count) {
      Stat(stats, entry.bytes) = maximum - 1u;
    }
    const ::rund::host::Event event{
        .kind = entry.kind,
        .status = ::rund::host::Status::Ok,
        .completed_bytes = 2u,
    };
    ::rund::node::record_detail::RecordNetworkStats(stats, event);
    ::rund::node::record_detail::RecordNetworkStats(stats, event);
    TEST_ASSERT(
        NetworkSlotsMatch(stats, entry.call, maximum, entry.bytes, maximum));
  }

  for (const NetworkCounterCase entry : kNetworkCounterCases) {
    StatStorage blocked{};
    ::rund::node::record_detail::RecordNetworkStats(
        blocked, ::rund::host::Event{
                     .kind = entry.kind,
                     .status = ::rund::host::Status::WouldBlock,
                     .completed_bytes = maximum,
                 });
    TEST_ASSERT(NetworkSlotsMatch(blocked, StatSlot::Count, 0u, StatSlot::Count,
                                  0u, 1u));
  }

  StatStorage would_block{};
  Stat(would_block, StatSlot::NetworkWouldBlock) = maximum - 1u;
  const ::rund::host::Event blocked_network{
      .kind = EventKind::NetRecv,
      .status = ::rund::host::Status::WouldBlock,
      .completed_bytes = maximum,
  };
  ::rund::node::record_detail::RecordNetworkStats(would_block, blocked_network);
  ::rund::node::record_detail::RecordNetworkStats(would_block, blocked_network);
  TEST_ASSERT(NetworkSlotsMatch(would_block, StatSlot::Count, 0u,
                                StatSlot::Count, 0u, maximum));

  StatStorage unrelated_would_block{};
  const ::rund::host::Event blocked_io{
      .kind = EventKind::IoRead,
      .status = ::rund::host::Status::WouldBlock,
      .completed_bytes = maximum,
  };
  ::rund::node::record_detail::RecordNetworkStats(unrelated_would_block,
                                                  blocked_io);
  TEST_ASSERT(NetworkSlotsMatch(unrelated_would_block));

  StatStorage failed{};
  ::rund::node::record_detail::RecordNetworkStats(
      failed, ::rund::host::Event{.kind = EventKind::NetSend,
                                  .status = ::rund::host::Status::SyscallFailed,
                                  .completed_bytes = maximum});
  TEST_ASSERT(NetworkSlotsMatch(failed));

  StatStorage unrelated{};
  ::rund::node::record_detail::RecordNetworkStats(
      unrelated, ::rund::host::Event{.kind = EventKind::IoRead,
                                     .status = ::rund::host::Status::Ok,
                                     .completed_bytes = maximum});
  TEST_ASSERT(NetworkSlotsMatch(unrelated));
  return 0;
}

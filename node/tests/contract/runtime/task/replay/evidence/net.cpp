#include "test/assert.hpp"

#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/replay.hpp>

#include <array>
#include <cerrno>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

int RunReplayNetEvidenceContract() {
  const std::array<std::byte, 3u> recv_payload{std::byte{'r'}, std::byte{'x'},
                                               std::byte{'1'}};
  const std::array<std::byte, 2u> send_payload{std::byte{'t'}, std::byte{'x'}};
  const std::vector<rund::host::Event> events{
      rund::host::Event{
          .sequence = 1u,
          .kind = rund::host::EventKind::NetRecv,
          .status = rund::host::Status::Ok,
          .host_handle_id = 41u,
          .requested_bytes = 8u,
          .completed_bytes = recv_payload.size(),
          .native_errno = 0,
          .payload_hash = rund::host::hash_bytes(recv_payload.data(),
                                                       recv_payload.size()),
      },
      rund::host::Event{
          .sequence = 2u,
          .kind = rund::host::EventKind::NetSend,
          .status = rund::host::Status::Ok,
          .host_handle_id = 42u,
          .requested_bytes = send_payload.size(),
          .completed_bytes = send_payload.size(),
          .native_errno = 0,
          .payload_hash = rund::host::hash_bytes(send_payload.data(),
                                                       send_payload.size()),
      },
      rund::host::Event{
          .sequence = 3u,
          .kind = rund::host::EventKind::IoSetNonblocking,
          .status = rund::host::Status::Ok,
          .host_handle_id = 43u,
          .completed_bytes = 1u,
          .native_errno = 0,
      },
      rund::host::Event{
          .sequence = 4u,
          .kind = rund::host::EventKind::NetAccept,
          .status = rund::host::Status::WouldBlock,
          .host_handle_id = 44u,
          .native_errno = EAGAIN,
      },
      rund::host::Event{
          .sequence = 5u,
          .kind = rund::host::EventKind::NetConnect,
          .status = rund::host::Status::Ok,
          .host_handle_id = 45u,
          .payload_hash = rund::StableHash{.value = 123u},
      },
      rund::host::Event{
          .sequence = 6u,
          .kind = rund::host::EventKind::NetSocket,
          .status = rund::host::Status::Ok,
          .host_handle_id = 46u,
      },
      rund::host::Event{
          .sequence = 7u,
          .kind = rund::host::EventKind::NetBind,
          .status = rund::host::Status::Ok,
          .host_handle_id = 46u,
          .payload_hash = rund::StableHash{.value = 124u},
      },
      rund::host::Event{
          .sequence = 8u,
          .kind = rund::host::EventKind::NetListen,
          .status = rund::host::Status::Ok,
          .host_handle_id = 46u,
          .completed_bytes = 64u,
      },
      rund::host::Event{
          .sequence = 9u,
          .kind = rund::host::EventKind::NetLocalAddress,
          .status = rund::host::Status::Ok,
          .host_handle_id = 46u,
          .payload_hash = rund::StableHash{.value = 125u},
      },
      rund::host::Event{
          .sequence = 10u,
          .kind = rund::host::EventKind::NetShutdown,
          .status = rund::host::Status::Ok,
          .host_handle_id = 46u,
          .completed_bytes = 2u,
      },
      rund::host::Event{
          .sequence = 11u,
          .kind = rund::host::EventKind::NetRecvDatagram,
          .status = rund::host::Status::Ok,
          .host_handle_id = 47u,
          .requested_bytes = 8u,
          .completed_bytes = recv_payload.size(),
          .name_hash = rund::StableHash{.value = 126u},
          .payload_hash = rund::host::hash_bytes(recv_payload.data(),
                                                       recv_payload.size()),
      },
      rund::host::Event{
          .sequence = 12u,
          .kind = rund::host::EventKind::NetSendDatagram,
          .status = rund::host::Status::Ok,
          .host_handle_id = 48u,
          .requested_bytes = send_payload.size(),
          .completed_bytes = send_payload.size(),
          .name_hash = rund::StableHash{.value = 127u},
          .payload_hash = rund::host::hash_bytes(send_payload.data(),
                                                       send_payload.size()),
      },
      rund::host::Event{
          .sequence = 13u,
          .kind = rund::host::EventKind::NetRecvVectored,
          .status = rund::host::Status::Ok,
          .host_handle_id = 49u,
          .requested_bytes = 8u,
          .completed_bytes = recv_payload.size(),
          .payload_hash = rund::host::hash_bytes(recv_payload.data(),
                                                       recv_payload.size()),
      },
      rund::host::Event{
          .sequence = 14u,
          .kind = rund::host::EventKind::NetSendVectored,
          .status = rund::host::Status::Ok,
          .host_handle_id = 50u,
          .requested_bytes = send_payload.size(),
          .completed_bytes = send_payload.size(),
          .payload_hash = rund::host::hash_bytes(send_payload.data(),
                                                       send_payload.size()),
      },
  };
  const std::vector<std::byte> encoded =
      rund::node::EncodeHostReplayEvents(events);
  std::vector<rund::host::Event> decoded{};
  TEST_ASSERT(rund::node::DecodeHostReplayEvents(encoded, decoded));
  TEST_ASSERT(decoded.size() == 14u);
  TEST_ASSERT(decoded[0].kind == rund::host::EventKind::NetRecv);
  TEST_ASSERT(decoded[1].kind == rund::host::EventKind::NetSend);
  TEST_ASSERT(decoded[2].kind == rund::host::EventKind::IoSetNonblocking);
  TEST_ASSERT(decoded[3].kind == rund::host::EventKind::NetAccept);
  TEST_ASSERT(decoded[3].status == rund::host::Status::WouldBlock);
  TEST_ASSERT(decoded[4].kind == rund::host::EventKind::NetConnect);
  TEST_ASSERT(decoded[5].kind == rund::host::EventKind::NetSocket);
  TEST_ASSERT(decoded[6].kind == rund::host::EventKind::NetBind);
  TEST_ASSERT(decoded[7].kind == rund::host::EventKind::NetListen);
  TEST_ASSERT(decoded[7].completed_bytes == 64u);
  TEST_ASSERT(decoded[8].kind == rund::host::EventKind::NetLocalAddress);
  TEST_ASSERT(decoded[9].kind == rund::host::EventKind::NetShutdown);
  TEST_ASSERT(decoded[9].completed_bytes == 2u);
  TEST_ASSERT(decoded[10].kind == rund::host::EventKind::NetRecvDatagram);
  TEST_ASSERT(decoded[10].name_hash.value == 126u);
  TEST_ASSERT(decoded[11].kind == rund::host::EventKind::NetSendDatagram);
  TEST_ASSERT(decoded[11].name_hash.value == 127u);
  TEST_ASSERT(decoded[12].kind == rund::host::EventKind::NetRecvVectored);
  TEST_ASSERT(
      decoded[12].payload_hash.value ==
      rund::host::hash_bytes(recv_payload.data(), recv_payload.size())
          .value);
  TEST_ASSERT(decoded[13].kind == rund::host::EventKind::NetSendVectored);
  TEST_ASSERT(
      decoded[13].payload_hash.value ==
      rund::host::hash_bytes(send_payload.data(), send_payload.size())
          .value);
  TEST_ASSERT(rund::host::hash_events(decoded).value ==
              rund::host::hash_events(events).value);

  return 0;
}

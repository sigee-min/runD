#include "test/assert.hpp"

#include "local.hpp"

#include <node/runtime/replay.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

[[nodiscard]] rund::host::Event
ValidReplayEvent(const rund::host::EventKind kind) {
  return rund::host::Event{
      .sequence = 17u,
      .kind = kind,
      .status = rund::host::Status::Ok,
      .task_id = 23u,
      .logical_time_ns = 31u,
      .stream_id = 41u,
      .draw_id = 43u,
      .host_handle_id = 47u,
      .offset = 53u,
      .requested_bytes = 59u,
      .completed_bytes = 61u,
      .native_errno = 0,
      .name_hash = rund::StableHash{67u},
      .path_hash = rund::StableHash{71u},
      .payload_hash = rund::StableHash{73u},
  };
}

[[nodiscard]] bool DecodeFailsWith(const std::span<const std::byte> encoded,
                                   const rund::replay::Code code) {
  const rund::node::replay_detail::HostReplayDecodeResult result =
      rund::node::replay_detail::DecodeHostReplayEvents(encoded);
  return !result.ok() && result.code == code;
}

} // namespace

int RunReplayHostRejectContract() {
  using rund::host::EventKind;

  const std::vector<rund::host::Event> events{
      rund::host::Event{.sequence = 1u,
                        .kind = rund::host::EventKind::TimerSleep,
                        .status = rund::host::Status::Ok,
                        .task_id = 7u,
                        .logical_time_ns = 11u}};
  const std::vector<std::byte> encoded =
      rund::node::EncodeHostReplayEvents(events);
  TEST_ASSERT(encoded.size() > 18u);

  TEST_ASSERT(rund::node::EncodeHostReplayEvents(
                  std::vector<rund::host::Event>{
                      rund::host::Event{.sequence = 2u,
                                        .kind = EventKind::TimerSleep,
                                        .status = rund::host::Status::Ok,
                                        .logical_time_ns = 2u},
                      rund::host::Event{.sequence = 1u,
                                        .kind = EventKind::TimerSleep,
                                        .status = rund::host::Status::Ok,
                                        .logical_time_ns = 3u}})
                  .empty());
  TEST_ASSERT(rund::node::EncodeHostReplayEvents(
                  std::vector<rund::host::Event>{
                      rund::host::Event{.sequence = 1u,
                                        .kind = EventKind::TimerSleep,
                                        .status = rund::host::Status::Ok,
                                        .logical_time_ns = 2u},
                      rund::host::Event{.sequence = 2u,
                                        .kind = EventKind::TimerSleep,
                                        .status = rund::host::Status::Ok,
                                        .logical_time_ns = 1u}})
                  .empty());

  std::vector<rund::host::Event> rejected_events{
      rund::host::Event{.sequence = 99u}};
  std::vector<std::byte> bad_magic = encoded;
  bad_magic.front() ^= std::byte{0xffu};
  TEST_ASSERT(!rund::node::DecodeHostReplayEvents(bad_magic, rejected_events));
  TEST_ASSERT(rejected_events.empty());

  for (const std::size_t cut : {std::size_t{0u}, std::size_t{7u},
                                encoded.size() / 2u, encoded.size() - 1u}) {
    TEST_ASSERT(!rund::node::DecodeHostReplayEvents(
        std::span<const std::byte>{encoded}.first(cut), rejected_events));
    TEST_ASSERT(rejected_events.empty());
  }

  // Host header (8) and count varuint (1) precede the fixed event hash.
  std::vector<std::byte> bad_hash = encoded;
  bad_hash[9] ^= std::byte{1u};
  TEST_ASSERT(!rund::node::DecodeHostReplayEvents(bad_hash, rejected_events));
  TEST_ASSERT(DecodeFailsWith(bad_hash, rund::replay::Code::HostHashInvalid));

  std::vector<std::byte> overlong_count = encoded;
  overlong_count[8] = std::byte{0x81u};
  overlong_count.insert(overlong_count.begin() + 9, std::byte{0u});
  TEST_ASSERT(
      DecodeFailsWith(overlong_count, rund::replay::Code::HostBadValue));

  const std::vector<std::byte> valid_encoded =
      rund::node::EncodeHostReplayEvents(
          std::vector<rund::host::Event>{ValidReplayEvent(EventKind::NetRecv)});
  std::vector<rund::host::Event> decoded{};
  TEST_ASSERT(rund::node::DecodeHostReplayEvents(valid_encoded, decoded));
  TEST_ASSERT(decoded.size() == 1u);

  rund::host::Event unsupported = ValidReplayEvent(EventKind::IoWrite);
  unsupported.status = rund::host::Status::Unsupported;
  const std::vector<std::byte> unsupported_encoded =
      rund::node::EncodeHostReplayEvents(
          std::vector<rund::host::Event>{unsupported});
  TEST_ASSERT(rund::node::DecodeHostReplayEvents(unsupported_encoded, decoded));
  TEST_ASSERT(decoded[0].status == rund::host::Status::Unsupported);

  rund::host::Event unknown_kind = ValidReplayEvent(static_cast<EventKind>(
      static_cast<std::uint16_t>(EventKind::NetSendVectored) + 1u));
  TEST_ASSERT(rund::node::EncodeHostReplayEvents(
                  std::vector<rund::host::Event>{unknown_kind})
                  .empty());

  rund::host::Event unknown_status = ValidReplayEvent(EventKind::NetRecv);
  unknown_status.status = static_cast<rund::host::Status>(7u);
  TEST_ASSERT(rund::node::EncodeHostReplayEvents(
                  std::vector<rund::host::Event>{unknown_status})
                  .empty());
  return 0;
}

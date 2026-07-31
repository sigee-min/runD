#include "local.hpp"
#include <rund/net/frame/io.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <span>

bool FrameIoSuccessfulWriteReadPreservesBytes() {
  FrameIoSocketPair pair{};
  FRAMEIO_ASSERT(OpenFrameIoNonblockingPair(pair));
  const std::array<std::byte, 5u> payload{std::byte{'f'}, std::byte{'r'},
                                          std::byte{'a'}, std::byte{'m'},
                                          std::byte{'e'}};
  std::array<std::byte, 5u> received{};
  rund::net::frame::WriteResult written{};
  rund::net::frame::ReadResult read{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(FrameIoRunSpec(), [&] {
    auto exchange = [&]() -> rund::task::Task<void> {
      const auto write = co_await rund::net::frame::write(
          pair.left.view(), std::span<const std::byte>{payload});
      if (write) {
        written = *write;
      }
      const auto receive = co_await rund::net::frame::read(
          pair.right.view(), std::span<std::byte>{received});
      if (receive) {
        read = *receive;
      }
    };
    const rund::task::Handle task =
        rund::task::spawn("net-frame-io-success", exchange());
    joined = rund::task::join(task);
  });

  FRAMEIO_ASSERT(report.ok());
  FRAMEIO_ASSERT(joined.ok());
  FRAMEIO_ASSERT(written);
  FRAMEIO_ASSERT(written.complete());
  FRAMEIO_ASSERT(written.bytes == payload.size());
  FRAMEIO_ASSERT(written.header_written);
  FRAMEIO_ASSERT(written.payload_written);
  FRAMEIO_ASSERT(written.header_bytes == 4u);
  FRAMEIO_ASSERT(written.payload_bytes == payload.size());
  const std::array<std::byte, 9u> framed{
      std::byte{0u},  std::byte{0u},  std::byte{0u},
      std::byte{5u},  std::byte{'f'}, std::byte{'r'},
      std::byte{'a'}, std::byte{'m'}, std::byte{'e'}};
  std::size_t send_events = 0u;
  for (const rund::host::Event &event : report.events()) {
    if (event.kind != rund::host::EventKind::NetSendVectored) {
      continue;
    }
    ++send_events;
    FRAMEIO_ASSERT(event.requested_bytes == framed.size());
    FRAMEIO_ASSERT(event.completed_bytes == framed.size());
    FRAMEIO_ASSERT(event.payload_hash.value ==
                   rund::host::hash_bytes(framed.data(), framed.size()).value);
  }
  FRAMEIO_ASSERT(send_events == 1u);
  FRAMEIO_ASSERT(report.tasks().network().vectored_send_calls() == 1u);
  FRAMEIO_ASSERT(report.tasks().network().send_calls() == 0u);
  FRAMEIO_ASSERT(report.tasks().network().bytes_sent() == framed.size());
  FRAMEIO_ASSERT(read);
  FRAMEIO_ASSERT(read.complete());
  FRAMEIO_ASSERT(read.bytes == payload.size());
  FRAMEIO_ASSERT(read.header_read);
  FRAMEIO_ASSERT(read.payload_read);
  FRAMEIO_ASSERT(read.header_bytes == 4u);
  FRAMEIO_ASSERT(read.payload_bytes == payload.size());
  FRAMEIO_ASSERT(received == payload);
  return true;
}

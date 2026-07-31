#include "local.hpp"
#include <rund/net/frame/io.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <span>

bool FrameIoZeroLengthPayloadSucceedsWithNoPayloadBytes() {
  FrameIoSocketPair pair{};
  FRAMEIO_ASSERT(OpenFrameIoNonblockingPair(pair));
  std::array<std::byte, 1u> payload{std::byte{'x'}};
  std::array<std::byte, 1u> received{std::byte{'y'}};
  rund::net::frame::WriteResult written{};
  rund::net::frame::ReadResult read{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run(FrameIoRunSpec(), [&] {
    auto exchange = [&]() -> rund::task::Task<void> {
      const auto write = co_await rund::net::frame::write(
          pair.left.view(), std::span<const std::byte>{payload.data(), 0u});
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
        rund::task::spawn("net-frame-io-empty", exchange());
    joined = rund::task::join(task);
  });

  FRAMEIO_ASSERT(report.ok());
  FRAMEIO_ASSERT(joined.ok());
  FRAMEIO_ASSERT(written);
  FRAMEIO_ASSERT(written.complete());
  FRAMEIO_ASSERT(written.bytes == 0u);
  FRAMEIO_ASSERT(written.header_written);
  FRAMEIO_ASSERT(written.payload_written);
  FRAMEIO_ASSERT(written.header_bytes == 4u);
  FRAMEIO_ASSERT(written.payload_bytes == 0u);
  FRAMEIO_ASSERT(read);
  FRAMEIO_ASSERT(read.complete());
  FRAMEIO_ASSERT(read.bytes == 0u);
  FRAMEIO_ASSERT(read.header_read);
  FRAMEIO_ASSERT(read.payload_read);
  FRAMEIO_ASSERT(read.header_bytes == 4u);
  FRAMEIO_ASSERT(read.payload_bytes == 0u);
  FRAMEIO_ASSERT(received[0] == std::byte{'y'});
  return true;
}

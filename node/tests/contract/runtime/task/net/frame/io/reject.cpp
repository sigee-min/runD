#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include <rund/net/frame/io.hpp>
#include <rund/net/frame/limit.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

bool FrameIoPayloadTooLargeRejectsBeforeWrite() {
  FrameIoSocketPair pair{};
  FRAMEIO_ASSERT(OpenFrameIoNonblockingPair(pair));
  const std::array<std::byte, 3u> payload{std::byte{'b'}, std::byte{'i'},
                                          std::byte{'g'}};
  rund::net::frame::WriteResult written{};
  rund::task::Status joined{};

  const rund::Session::Result report = rund::run([&] {
    auto reject = [&]() -> rund::task::Task<void> {
      const auto result = co_await rund::net::frame::write(
          pair.left.view(), std::span<const std::byte>{payload},
          rund::net::frame::IoLimit{
              .frame = rund::net::frame::Limit{.max_bytes = 2u}});
      if (result) {
        written = *result;
      }
    };
    const rund::task::Handle task =
        rund::task::spawn("net-frame-io-too-large", reject());
    joined = rund::task::join(task);
  });

  FRAMEIO_ASSERT(report.ok());
  FRAMEIO_ASSERT(joined.ok());
  FRAMEIO_ASSERT(!written);
  FRAMEIO_ASSERT(written.code() == rund::ReasonCode::NetFrameTooLarge);
  FRAMEIO_ASSERT(std::string_view{written.error()} == "net_frame_too_large");
  FRAMEIO_ASSERT(written.bytes == 0u);
  FRAMEIO_ASSERT(written.header_bytes == 0u);
  FRAMEIO_ASSERT(written.payload_bytes == 0u);

  auto ticket = rund::node::test::net::ticket(
      pair.left.view(), rund::net::ready::Interest::Writable);
  const rund::net::frame::WriteResult direct = rund::net::frame::write(
      std::move(ticket), payload,
      rund::net::frame::IoLimit{.frame =
                                    rund::net::frame::Limit{.max_bytes = 2u}});
  FRAMEIO_ASSERT(direct.code() == rund::ReasonCode::NetFrameTooLarge);
  FRAMEIO_ASSERT(ticket.consumed());
  FRAMEIO_ASSERT(rund::net::frame::write(std::move(ticket), payload).code() ==
                 rund::ReasonCode::NetTicketConsumed);
  return true;
}

bool FrameIoVectoredCapacityRejectsDirectly() {
  FrameIoSocketPair pair{};
  FRAMEIO_ASSERT(OpenFrameIoNonblockingPair(pair));
  const std::array<std::byte, 3u> payload{std::byte{'i'}, std::byte{'o'},
                                          std::byte{'v'}};
  rund::net::frame::WriteResult written{};
  rund::task::Status joined{};
  rund::SessionConfig spec = FrameIoRunSpec();
  spec.scheduler.net_iov_capacity = 1u;

  const rund::Session::Result report = rund::run(spec, [&] {
    auto reject = [&]() -> rund::task::Task<void> {
      const auto result = co_await rund::net::frame::write(
          pair.left.view(), std::span<const std::byte>{payload});
      if (result) {
        written = *result;
      }
    };
    const rund::task::Handle task =
        rund::task::spawn("net-frame-io-iov-capacity", reject());
    joined = rund::task::join(task);
  });

  FRAMEIO_ASSERT(report.ok());
  FRAMEIO_ASSERT(joined.ok());
  FRAMEIO_ASSERT(!written);
  FRAMEIO_ASSERT(written.code() == rund::ReasonCode::TaskInvalid);
  FRAMEIO_ASSERT(written.header_bytes == 0u);
  FRAMEIO_ASSERT(written.payload_bytes == 0u);
  for (const rund::host::Event &event : report.events()) {
    FRAMEIO_ASSERT(event.kind != rund::host::EventKind::NetSend);
    FRAMEIO_ASSERT(event.kind != rund::host::EventKind::NetSendVectored);
  }
  FRAMEIO_ASSERT(report.tasks().network().send_calls() == 0u);
  FRAMEIO_ASSERT(report.tasks().network().vectored_send_calls() == 0u);
  return true;
}

bool FrameIoDestinationBufferTooSmallRejectsAfterHeader() {
  FrameIoSocketPair pair{};
  FRAMEIO_ASSERT(OpenFrameIoNonblockingPair(pair));
  const std::array<std::byte, 3u> payload{std::byte{'s'}, std::byte{'m'},
                                          std::byte{'l'}};
  std::array<std::byte, 2u> received{};
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
        rund::task::spawn("net-frame-io-small-buffer", exchange());
    joined = rund::task::join(task);
  });

  FRAMEIO_ASSERT(report.ok());
  FRAMEIO_ASSERT(joined.ok());
  FRAMEIO_ASSERT(written);
  FRAMEIO_ASSERT(!read);
  FRAMEIO_ASSERT(read.code() == rund::ReasonCode::NetFrameBufferTooSmall);
  FRAMEIO_ASSERT(std::string_view{read.error()} ==
                 "net_frame_buffer_too_small");
  FRAMEIO_ASSERT(read.header_read);
  FRAMEIO_ASSERT(!read.payload_read);
  FRAMEIO_ASSERT(read.header_bytes == 4u);
  FRAMEIO_ASSERT(read.payload_bytes == 0u);
  FRAMEIO_ASSERT(read.bytes == 0u);
  return true;
}

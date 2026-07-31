#include "test/assert.hpp"

#include "../../coroutine/allocation.hpp"
#include "local.hpp"

#include <rund/host/io.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <span>

#include <fcntl.h>
#include <unistd.h>

namespace runtime_task_host_io {
namespace {

void VerifyDiscard() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  Fd read_cleanup{pipe_fds[0]};
  Fd write_cleanup{pipe_fds[1]};
  const std::array<std::byte, 1u> source{std::byte{0x51}};
  TEST_ASSERT(::write(write_cleanup.native, source.data(), source.size()) == 1);

  std::array<std::byte, 1u> buffer{};
  rund::task::Status joined{};
  rund::host::io::Fd fd =
      rund::host::io::take_native_fd(::dup(read_cleanup.native));
  TEST_ASSERT(fd);
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_io_capacity = 1u,
                  .host_event_capacity = 1u,
                  .host_payload_capacity_bytes = 1u,
              },
      },
      [&] {
        auto body = [&]() -> rund::task::Task<void> {
          auto discarded = rund::host::io::read_some(
              fd.view(), std::span<std::byte>{buffer});
          static_cast<void>(discarded);
          co_return;
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-discarded-token", body());
        joined = rund::task::join(task);
      });
  TEST_ASSERT(report);
  TEST_ASSERT(joined);
  TEST_ASSERT(report.events().empty());
  TEST_ASSERT(report.tasks().external_parks() == 0u);
  TEST_ASSERT(report.tasks().external_wakes() == 0u);
  TEST_ASSERT(buffer[0] == std::byte{0});
  std::array<std::byte, 1u> remaining{};
  TEST_ASSERT(
      ::read(read_cleanup.native, remaining.data(), remaining.size()) == 1);
  TEST_ASSERT(remaining == source);
}

void VerifyPayloadLimit() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  Fd read_cleanup{pipe_fds[0]};
  Fd write_cleanup{pipe_fds[1]};
  const std::array<std::byte, 1u> source{std::byte{0x42}};
  TEST_ASSERT(::write(write_cleanup.native, source.data(), source.size()) == 1);

  std::array<std::byte, 1u> buffer{};
  rund::host::io::ReadResult read{};
  rund::task::Status joined{};
  rund::host::io::Fd fd =
      rund::host::io::take_native_fd(::dup(read_cleanup.native));
  TEST_ASSERT(fd);
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_event_capacity = 2u,
                  .host_payload_capacity_bytes = 0u,
              },
      },
      [&] {
        auto body = [&]() -> rund::task::Task<void> {
          read = co_await rund::host::io::read_some(
              fd.view(), std::span<std::byte>{buffer});
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-capacity-preflight", body());
        joined = rund::task::join(task);
      });
  TEST_ASSERT(report);
  TEST_ASSERT(!joined);
  TEST_ASSERT(joined.code() == rund::ReasonCode::TaskCapacityExceeded);
  TEST_ASSERT(!read);
  TEST_ASSERT(read.code() == rund::ReasonCode::TaskCapacityExceeded);
  TEST_ASSERT(buffer[0] == std::byte{0});
  std::array<std::byte, 1u> remaining{};
  TEST_ASSERT(
      ::read(read_cleanup.native, remaining.data(), remaining.size()) == 1);
  TEST_ASSERT(remaining == source);
}

void VerifyQueueLimit() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  Fd read_cleanup{pipe_fds[0]};
  Fd write_cleanup{pipe_fds[1]};
  const int flags = ::fcntl(read_cleanup.native, F_GETFL, 0);
  TEST_ASSERT(flags >= 0);
  TEST_ASSERT(
      ::fcntl(read_cleanup.native, F_SETFL, flags | O_NONBLOCK) == 0);

  const std::array<std::byte, 1u> payload{std::byte{0x24}};
  rund::host::io::WriteResult write{};
  rund::task::Status joined{};
  rund::host::io::Fd fd =
      rund::host::io::take_native_fd(write_cleanup.native);
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_io_capacity = 0u,
                  .host_event_capacity = 2u,
                  .host_payload_capacity_bytes = 1u,
              },
      },
      [&] {
        auto body = [&]() -> rund::task::Task<void> {
          write = co_await rund::host::io::write_some(
              fd.view(), std::span<const std::byte>{payload});
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-queue-capacity", body());
        joined = rund::task::join(task);
      });
  TEST_ASSERT(report);
  TEST_ASSERT(!joined);
  TEST_ASSERT(joined.code() == rund::ReasonCode::TaskCapacityExceeded);
  TEST_ASSERT(!write);
  TEST_ASSERT(write.code() == rund::ReasonCode::TaskCapacityExceeded);
  std::array<std::byte, 1u> probe{};
  errno = 0;
  TEST_ASSERT(::read(read_cleanup.native, probe.data(), probe.size()) == -1);
  TEST_ASSERT(errno == EAGAIN || errno == EWOULDBLOCK);
}

void VerifyAllocation() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  Fd read_cleanup{pipe_fds[0]};
  Fd write_cleanup{pipe_fds[1]};
  const int flags = ::fcntl(read_cleanup.native, F_GETFL, 0);
  TEST_ASSERT(flags >= 0);
  TEST_ASSERT(
      ::fcntl(read_cleanup.native, F_SETFL, flags | O_NONBLOCK) == 0);

  rund::host::io::Fd fd =
      rund::host::io::take_native_fd(read_cleanup.native);
  std::array<std::byte, 1u> buffer{};
  rund::host::io::ReadResult warm{};
  rund::host::io::ReadResult measured{};
  std::uint64_t allocations = 0u;
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_io_capacity = 1u,
                  .host_event_capacity = 2u,
                  .host_payload_capacity_bytes = 1u,
              },
      },
      [&] {
        auto body = [&]() -> rund::task::Task<void> {
          warm = co_await rund::host::io::read_some(
              fd.view(), std::span<std::byte>{buffer});
          runtime_task_allocation::Start();
          measured = co_await rund::host::io::read_some(
              fd.view(), std::span<std::byte>{buffer});
          runtime_task_allocation::Stop();
          allocations = runtime_task_allocation::Count();
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-warm-allocation", body());
        joined = rund::task::join(task);
      });

  runtime_task_allocation::Stop();
  TEST_ASSERT(report);
  TEST_ASSERT(joined);
  TEST_ASSERT(!warm);
  TEST_ASSERT(warm.code() == rund::ReasonCode::IoWouldBlock);
  TEST_ASSERT(!measured);
  TEST_ASSERT(measured.code() == rund::ReasonCode::IoWouldBlock);
  TEST_ASSERT(allocations == 0u);
  TEST_ASSERT(report.events().size() == 2u);
}

} // namespace

void Admission() {
  VerifyDiscard();
  VerifyPayloadLimit();
  VerifyQueueLimit();
  VerifyAllocation();
}

} // namespace runtime_task_host_io

#include "test/assert.hpp"

#include "local.hpp"

#include <rund/host.hpp>
#include <rund/host/io.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <unistd.h>

namespace runtime_task_host_io {
namespace {

rund::task::Task<void> WriteOne(const rund::host::io::FdView fd,
                                const std::byte *const byte,
                                rund::host::io::WriteResult *const result) {
  *result = co_await rund::host::io::write_some(
      fd, std::span<const std::byte>{byte, 1u});
}

void VerifyReuse() {
  constexpr std::size_t kIterations = 32u;
  int source_pipe[2] = {-1, -1};
  int sink_pipe[2] = {-1, -1};
  TEST_ASSERT(::pipe(source_pipe) == 0);
  TEST_ASSERT(::pipe(sink_pipe) == 0);
  Fd source_read{source_pipe[0]};
  Fd source_write{source_pipe[1]};
  Fd sink_read{sink_pipe[0]};
  Fd sink_write{sink_pipe[1]};

  std::array<std::byte, kIterations> source{};
  for (std::size_t index = 0u; index < source.size(); ++index) {
    source[index] = static_cast<std::byte>(index + 1u);
  }
  TEST_ASSERT(::write(source_write.native, source.data(), source.size()) ==
              static_cast<ssize_t>(source.size()));
  rund::host::io::Fd read_fd =
      rund::host::io::take_native_fd(source_read.native);
  rund::host::io::Fd write_fd =
      rund::host::io::take_native_fd(sink_write.native);
  bool operations_ok = true;
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 2u,
          .scheduler =
              {
                  .task_workers = 2u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_io_capacity = 1u,
                  .host_event_capacity =
                      static_cast<std::uint32_t>(kIterations * 2u),
                  .host_payload_capacity_bytes = kIterations * 2u,
              },
      },
      [&] {
        auto body = [&]() -> rund::task::Task<void> {
          std::array<std::byte, 1u> byte{};
          for (std::size_t index = 0u; index < kIterations; ++index) {
            const rund::host::io::ReadResult read =
                co_await rund::host::io::read_some(
                    read_fd.view(), std::span<std::byte>{byte});
            operations_ok = operations_ok && read && read.bytes == 1 &&
                            byte[0] == source[index];
            const rund::host::io::WriteResult write =
                co_await rund::host::io::write_some(
                    write_fd.view(), std::span<const std::byte>{byte});
            operations_ok = operations_ok && write && write.bytes == 1;
          }
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-single-slot-reuse", body());
        joined = rund::task::join(task);
      });
  TEST_ASSERT(report);
  TEST_ASSERT(joined);
  TEST_ASSERT(operations_ok);
  TEST_ASSERT(report.events().size() == kIterations * 2u);
  TEST_ASSERT(report.tasks().external_parks() == kIterations * 2u);
  TEST_ASSERT(report.tasks().external_wakes() == kIterations * 2u);
  for (std::size_t index = 0u; index < report.events().size(); ++index) {
    TEST_ASSERT(report.events()[index].sequence == index + 1u);
    TEST_ASSERT(report.events()[index].kind ==
                (index % 2u == 0u ? rund::host::EventKind::IoRead
                                  : rund::host::EventKind::IoWrite));
  }
  std::array<std::byte, kIterations> sink{};
  TEST_ASSERT(::read(sink_read.native, sink.data(), sink.size()) ==
              static_cast<ssize_t>(sink.size()));
  TEST_ASSERT(sink == source);
}

void VerifyFifo() {
  constexpr std::size_t kOperations = 8u;
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  Fd read_cleanup{pipe_fds[0]};
  Fd write_cleanup{pipe_fds[1]};
  rund::host::io::Fd fd =
      rund::host::io::take_native_fd(write_cleanup.native);

  std::array<std::byte, kOperations> payload{};
  for (std::size_t index = 0u; index < payload.size(); ++index) {
    payload[index] = static_cast<std::byte>(0x60u + index);
  }
  std::array<rund::host::io::WriteResult, kOperations> writes{};
  std::array<rund::task::Handle, kOperations> handles{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 4u,
          .scheduler =
              {
                  .task_workers = 4u,
                  .task_capacity = 16u,
                  .ready_queue_capacity = 16u,
                  .host_io_capacity = static_cast<std::uint32_t>(kOperations),
                  .host_event_capacity =
                      static_cast<std::uint32_t>(kOperations),
                  .host_payload_capacity_bytes = kOperations,
              },
      },
      [&] {
        for (std::size_t index = 0u; index < handles.size(); ++index) {
          handles[index] =
              rund::task::spawn("hostio-ordered-write",
                                WriteOne(fd.view(), payload.data() + index,
                                         writes.data() + index));
        }
        joined =
            rund::task::join_all(std::span<const rund::task::Handle>{handles});
      });
  TEST_ASSERT(report);
  TEST_ASSERT(joined);
  TEST_ASSERT(report.events().size() == kOperations);
  TEST_ASSERT(report.tasks().external_parks() == kOperations);
  TEST_ASSERT(report.tasks().external_wakes() == kOperations);
  TEST_ASSERT(report.tasks().global_ready_queue_pushes() == kOperations);
  TEST_ASSERT(report.tasks().global_ready_queue_pops() == kOperations);
  for (std::size_t index = 0u; index < writes.size(); ++index) {
    TEST_ASSERT(writes[index]);
    TEST_ASSERT(writes[index].bytes == 1);
    TEST_ASSERT(report.events()[index].sequence == index + 1u);
    TEST_ASSERT(report.events()[index].kind == rund::host::EventKind::IoWrite);
    TEST_ASSERT(report.events()[index].task_id == index + 1u);
  }
  std::array<std::byte, kOperations> sink{};
  TEST_ASSERT(::read(read_cleanup.native, sink.data(), sink.size()) ==
              static_cast<ssize_t>(sink.size()));
  TEST_ASSERT(sink == payload);
}

} // namespace

void Order() {
  VerifyReuse();
  VerifyFifo();
}

} // namespace runtime_task_host_io

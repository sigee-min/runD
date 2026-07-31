#include "test/assert.hpp"

#include "local.hpp"

#include <rund/host.hpp>
#include <rund/host/io.hpp>
#include <rund/replay.hpp>
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

void VerifyPayload() {
  int source_pipe[2] = {-1, -1};
  int sink_pipe[2] = {-1, -1};
  TEST_ASSERT(::pipe(source_pipe) == 0);
  TEST_ASSERT(::pipe(sink_pipe) == 0);
  Fd source_read{source_pipe[0]};
  Fd source_write{source_pipe[1]};
  Fd sink_read{sink_pipe[0]};
  Fd sink_write{sink_pipe[1]};

  const std::array<std::byte, 4u> source{std::byte{'r'}, std::byte{'u'},
                                         std::byte{'n'}, std::byte{'D'}};
  const std::array<std::byte, 3u> written{std::byte{'i'}, std::byte{'o'},
                                          std::byte{'!'}};
  TEST_ASSERT(::write(source_write.native, source.data(), source.size()) ==
              static_cast<ssize_t>(source.size()));

  rund::host::io::Fd record_read_fd =
      rund::host::io::take_native_fd(source_read.native);
  rund::host::io::Fd record_write_fd =
      rund::host::io::take_native_fd(sink_write.native);
  std::array<std::byte, 4u> recorded_bytes{};
  rund::host::io::ReadResult recorded_read{};
  rund::host::io::WriteResult recorded_write{};
  rund::task::Status recorded_join{};
  const rund::SessionConfig config{
      .workers = 2u,
      .scheduler =
          {
              .task_workers = 2u,
              .task_capacity = 4u,
              .ready_queue_capacity = 4u,
              .net_socket_registry_capacity = 0u,
              .host_handle_capacity = 2u,
              .host_event_capacity = 4u,
              .host_payload_capacity_bytes = source.size() + written.size(),
          },
  };
  rund::Session session{};
  TEST_ASSERT(session.open(config));
  const rund::replay::Record recorded =
      rund::replay::record(session, [&](rund::replay::Context &) {
        auto body = [&]() -> rund::task::Task<void> {
          recorded_read = co_await rund::host::io::read_some(
              record_read_fd.view(), std::span<std::byte>{recorded_bytes});
          recorded_write = co_await rund::host::io::write_some(
              record_write_fd.view(), std::span<const std::byte>{written});
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-record-replay", body());
        recorded_join = rund::task::join(task);
      });
  TEST_ASSERT(recorded);
  TEST_ASSERT(recorded_join);
  TEST_ASSERT(recorded_read);
  TEST_ASSERT(recorded_read.bytes == static_cast<std::int64_t>(source.size()));
  TEST_ASSERT(recorded_write);
  TEST_ASSERT(recorded_write.bytes ==
              static_cast<std::int64_t>(written.size()));
  TEST_ASSERT(recorded_bytes == source);
  TEST_ASSERT(recorded.host_event_count() == 2u);
  TEST_ASSERT(recorded.storage_report().logical_bytes ==
              source.size() + written.size());
  TEST_ASSERT(recorded.tasks().external_parks() == 2u);
  TEST_ASSERT(recorded.tasks().external_wakes() == 2u);

  std::array<std::byte, 3u> sink_bytes{};
  TEST_ASSERT(::read(sink_read.native, sink_bytes.data(), sink_bytes.size()) ==
              static_cast<ssize_t>(sink_bytes.size()));
  TEST_ASSERT(sink_bytes == written);

  std::array<std::byte, 4u> replayed_bytes{};
  rund::host::io::ReadResult replayed_read{};
  rund::host::io::WriteResult replayed_write{};
  rund::task::Status replayed_join{};
  constexpr std::uint64_t replay_read_handle = 101u;
  constexpr std::uint64_t replay_write_handle = 202u;
  rund::host::io::Fd replay_read_fd =
      rund::host::io::replay_fd(replay_read_handle);
  rund::host::io::Fd replay_write_fd =
      rund::host::io::replay_fd(replay_write_handle);
  TEST_ASSERT(replay_read_fd.id() == replay_read_handle);
  TEST_ASSERT(replay_write_fd.id() == replay_write_handle);
  const rund::replay::Check replayed =
      rund::replay::run(session, recorded, [&](rund::replay::Context &) {
        auto body = [&]() -> rund::task::Task<void> {
          replayed_read = co_await rund::host::io::read_some(
              replay_read_fd.view(), std::span<std::byte>{replayed_bytes});
          replayed_write = co_await rund::host::io::write_some(
              replay_write_fd.view(), std::span<const std::byte>{written});
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-record-replay", body());
        replayed_join = rund::task::join(task);
      });
  TEST_ASSERT(replayed);
  TEST_ASSERT(replayed_join);
  TEST_ASSERT(replayed_read);
  TEST_ASSERT(replayed_write);
  TEST_ASSERT(replayed_bytes == source);
  TEST_ASSERT(replayed.actual().has_value());
  TEST_ASSERT(replayed.actual()->host_event_hash() ==
              recorded.host_event_hash());
  TEST_ASSERT(replayed.actual()->tasks().external_parks() == 0u);
  TEST_ASSERT(replayed.actual()->tasks().external_wakes() == 0u);

  std::array<std::byte, 4u> mismatch_read_bytes{};
  const std::array<std::byte, 3u> wrong_write{std::byte{'n'}, std::byte{'o'},
                                              std::byte{'!'}};
  rund::host::io::WriteResult mismatch_write{};
  rund::task::Status mismatch_join{};
  rund::host::io::Fd mismatch_read_fd =
      rund::host::io::replay_fd(replay_read_handle);
  rund::host::io::Fd mismatch_write_fd =
      rund::host::io::replay_fd(replay_write_handle);
  const rund::replay::Check mismatch =
      rund::replay::run(session, recorded, [&](rund::replay::Context &) {
        auto body = [&]() -> rund::task::Task<void> {
          const rund::host::io::ReadResult read =
              co_await rund::host::io::read_some(
                  mismatch_read_fd.view(),
                  std::span<std::byte>{mismatch_read_bytes});
          TEST_ASSERT(read);
          mismatch_write = co_await rund::host::io::write_some(
              mismatch_write_fd.view(),
              std::span<const std::byte>{wrong_write});
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-record-replay", body());
        mismatch_join = rund::task::join(task);
      });
  TEST_ASSERT(!mismatch);
  TEST_ASSERT(!mismatch_join);
  TEST_ASSERT(mismatch_join.code() ==
              rund::ReasonCode::HostReplayPayloadMismatch);
  TEST_ASSERT(!mismatch_write);
  TEST_ASSERT(mismatch_write.code() ==
              rund::ReasonCode::HostReplayPayloadMismatch);
  TEST_ASSERT(session.close());
}

void VerifyWouldBlock() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  Fd read_cleanup{pipe_fds[0]};
  Fd write_cleanup{pipe_fds[1]};
  const int flags = ::fcntl(read_cleanup.native, F_GETFL, 0);
  TEST_ASSERT(flags >= 0);
  TEST_ASSERT(
      ::fcntl(read_cleanup.native, F_SETFL, flags | O_NONBLOCK) == 0);

  rund::host::io::Fd record_fd =
      rund::host::io::take_native_fd(read_cleanup.native);
  const rund::SessionConfig config{
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
  };
  std::array<std::byte, 1u> record_buffer{};
  rund::host::io::ReadResult record_read{};
  rund::task::Status record_join{};
  rund::Session session{};
  TEST_ASSERT(session.open(config));
  const rund::replay::Record recorded =
      rund::replay::record(session, [&](rund::replay::Context &) {
        auto body = [&]() -> rund::task::Task<void> {
          record_read = co_await rund::host::io::read_some(
              record_fd.view(), std::span<std::byte>{record_buffer});
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-would-block", body());
        record_join = rund::task::join(task);
      });
  TEST_ASSERT(recorded);
  TEST_ASSERT(record_join);
  TEST_ASSERT(!record_read);
  TEST_ASSERT(record_read.code() == rund::ReasonCode::IoWouldBlock);
  TEST_ASSERT(record_read.bytes == -1);
  TEST_ASSERT(record_read.native_error == EAGAIN ||
              record_read.native_error == EWOULDBLOCK);
  TEST_ASSERT(recorded.host_event_count() == 1u);
  TEST_ASSERT(recorded.storage_report().logical_bytes == 0u);
  TEST_ASSERT(recorded.tasks().external_parks() == 1u);
  TEST_ASSERT(recorded.tasks().external_wakes() == 1u);

  std::array<std::byte, 1u> replay_buffer{};
  rund::host::io::ReadResult replay_read{};
  rund::task::Status replay_join{};
  rund::host::io::Fd replay_fd = rund::host::io::replay_fd(303u);
  const rund::replay::Check replayed =
      rund::replay::run(session, recorded, [&](rund::replay::Context &) {
        auto body = [&]() -> rund::task::Task<void> {
          replay_read = co_await rund::host::io::read_some(
              replay_fd.view(), std::span<std::byte>{replay_buffer});
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-would-block", body());
        replay_join = rund::task::join(task);
      });
  TEST_ASSERT(replayed);
  TEST_ASSERT(replay_join);
  TEST_ASSERT(!replay_read);
  TEST_ASSERT(replay_read.code() == rund::ReasonCode::IoWouldBlock);
  TEST_ASSERT(replay_read.bytes == -1);
  TEST_ASSERT(replay_read.native_error == record_read.native_error);
  TEST_ASSERT(replayed.actual().has_value());
  TEST_ASSERT(replayed.actual()->tasks().external_parks() == 0u);
  TEST_ASSERT(replayed.actual()->tasks().external_wakes() == 0u);
  TEST_ASSERT(session.close());
}

} // namespace

void Replay() {
  VerifyPayload();
  VerifyWouldBlock();
}

} // namespace runtime_task_host_io

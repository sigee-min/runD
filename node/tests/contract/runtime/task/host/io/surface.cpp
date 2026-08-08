#include "test/assert.hpp"

#include "../../../../../../src/runtime/task/scheduler/host/io/local.hpp"
#include "local.hpp"

#include <rund/host/io.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace runtime_task_host_io {
namespace {

static_assert(std::is_nothrow_move_constructible_v<rund::host::io::ReadOp>);
static_assert(!std::is_copy_constructible_v<rund::host::io::ReadOp>);
static_assert(!std::is_copy_assignable_v<rund::host::io::ReadOp>);
static_assert(std::is_nothrow_move_constructible_v<rund::host::io::WriteOp>);
static_assert(!std::is_copy_constructible_v<rund::host::io::WriteOp>);
static_assert(!std::is_copy_assignable_v<rund::host::io::WriteOp>);
static_assert(!std::is_copy_constructible_v<rund::host::io::Fd>);
static_assert(!std::is_copy_assignable_v<rund::host::io::Fd>);
static_assert(std::is_nothrow_move_constructible_v<rund::host::io::Fd>);
static_assert(std::is_nothrow_move_assignable_v<rund::host::io::Fd>);
static_assert(std::is_trivially_copyable_v<rund::host::io::FdView>);

using rund::node::HostIoOutcome;
using rund::node::HostIoOutcomeDisposition;

static_assert(!std::is_aggregate_v<HostIoOutcome>);
static_assert(!std::is_default_constructible_v<HostIoOutcome>);
static_assert(std::is_trivially_copyable_v<HostIoOutcome>);
static_assert(sizeof(HostIoOutcome) == 16u);

constexpr HostIoOutcome kPendingOutcome = HostIoOutcome::pending();
static_assert(kPendingOutcome.disposition() ==
              HostIoOutcomeDisposition::Pending);
static_assert(kPendingOutcome.value() == -1);
static_assert(kPendingOutcome.native_error() == 0);

constexpr HostIoOutcome kCompleteOutcome = HostIoOutcome::complete(7);
static_assert(kCompleteOutcome.disposition() ==
              HostIoOutcomeDisposition::Complete);
static_assert(kCompleteOutcome.value() == 7);
static_assert(kCompleteOutcome.native_error() == 0);

constexpr HostIoOutcome kFailedOutcome = HostIoOutcome::failed(EAGAIN);
static_assert(kFailedOutcome.disposition() == HostIoOutcomeDisposition::Failed);
static_assert(kFailedOutcome.value() == -1);
static_assert(kFailedOutcome.native_error() == EAGAIN);

constexpr HostIoOutcome kInvalidOutcome = HostIoOutcome::invalid_buffer(EINVAL);
static_assert(kInvalidOutcome.disposition() ==
              HostIoOutcomeDisposition::InvalidBuffer);
static_assert(kInvalidOutcome.value() == -1);
static_assert(kInvalidOutcome.native_error() == EINVAL);

constexpr HostIoOutcome kUnsupportedOutcome = HostIoOutcome::unsupported();
static_assert(kUnsupportedOutcome.disposition() ==
              HostIoOutcomeDisposition::Unsupported);
static_assert(kUnsupportedOutcome.value() == -1);
static_assert(kUnsupportedOutcome.native_error() == 0);

template <class T>
concept RvalueView = requires(T value) { std::move(value).view(); };

static_assert(!RvalueView<rund::host::io::Fd>);

template <class T>
concept MutableCode = requires(T value) { value.code = rund::ReasonCode::Ok; };

static_assert(!MutableCode<rund::host::io::ReadResult>);
static_assert(!MutableCode<rund::host::io::WriteResult>);
static_assert(!MutableCode<rund::host::io::OpenResult>);
static_assert(!MutableCode<rund::host::io::CloseResult>);
static_assert(
    !std::is_constructible_v<rund::host::io::ReadResult, rund::ReasonCode>);
static_assert(
    !std::is_constructible_v<rund::host::io::WriteResult, rund::ReasonCode>);
static_assert(
    !std::is_constructible_v<rund::host::io::OpenResult, rund::ReasonCode>);
static_assert(
    !std::is_constructible_v<rund::host::io::CloseResult, rund::ReasonCode>);
static_assert(std::is_trivially_copyable_v<rund::host::io::ReadResult>);
static_assert(std::is_trivially_copyable_v<rund::host::io::WriteResult>);
static_assert(!std::is_copy_constructible_v<rund::host::io::OpenResult>);
static_assert(std::is_nothrow_move_constructible_v<rund::host::io::OpenResult>);
static_assert(std::is_trivially_copyable_v<rund::host::io::CloseResult>);
static_assert(sizeof(rund::host::io::ReadResult) == 24u);
static_assert(sizeof(rund::host::io::WriteResult) == 24u);
static_assert(sizeof(rund::host::io::OpenResult) == 32u);
static_assert(sizeof(rund::host::io::CloseResult) == 8u);
static_assert(std::is_same_v<decltype(rund::host::io::ReadResult{}.code()),
                             rund::ReasonCode>);

template <class T>
concept ReadinessReadable =
    requires(T value) { rund::host::io::readable(value); };

template <class T>
concept ReadinessWritable =
    requires(T value) { rund::host::io::writable(value); };

static_assert(!ReadinessReadable<rund::host::io::Fd>);
static_assert(!ReadinessWritable<rund::host::io::Fd>);
static_assert(ReadinessReadable<rund::host::io::FdView>);
static_assert(ReadinessWritable<rund::host::io::FdView>);
static_assert(!ReadinessReadable<int>);
static_assert(!ReadinessWritable<int>);

void VerifyResult() {
  const rund::host::io::ReadResult read{};
  const rund::host::io::WriteResult write{};
  const rund::host::io::OpenResult open{};
  const rund::host::io::CloseResult close{};
  TEST_ASSERT(!read && !write && !open && !close);
  TEST_ASSERT(read.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(write.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(open.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(close.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(read.exit_code() == 1 && write.exit_code() == 1 &&
              open.exit_code() == 1 && close.exit_code() == 1);
  TEST_ASSERT(
      read.error() == "task_invalid" && write.error() == "task_invalid" &&
      open.error() == "task_invalid" && close.error() == "task_invalid");
}

void VerifyInternalOutcomeProjection() {
  TEST_ASSERT(rund::node::host_io::OutcomeCode(HostIoOutcome::pending()) ==
              rund::ReasonCode::IoSyscallFailed);
  TEST_ASSERT(rund::node::host_io::OutcomeCode(HostIoOutcome::complete(7)) ==
              rund::ReasonCode::Ok);
  TEST_ASSERT(rund::node::host_io::OutcomeCode(HostIoOutcome::failed(EAGAIN)) ==
              rund::ReasonCode::IoWouldBlock);
  TEST_ASSERT(rund::node::host_io::OutcomeCode(HostIoOutcome::failed(EIO)) ==
              rund::ReasonCode::IoSyscallFailed);
  TEST_ASSERT(rund::node::host_io::OutcomeCode(HostIoOutcome::invalid_buffer(
                  EINVAL)) == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(rund::node::host_io::OutcomeCode(HostIoOutcome::unsupported()) ==
              rund::ReasonCode::IoUnsupported);
}

void VerifyOwner() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  Fd read_cleanup{pipe_fds[0]};
  Fd write_cleanup{pipe_fds[1]};

  int moved_native = ::dup(read_cleanup.native);
  TEST_ASSERT(moved_native >= 0);
  const int moved_probe = moved_native;
  {
    rund::host::io::Fd owner = rund::host::io::take_native_fd(moved_native);
    TEST_ASSERT(moved_native == -1);
    TEST_ASSERT(owner);
    const rund::host::io::FdView borrowed = owner.view();
    rund::host::io::Fd moved = std::move(owner);
    TEST_ASSERT(!owner);
    TEST_ASSERT(moved);
    TEST_ASSERT(borrowed == moved.view());
    TEST_ASSERT(::fcntl(moved_probe, F_GETFD) >= 0);
  }
  errno = 0;
  TEST_ASSERT(::fcntl(moved_probe, F_GETFD) == -1);
  TEST_ASSERT(errno == EBADF);

  int replaced_native = ::dup(read_cleanup.native);
  int replacement_native = ::dup(read_cleanup.native);
  TEST_ASSERT(replaced_native >= 0);
  TEST_ASSERT(replacement_native >= 0);
  const int replaced_probe = replaced_native;
  const int replacement_probe = replacement_native;
  {
    rund::host::io::Fd target = rund::host::io::take_native_fd(replaced_native);
    rund::host::io::Fd replacement =
        rund::host::io::take_native_fd(replacement_native);
    TEST_ASSERT(replaced_native == -1);
    TEST_ASSERT(replacement_native == -1);
    target = std::move(replacement);
    TEST_ASSERT(!replacement);
    TEST_ASSERT(target.id() ==
                static_cast<std::uint64_t>(replacement_probe) + 1u);
    errno = 0;
    TEST_ASSERT(::fcntl(replaced_probe, F_GETFD) == -1);
    TEST_ASSERT(errno == EBADF);
    TEST_ASSERT(::fcntl(replacement_probe, F_GETFD) >= 0);
  }
  errno = 0;
  TEST_ASSERT(::fcntl(replacement_probe, F_GETFD) == -1);
  TEST_ASSERT(errno == EBADF);

  int invalid_native = -1;
  TEST_ASSERT(!rund::host::io::take_native_fd(invalid_native));
  TEST_ASSERT(invalid_native == -1);
}

void VerifyBlocking() {
  int pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(pipe_fds) == 0);
  Fd read_cleanup{pipe_fds[0]};
  Fd write_cleanup{pipe_fds[1]};
  const std::uint64_t read_id =
      static_cast<std::uint64_t>(read_cleanup.native) + 1u;
  const std::uint64_t write_id =
      static_cast<std::uint64_t>(write_cleanup.native) + 1u;

  rund::host::io::Fd read_fd =
      rund::host::io::take_native_fd(read_cleanup.native);
  rund::host::io::Fd write_fd =
      rund::host::io::take_native_fd(write_cleanup.native);
  TEST_ASSERT(read_fd);
  TEST_ASSERT(write_fd);
  TEST_ASSERT(read_cleanup.native == -1);
  TEST_ASSERT(write_cleanup.native == -1);
  TEST_ASSERT(read_fd.id() == read_id);
  TEST_ASSERT(write_fd.id() == write_id);
  TEST_ASSERT(!rund::host::io::take_native_fd(-1));

  std::array<std::byte, 1u> out{std::byte{0x5a}};
  const rund::host::io::WriteResult write = rund::host::io::write_some_blocking(
      write_fd.view(), std::span<const std::byte>{out});
  TEST_ASSERT(write);
  TEST_ASSERT(write.bytes == 1);

  std::array<std::byte, 1u> in{};
  const rund::host::io::ReadResult read = rund::host::io::read_some_blocking(
      read_fd.view(), std::span<std::byte>{in});
  TEST_ASSERT(read);
  TEST_ASSERT(read.bytes == 1);
  TEST_ASSERT(in[0] == std::byte{0x5a});

  const rund::host::io::WriteResult forged_probe_write =
      rund::host::io::write_some_blocking(write_fd.view(),
                                          std::span<const std::byte>{out});
  TEST_ASSERT(forged_probe_write);
  TEST_ASSERT(forged_probe_write.bytes == 1);
  const rund::host::io::FdView invalid_view{};
  const rund::host::io::ReadResult invalid_view_read =
      rund::host::io::read_some_blocking(invalid_view,
                                         std::span<std::byte>{in});
  TEST_ASSERT(!invalid_view_read);
  TEST_ASSERT(invalid_view_read.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(invalid_view_read.bytes == -1);
  const rund::task::IoOp invalid_view_readable =
      rund::host::io::readable(invalid_view);
  TEST_ASSERT(!invalid_view_readable);
  TEST_ASSERT(invalid_view_readable.code() == rund::ReasonCode::IoFdInvalid);

  TEST_ASSERT(read_fd.close());
  TEST_ASSERT(write_fd.close());
  TEST_ASSERT(!read_fd && !write_fd);

  std::array<std::byte, 1u> invalid_buffer{};
  const rund::host::io::ReadResult invalid_read =
      rund::host::io::read_some_blocking(rund::host::io::FdView{},
                                         std::span<std::byte>{invalid_buffer});
  TEST_ASSERT(!invalid_read);
  TEST_ASSERT(invalid_read.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(std::string_view{invalid_read.error()} == "io_fd_invalid");
  TEST_ASSERT(invalid_read.bytes == -1);

  const rund::host::io::WriteResult invalid_write =
      rund::host::io::write_some_blocking(rund::host::io::FdView{},
                                          std::span<const std::byte>{out});
  TEST_ASSERT(!invalid_write);
  TEST_ASSERT(invalid_write.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(std::string_view{invalid_write.error()} == "io_fd_invalid");
  TEST_ASSERT(invalid_write.bytes == -1);

  static_cast<void>(::mkdir(".cache", 0777));
  Temp temp_file{.path = Path("file")};
  rund::host::io::OpenResult opened = rund::host::io::open_file(
      temp_file.path, rund::host::io::OpenOptions{
                          .flags = O_CREAT | O_TRUNC | O_RDWR,
                          .mode = 0600,
                      });
  TEST_ASSERT(opened);
  TEST_ASSERT(opened.fd);
  TEST_ASSERT(opened.fd.id() != 0u);

  const rund::host::io::ReadResult null_nonempty_read =
      rund::host::io::read_some_blocking(
          opened.fd.view(),
          std::span<std::byte>{static_cast<std::byte *>(nullptr), 1u});
  TEST_ASSERT(!null_nonempty_read);
  TEST_ASSERT(null_nonempty_read.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(null_nonempty_read.bytes == -1);

  const rund::host::io::WriteResult null_nonempty_write =
      rund::host::io::write_some_blocking(
          opened.fd.view(), std::span<const std::byte>{
                                static_cast<const std::byte *>(nullptr), 1u});
  TEST_ASSERT(!null_nonempty_write);
  TEST_ASSERT(null_nonempty_write.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(null_nonempty_write.bytes == -1);

  const std::array<std::byte, 3u> file_bytes{
      std::byte{'a'},
      std::byte{'b'},
      std::byte{'c'},
  };
  const rund::host::io::WriteResult file_write =
      rund::host::io::write_some_blocking(
          opened.fd.view(), std::span<const std::byte>{file_bytes});
  TEST_ASSERT(file_write);
  TEST_ASSERT(file_write.bytes == 3);

  std::array<std::byte, 1u> pread_buffer{};
  const rund::host::io::ReadResult pread = rund::host::io::pread_some(
      opened.fd.view(), std::span<std::byte>{pread_buffer}, 1u);
  TEST_ASSERT(pread);
  TEST_ASSERT(pread.bytes == 1);
  TEST_ASSERT(pread_buffer[0] == std::byte{'b'});

  constexpr auto kOverflowOffset =
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) + 1u;
  const rund::host::io::ReadResult pread_overflow = rund::host::io::pread_some(
      opened.fd.view(), std::span<std::byte>{pread_buffer}, kOverflowOffset);
  TEST_ASSERT(!pread_overflow);
  TEST_ASSERT(pread_overflow.code() == rund::ReasonCode::IoSyscallFailed);
  TEST_ASSERT(pread_overflow.native_error == EINVAL);
  TEST_ASSERT(pread_overflow.bytes == -1);

  TEST_ASSERT(opened.fd.close());

  const rund::host::io::OpenResult empty_path = rund::host::io::open_file(
      std::string_view{},
      rund::host::io::OpenOptions{.flags = O_RDONLY, .mode = 0});
  TEST_ASSERT(!empty_path);
  TEST_ASSERT(empty_path.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{empty_path.error()} == "task_invalid");

  const std::string missing_path = Path("missing");
  static_cast<void>(::unlink(missing_path.c_str()));
  const rund::host::io::OpenResult missing_file = rund::host::io::open_file(
      missing_path, rund::host::io::OpenOptions{.flags = O_RDONLY, .mode = 0});
  TEST_ASSERT(!missing_file);
  TEST_ASSERT(missing_file.code() == rund::ReasonCode::IoSyscallFailed);
  TEST_ASSERT(missing_file.native_error == ENOENT);

  int guard_pipe_fds[2] = {-1, -1};
  TEST_ASSERT(::pipe(guard_pipe_fds) == 0);
  Fd guard_read_cleanup{guard_pipe_fds[0]};
  Fd guard_write_cleanup{guard_pipe_fds[1]};
  const std::array<std::byte, 1u> source{std::byte{0x21}};
  TEST_ASSERT(::write(guard_write_cleanup.native, source.data(), source.size()) ==
              1);
  rund::host::io::Fd guard_read_fd =
      rund::host::io::take_native_fd(guard_read_cleanup.native);
  rund::host::io::Fd guard_write_fd =
      rund::host::io::take_native_fd(guard_write_cleanup.native);

  rund::host::io::OpenResult guard_file = rund::host::io::open_file(
      temp_file.path, rund::host::io::OpenOptions{.flags = O_RDWR, .mode = 0});
  TEST_ASSERT(guard_file);
  int close_probe_native = ::open(temp_file.path.c_str(), O_RDWR);
  TEST_ASSERT(close_probe_native >= 0);
  rund::host::io::Fd close_probe_fd =
      rund::host::io::take_native_fd(close_probe_native);
  TEST_ASSERT(close_probe_native == -1);

  std::array<std::byte, 1u> task_read_buffer{};
  const std::array<std::byte, 1u> task_write_buffer{std::byte{0x21}};
  rund::host::io::ReadResult task_read{};
  rund::host::io::WriteResult task_write{};
  rund::host::io::ReadResult task_pread{};
  rund::host::io::CloseResult task_close{};
  rund::host::io::OpenResult task_open{};
  rund::task::Status task_join{};
  const rund::Session::Result task_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
              },
      },
      [&] {
        auto body = [&]() -> rund::task::Task<void> {
          task_read = rund::host::io::read_some_blocking(
              guard_read_fd.view(), std::span<std::byte>{task_read_buffer});
          task_write = rund::host::io::write_some_blocking(
              guard_write_fd.view(),
              std::span<const std::byte>{task_write_buffer});
          task_pread = rund::host::io::pread_some(
              guard_file.fd.view(), std::span<std::byte>{task_read_buffer}, 0u);
          task_close = close_probe_fd.close();
          task_open = rund::host::io::open_file(
              Path("task-guard"),
              rund::host::io::OpenOptions{.flags = O_RDONLY, .mode = 0});
          co_return;
        };
        const rund::task::Handle task =
            rund::task::spawn("hostio-direct-guard", body());
        task_join = rund::task::join(task);
      });
  TEST_ASSERT(task_report.ok());
  TEST_ASSERT(task_join.ok());
  TEST_ASSERT(!task_read);
  TEST_ASSERT(task_read.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{task_read.error()} == "task_invalid");
  TEST_ASSERT(!task_write);
  TEST_ASSERT(task_write.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{task_write.error()} == "task_invalid");
  TEST_ASSERT(!task_pread);
  TEST_ASSERT(task_pread.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{task_pread.error()} == "task_invalid");
  TEST_ASSERT(task_close);
  TEST_ASSERT(!close_probe_fd);
  TEST_ASSERT(!task_open);
  TEST_ASSERT(task_open.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(std::string_view{task_open.error()} == "task_invalid");

  const rund::host::io::CloseResult close_probe_after_task =
      close_probe_fd.close();
  TEST_ASSERT(!close_probe_after_task);
  TEST_ASSERT(close_probe_after_task.code() == rund::ReasonCode::IoFdInvalid);
}

} // namespace

void Surface() {
  VerifyResult();
  VerifyInternalOutcomeProjection();
  VerifyOwner();
  VerifyBlocking();
}

} // namespace runtime_task_host_io

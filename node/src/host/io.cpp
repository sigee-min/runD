#include <rund/host/io.hpp>

#include "../runtime/platform/io.hpp"
#include "../runtime/task/scheduler/host.hpp"
#include "../runtime/task/scheduler/state.hpp"
#include "io/access.hpp"
#include "io/validation.hpp"

#include <utility>

namespace rund::host::io {
namespace {

[[nodiscard]] ReadResult FailRead(const ReasonCode code,
                                  const int err = 0) noexcept {
  return detail::Access::read(code, err);
}

[[nodiscard]] WriteResult FailWrite(const ReasonCode code,
                                    const int err = 0) noexcept {
  return detail::Access::write(code, err);
}

[[nodiscard]] OpenResult FailOpen(const ReasonCode code,
                                  const int err = 0) noexcept {
  return detail::Access::open(code, err);
}

[[nodiscard]] CloseResult FailClose(const ReasonCode code,
                                    const int err = 0) noexcept {
  return detail::Access::close(code, err);
}

[[nodiscard]] bool InvalidBuffer(const void *const data,
                                 const std::size_t size) noexcept {
  return data == nullptr && size != 0u;
}

[[nodiscard]] bool InActiveSchedulerTask() noexcept {
  return ::rund::node::scheduler_host::ActiveTask();
}

[[nodiscard]] ReadResult
CompleteNativeRead(const ::rund::node::NativeIoResult native) noexcept {
  using ::rund::node::NativeIoDisposition;
  switch (native.disposition()) {
  case NativeIoDisposition::Complete:
    return detail::Access::read(native.value());
  case NativeIoDisposition::InvalidBuffer:
    return FailRead(ReasonCode::TaskInvalid, native.native_error());
  case NativeIoDisposition::Failed:
    return FailRead(ReasonCode::IoSyscallFailed, native.native_error());
  case NativeIoDisposition::Unsupported:
    return FailRead(ReasonCode::IoUnsupported);
  }
  std::abort();
}

[[nodiscard]] WriteResult
CompleteNativeWrite(const ::rund::node::NativeIoResult native) noexcept {
  using ::rund::node::NativeIoDisposition;
  switch (native.disposition()) {
  case NativeIoDisposition::Complete:
    return detail::Access::write(native.value());
  case NativeIoDisposition::InvalidBuffer:
    return FailWrite(ReasonCode::TaskInvalid, native.native_error());
  case NativeIoDisposition::Failed:
    return FailWrite(ReasonCode::IoSyscallFailed, native.native_error());
  case NativeIoDisposition::Unsupported:
    return FailWrite(ReasonCode::IoUnsupported);
  }
  std::abort();
}

} // namespace

Fd take_native_fd(int &fd) noexcept {
  if (fd < 0) {
    return {};
  }
  const int owned = std::exchange(fd, -1);
  return Fd{owned, static_cast<std::uint64_t>(owned) + 1u};
}

Fd take_native_fd(int &&fd) noexcept { return take_native_fd(fd); }

Fd replay_fd(const std::uint64_t host_id) noexcept { return Fd{-1, host_id}; }

Fd::~Fd() noexcept {
  if (valid()) {
    static_cast<void>(close());
  }
}

Fd &Fd::operator=(Fd &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (valid()) {
    static_cast<void>(close());
  }
  native_ = std::exchange(other.native_, -1);
  host_id_ = std::exchange(other.host_id_, 0u);
  return *this;
}

CloseResult Fd::close() noexcept {
  const FdView owned = view();
  const detail::FdIdentity identity = detail::Project(owned);
  if (!identity.live() && !identity.replay()) {
    return FailClose(ReasonCode::IoFdInvalid);
  }
  native_ = -1;
  host_id_ = 0u;

  if (identity.replay()) {
    return detail::Access::close();
  }
  if (InActiveSchedulerTask()) {
    ::rund::node::Scheduler *const scheduler =
        ::rund::node::Scheduler::Active();
    if (scheduler != nullptr) {
      return scheduler->CloseHostFd(owned);
    }
  }
  const ::rund::node::NativeIoResult native =
      ::rund::node::NativeClose(identity.native);
  switch (native.disposition()) {
  case ::rund::node::NativeIoDisposition::Complete:
    return detail::Access::close();
  case ::rund::node::NativeIoDisposition::Unsupported:
    return FailClose(ReasonCode::IoUnsupported);
  case ::rund::node::NativeIoDisposition::InvalidBuffer:
  case ::rund::node::NativeIoDisposition::Failed:
    return FailClose(ReasonCode::IoSyscallFailed, native.native_error());
  }
  std::abort();
}

ReadOp::ReadOp(ReadOp &&other) noexcept
    : fd_(other.fd_), buffer_(other.buffer_), scheduler_(other.scheduler_),
      token_(other.token_), result_(other.result_), started_(other.started_) {
  other.fd_ = {};
  other.buffer_ = {};
  other.scheduler_ = nullptr;
  other.token_ = nullptr;
  other.started_ = false;
}

bool ReadOp::await_suspend(std::coroutine_handle<>) noexcept {
  if (started_) {
    result_ = FailRead(ReasonCode::TaskInvalid);
    return false;
  }
  started_ = true;
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    result_ = FailRead(ReasonCode::NodeRuntimeMissing);
    return false;
  }
  scheduler_ = scheduler;
  const bool suspended =
      scheduler->SuspendHostIoRead(fd_, buffer_, &token_, &result_);
  fd_ = {};
  buffer_ = {};
  return suspended;
}

ReadResult ReadOp::await_resume() noexcept {
  if (!started_) {
    return FailRead(ReasonCode::TaskInvalid);
  }
  if (token_ == nullptr) {
    return result_;
  }
  auto *const scheduler = static_cast<::rund::node::Scheduler *>(scheduler_);
  if (scheduler == nullptr || ::rund::node::Scheduler::Active() != scheduler) {
    return FailRead(ReasonCode::TaskContextMissing);
  }
  result_ = scheduler->CompleteHostIoRead(std::exchange(token_, nullptr));
  return result_;
}

WriteOp::WriteOp(WriteOp &&other) noexcept
    : fd_(other.fd_), buffer_(other.buffer_), scheduler_(other.scheduler_),
      token_(other.token_), result_(other.result_), started_(other.started_) {
  other.fd_ = {};
  other.buffer_ = {};
  other.scheduler_ = nullptr;
  other.token_ = nullptr;
  other.started_ = false;
}

bool WriteOp::await_suspend(std::coroutine_handle<>) noexcept {
  if (started_) {
    result_ = FailWrite(ReasonCode::TaskInvalid);
    return false;
  }
  started_ = true;
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    result_ = FailWrite(ReasonCode::NodeRuntimeMissing);
    return false;
  }
  scheduler_ = scheduler;
  const bool suspended =
      scheduler->SuspendHostIoWrite(fd_, buffer_, &token_, &result_);
  fd_ = {};
  buffer_ = {};
  return suspended;
}

WriteResult WriteOp::await_resume() noexcept {
  if (!started_) {
    return FailWrite(ReasonCode::TaskInvalid);
  }
  if (token_ == nullptr) {
    return result_;
  }
  auto *const scheduler = static_cast<::rund::node::Scheduler *>(scheduler_);
  if (scheduler == nullptr || ::rund::node::Scheduler::Active() != scheduler) {
    return FailWrite(ReasonCode::TaskContextMissing);
  }
  result_ = scheduler->CompleteHostIoWrite(std::exchange(token_, nullptr));
  return result_;
}

ReadOp read_some(const FdView fd, const std::span<std::byte> buffer) noexcept {
  return ReadOp{fd, buffer};
}

WriteOp write_some(const FdView fd,
                   const std::span<const std::byte> buffer) noexcept {
  return WriteOp{fd, buffer};
}

ReadResult read_some_blocking(const FdView fd,
                              const std::span<std::byte> buffer) noexcept {
  if (InActiveSchedulerTask()) {
    return FailRead(ReasonCode::TaskInvalid);
  }
  const detail::FdIdentity identity = detail::Project(fd);
  if (!identity.live()) {
    return FailRead(ReasonCode::IoFdInvalid);
  }
  if (InvalidBuffer(buffer.data(), buffer.size())) {
    return FailRead(ReasonCode::TaskInvalid);
  }
  const ::rund::node::NativeIoResult native =
      ::rund::node::NativeRead(identity.native, buffer);
  return CompleteNativeRead(native);
}

WriteResult
write_some_blocking(const FdView fd,
                    const std::span<const std::byte> buffer) noexcept {
  if (InActiveSchedulerTask()) {
    return FailWrite(ReasonCode::TaskInvalid);
  }
  const detail::FdIdentity identity = detail::Project(fd);
  if (!identity.live()) {
    return FailWrite(ReasonCode::IoFdInvalid);
  }
  if (InvalidBuffer(buffer.data(), buffer.size())) {
    return FailWrite(ReasonCode::TaskInvalid);
  }
  const ::rund::node::NativeIoResult native =
      ::rund::node::NativeWrite(identity.native, buffer);
  return CompleteNativeWrite(native);
}

ReadResult pread_some(const FdView fd, const std::span<std::byte> buffer,
                      const std::uint64_t offset) noexcept {
  const detail::FdIdentity identity = detail::Project(fd);
  if (!identity.live()) {
    return FailRead(ReasonCode::IoFdInvalid);
  }
  if (InActiveSchedulerTask()) {
    return FailRead(ReasonCode::TaskInvalid);
  }
  if (InvalidBuffer(buffer.data(), buffer.size())) {
    return FailRead(ReasonCode::TaskInvalid);
  }
  const ::rund::node::NativeIoResult native =
      ::rund::node::NativePread(identity.native, buffer, offset);
  return CompleteNativeRead(native);
}

OpenResult open_file(const std::string_view path,
                     const OpenOptions options) noexcept {
  if (InActiveSchedulerTask()) {
    return FailOpen(ReasonCode::TaskInvalid);
  }
  if (path.empty()) {
    return FailOpen(ReasonCode::TaskInvalid);
  }
  const ::rund::node::NativeIoResult native =
      ::rund::node::NativeOpen(path, options.flags, options.mode);
  switch (native.disposition()) {
  case ::rund::node::NativeIoDisposition::Complete:
    return detail::Access::open(
        take_native_fd(static_cast<int>(native.value())));
  case ::rund::node::NativeIoDisposition::Unsupported:
    return FailOpen(ReasonCode::IoUnsupported);
  case ::rund::node::NativeIoDisposition::InvalidBuffer:
  case ::rund::node::NativeIoDisposition::Failed:
    return FailOpen(ReasonCode::IoSyscallFailed, native.native_error());
  }
  std::abort();
}

} // namespace rund::host::io

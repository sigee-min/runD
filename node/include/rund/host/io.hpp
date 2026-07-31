#pragma once

#include <rund/task/results.hpp>
#include <rund/host/io/fd.hpp>
#include <rund/reason.hpp>

#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rund::host::io {

namespace detail {
struct Access;
}

struct OpenOptions {
  int flags = 0;
  std::uint32_t mode = 0u;
};

class Status {
public:
  constexpr Status() noexcept = default;

  [[nodiscard]] constexpr ReasonCode code() const noexcept { return code_; }
  [[nodiscard]] constexpr bool ok() const noexcept {
    return ::rund::detail::status_ok(code_);
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::detail::status_error(code_);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::detail::status_exit(code_);
  }

private:
  friend struct detail::Access;
  ReasonCode code_ = ReasonCode::TaskInvalid;
};

struct ReadResult final : Status {
  std::int64_t bytes = -1;
  int native_error = 0;
};

struct WriteResult final : Status {
  std::int64_t bytes = -1;
  int native_error = 0;
};

struct OpenResult final : Status {
  OpenResult() noexcept = default;
  OpenResult(const OpenResult &) = delete;
  OpenResult &operator=(const OpenResult &) = delete;
  OpenResult(OpenResult &&) noexcept = default;
  OpenResult &operator=(OpenResult &&) noexcept = default;

  Fd fd{};
  int native_error = 0;
};

struct CloseResult final : Status {
  int native_error = 0;
};

class ReadOp final {
public:
  ReadOp(const ReadOp &) = delete;
  ReadOp &operator=(const ReadOp &) = delete;
  ReadOp(ReadOp &&other) noexcept;
  ReadOp &operator=(ReadOp &&) = delete;

  [[nodiscard]] constexpr bool await_ready() const noexcept { return false; }
  [[nodiscard]] bool await_suspend(std::coroutine_handle<>) noexcept;
  [[nodiscard]] ReadResult await_resume() noexcept;

private:
  friend ReadOp read_some(FdView, std::span<std::byte>) noexcept;

  explicit ReadOp(FdView fd, std::span<std::byte> buffer) noexcept
      : fd_(fd), buffer_(buffer) {}

  FdView fd_{};
  std::span<std::byte> buffer_{};
  void *scheduler_ = nullptr;
  void *token_ = nullptr;
  ReadResult result_{};
  bool started_ = false;
};

class WriteOp final {
public:
  WriteOp(const WriteOp &) = delete;
  WriteOp &operator=(const WriteOp &) = delete;
  WriteOp(WriteOp &&other) noexcept;
  WriteOp &operator=(WriteOp &&) = delete;

  [[nodiscard]] constexpr bool await_ready() const noexcept { return false; }
  [[nodiscard]] bool await_suspend(std::coroutine_handle<>) noexcept;
  [[nodiscard]] WriteResult await_resume() noexcept;

private:
  friend WriteOp write_some(FdView, std::span<const std::byte>) noexcept;

  explicit WriteOp(FdView fd, std::span<const std::byte> buffer) noexcept
      : fd_(fd), buffer_(buffer) {}

  FdView fd_{};
  std::span<const std::byte> buffer_{};
  void *scheduler_ = nullptr;
  void *token_ = nullptr;
  WriteResult result_{};
  bool started_ = false;
};

[[nodiscard]] Fd take_native_fd(int &fd) noexcept;
[[nodiscard]] Fd take_native_fd(int &&fd) noexcept;
[[nodiscard]] Fd replay_fd(std::uint64_t host_id) noexcept;
[[nodiscard]] task::IoOp readable(FdView fd) noexcept;
[[nodiscard]] task::IoOp writable(FdView fd) noexcept;
[[nodiscard]] ReadOp read_some(FdView fd, std::span<std::byte> buffer) noexcept;
[[nodiscard]] WriteOp write_some(FdView fd,
                                 std::span<const std::byte> buffer) noexcept;
[[nodiscard]] ReadResult
read_some_blocking(FdView fd, std::span<std::byte> buffer) noexcept;
[[nodiscard]] WriteResult
write_some_blocking(FdView fd, std::span<const std::byte> buffer) noexcept;
[[nodiscard]] ReadResult pread_some(FdView fd, std::span<std::byte> buffer,
                                    std::uint64_t offset) noexcept;
[[nodiscard]] OpenResult open_file(std::string_view path,
                                   OpenOptions options) noexcept;

} // namespace rund::host::io

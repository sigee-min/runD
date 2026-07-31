#pragma once

#include <rund/host/io.hpp>

#include <utility>

namespace rund::host::io::detail {

struct Access final {
  [[nodiscard]] static constexpr int native(const FdView fd) noexcept {
    return fd.native_;
  }

  [[nodiscard]] static constexpr std::uint64_t id(const FdView fd) noexcept {
    return fd.host_id_;
  }

  [[nodiscard]] static constexpr ReadResult
  read(const std::int64_t bytes, const int native_error = 0) noexcept {
    if (bytes < 0) {
      return read(ReasonCode::TaskInvalid, native_error);
    }
    ReadResult result{};
    result.code_ = ReasonCode::Ok;
    result.bytes = bytes;
    result.native_error = native_error;
    return result;
  }

  [[nodiscard]] static constexpr ReadResult
  read(const ReasonCode code, const int native_error = 0) noexcept {
    ReadResult result{};
    result.code_ = failure(code);
    result.native_error = native_error;
    return result;
  }

  [[nodiscard]] static constexpr WriteResult
  write(const std::int64_t bytes, const int native_error = 0) noexcept {
    if (bytes < 0) {
      return write(ReasonCode::TaskInvalid, native_error);
    }
    WriteResult result{};
    result.code_ = ReasonCode::Ok;
    result.bytes = bytes;
    result.native_error = native_error;
    return result;
  }

  [[nodiscard]] static constexpr WriteResult
  write(const ReasonCode code, const int native_error = 0) noexcept {
    WriteResult result{};
    result.code_ = failure(code);
    result.native_error = native_error;
    return result;
  }

  [[nodiscard]] static OpenResult open(Fd fd) noexcept {
    if (!fd || native(fd.view()) < 0) {
      return open(ReasonCode::TaskInvalid);
    }
    OpenResult result{};
    result.code_ = ReasonCode::Ok;
    result.fd = std::move(fd);
    return result;
  }

  [[nodiscard]] static OpenResult open(const ReasonCode code,
                                       const int native_error = 0) noexcept {
    OpenResult result{};
    result.code_ = failure(code);
    result.native_error = native_error;
    return result;
  }

  [[nodiscard]] static constexpr CloseResult close() noexcept {
    CloseResult result{};
    result.code_ = ReasonCode::Ok;
    return result;
  }

  [[nodiscard]] static constexpr CloseResult
  close(const ReasonCode code, const int native_error = 0) noexcept {
    CloseResult result{};
    result.code_ = failure(code);
    result.native_error = native_error;
    return result;
  }

private:
  [[nodiscard]] static constexpr ReasonCode
  failure(const ReasonCode code) noexcept {
    return code == ReasonCode::Ok ? ReasonCode::TaskInvalid : code;
  }
};

} // namespace rund::host::io::detail

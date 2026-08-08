#pragma once

#include "../../model/wake.hpp"

#include <rund/reason.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

namespace rund::node {

enum class HostIoKind : std::uint8_t {
  Read,
  Write,
};

enum class HostIoSlotState : std::uint8_t {
  Free,
  Admitting,
  Queued,
  Running,
  Complete,
};

enum class HostIoOutcomeDisposition : std::uint8_t {
  Pending,
  Complete,
  Failed,
  InvalidBuffer,
  Unsupported,
};

struct HostIoOperation final {
  const std::byte *data = nullptr;
  std::uint64_t host_id = 0u;
  std::uint64_t sequence = 0u;
  std::size_t size = 0u;
  int native = -1;
  HostIoKind kind = HostIoKind::Read;

  [[nodiscard]] std::span<std::byte> read_buffer() const noexcept {
    return {const_cast<std::byte *>(data), size};
  }

  [[nodiscard]] std::span<const std::byte> write_buffer() const noexcept {
    return {data, size};
  }
};

class HostIoOutcome final {
public:
  HostIoOutcome() = delete;

  [[nodiscard]] static constexpr HostIoOutcome pending() noexcept {
    return HostIoOutcome{-1, 0, HostIoOutcomeDisposition::Pending};
  }

  [[nodiscard]] static constexpr HostIoOutcome
  complete(const std::int64_t value) noexcept {
    if (value < 0) {
      std::abort();
    }
    return HostIoOutcome{value, 0, HostIoOutcomeDisposition::Complete};
  }

  [[nodiscard]] static constexpr HostIoOutcome
  failed(const int native_error) noexcept {
    if (native_error == 0) {
      std::abort();
    }
    return HostIoOutcome{-1, native_error, HostIoOutcomeDisposition::Failed};
  }

  [[nodiscard]] static constexpr HostIoOutcome
  invalid_buffer(const int native_error) noexcept {
    if (native_error == 0) {
      std::abort();
    }
    return HostIoOutcome{-1, native_error,
                         HostIoOutcomeDisposition::InvalidBuffer};
  }

  [[nodiscard]] static constexpr HostIoOutcome unsupported() noexcept {
    return HostIoOutcome{-1, 0, HostIoOutcomeDisposition::Unsupported};
  }

  [[nodiscard]] constexpr HostIoOutcomeDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr std::int64_t value() const noexcept { return value_; }

  [[nodiscard]] constexpr int native_error() const noexcept {
    return native_error_;
  }

private:
  constexpr HostIoOutcome(const std::int64_t value, const int native_error,
                          const HostIoOutcomeDisposition disposition) noexcept
      : value_(value), native_error_(native_error), disposition_(disposition) {}

  std::int64_t value_;
  int native_error_;
  HostIoOutcomeDisposition disposition_;
};

struct HostIoCompletion final {
  ReasonCode code = ReasonCode::TaskContextMissing;
  std::int64_t bytes = -1;
  int native_error = 0;
};

struct HostIoSlot {
  HostIoSlot *next = nullptr;
  ExternalWake wake{};
  HostIoOperation operation{};
  HostIoOutcome outcome = HostIoOutcome::pending();
  std::uint64_t released_sequence = 0u;
  std::atomic<std::uint8_t> phase{0u};
  std::atomic<HostIoSlotState> state{HostIoSlotState::Free};
};

static_assert(sizeof(void *) != 8u || sizeof(HostIoSlot) <= 104u,
              "64-bit HostIoSlot must remain within 104 bytes");

} // namespace rund::node

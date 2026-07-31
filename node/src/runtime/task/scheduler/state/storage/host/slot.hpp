#pragma once

#include "../../model/wake.hpp"

#include <rund/reason.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
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

enum class HostIoOutcomeKind : std::uint8_t {
  Pending,
  Ready,
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

struct HostIoOutcome final {
  std::int64_t value = -1;
  int error = 0;
  HostIoOutcomeKind kind = HostIoOutcomeKind::Pending;
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
  HostIoOutcome outcome{};
  std::uint64_t released_sequence = 0u;
  std::atomic<std::uint8_t> phase{0u};
  std::atomic<HostIoSlotState> state{HostIoSlotState::Free};
};

static_assert(sizeof(void *) != 8u || sizeof(HostIoSlot) <= 104u,
              "64-bit HostIoSlot must remain within 104 bytes");

} // namespace rund::node

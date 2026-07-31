#pragma once

#include <accel/buffer.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>
#include <memory>

namespace rund {

struct AccelBuffer {
  AccelCheck check{false, "accel_context_buffer_invalid"};
  std::uint64_t context_id = 0u;
  Buffer buffer{};
  rund::kernel::ResidentBufferRef resident{};
  std::uint64_t byte_extent = 0u;
  std::uint64_t scalar_width_bytes = 0u;
  std::uint64_t count = 0u;
  BufferUsage usage = BufferUsage::ReadWrite;
  std::shared_ptr<void> owner{};
  std::shared_ptr<void> handle{};
  const char* reason = "accel_context_buffer_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return check.ok;
  }
};

}  // namespace rund

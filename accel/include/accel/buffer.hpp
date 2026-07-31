#pragma once

#include <accel/check.hpp>

#include <cstdint>
#include <memory>

namespace rund {

enum class BufferUsage : std::uint8_t {
  ReadOnly,
  WriteOnly,
  ReadWrite,
};

struct BufferDesc {
  std::uint64_t bytes = 0u;
  BufferUsage usage = BufferUsage::ReadWrite;
  std::uint32_t alignment = 16u;
};

struct Buffer {
  AccelCheck check{};
  std::uint64_t id = 0u;
  std::uint64_t bytes = 0u;
  std::uint64_t element_bytes = 0u;
  std::uint64_t stride_bytes = 0u;
  std::uint64_t count = 0u;
  std::uint64_t storage_bytes = 0u;
  bool storage_reused = false;
  BufferUsage usage = BufferUsage::ReadWrite;
  std::shared_ptr<void> owner{};
  std::shared_ptr<void> handle{};
};

}  // namespace rund

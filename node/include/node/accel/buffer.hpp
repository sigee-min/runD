#pragma once

#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>


#include <memory>

namespace rund::node::accel {

struct AccelMemoryCounter final {
  std::uint64_t current{};
  std::uint64_t peak{};
  std::uint64_t cumulative{};
  std::uint64_t reused{};
  std::uint64_t budget{};
};

struct AccelMemoryStats final {
  AccelMemoryCounter staging{};
};

[[nodiscard]] rund::Buffer CreateBuffer(const rund::AccelDevice& pick,
                                  const rund::BufferDesc& desc);

[[nodiscard]] rund::AccelCheck UploadBuffer(const rund::AccelDevice& pick,
                                 const rund::Buffer& buffer,
                                 const void* data,
                                 std::uint64_t bytes,
                                 std::uint64_t offset = 0u);

[[nodiscard]] rund::AccelCheck DownloadBuffer(const rund::AccelDevice& pick,
                                   const rund::Buffer& buffer,
                                   void* data,
                                   std::uint64_t bytes,
                                   std::uint64_t offset = 0u);

[[nodiscard]] rund::kernel::ResidentBufferRef ResidentRef(
    const rund::Buffer& buffer);

[[nodiscard]] rund::kernel::ResidentBufferRef ResidentRead(
    const rund::Buffer& buffer,
    std::uint64_t element_bytes,
    std::uint64_t count);

[[nodiscard]] rund::kernel::ResidentBufferRef ResidentWrite(
    const rund::Buffer& buffer,
    std::uint64_t element_bytes,
    std::uint64_t count);

[[nodiscard]] std::shared_ptr<void> ResidentHandle(const rund::Buffer& buffer);

[[nodiscard]] rund::RuntimeStats ReadRuntimeStats(const rund::AccelDevice& pick);
[[nodiscard]] AccelMemoryStats ReadAccelMemoryStats(
    const rund::AccelDevice& pick) noexcept;

void ResetRuntimeStats(const rund::AccelDevice& pick);

}  // namespace rund::node::accel

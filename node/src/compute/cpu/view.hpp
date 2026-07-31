#pragma once

#include <cstddef>
#include <optional>

namespace rund::compute::detail {

struct BufferState;
struct JobBufferView;

struct CpuFootprint final {
  std::size_t base{};
  std::size_t stride{};
  std::size_t bytes{};
  std::size_t count{};
  std::size_t width{};

  [[nodiscard]] bool dense() const noexcept { return stride == width; }
};

struct CpuView final {
  std::byte *data{};
  CpuFootprint footprint{};
};

[[nodiscard]] std::optional<CpuFootprint>
cpu_footprint(std::size_t storage_bytes, std::size_t offset, std::size_t count,
              std::size_t stride, std::size_t width) noexcept;

[[nodiscard]] std::optional<CpuView>
cpu_view(const BufferState *buffer, std::size_t offset, std::size_t count,
         std::size_t stride, std::size_t width) noexcept;

[[nodiscard]] std::optional<CpuView>
cpu_view(const BufferState *buffer, const JobBufferView &view) noexcept;

} // namespace rund::compute::detail

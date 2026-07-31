#include "view.hpp"

#include "../device/state.hpp"
#include "../job/state.hpp"
#include "../size.hpp"

namespace rund::compute::detail {

std::optional<CpuFootprint> cpu_footprint(const std::size_t storage_bytes,
                                          const std::size_t offset,
                                          const std::size_t count,
                                          const std::size_t stride,
                                          const std::size_t width) noexcept {
  std::size_t base = 0u;
  if (width == 0u || stride == 0u || !size::multiply(offset, width, base)) {
    return std::nullopt;
  }
  if (base > storage_bytes) {
    return std::nullopt;
  }
  if (count == 0u) {
    return CpuFootprint{
        .base = base,
        .stride = width,
        .count = 0u,
        .width = width,
    };
  }

  const std::size_t remaining = storage_bytes - base;
  if (width > remaining) {
    return std::nullopt;
  }
  const std::size_t tail = count - 1u;
  const std::size_t distance = (remaining - width) / width;
  if (tail > distance / stride) {
    return std::nullopt;
  }

  return CpuFootprint{
      .base = base,
      .stride = tail == 0u ? width : stride * width,
      .bytes = width + tail * width,
      .count = count,
      .width = width,
  };
}

std::optional<CpuView> cpu_view(const BufferState *const buffer,
                                const std::size_t offset,
                                const std::size_t count,
                                const std::size_t stride,
                                const std::size_t width) noexcept {
  const CpuBufferState *const cpu =
      buffer == nullptr ? nullptr : cpu_buffer(*buffer);
  if (cpu == nullptr || (cpu->data == nullptr && cpu->bytes != 0u)) {
    return std::nullopt;
  }
  std::optional<CpuFootprint> footprint =
      cpu_footprint(cpu->bytes, offset, count, stride, width);
  if (!footprint || (count != 0u && cpu->data == nullptr)) {
    return std::nullopt;
  }
  return CpuView{
      .data =
          cpu->data == nullptr ? nullptr : cpu->data.get() + footprint->base,
      .footprint = *footprint,
  };
}

std::optional<CpuView> cpu_view(const BufferState *const buffer,
                                const JobBufferView &view) noexcept {
  return cpu_view(buffer, view.offset, view.count, view.stride,
                  view.element_bytes);
}

} // namespace rund::compute::detail

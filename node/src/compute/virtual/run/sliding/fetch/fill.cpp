#include "fill.hpp"

#include <array>
#include <cstring>

namespace rund::compute::detail::sliding_product_detail {
namespace {

void repeat(std::byte *const target, const std::size_t bytes,
            const std::byte *const value,
            const std::size_t element_bytes) noexcept {
  for (std::size_t offset = 0u; offset < bytes; offset += element_bytes) {
    std::memcpy(target + offset, value, element_bytes);
  }
}

} // namespace

bool fill_fetch_frame(
    std::byte *const frame,
    const residency::execution::FetchSource &source) noexcept {
  using residency::execution::FetchFill;
  if (frame == nullptr || !source.materializes_frame()) {
    return false;
  }
  if (source.complete_frame()) {
    return true;
  }
  const std::size_t frame_bytes = static_cast<std::size_t>(source.frame_bytes);
  const std::size_t target = static_cast<std::size_t>(source.target_offset);
  const std::size_t bytes = static_cast<std::size_t>(source.bytes);
  if (source.fill == FetchFill::ZeroInactiveTail) {
    std::memset(frame + bytes, 0, frame_bytes - bytes);
    return true;
  }
  const std::size_t element =
      static_cast<std::size_t>(source.fill_element_bytes);
  const std::size_t suffix = frame_bytes - target - bytes;
  if (source.fill == FetchFill::RepeatBoundary) {
    repeat(frame, target, frame + target, element);
    repeat(frame + target + bytes, suffix, frame + target + bytes - element,
           element);
    return true;
  }
  if (source.fill != FetchFill::ConstantBoundary) {
    return false;
  }
  std::array<std::byte, sizeof(source.fill_value)> value{};
  std::memcpy(value.data(), &source.fill_value, element);
  repeat(frame, target, value.data(), element);
  repeat(frame + target + bytes, suffix, value.data(), element);
  return true;
}

} // namespace rund::compute::detail::sliding_product_detail

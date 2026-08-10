#include "backing.hpp"
#include "model.hpp"

#include <algorithm>
#include <cstring>

namespace rund::measure::compute::virtual_residency {

MemoryBacking::MemoryBacking(const std::size_t bytes)
    : bytes_(bytes, TailPoison) {}

std::uint64_t MemoryBacking::size_bytes() const noexcept {
  return bytes_.size();
}

bool MemoryBacking::contains(const std::uint64_t offset,
                             const std::size_t bytes) const noexcept {
  return offset <= bytes_.size() && bytes <= bytes_.size() - offset;
}

::rund::compute::Status
MemoryBacking::read(const std::uint64_t offset,
                    const std::span<std::byte> output) noexcept {
  if (!contains(offset, output.size())) {
    return ::rund::compute::Status::fail(
        ::rund::compute::Reason::ShapeMismatch);
  }
  if (!output.empty()) {
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
  }
  return ::rund::compute::Status::success();
}

::rund::compute::Status
MemoryBacking::write(const std::uint64_t offset,
                     const std::span<const std::byte> input) noexcept {
  if (!contains(offset, input.size())) {
    return ::rund::compute::Status::fail(
        ::rund::compute::Reason::ShapeMismatch);
  }
  if (!input.empty()) {
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
  }
  return ::rund::compute::Status::success();
}

void MemoryBacking::reset(const std::byte value) noexcept {
  std::fill(bytes_.begin(), bytes_.end(), value);
}

} // namespace rund::measure::compute::virtual_residency

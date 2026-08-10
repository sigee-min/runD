#include "backing.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rund_node_test_virtual::product {
namespace {

[[nodiscard]] std::size_t rounded_capacity(const std::size_t logical,
                                           const std::size_t page) noexcept {
  if (page == 0u) {
    return 0u;
  }
  const std::size_t remainder = logical % page;
  const std::size_t padding = remainder == 0u ? page : page - remainder;
  return logical <= std::numeric_limits<std::size_t>::max() - padding
             ? logical + padding
             : 0u;
}

} // namespace

MemoryVirtualBacking::MemoryVirtualBacking(const std::size_t logical_bytes,
                                           const std::size_t page_bytes)
    : bytes_(rounded_capacity(logical_bytes, page_bytes), TailPoison),
      logical_bytes_(logical_bytes), page_bytes_(page_bytes) {}

bool MemoryVirtualBacking::contains(const std::uint64_t offset,
                                    const std::size_t bytes) const noexcept {
  return offset <= logical_bytes_ &&
         bytes <= logical_bytes_ - static_cast<std::size_t>(offset);
}

rund::compute::Status
MemoryVirtualBacking::read(const std::uint64_t offset,
                           const std::span<std::byte> output) noexcept {
  if (read_failure_ != rund::compute::Reason::Ok) {
    const rund::compute::Reason reason = read_failure_;
    read_failure_ = rund::compute::Reason::Ok;
    ++facts_.read_failure_count;
    return rund::compute::Status::fail(reason);
  }
  if (!contains(offset, output.size())) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  if (!output.empty()) {
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
  }
  ++facts_.read_count;
  facts_.read_bytes += output.size();
  return rund::compute::Status::success();
}

rund::compute::Status
MemoryVirtualBacking::write(const std::uint64_t offset,
                            const std::span<const std::byte> input) noexcept {
  if (write_failure_ != rund::compute::Reason::Ok) {
    const rund::compute::Reason reason = write_failure_;
    write_failure_ = rund::compute::Reason::Ok;
    const std::size_t changed = std::min(write_failure_prefix_, input.size());
    write_failure_prefix_ = 0u;
    if (changed != 0u && contains(offset, changed)) {
      std::memcpy(bytes_.data() + offset, input.data(), changed);
      facts_.partial_write_bytes += changed;
    }
    ++facts_.write_failure_count;
    return rund::compute::Status::fail(reason);
  }
  if (!contains(offset, input.size())) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  if (!input.empty()) {
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
  }
  ++facts_.write_count;
  facts_.write_bytes += input.size();
  return rund::compute::Status::success();
}

bool MemoryVirtualBacking::seed(
    const std::span<const std::byte> input) noexcept {
  if (input.size() != logical_bytes_) {
    return false;
  }
  if (!input.empty()) {
    std::memcpy(bytes_.data(), input.data(), input.size());
  }
  return true;
}

bool MemoryVirtualBacking::observe(const std::span<std::byte> output) noexcept {
  if (output.size() != logical_bytes_) {
    return false;
  }
  if (!output.empty()) {
    std::memcpy(output.data(), bytes_.data(), output.size());
  }
  ++facts_.observation_count;
  facts_.observation_bytes += output.size();
  return true;
}

void MemoryVirtualBacking::reset(const std::byte value) noexcept {
  std::fill(bytes_.begin(), bytes_.end(), value);
}

void MemoryVirtualBacking::fail_next_read(
    const rund::compute::Reason reason) noexcept {
  read_failure_ = reason == rund::compute::Reason::Ok
                      ? rund::compute::Reason::BackendFailed
                      : reason;
}

void MemoryVirtualBacking::fail_next_write(
    const rund::compute::Reason reason) noexcept {
  fail_next_write_after(0u, reason);
}

void MemoryVirtualBacking::fail_next_write_after(
    const std::size_t prefix_bytes,
    const rund::compute::Reason reason) noexcept {
  write_failure_ = reason == rund::compute::Reason::Ok
                       ? rund::compute::Reason::BackendFailed
                       : reason;
  write_failure_prefix_ = prefix_bytes;
}

bool MemoryVirtualBacking::tail_poisoned() const noexcept {
  return logical_bytes_ <= bytes_.size() &&
         std::all_of(bytes_.begin() +
                         static_cast<std::ptrdiff_t>(logical_bytes_),
                     bytes_.end(),
                     [](const std::byte value) { return value == TailPoison; });
}

bool MemoryVirtualBacking::valid() const noexcept {
  return page_bytes_ != 0u && logical_bytes_ <= bytes_.size() &&
         bytes_.size() - logical_bytes_ != 0u &&
         bytes_.size() % page_bytes_ == 0u && tail_poisoned();
}

} // namespace rund_node_test_virtual::product

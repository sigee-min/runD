#include "backing.hpp"

#include <cstring>

namespace rund_node_test_persistent_product {

PersistentProductBacking::PersistentProductBacking(const std::size_t bytes)
    : bytes_(bytes) {}

std::uint64_t PersistentProductBacking::size_bytes() const noexcept {
  return bytes_.size();
}

rund::compute::Status
PersistentProductBacking::read(const std::uint64_t offset,
                               const std::span<std::byte> output) noexcept {
  if (fail_read_once_) {
    fail_read_once_ = false;
    return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
  }
  if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(output.data(), bytes_.data() + offset, output.size());
  return rund::compute::Status::success();
}

rund::compute::Status PersistentProductBacking::read_batch(
    const std::span<const rund::compute::VirtualRead> ranges) noexcept {
  ++read_batch_calls_;
  read_batch_ranges_ += ranges.size();
  return VirtualBacking::read_batch(ranges);
}

rund::compute::Status PersistentProductBacking::write(
    const std::uint64_t offset,
    const std::span<const std::byte> input) noexcept {
  if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(bytes_.data() + offset, input.data(), input.size());
  return rund::compute::Status::success();
}

bool PersistentProductBacking::seed(
    const std::span<const std::byte> input) noexcept {
  if (input.size() != bytes_.size()) {
    return false;
  }
  std::memcpy(bytes_.data(), input.data(), input.size());
  return true;
}

bool PersistentProductBacking::observe(
    const std::span<std::byte> output) const noexcept {
  if (output.size() != bytes_.size()) {
    return false;
  }
  std::memcpy(output.data(), bytes_.data(), output.size());
  return true;
}

void PersistentProductBacking::fail_next_read() noexcept {
  fail_read_once_ = true;
}

std::uint64_t PersistentProductBacking::read_batch_calls() const noexcept {
  return read_batch_calls_;
}

std::uint64_t PersistentProductBacking::read_batch_ranges() const noexcept {
  return read_batch_ranges_;
}

} // namespace rund_node_test_persistent_product

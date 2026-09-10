#pragma once

#include <rund/compute/virtual.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace rund::compute::detail {
struct PipelineState;
}

namespace rund_node_test_pipeline {

enum class ProductFault : std::uint8_t { None, KnownInput, DeviceLoss };

[[nodiscard]] std::uint64_t ReadVulkanResidencyQueueSubmits(
    const std::shared_ptr<rund::compute::detail::PipelineState> &pipeline);

class WindowBacking final : public rund::compute::VirtualBacking {
public:
  explicit WindowBacking(const std::size_t bytes) : bytes_(bytes) {}

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override {
    return bytes_.size();
  }

  [[nodiscard]] rund::compute::Status
  read(const std::uint64_t offset,
       const std::span<std::byte> output) noexcept override {
    if (fail_read_.exchange(false, std::memory_order_acq_rel)) {
      return rund::compute::Status::fail(rund::compute::Reason::BackendFailed);
    }
    ++reads_;
    if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
    return rund::compute::Status::success();
  }

  [[nodiscard]] rund::compute::Status
  write(const std::uint64_t offset,
        const std::span<const std::byte> input) noexcept override {
    ++writes_;
    if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
    return rund::compute::Status::success();
  }

  [[nodiscard]] bool all_u32(const std::uint32_t expected) const noexcept {
    if (bytes_.size() % sizeof(expected) != 0u) {
      return false;
    }
    for (std::size_t offset = 0u; offset < bytes_.size();
         offset += sizeof(expected)) {
      std::uint32_t value = 0u;
      std::memcpy(&value, bytes_.data() + offset, sizeof(value));
      if (value != expected) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::uint64_t reads() const noexcept { return reads_; }
  [[nodiscard]] std::uint64_t writes() const noexcept { return writes_; }

  void fail_next_read() noexcept {
    fail_read_.store(true, std::memory_order_release);
  }

private:
  std::vector<std::byte> bytes_;
  std::atomic_bool fail_read_{false};
  std::uint64_t reads_{};
  std::uint64_t writes_{};
};

} // namespace rund_node_test_pipeline

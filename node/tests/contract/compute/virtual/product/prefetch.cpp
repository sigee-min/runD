#include "local.hpp"

#include "golden.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"
#include "../../allocation.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <thread>
#include <vector>

namespace rund_node_test_virtual::product {
namespace {

class DelayedBacking final : public rund::compute::VirtualBacking {
public:
  DelayedBacking(const std::size_t bytes,
                 const std::chrono::milliseconds read_delay,
                 const std::chrono::milliseconds write_delay)
      : bytes_(bytes), read_delay_(read_delay), write_delay_(write_delay) {}

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override {
    return bytes_.size();
  }
  [[nodiscard]] rund::compute::VirtualBackingTier
  tier() const noexcept override {
    return rund::compute::VirtualBackingTier::Persistent;
  }
  [[nodiscard]] std::uint32_t max_parallel_reads() const noexcept override {
    return 2u;
  }

  [[nodiscard]] rund::compute::Status
  read(const std::uint64_t offset,
       const std::span<std::byte> output) noexcept override {
    if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    const std::uint32_t active =
        active_reads_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    std::uint32_t maximum = max_active_reads_.load(std::memory_order_relaxed);
    while (maximum < active && !max_active_reads_.compare_exchange_weak(
                                   maximum, active, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
    }
    std::this_thread::sleep_for(read_delay_);
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
    active_reads_.fetch_sub(1u, std::memory_order_release);
    return rund::compute::Status::success();
  }

  [[nodiscard]] rund::compute::Status
  write(const std::uint64_t offset,
        const std::span<const std::byte> input) noexcept override {
    if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    std::this_thread::sleep_for(write_delay_);
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
    return rund::compute::Status::success();
  }

  void seed(const std::span<const std::int32_t> values) noexcept {
    std::memcpy(bytes_.data(), values.data(), bytes_.size());
  }

  [[nodiscard]] bool matches_golden() const noexcept {
    for (std::size_t index = 0u; index < LogicalElements; ++index) {
      std::int32_t value = 0;
      std::memcpy(&value, bytes_.data() + index * sizeof(value), sizeof(value));
      if (value != ProductValue(SeedValue(index))) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::uint32_t max_active_reads() const noexcept {
    return max_active_reads_.load(std::memory_order_acquire);
  }

private:
  std::vector<std::byte> bytes_;
  std::chrono::milliseconds read_delay_{};
  std::chrono::milliseconds write_delay_{};
  std::atomic<std::uint32_t> active_reads_{};
  std::atomic<std::uint32_t> max_active_reads_{};
};

} // namespace

int CheckProductPrefetch(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 1;
  }
  auto program =
      on(*device)
          .map<std::int32_t>("virtual-product-prefetch", PageElements,
                             [](auto value) { return (value + 5) * 3; })
          .compile();
  if (!program) {
    return 2;
  }
  auto input_backing = std::make_shared<DelayedBacking>(
      LogicalBytes, std::chrono::milliseconds{20},
      std::chrono::milliseconds{0});
  auto output_backing = std::make_shared<DelayedBacking>(
      LogicalBytes, std::chrono::milliseconds{0}, std::chrono::milliseconds{5});
  std::vector<std::int32_t> seeded(LogicalElements);
  SeedInput(seeded);
  input_backing->seed(seeded);
  auto input = virtual_buffer<std::int32_t>(LogicalElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(LogicalElements, output_backing);
  auto prepared =
      input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return 3;
  }
  if (!prepared->run()) {
    return 4;
  }
  node_compute_allocation::Start();
  const Status warm = prepared->run();
  node_compute_allocation::Stop();
  const Stats stats = prepared->stats();
  const ResidencyStats &residency = stats.pipeline.residency;
  if (!warm || node_compute_allocation::Count() != 0u ||
      !output_backing->matches_golden() ||
      residency.page_in_count != PageCount ||
      residency.late_page_count != FrameCapacity ||
      residency.prefetch_count != PageCount - FrameCapacity ||
      residency.late_page_count + residency.prefetch_count !=
          residency.page_in_count ||
      residency.overlap_ns < 1'000'000u ||
      residency.overlap_ns > residency.backing_io_ns ||
      input_backing->max_active_reads() < 2u ||
      stats.output_hash != GoldenHash) {
    return 5;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product

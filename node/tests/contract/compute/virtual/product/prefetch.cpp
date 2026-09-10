#include "local.hpp"

#include "golden.hpp"
#include "model.hpp"
#include "route.hpp"

#include "../../../target/selection.hpp"
#include "../../allocation.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <thread>
#include <vector>

namespace rund_node_test_virtual::product {
namespace {

[[nodiscard]] bool
same_counter_shape(const rund::compute::MemoryCounter left,
                   const rund::compute::MemoryCounter right) noexcept {
  return left.current == right.current && left.peak == right.peak &&
         left.budget == right.budget;
}

[[nodiscard]] bool
same_fixed_memory(const rund::compute::MemoryStats &left,
                  const rund::compute::MemoryStats &right) noexcept {
  return left.backend == right.backend && left.scope == right.scope &&
         same_counter_shape(left.host, right.host) &&
         same_counter_shape(left.frame, right.frame) &&
         same_counter_shape(left.tile, right.tile) &&
         same_counter_shape(left.resident, right.resident) &&
         same_counter_shape(left.staging, right.staging) &&
         same_counter_shape(left.device, right.device) &&
         same_counter_shape(left.transfer, right.transfer);
}

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
    read_count_.fetch_add(1u, std::memory_order_relaxed);
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

  [[nodiscard]] std::uint64_t read_count() const noexcept {
    return read_count_.load(std::memory_order_acquire);
  }

private:
  std::vector<std::byte> bytes_;
  std::chrono::milliseconds read_delay_{};
  std::chrono::milliseconds write_delay_{};
  std::atomic<std::uint32_t> active_reads_{};
  std::atomic<std::uint32_t> max_active_reads_{};
  std::atomic<std::uint64_t> read_count_{};
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
  const PipelinePlan warm_plan = prepared->plan();
  const MemoryStats warm_memory = prepared->memory();
  if (!prepared->begin_samples()) {
    return 5;
  }
  node_compute_allocation::Start();
  Status warm = Status::success();
  Stats stats{};
  bool transfer_overlap_observed = false;
  bool h2d_overlap_observed = false;
  bool d2h_overlap_observed = false;
  std::uint64_t warm_read_count = 0u;
  for (std::size_t attempt = 0u; attempt < 8u; ++attempt) {
    const std::uint64_t reads_before = input_backing->read_count();
    warm = prepared->run();
    if (!warm) {
      break;
    }
    stats = prepared->stats();
    transfer_overlap_observed =
        transfer_overlap_observed || stats.pipeline.residency.overlap_ns != 0u;
    h2d_overlap_observed =
        h2d_overlap_observed || stats.pipeline.residency.h2d_overlap_ns != 0u;
    d2h_overlap_observed =
        d2h_overlap_observed || stats.pipeline.residency.d2h_overlap_ns != 0u;
    warm_read_count = input_backing->read_count() - reads_before;
  }
  node_compute_allocation::Stop();
  if (!prepared->end_samples()) {
    return 5;
  }
  const ResidencyStats &residency = stats.pipeline.residency;
  const bool cpu_direct = backend == Backend::Cpu;
  const RouteKind mode = ClassifyMode(backend, residency, residency.page_count);
  const bool device_vsm = mode == RouteKind::DeviceVsm;
  const bool zero_download = stats.downloaded_bytes == 0u &&
                             stats.transfer_submissions.device_to_host == 0u;
  const bool host_cache_accelerator = !cpu_direct && zero_download;
  const bool physical_vulkan = backend == Backend::Vulkan && !zero_download;
  const bool host_supply_exact = host_cache_accelerator &&
                                 stats.uploaded_bytes == ElementPageBytes &&
                                 stats.transfer_submissions.host_to_device ==
                                     (backend == Backend::Vulkan ? 1u : 0u);
  const bool cpu_transfer_exact =
      !cpu_direct ||
      (stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u &&
       stats.transfer_submissions.host_to_device == 0u &&
       stats.transfer_submissions.device_to_host == 0u);
  const std::uint64_t expected_reads = host_cache_accelerator ? 1u : 2u;
  const std::uint64_t expected_read_bytes =
      host_cache_accelerator ? ElementPageBytes : WarmBackingReadBytes;
  const bool owned_allocation_free = device_vsm || backend == Backend::Vulkan ||
                                     node_compute_allocation::Count() == 0u;
  const bool device_vsm_exact =
      device_vsm && residency.page_in_count == PageCount &&
      residency.cache_hit_count == 0u && residency.late_page_count == 0u &&
      residency.prefetch_count == 0u && warm_read_count == PageCount &&
      residency.backing_read_bytes == LogicalBytes && zero_download &&
      stats.uploaded_bytes == 0u &&
      stats.transfer_submissions.host_to_device == 0u &&
      !transfer_overlap_observed && !h2d_overlap_observed &&
      !d2h_overlap_observed && residency.stall_ns == 0u &&
      input_backing->max_active_reads() == 1u;
  const bool rolling_exact =
      !device_vsm && residency.page_in_count == 2u &&
      residency.cache_hit_count == 3u &&
      residency.late_page_count == (backend == Backend::Cpu ? 2u : 1u) &&
      residency.prefetch_count == (physical_vulkan ? 1u : 0u) &&
      residency.late_page_count + residency.prefetch_count +
              (host_cache_accelerator ? 1u : 0u) ==
          residency.page_in_count &&
      warm_read_count == expected_reads &&
      residency.backing_read_bytes == expected_read_bytes &&
      cpu_transfer_exact && (!host_cache_accelerator || host_supply_exact) &&
      (cpu_direct || host_cache_accelerator || physical_vulkan) &&
      (!cpu_direct || !transfer_overlap_observed) &&
      transfer_overlap_observed == physical_vulkan &&
      residency.stall_ns != 0u &&
      (!(cpu_direct || host_cache_accelerator) || !h2d_overlap_observed) &&
      d2h_overlap_observed == physical_vulkan &&
      (backend == Backend::Cpu ? input_backing->max_active_reads() == 1u
                               : input_backing->max_active_reads() >= 2u);
  if (!warm || !owned_allocation_free || prepared->plan() != warm_plan ||
      !same_fixed_memory(prepared->memory(), warm_memory) ||
      !residency.samples_allocation_free(8u) ||
      !output_backing->matches_golden() ||
      // Two canonical banks retain four of five pages. The deterministic
      // cyclic future-use policy keeps pages 0,2,3 plus the last page, yielding
      // three warm hits. With a HostVisible output owner, one Device miss is
      // supplied by the authoritative Host cache and only the other is a late
      // Persistent read. Promotion is either a direct Host write or one exact
      // H2D receipt; neither is fabricated overlap. CPU reads both misses
      // directly. A private Vulkan output owner retains one speculative and
      // one late Persistent read plus its physical D2H receipt.
      (!device_vsm_exact && !rolling_exact) ||
      !residency.directional_overlap_exact() ||
      stats.output_hash != GoldenHash) {
    std::fprintf(
        stderr,
        "virtual prefetch loads=%llu hits=%llu late=%llu prefetch=%llu "
        "overlap=%llu h2d_overlap=%llu d2h_overlap=%llu stall=%llu "
        "max_reads=%u warm_reads=%llu read_bytes=%llu transfers=%llu/%llu "
        "bytes=%llu/%llu allocations=%llu hash=%llu/%llu plan=%u memory=%u "
        "samples=%u/%u\n",
        static_cast<unsigned long long>(residency.page_in_count),
        static_cast<unsigned long long>(residency.cache_hit_count),
        static_cast<unsigned long long>(residency.late_page_count),
        static_cast<unsigned long long>(residency.prefetch_count),
        static_cast<unsigned long long>(residency.overlap_ns),
        static_cast<unsigned long long>(residency.h2d_overlap_ns),
        static_cast<unsigned long long>(residency.d2h_overlap_ns),
        static_cast<unsigned long long>(residency.stall_ns),
        input_backing->max_active_reads(),
        static_cast<unsigned long long>(warm_read_count),
        static_cast<unsigned long long>(residency.backing_read_bytes),
        static_cast<unsigned long long>(
            stats.transfer_submissions.host_to_device),
        static_cast<unsigned long long>(
            stats.transfer_submissions.device_to_host),
        static_cast<unsigned long long>(stats.uploaded_bytes),
        static_cast<unsigned long long>(stats.downloaded_bytes),
        static_cast<unsigned long long>(node_compute_allocation::Count()),
        static_cast<unsigned long long>(stats.output_hash),
        static_cast<unsigned long long>(GoldenHash),
        static_cast<unsigned>(prepared->plan() == warm_plan),
        static_cast<unsigned>(
            same_fixed_memory(prepared->memory(), warm_memory)),
        residency.sampled_runs, residency.allocation_free_runs);
    const MemoryStats observed_memory = prepared->memory();
    std::fprintf(
        stderr,
        "virtual prefetch memory host=%llu/%llu frame=%llu/%llu "
        "resident=%llu/%llu staging=%llu/%llu device=%llu/%llu "
        "transfer=%llu/%llu\n",
        static_cast<unsigned long long>(warm_memory.host.current),
        static_cast<unsigned long long>(observed_memory.host.current),
        static_cast<unsigned long long>(warm_memory.frame.current),
        static_cast<unsigned long long>(observed_memory.frame.current),
        static_cast<unsigned long long>(warm_memory.resident.current),
        static_cast<unsigned long long>(observed_memory.resident.current),
        static_cast<unsigned long long>(warm_memory.staging.current),
        static_cast<unsigned long long>(observed_memory.staging.current),
        static_cast<unsigned long long>(warm_memory.device.current),
        static_cast<unsigned long long>(observed_memory.device.current),
        static_cast<unsigned long long>(warm_memory.transfer.current),
        static_cast<unsigned long long>(observed_memory.transfer.current));
    return 5;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product

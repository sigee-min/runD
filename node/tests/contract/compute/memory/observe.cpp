#include "model.hpp"

#include "../../../../src/accel/kernel/memory.hpp"
#include "../../../../src/compute/cpu/graph.hpp"
#include "../../../../src/compute/job/state.hpp"
#include "../../../../src/compute/memory/local.hpp"
#include <rund/counter.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>

namespace rund_node_memory_contract {

template <class Meter>
concept HasCurrent = requires(Meter &meter) { meter.current; };
template <class Meter>
concept HasReused = requires(Meter &meter) { meter.reused; };
template <class Meter>
concept HasBudget = requires(Meter &meter) { meter.budget; };

static_assert(HasCurrent<rund::compute::detail::AllocationMeter>);
static_assert(HasReused<rund::compute::detail::AllocationMeter>);
static_assert(!HasBudget<rund::compute::detail::AllocationMeter>);
static_assert(!HasCurrent<rund::compute::detail::TrafficMeter>);
static_assert(!HasReused<rund::compute::detail::TrafficMeter>);
static_assert(!HasBudget<rund::compute::detail::TrafficMeter>);

[[nodiscard]] bool
ValidStats(const rund::compute::MemoryStats &stats) noexcept {
  return ValidCounter(stats.host) && ValidCounter(stats.frame) &&
         ValidCounter(stats.tile) && ValidCounter(stats.resident) &&
         ValidCounter(stats.staging) && ValidCounter(stats.device) &&
         ValidCounter(stats.transfer);
}

[[nodiscard]] bool CheckCounterSaturation() {
  using namespace rund::compute::detail;
  std::uint64_t scalar = kCounterMaximum - 1u;
  ::rund::detail::counter::Accumulate(scalar, 2u);
  ::rund::detail::counter::Release(scalar, 4u);
  if (scalar != kCounterMaximum ||
      ::rund::detail::counter::Remaining(kCounterMaximum, kCounterMaximum) !=
          kCounterMaximum) {
    return false;
  }

  AllocationMeter allocation{};
  allocation.current = 11u;
  allocation.peak = 13u;
  allocation.cumulative = 17u;
  allocation.reused = 5u;
  const rund::compute::MemoryCounter allocation_memory =
      allocation_meter_memory(allocation);
  TrafficMeter traffic{};
  traffic.peak = 19u;
  traffic.cumulative = 23u;
  const rund::compute::MemoryCounter traffic_memory =
      traffic_meter_memory(traffic);
  if (allocation_memory.current != 11u || allocation_memory.peak != 13u ||
      allocation_memory.cumulative != 17u || allocation_memory.reused != 5u ||
      allocation_memory.budget != 0u || traffic_memory.current != 0u ||
      traffic_memory.peak != 19u || traffic_memory.cumulative != 23u ||
      traffic_memory.reused != 0u || traffic_memory.budget != 0u) {
    return false;
  }
  auto owner = std::make_shared<BufferState>();
  owner->bytes = 3u;
  owner->physical_bytes = 7u;
  const BufferMemory owner_memory = measure_buffer(owner);
  if (owner_memory.resident != 3u || owner_memory.physical != 7u ||
      owner_memory.reused != 0u) {
    return false;
  }

  rund::node::accel::detail::PreparedMemory total{
      .current = kCounterMaximum - 1u,
      .peak = kCounterMaximum - 1u,
      .cumulative = kCounterMaximum - 1u,
      .reused = kCounterMaximum - 1u};
  rund::node::accel::detail::accumulate_memory(
      total, rund::node::accel::detail::PreparedMemory{
                 .current = 2u, .peak = 2u, .cumulative = 2u, .reused = 2u});
  if (total.current != kCounterMaximum || total.peak != kCounterMaximum ||
      total.cumulative != kCounterMaximum || total.reused != kCounterMaximum) {
    return false;
  }

  rund::node::accel::detail::PreparedMemory serial{
      .current = 100u,
      .peak = 180u,
      .cumulative = 190u,
      .reused = 11u,
      .budget = 220u,
  };
  rund::node::accel::detail::accumulate_serial_memory(
      serial, rund::node::accel::detail::PreparedMemory{
                  .current = 40u,
                  .peak = 55u,
                  .cumulative = 60u,
                  .reused = 7u,
                  .budget = 200u,
              });
  if (serial.current != 140u || serial.peak != 220u ||
      serial.cumulative != 250u || serial.reused != 18u ||
      serial.budget != 220u) {
    return false;
  }
  rund::node::accel::detail::PreparedMemory serial_saturated{
      .current = kCounterMaximum - 4u,
      .peak = kCounterMaximum - 1u,
  };
  rund::node::accel::detail::accumulate_serial_memory(
      serial_saturated,
      rund::node::accel::detail::PreparedMemory{.current = 10u, .peak = 20u});
  if (serial_saturated.current != kCounterMaximum ||
      serial_saturated.peak != kCounterMaximum) {
    return false;
  }

  CpuMapRun simd{};
  std::array<CpuSimdCount, 2u> simd_counts{};
  simd.simd = simd_counts;
  simd.simd[0u] = CpuSimdCount{.vectors = kCounterMaximum - 1u,
                               .tails = kCounterMaximum - 1u};
  record_simd(simd, 0u,
              rund::node::accel::CpuSimdRunResult{.vector_chunk_count = 2u,
                                                  .tail_chunk_count = 2u});
  simd.simd[1u] = CpuSimdCount{.vectors = 1u, .tails = 1u};
  const CpuSimdCount summed = sum_simd(simd);
  if (summed.vectors != kCounterMaximum || summed.tails != kCounterMaximum) {
    return false;
  }

  auto job = std::make_shared<JobState>();
  job->frame_current = kCounterMaximum - 1u;
  job->frame_bytes = kCounterMaximum - 1u;
  job->frame_reused = kCounterMaximum - 1u;
  job->run_count = kCounterMaximum - 1u;
  record_job_frame(job, 2u, true, 16u);
  release_job_frame(job, 2u);
  if (job->frame_current != kCounterMaximum ||
      job->frame_bytes != kCounterMaximum ||
      job->frame_reused != kCounterMaximum) {
    return false;
  }
  const rund::compute::Status finished =
      finish_job(job, rund::compute::Result<RunState>::success(RunState{}));
  return finished && job->run_count == kCounterMaximum;
}
[[nodiscard]] bool CheckPreparedMemorySnapshot() {
  using rund::node::accel::detail::PreparedMemory;
  using rund::node::accel::detail::PreparedMemoryMeter;

  constexpr std::uint32_t writers = 4u;
  constexpr std::uint32_t iterations = 2'000u;
  constexpr std::uint64_t total = writers * iterations;
  PreparedMemoryMeter meter{};
  std::atomic<std::uint32_t> finished{};
  std::array<std::thread, writers> threads{};
  for (auto &thread : threads) {
    thread = std::thread{[&] {
      for (std::uint32_t index = 0u; index < iterations; ++index) {
        meter.add(PreparedMemory{.current = 1u,
                                 .peak = 1u,
                                 .cumulative = 1u,
                                 .reused = 1u,
                                 .budget = total});
      }
      finished.fetch_add(1u, std::memory_order_release);
    }};
  }

  bool coherent = true;
  while (finished.load(std::memory_order_acquire) != writers) {
    const PreparedMemory snapshot = meter.read();
    coherent = coherent && snapshot.current <= snapshot.peak &&
               snapshot.current == snapshot.cumulative &&
               snapshot.current == snapshot.reused && snapshot.budget <= total;
  }
  for (auto &thread : threads) {
    thread.join();
  }
  const PreparedMemory snapshot = meter.read();
  return coherent && snapshot.current == total && snapshot.peak == total &&
         snapshot.cumulative == total && snapshot.reused == total &&
         snapshot.budget == total;
}

} // namespace rund_node_memory_contract

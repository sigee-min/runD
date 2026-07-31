#include "model.hpp"

#include "../../../../src/accel/kernel/memory.hpp"
#include "../../../../src/compute/cpu/graph.hpp"
#include "../../../../src/compute/job/state.hpp"
#include <rund/counter.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>

namespace rund_node_memory_contract {

[[nodiscard]] bool ValidStats(const rund::compute::MemoryStats &stats) noexcept {
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
  if (scalar != kCounterMaximum) {
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

  CpuMapRun simd{};
  simd.simd.resize(2u);
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

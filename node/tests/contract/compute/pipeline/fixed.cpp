#include "local.hpp"

#include "../allocation.hpp"

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckWideFixed(rund::compute::Device &device,
                                 const Backend backend,
                                 rund::compute::graph::Fingerprint &fingerprint,
                                 std::uint64_t &output_hash) {
  using namespace rund::compute;
  using Real = Fixed<20, 44>;
  constexpr std::array<Real, 4u> position{Real::from_raw(3), Real::from_raw(6),
                                          Real::from_raw(9),
                                          Real::from_raw(12)};
  constexpr std::array<Real, 4u> velocity{Real::from_raw(2), Real::from_raw(4),
                                          Real::from_raw(6), Real::from_raw(8)};
  constexpr std::array<Real, 4u> force{Real::from_raw(1), Real::from_raw(3),
                                       Real::from_raw(5), Real::from_raw(7)};
  auto integrate =
      on(device)
          .input<Real>(position.size())
          .zip_input<Real>(velocity.size())
          .zip_input<Real>(force.size())
          .map("pipeline-wide-integrate",
               [](auto p, auto v, auto f) { return quantize<Real>(p + v + f); })
          .compile();
  auto advance =
      on(device)
          .map<Real>("pipeline-wide-advance", position.size(),
                     [](auto value) { return quantize<Real>(value + value); })
          .compile();
  auto p = Upload(device, position);
  auto v = Upload(device, velocity);
  auto f = Upload(device, force);
  auto middle = device.buffer<Real>(position.size());
  auto output = device.buffer<Real>(position.size());
  if (!integrate || !advance || !p || !v || !f || !middle || !output) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .then(*integrate, read(*p, *v, *f), write(*middle))
                      .then(*advance, read(*middle), write(*output))
                      .prepare();
  if (!prepared || !prepared->fingerprint() || !prepared->run()) {
    return 2;
  }
  const Stats first = prepared->stats();
  const std::uint64_t expected_submits = backend == Backend::Cpu ? 0u : 1u;
  if (first.backend != backend || first.pipeline.step_count != 2u ||
      first.pipeline.resource_count != 5u ||
      first.pipeline.barrier_count != 1u ||
      first.pipeline.verified_step_count != 2u ||
      first.pipeline.failed_step_index !=
          rund::compute::PipelineStats::no_failed_step ||
      first.command_submits != expected_submits || first.dispatches != 2u ||
      !WarmCountersClean(first)) {
    return 3;
  }
  if (backend == Backend::Cpu) {
    if (first.pipeline.control_byte_count != 0u) {
      return 4;
    }
  } else if (first.pipeline.control_byte_count != 80u ||
             first.pipeline.status_entry_count != 0u ||
             first.pipeline.control_command_count != 2u) {
    return 5;
  }

  std::array<MemoryEntry, 128u> before_entries{};
  const MemorySnapshot before = prepared->memory_snapshot(before_entries);
  // The process-wide operator-new hook is an exact SDK boundary on CPU. On
  // native accelerators it also observes opaque command-buffer allocations
  // inside the platform driver, which the SDK neither owns nor can retain.
  // GPU warm-growth is therefore proven by public SDK counters plus the exact
  // retained-memory snapshot below.
  const bool count_sdk_allocations = backend == Backend::Cpu;
  if (count_sdk_allocations) {
    node_compute_allocation::Start();
  }
  const auto second_run = prepared->run();
  const auto third_run = prepared->run();
  if (count_sdk_allocations) {
    node_compute_allocation::Stop();
  }
  if (!second_run || !third_run ||
      (count_sdk_allocations && node_compute_allocation::Count() != 0u)) {
    return 6;
  }
  std::array<MemoryEntry, 128u> after_entries{};
  const MemorySnapshot after = prepared->memory_snapshot(after_entries);
  if (before.truncated() || after.truncated() || before.total != after.total ||
      before.written != after.written ||
      !SameMemoryEntries(
          std::span<const MemoryEntry>{before_entries.data(), before.written},
          std::span<const MemoryEntry>{after_entries.data(), after.written}) ||
      !SameMemory(before.summary, after.summary) ||
      prepared->generation() != 3u || !WarmCountersClean(prepared->stats())) {
    return 7;
  }

  std::array<Real, 4u> observed{};
  if (!ReadExact(*prepared, *output, observed) ||
      observed != std::array<Real, 4u>{Real::from_raw(12), Real::from_raw(26),
                                       Real::from_raw(40),
                                       Real::from_raw(54)} ||
      prepared->stats().output_hash != 0u) {
    return 8;
  }
  std::array<Real, 4u> observed_middle{};
  if (!ReadExact(*prepared, *middle, observed_middle) ||
      observed_middle !=
          std::array<Real, 4u>{Real::from_raw(6), Real::from_raw(13),
                               Real::from_raw(20), Real::from_raw(27)} ||
      prepared->stats().output_hash == 0u) {
    return 8;
  }
  const auto current_fingerprint = prepared->fingerprint();
  const auto current_output_hash = prepared->stats().output_hash;
  if (fingerprint) {
    if (fingerprint != current_fingerprint ||
        output_hash != current_output_hash) {
      return 9;
    }
  } else {
    fingerprint = current_fingerprint;
    output_hash = current_output_hash;
  }
  return 0;
}

} // namespace rund_node_test_pipeline

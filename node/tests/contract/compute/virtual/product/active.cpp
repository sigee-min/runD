#include "local.hpp"

#include "backing.hpp"
#include "golden.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"
#include "../../allocation.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {
namespace {

[[nodiscard]] BackingFacts delta(const BackingFacts after,
                                 const BackingFacts before) noexcept {
  return BackingFacts{
      .read_count = after.read_count - before.read_count,
      .read_bytes = after.read_bytes - before.read_bytes,
      .write_count = after.write_count - before.write_count,
      .write_bytes = after.write_bytes - before.write_bytes,
      .observation_count = after.observation_count - before.observation_count,
      .observation_bytes = after.observation_bytes - before.observation_bytes,
      .read_failure_count =
          after.read_failure_count - before.read_failure_count,
      .write_failure_count =
          after.write_failure_count - before.write_failure_count,
      .partial_write_bytes =
          after.partial_write_bytes - before.partial_write_bytes,
  };
}

[[nodiscard]] bool same_capacity(const rund::compute::MemoryStats &left,
                                 const rund::compute::MemoryStats &right) {
  const auto same = [](const rund::compute::MemoryCounter a,
                       const rund::compute::MemoryCounter b) {
    return a.current == b.current;
  };
  return left.backend == right.backend && left.scope == right.scope &&
         same(left.host, right.host) && same(left.frame, right.frame) &&
         same(left.tile, right.tile) && same(left.resident, right.resident) &&
         same(left.staging, right.staging) && same(left.device, right.device) &&
         same(left.transfer, right.transfer);
}

[[nodiscard]] bool prefix_and_tail(const std::span<const std::byte> observed,
                                   const std::size_t active_count) noexcept {
  const std::size_t active_bytes = active_count * sizeof(std::int32_t);
  if (observed.size() != LogicalBytes || active_bytes > observed.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < active_count; ++index) {
    std::int32_t value = 0;
    std::memcpy(&value, observed.data() + index * sizeof(value), sizeof(value));
    if (value != ProductValue(SeedValue(index))) {
      return false;
    }
  }
  return std::all_of(observed.begin() +
                         static_cast<std::ptrdiff_t>(active_bytes),
                     observed.end(),
                     [](const std::byte value) { return value == TailPoison; });
}

[[nodiscard]] bool exact_active_stats(const rund::compute::Stats &stats,
                                      const rund::compute::Backend backend,
                                      const std::size_t active_count) noexcept {
  const std::uint64_t pages =
      active_count / PageElements +
      static_cast<std::uint64_t>(active_count % PageElements != 0u);
  const std::uint64_t waves =
      pages / SlotCapacity +
      static_cast<std::uint64_t>(pages % SlotCapacity != 0u);
  const std::uint64_t physical = waves * SlotCapacity * ElementPageBytes;
  const std::uint64_t submits =
      backend == rund::compute::Backend::Cpu ? 0u : waves;
  const auto &residency = stats.pipeline.residency;
  return residency.logical_bytes == LogicalBytes * 2u &&
         residency.active_count == active_count &&
         residency.page_bytes == ResidencyPageBytes &&
         residency.page_count == PageCount &&
         residency.slot_capacity == SlotCapacity &&
         residency.active_slots_peak ==
             std::min(pages, static_cast<std::uint64_t>(SlotCapacity)) &&
         residency.wave_count == waves && residency.load_count == pages &&
         residency.writeback_count == pages &&
         residency.backing_read_bytes == active_count * sizeof(std::int32_t) &&
         residency.backing_write_bytes == active_count * sizeof(std::int32_t) &&
         residency.samples_allocation_free(WarmRuns) &&
         residency.failed_page ==
             rund::compute::ResidencyStats::no_failed_page &&
         stats.dispatches == waves * SlotCapacity &&
         stats.command_submits == submits && stats.uploaded_bytes == physical &&
         stats.downloaded_bytes == physical &&
         stats.transfer_submissions.host_to_device == 0u &&
         stats.transfer_submissions.device_to_host == 0u &&
         stats.transfer_submissions.device_to_device == 0u;
}

} // namespace

int CheckProductActiveCount(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-product-active", PageElements,
                             [](auto value) { return (value + 5) * 3; })
          .compile();
  if (!program) {
    return 2;
  }
  auto input_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  std::array<std::int32_t, LogicalElements> seeded{};
  std::array<std::int32_t, LogicalElements> golden{};
  SeedInput(seeded);
  for (std::size_t index = 0u; index < golden.size(); ++index) {
    golden[index] = ProductValue(seeded[index]);
  }
  if (!input_backing->seed(std::as_bytes(std::span{seeded}))) {
    return 3;
  }
  auto input = virtual_buffer<std::int32_t>(LogicalElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(LogicalElements, output_backing);
  auto prepared =
      input && output
          ? virtual_pipeline(*program, *input, *output,
                             ResidencyConfig{.slots = SlotCapacity})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return 4;
  }
  const PipelinePlan plan = prepared->plan();
  const MemoryStats memory = prepared->memory();
  const void *const input_identity = input_backing->identity();
  const void *const output_identity = output_backing->identity();
  const BackingFacts overflow_input = input_backing->facts();
  const BackingFacts overflow_output = output_backing->facts();
  if (prepared->run(LogicalElements + 1u).reason() != Reason::ShapeMismatch ||
      input_backing->facts().read_count != overflow_input.read_count ||
      output_backing->facts().write_count != overflow_output.write_count) {
    return 5;
  }

  std::array<std::byte, LogicalBytes> cold_observed{};
  if (!prepared->run() || !output_backing->observe(cold_observed)) {
    return 6;
  }
  const Stats cold_stats = prepared->stats();
  const auto cold_profile = prepared->profile();
  if (!cold_profile || !prefix_and_tail(cold_observed, LogicalElements) ||
      cold_stats.pipeline.residency.active_count != LogicalElements ||
      cold_stats.output_hash != HashValues(golden) ||
      cold_profile->execution().output_hash != cold_stats.output_hash ||
      prepared->plan() != plan || input_backing->identity() != input_identity ||
      output_backing->identity() != output_identity) {
    return 7;
  }
  output_backing->reset(TailPoison);

  constexpr std::array<std::size_t, 4u> active_counts{0u, 7u, 35u,
                                                      LogicalElements};
  for (const std::size_t active_count : active_counts) {
    const BackingFacts input_before = input_backing->facts();
    const BackingFacts output_before = output_backing->facts();
    if (!prepared->run(active_count) || !prepared->begin_samples()) {
      return 8;
    }
    node_compute_allocation::Start();
    bool warm_ok = true;
    for (std::size_t sample = 0u; sample < WarmRuns; ++sample) {
      warm_ok = warm_ok && static_cast<bool>(prepared->run(active_count));
    }
    node_compute_allocation::Stop();
    const std::uint64_t allocations = node_compute_allocation::Count();
    if (!warm_ok || !prepared->end_samples()) {
      return 9;
    }
    std::array<std::byte, LogicalBytes> observed{};
    if (!output_backing->observe(observed)) {
      return 10;
    }
    const Stats stats = prepared->stats();
    const auto profile = prepared->profile();
    const std::uint64_t pages =
        active_count / PageElements +
        static_cast<std::uint64_t>(active_count % PageElements != 0u);
    const std::uint64_t waves =
        pages / SlotCapacity +
        static_cast<std::uint64_t>(pages % SlotCapacity != 0u);
    const BackingFacts input_delta =
        delta(input_backing->facts(), input_before);
    const BackingFacts output_delta =
        delta(output_backing->facts(), output_before);
    const auto expected = std::span{golden}.first(active_count);
    const bool capacity_same = same_capacity(memory, prepared->memory());
    const bool stats_exact = exact_active_stats(stats, backend, active_count);
    const bool content_exact = prefix_and_tail(observed, active_count);
    if (!profile || allocations != 0u || prepared->plan() != plan ||
        input_backing->identity() != input_identity ||
        output_backing->identity() != output_identity || !capacity_same ||
        !stats_exact ||
        profile->execution().pipeline.residency.active_count != active_count ||
        profile->execution().output_hash != stats.output_hash ||
        stats.output_hash != HashValues(expected) ||
        input_delta.read_count != (WarmRuns + 1u) * waves ||
        input_delta.read_bytes !=
            (WarmRuns + 1u) * active_count * sizeof(std::int32_t) ||
        input_delta.write_count != 0u || input_delta.observation_count != 0u ||
        output_delta.write_count != (WarmRuns + 1u) * waves ||
        output_delta.write_bytes !=
            (WarmRuns + 1u) * active_count * sizeof(std::int32_t) ||
        output_delta.observation_count != 1u ||
        output_delta.observation_bytes != LogicalBytes ||
        input_delta.read_failure_count != 0u ||
        output_delta.write_failure_count != 0u ||
        output_delta.partial_write_bytes != 0u || !content_exact ||
        !input_backing->tail_poisoned() || !output_backing->tail_poisoned()) {
      std::fprintf(
          stderr,
          "virtual active backend=%u n=%zu alloc=%llu capacity=%u stats=%u "
          "content=%u hash=%llx expected=%llx read=%llu/%llu "
          "write=%llu/%llu observe=%llu/%llu sampled=%u/%u\n",
          static_cast<unsigned>(backend), active_count,
          static_cast<unsigned long long>(allocations),
          static_cast<unsigned>(capacity_same),
          static_cast<unsigned>(stats_exact),
          static_cast<unsigned>(content_exact),
          static_cast<unsigned long long>(stats.output_hash),
          static_cast<unsigned long long>(HashValues(expected)),
          static_cast<unsigned long long>(input_delta.read_count),
          static_cast<unsigned long long>(input_delta.read_bytes),
          static_cast<unsigned long long>(output_delta.write_count),
          static_cast<unsigned long long>(output_delta.write_bytes),
          static_cast<unsigned long long>(output_delta.observation_count),
          static_cast<unsigned long long>(output_delta.observation_bytes),
          stats.pipeline.residency.sampled_runs,
          stats.pipeline.residency.allocation_free_runs);
      return 11;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product

#include "support.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace rund_node_test_virtual::product::scan {

[[nodiscard]] int RunBasic(ScanFixture &fixture) {
  using namespace rund::compute;
  const Backend backend = fixture.backend;
  auto inclusive_flow =
      on(*fixture.device).input<std::uint32_t>(ScanFrameElements);
  auto inclusive_program_result =
      std::move(inclusive_flow)
          .branch([](auto values) { return values.scan(Scan::InclusiveSum); })
          .compile();
  auto exclusive_flow =
      on(*fixture.device).input<std::uint32_t>(ScanFrameElements);
  auto exclusive_program_result =
      std::move(exclusive_flow)
          .branch([](auto values) { return values.scan(Scan::ExclusiveSum); })
          .compile();
  if (!inclusive_program_result || !exclusive_program_result ||
      inclusive_program_result->graph().nodes.size() != 1u ||
      exclusive_program_result->graph().nodes.size() != 1u) {
    return 2;
  }
  fixture.inclusive_program.emplace(
      std::move(inclusive_program_result).value());
  fixture.exclusive_program.emplace(
      std::move(exclusive_program_result).value());
  auto &inclusive_program = fixture.inclusive_program;
  auto &exclusive_program = fixture.exclusive_program;
  const auto values = fixture.values;
  auto input_backing = fixture.input_backing;
  if (!input_backing) {
    return 3;
  }
  auto input = virtual_buffer<std::uint32_t>(ScanElements, input_backing);
  auto inclusive_backing =
      std::make_shared<MemoryVirtualBacking>(sizeof(values), sizeof(values));
  fixture.inclusive_backing = inclusive_backing;
  auto inclusive_output =
      virtual_buffer<std::uint32_t>(ScanElements, inclusive_backing);
  auto inclusive =
      input && inclusive_output
          ? virtual_pipeline(*inclusive_program, *input, *inclusive_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                Reason::PipelineInvalid);
  const std::uint64_t inclusive_version_before =
      detail::VirtualBackingAccess::version(*inclusive_backing);
  const Status inclusive_status =
      inclusive ? inclusive->run() : Status::fail(inclusive.reason());
  const bool inclusive_matches =
      inclusive_status && scan_matches(*inclusive_backing, values, true);
  const BackingFacts inclusive_facts = inclusive_backing->facts();
  const std::uint64_t inclusive_version =
      detail::VirtualBackingAccess::version(*inclusive_backing);
  if (!inclusive || !inclusive_status || !inclusive_matches) {
    std::fprintf(stderr,
                 "virtual scan inclusive prepared=%u reason=%.*s match=%u "
                 "commits=%llu version=%llu/%llu\n",
                 static_cast<unsigned>(static_cast<bool>(inclusive)),
                 static_cast<int>(inclusive_status.error().size()),
                 inclusive_status.error().data(),
                 static_cast<unsigned>(inclusive_matches),
                 static_cast<unsigned long long>(inclusive_facts.write_count),
                 static_cast<unsigned long long>(inclusive_version),
                 static_cast<unsigned long long>(inclusive_version_before));
    return 4;
  }
  constexpr std::size_t InclusivePages =
      (ScanElements + ScanFrameElements - 1u) / ScanFrameElements;
  constexpr std::size_t InclusiveEpochs =
      (InclusivePages + FrameCapacity - 1u) / FrameCapacity;
  const Stats inclusive_stats = inclusive->stats();
  const bool inclusive_device_vsm =
      uses_device_vsm(backend, inclusive_stats.pipeline.residency);
  const std::size_t scan_dispatches_per_frame =
      backend == Backend::Cpu ? 4u : 2u;
  const std::size_t expected_inclusive_dispatches =
      inclusive_device_vsm
          ? 1u
          : InclusiveEpochs * FrameCapacity * scan_dispatches_per_frame;
  if (inclusive->plan().residency.frame_capacity != FrameCapacity ||
      inclusive_stats.pipeline.residency.epoch_count != InclusiveEpochs ||
      inclusive_stats.pipeline.residency.page_in_count != InclusivePages ||
      inclusive_stats.pipeline.residency.page_out_count != InclusivePages ||
      inclusive_stats.dispatches != expected_inclusive_dispatches ||
      inclusive_stats.pipeline.residency.backing_read_bytes != sizeof(values) ||
      inclusive_stats.pipeline.residency.backing_write_bytes !=
          sizeof(values)) {
    std::fprintf(stderr,
                 "virtual scan inclusive frames=%llu epochs=%llu in=%llu "
                 "out=%llu dispatch=%llu read=%llu write=%llu\n",
                 static_cast<unsigned long long>(
                     inclusive->plan().residency.frame_capacity),
                 static_cast<unsigned long long>(
                     inclusive_stats.pipeline.residency.epoch_count),
                 static_cast<unsigned long long>(
                     inclusive_stats.pipeline.residency.page_in_count),
                 static_cast<unsigned long long>(
                     inclusive_stats.pipeline.residency.page_out_count),
                 static_cast<unsigned long long>(inclusive_stats.dispatches),
                 static_cast<unsigned long long>(
                     inclusive_stats.pipeline.residency.backing_read_bytes),
                 static_cast<unsigned long long>(
                     inclusive_stats.pipeline.residency.backing_write_bytes));
    return 5;
  }
  const PipelinePlan inclusive_plan = inclusive->plan();
  const std::uint64_t inclusive_hash = inclusive_stats.output_hash;
  const Status inclusive_warm_status = inclusive->run();
  if (!inclusive_warm_status) {
    std::fprintf(stderr, "virtual scan warm run reason=%.*s\n",
                 static_cast<int>(inclusive_warm_status.error().size()),
                 inclusive_warm_status.error().data());
    return 6;
  }
  const Stats inclusive_warm = inclusive->stats();
  const PipelinePlan inclusive_warm_plan = inclusive->plan();
  const bool inclusive_warm_matches =
      scan_matches(*inclusive_backing, values, true);
  const ResidencyStats &inclusive_warm_residency =
      inclusive_warm.pipeline.residency;
  const bool inclusive_warm_io_exact =
      inclusive_device_vsm
          ? inclusive_warm_residency.cache_hit_count == 0u &&
                inclusive_warm_residency.page_in_count == InclusivePages &&
                inclusive_warm_residency.backing_read_bytes == sizeof(values)
          : inclusive_warm_residency.cache_hit_count == InclusivePages &&
                inclusive_warm_residency.page_in_count == 0u &&
                inclusive_warm_residency.backing_read_bytes == 0u;
  if (!inclusive_warm_matches || !inclusive_warm_io_exact ||
      inclusive_warm.buffer_allocations != 0u ||
      inclusive_warm.output_hash != inclusive_hash ||
      inclusive_warm_plan.residency.identity_hi !=
          inclusive_plan.residency.identity_hi ||
      inclusive_warm_plan.residency.identity_lo !=
          inclusive_plan.residency.identity_lo) {
    std::array<std::uint32_t, ScanElements> warm_values{};
    const bool warm_observed = inclusive_backing->observe(
        std::as_writable_bytes(std::span{warm_values}));
    std::fprintf(
        stderr,
        "virtual scan warm hits=%llu in=%llu read=%llu alloc=%llu "
        "hash=%llx/%llx plan=%llx:%llx/%llx:%llx\n",
        static_cast<unsigned long long>(
            inclusive_warm.pipeline.residency.cache_hit_count),
        static_cast<unsigned long long>(
            inclusive_warm.pipeline.residency.page_in_count),
        static_cast<unsigned long long>(
            inclusive_warm.pipeline.residency.backing_read_bytes),
        static_cast<unsigned long long>(inclusive_warm.buffer_allocations),
        static_cast<unsigned long long>(inclusive_warm.output_hash),
        static_cast<unsigned long long>(inclusive_hash),
        static_cast<unsigned long long>(
            inclusive_warm_plan.residency.identity_hi),
        static_cast<unsigned long long>(
            inclusive_warm_plan.residency.identity_lo),
        static_cast<unsigned long long>(inclusive_plan.residency.identity_hi),
        static_cast<unsigned long long>(inclusive_plan.residency.identity_lo));
    if (warm_observed) {
      std::fprintf(stderr, "virtual scan warm values=%u,%u,%u,%u,%u,%u,%u,%u\n",
                   warm_values[0], warm_values[1], warm_values[2],
                   warm_values[3], warm_values[4], warm_values[5],
                   warm_values[6], warm_values[7]);
    }
    return 7;
  }

  auto exclusive_backing =
      std::make_shared<MemoryVirtualBacking>(sizeof(values), sizeof(values));
  auto exclusive_output =
      virtual_buffer<std::uint32_t>(ScanElements, exclusive_backing);
  auto exclusive =
      input && exclusive_output
          ? virtual_pipeline(*exclusive_program, *input, *exclusive_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!exclusive || !exclusive->run() ||
      !scan_matches(*exclusive_backing, values, false)) {
    return 8;
  }
  constexpr std::size_t ExclusivePayload = ScanFrameElements - 1u;
  constexpr std::size_t ExclusivePages =
      (ScanElements + ExclusivePayload - 1u) / ExclusivePayload;
  constexpr std::size_t ExclusiveEpochs =
      (ExclusivePages + FrameCapacity - 1u) / FrameCapacity;
  const Stats exclusive_stats = exclusive->stats();
  const bool exclusive_device_vsm =
      uses_device_vsm(backend, exclusive_stats.pipeline.residency);
  const std::size_t expected_exclusive_dispatches =
      exclusive_device_vsm
          ? 1u
          : ExclusiveEpochs * FrameCapacity * scan_dispatches_per_frame;
  if (exclusive->plan().residency.page_count != ExclusivePages ||
      exclusive_stats.pipeline.residency.epoch_count != ExclusiveEpochs ||
      exclusive_stats.pipeline.residency.page_in_count != ExclusivePages ||
      exclusive_stats.pipeline.residency.page_out_count != ExclusivePages ||
      exclusive_stats.dispatches != expected_exclusive_dispatches ||
      exclusive_stats.pipeline.residency.backing_write_bytes !=
          sizeof(values)) {
    return 9;
  }
  fixture.inclusive.emplace(std::move(inclusive).value());
  return 0;
}

} // namespace rund_node_test_virtual::product::scan

#include "support.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace rund_node_test_virtual::product::scan {

[[nodiscard]] int RunTieredCold(ScanFixture &fixture) {
  using namespace rund::compute;
  if (fixture.backend != Backend::Cpu && fixture.tiered_route &&
      *fixture.tiered_route) {
    constexpr std::size_t TieredElements = 97u;
    constexpr std::size_t TieredPages =
        (TieredElements + ExclusivePayload - 1u) / ExclusivePayload;
    std::array<std::uint32_t, TieredElements> tiered_values{};
    for (std::size_t index = 0u; index < tiered_values.size(); ++index) {
      tiered_values[index] = static_cast<std::uint32_t>(index % 11u + 1u);
    }
    auto tiered_input_backing =
        std::make_shared<PersistentScanBacking>(sizeof(tiered_values));
    auto tiered_output_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(tiered_values), sizeof(tiered_values));
    if (!tiered_input_backing->seed(std::as_bytes(std::span{tiered_values}))) {
      return 10;
    }
    auto tiered_input =
        virtual_buffer<std::uint32_t>(TieredElements, tiered_input_backing);
    auto tiered_output =
        virtual_buffer<std::uint32_t>(TieredElements, tiered_output_backing);
    constexpr std::uint64_t ScanPageBytes =
        ScanFrameElements * sizeof(std::uint32_t);
    const ResidencyConfig tiered_config{
        .device_resident_bytes = ScanPageBytes * 8u,
        .host_resident_bytes = ScanPageBytes * 14u,
    };
    auto tiered =
        tiered_input && tiered_output
            ? virtual_pipeline(*fixture.exclusive_program, *tiered_input,
                               *tiered_output, tiered_config)
            : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                  Reason::PipelineInvalid);
    const auto tiered_state =
        tiered ? detail::VirtualPipelineAccess::state(*tiered) : nullptr;
    const ScanGenerations tiered_generations_before =
        scan_generations(tiered_state);
    const std::uint64_t tiered_version_before =
        detail::VirtualBackingAccess::version(*tiered_output_backing);
    const BackingFacts tiered_facts_before = tiered_output_backing->facts();
    const Status tiered_cold =
        tiered ? tiered->run() : Status::fail(tiered.reason());
    const bool tiered_cold_matches =
        tiered && tiered_cold &&
        scan_matches(*tiered_output_backing, tiered_values, false);
    const BackingFacts tiered_facts = tiered_output_backing->facts();
    const std::uint64_t tiered_version =
        detail::VirtualBackingAccess::version(*tiered_output_backing);
    constexpr std::size_t TieredEpochs =
        (TieredPages + FrameCapacity - 1u) / FrameCapacity;
    constexpr std::size_t TieredBank0Terminals =
        ((TieredEpochs + 1u) / 2u) * 2u;
    constexpr std::size_t TieredBank1Terminals = (TieredEpochs / 2u) * 2u;
    const ScanGenerations tiered_generations = scan_generations(tiered_state);
    const bool tiered_generation_delta =
        tiered_generations[0u] ==
            tiered_generations_before[0u] + TieredBank0Terminals &&
        tiered_generations[1u] ==
            tiered_generations_before[1u] + TieredBank1Terminals;
    const bool tiered_transaction =
        tiered_facts.write_count == tiered_facts_before.write_count + 1u &&
        tiered_facts.write_bytes ==
            tiered_facts_before.write_bytes + sizeof(tiered_values) &&
        tiered_version == tiered_version_before + 1u;
    if (!tiered || tiered->plan().residency.frame_capacity != FrameCapacity ||
        tiered->plan().residency.page_count != TieredPages || !tiered_cold ||
        !tiered_cold_matches || !tiered_transaction ||
        !tiered_generation_delta) {
      std::fprintf(
          stderr,
          "virtual scan tiered cold prepared=%u reason=%.*s frames=%llu "
          "pages=%llu/%llu match=%u commits=%llu version=%llu/%llu "
          "generation=%llu/%llu,%llu/%llu\n",
          static_cast<unsigned>(static_cast<bool>(tiered)),
          static_cast<int>(tiered_cold.error().size()),
          tiered_cold.error().data(),
          static_cast<unsigned long long>(
              tiered ? tiered->plan().residency.frame_capacity : 0u),
          static_cast<unsigned long long>(
              tiered ? tiered->plan().residency.page_count : 0u),
          static_cast<unsigned long long>(TieredPages),
          static_cast<unsigned>(tiered_cold_matches),
          static_cast<unsigned long long>(tiered_facts.write_count),
          static_cast<unsigned long long>(tiered_version),
          static_cast<unsigned long long>(tiered_version_before),
          static_cast<unsigned long long>(tiered_generations[0u]),
          static_cast<unsigned long long>(tiered_generations_before[0u]),
          static_cast<unsigned long long>(tiered_generations[1u]),
          static_cast<unsigned long long>(tiered_generations_before[1u]));
      return 10;
    }
    const std::uint64_t populated = tiered_input_backing->read_bytes();
    const Status tiered_warm = tiered->run();
    const bool tiered_warm_matches =
        tiered_warm &&
        scan_matches(*tiered_output_backing, tiered_values, false);
    const Stats tiered_stats = tiered->stats();
    if (populated != sizeof(tiered_values) || !tiered_warm ||
        !tiered_warm_matches ||
        tiered_input_backing->read_bytes() != populated ||
        tiered_stats.pipeline.residency.backing_read_bytes != 0u ||
        tiered_stats.pipeline.residency.late_page_count != 0u ||
        tiered_stats.pipeline.residency.prefetch_count != 0u ||
        tiered_stats.buffer_allocations != 0u) {
      std::fprintf(
          stderr,
          "virtual scan tiered warm reason=%.*s match=%u populated=%llu/%llu "
          "reads=%llu read_bytes=%llu late=%llu prefetch=%llu alloc=%llu\n",
          static_cast<int>(tiered_warm.error().size()),
          tiered_warm.error().data(),
          static_cast<unsigned>(tiered_warm_matches),
          static_cast<unsigned long long>(populated),
          static_cast<unsigned long long>(sizeof(tiered_values)),
          static_cast<unsigned long long>(tiered_input_backing->read_bytes()),
          static_cast<unsigned long long>(
              tiered_stats.pipeline.residency.backing_read_bytes),
          static_cast<unsigned long long>(
              tiered_stats.pipeline.residency.late_page_count),
          static_cast<unsigned long long>(
              tiered_stats.pipeline.residency.prefetch_count),
          static_cast<unsigned long long>(tiered_stats.buffer_allocations));
      return 10;
    }
  }
  return 0;
}

[[nodiscard]] int RunTieredWide(ScanFixture &fixture) {
  using namespace rund::compute;
  if (fixture.backend != Backend::Cpu && fixture.tiered_route &&
      *fixture.tiered_route) {
    constexpr std::size_t WidePages = 129u;
    constexpr std::size_t WideFrameCapacity = 32u;
    constexpr std::size_t WideElements = ExclusivePayload * WidePages;
    constexpr std::size_t WidePageBytes =
        ScanFrameElements * sizeof(std::uint32_t);
    constexpr std::size_t WideHostOutputCapacity = 65u;
    constexpr std::size_t WideHostUnits = 388u;
    constexpr std::size_t WideDeviceUnits = 128u;
    std::array<std::uint32_t, WideElements> wide_values{};
    for (std::size_t index = 0u; index < wide_values.size(); ++index) {
      wide_values[index] = static_cast<std::uint32_t>(index % 13u + 1u);
    }
    auto wide_input_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(wide_values), WidePageBytes);
    auto wide_output_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(wide_values), WidePageBytes);
    if (!wide_input_backing->seed(std::as_bytes(std::span{wide_values}))) {
      return 14;
    }
    auto wide_input =
        virtual_buffer<std::uint32_t>(WideElements, wide_input_backing);
    auto wide_output =
        virtual_buffer<std::uint32_t>(WideElements, wide_output_backing);
    const ResidencyConfig wide_config{
        .device_resident_bytes = WidePageBytes * WideDeviceUnits,
        .host_resident_bytes = WidePageBytes * WideHostUnits,
    };
    auto wide =
        wide_input && wide_output
            ? virtual_pipeline(*fixture.exclusive_program, *wide_input,
                               *wide_output, wide_config)
            : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                  Reason::PipelineInvalid);
    const auto wide_state =
        wide ? detail::VirtualPipelineAccess::state(*wide) : nullptr;
    const std::uint64_t wide_host_output =
        wide_state != nullptr && wide_state->pipeline != nullptr &&
                wide_state->pipeline->residency_pool != nullptr
            ? wide_state->pipeline->residency_pool->layout
                  .host_output_frame_capacity
            : 0u;
    const BackingFacts wide_facts_before = wide_output_backing->facts();
    const std::uint64_t wide_version_before =
        detail::VirtualBackingAccess::version(*wide_output_backing);
    const Status wide_status = wide ? wide->run() : Status::fail(wide.reason());
    const bool wide_matches =
        wide_status && scan_matches(*wide_output_backing, wide_values, false);
    const BackingFacts wide_facts = wide_output_backing->facts();
    const std::uint64_t wide_version =
        detail::VirtualBackingAccess::version(*wide_output_backing);
    const Stats wide_stats = wide ? wide->stats() : Stats{};
    const bool wide_ok =
        wide && wide_status && wide_matches &&
        wide->plan().residency.page_count == WidePages &&
        wide->plan().residency.frame_capacity == WideFrameCapacity &&
        wide_host_output >= WideHostOutputCapacity &&
        wide_facts.write_count == wide_facts_before.write_count + 1u &&
        wide_facts.write_bytes ==
            wide_facts_before.write_bytes + sizeof(wide_values) &&
        wide_version == wide_version_before + 1u &&
        wide_stats.pipeline.residency.backing_write_bytes ==
            sizeof(wide_values);
    if (!wide_ok) {
      std::fprintf(stderr,
                   "virtual scan wide status=%u reason=%.*s frames=%llu/%u "
                   "pages=%llu/%u host=%llu/%u match=%u writes=%llu/%llu "
                   "version=%llu/%llu\n",
                   static_cast<unsigned>(wide_status.reason()),
                   static_cast<int>(wide_status.error().size()),
                   wide_status.error().data(),
                   static_cast<unsigned long long>(
                       wide ? wide->plan().residency.frame_capacity : 0u),
                   static_cast<unsigned>(WideFrameCapacity),
                   static_cast<unsigned long long>(
                       wide ? wide->plan().residency.page_count : 0u),
                   static_cast<unsigned>(WidePages),
                   static_cast<unsigned long long>(wide_host_output),
                   static_cast<unsigned>(WideHostOutputCapacity),
                   static_cast<unsigned>(wide_matches),
                   static_cast<unsigned long long>(wide_facts.write_count),
                   static_cast<unsigned long long>(wide_facts.write_bytes),
                   static_cast<unsigned long long>(wide_version),
                   static_cast<unsigned long long>(wide_version_before));
      return 14;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::scan

#include "support.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace rund_node_test_virtual::product::scan {

[[nodiscard]] int RunOverflow(ScanFixture &fixture) {
  using namespace rund::compute;
  if (fixture.backend != Backend::Cpu && fixture.tiered_route &&
      *fixture.tiered_route) {
    std::array<std::uint32_t, ScanElements> overflow_values{};
    overflow_values.fill(std::numeric_limits<std::uint32_t>::max() / 40u);
    auto overflow_input_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(overflow_values), ScanFrameElements * sizeof(std::uint32_t));
    auto overflow_output_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(overflow_values), sizeof(overflow_values));
    if (!overflow_input_backing->seed(
            std::as_bytes(std::span{overflow_values}))) {
      return 10;
    }
    auto overflow_input =
        virtual_buffer<std::uint32_t>(ScanElements, overflow_input_backing);
    auto overflow_output =
        virtual_buffer<std::uint32_t>(ScanElements, overflow_output_backing);
    auto overflow =
        overflow_input && overflow_output
            ? virtual_pipeline(*fixture.exclusive_program, *overflow_input,
                               *overflow_output, ResidencyConfig{})
            : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                  Reason::PipelineInvalid);
    const auto overflow_state =
        overflow ? detail::VirtualPipelineAccess::state(*overflow) : nullptr;
    std::array<std::byte, sizeof(overflow_values)> overflow_bytes_before{};
    const bool overflow_before_observed =
        overflow_output_backing->observe(overflow_bytes_before);
    const std::uint64_t version_before =
        detail::VirtualBackingAccess::version(*overflow_output_backing);
    const PublicationStats publication_before =
        overflow ? overflow->stats().publication : PublicationStats{};
    const ScanGenerations generations_before = scan_generations(overflow_state);
    const ScanControls controls_before = scan_controls(overflow_state);
    const Status overflow_status =
        overflow ? overflow->run() : Status::fail(overflow.reason());
    std::array<std::byte, sizeof(overflow_values)> first_bytes{};
    const bool first_bytes_observed =
        overflow_output_backing->observe(first_bytes);
    const Stats first = overflow ? overflow->stats() : Stats{};
    const auto first_output = overflow_output_backing->facts();
    const std::uint64_t first_version =
        detail::VirtualBackingAccess::version(*overflow_output_backing);
    const ScanGenerations first_generations = scan_generations(overflow_state);
    const ScanControls first_controls = scan_controls(overflow_state);
    const Status retry_status =
        overflow ? overflow->run() : Status::fail(overflow.reason());
    std::array<std::byte, sizeof(overflow_values)> retry_bytes{};
    const bool retry_bytes_observed =
        overflow_output_backing->observe(retry_bytes);
    const Stats retry = overflow ? overflow->stats() : Stats{};
    const auto retry_output = overflow_output_backing->facts();
    const std::uint64_t retry_version =
        detail::VirtualBackingAccess::version(*overflow_output_backing);
    const ScanGenerations retry_generations = scan_generations(overflow_state);
    const ScanControls retry_controls = scan_controls(overflow_state);
    const bool first_ok =
        overflow && overflow_status.reason() == Reason::ScanSumOverflow &&
        first.pipeline.residency.failed_page == 2u &&
        first_output.write_count == 0u && first_output.write_bytes == 0u;
    const bool retry_ok =
        overflow && retry_status.reason() == Reason::ScanSumOverflow &&
        retry.pipeline.residency.failed_page == 2u &&
        retry_output.write_count == 0u && retry_output.write_bytes == 0u;
    const bool unchanged =
        overflow_before_observed && first_bytes_observed &&
        retry_bytes_observed &&
        std::memcmp(overflow_bytes_before.data(), first_bytes.data(),
                    overflow_bytes_before.size()) == 0 &&
        std::memcmp(overflow_bytes_before.data(), retry_bytes.data(),
                    overflow_bytes_before.size()) == 0 &&
        first_version == version_before && retry_version == version_before &&
        first.publication == publication_before &&
        retry.publication == publication_before &&
        first_generations == generations_before &&
        retry_generations == generations_before &&
        first_controls == controls_before && retry_controls == controls_before;
    if (!first_ok || !retry_ok || !unchanged) {
      std::fprintf(
          stderr,
          "virtual scan overflow first=%u/%u page=%llu retry=%u/%u "
          "page=%llu writes=%llu/%llu version=%llu/%llu "
          "control=%llu:%u,%llu:%u retry=%llu:%u,%llu:%u\n",
          static_cast<unsigned>(overflow_status.reason()),
          static_cast<unsigned>(overflow_status.code()),
          static_cast<unsigned long long>(first.pipeline.residency.failed_page),
          static_cast<unsigned>(retry_status.reason()),
          static_cast<unsigned>(retry_status.code()),
          static_cast<unsigned long long>(retry.pipeline.residency.failed_page),
          static_cast<unsigned long long>(first_output.write_count),
          static_cast<unsigned long long>(retry_output.write_count),
          static_cast<unsigned long long>(version_before),
          static_cast<unsigned long long>(retry_version),
          static_cast<unsigned long long>(first_controls[0u].generation),
          static_cast<unsigned>(first_controls[0u].parity),
          static_cast<unsigned long long>(first_controls[1u].generation),
          static_cast<unsigned>(first_controls[1u].parity),
          static_cast<unsigned long long>(retry_controls[0u].generation),
          static_cast<unsigned>(retry_controls[0u].parity),
          static_cast<unsigned long long>(retry_controls[1u].generation),
          static_cast<unsigned>(retry_controls[1u].parity));
      return 11;
    }
  }
  return 0;
}

[[nodiscard]] int RunUnknown(ScanFixture &fixture) {
  using namespace rund::compute;
  const auto values = fixture.values;
  auto &inclusive_program = fixture.inclusive_program;
  if (fixture.backend != Backend::Cpu && fixture.tiered_route &&
      *fixture.tiered_route) {
    auto unknown_input_backing = std::make_shared<MemoryVirtualBacking>(
        sizeof(values), ScanFrameElements * sizeof(std::uint32_t));
    auto unknown_output_backing =
        std::make_shared<MemoryVirtualBacking>(sizeof(values), sizeof(values));
    if (!unknown_input_backing->seed(std::as_bytes(std::span{values}))) {
      return 13;
    }
    auto unknown_input =
        virtual_buffer<std::uint32_t>(ScanElements, unknown_input_backing);
    auto unknown_output =
        virtual_buffer<std::uint32_t>(ScanElements, unknown_output_backing);
    auto unknown =
        unknown_input && unknown_output
            ? virtual_pipeline(*inclusive_program, *unknown_input,
                               *unknown_output, ResidencyConfig{})
            : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                  Reason::PipelineInvalid);
    const auto unknown_state =
        unknown ? detail::VirtualPipelineAccess::state(*unknown) : nullptr;
    std::array<std::byte, sizeof(values)> unknown_bytes_before{};
    const bool unknown_before_observed =
        unknown_output_backing->observe(unknown_bytes_before);
    const BackingFacts unknown_facts_before = unknown_output_backing->facts();
    const PublicationStats unknown_publication_before =
        unknown ? unknown->stats().publication : PublicationStats{};
    const ScanGenerations unknown_generations_before =
        scan_generations(unknown_state);
    const ScanControls unknown_controls_before = scan_controls(unknown_state);
    unknown_output_backing->fail_next_transaction_unknown();
    const Status unknown_status =
        unknown ? unknown->run() : Status::fail(unknown.reason());
    const Stats unknown_first = unknown ? unknown->stats() : Stats{};
    std::array<std::byte, sizeof(values)> unknown_bytes_after{};
    const bool unknown_after_observed =
        unknown_output_backing->observe(unknown_bytes_after);
    const BackingFacts unknown_facts = unknown_output_backing->facts();
    const ScanGenerations unknown_generations = scan_generations(unknown_state);
    const ScanControls unknown_controls = scan_controls(unknown_state);
    const Status unknown_retry_status =
        unknown ? unknown->run() : Status::fail(unknown.reason());
    const bool unknown_failure =
        unknown && unknown_status.reason() == Reason::DeviceLost &&
        unknown_retry_status.reason() == Reason::PipelinePoisoned &&
        unknown_state != nullptr &&
        unknown_state->phase == detail::VirtualPipelinePhase::Poisoned &&
        unknown_output_backing->transaction_quarantined();
    const bool unknown_unchanged =
        unknown_before_observed && unknown_after_observed &&
        std::memcmp(unknown_bytes_before.data(), unknown_bytes_after.data(),
                    unknown_bytes_before.size()) == 0 &&
        unknown_facts.write_count == unknown_facts_before.write_count &&
        unknown_facts.write_bytes == unknown_facts_before.write_bytes &&
        unknown_publication_before.generation ==
            unknown_first.publication.generation &&
        unknown_publication_before.commit_count ==
            unknown_first.publication.commit_count &&
        unknown_publication_before.discard_count ==
            unknown_first.publication.discard_count &&
        unknown_generations == unknown_generations_before &&
        unknown_controls[0u].poisoned && unknown_controls[1u].poisoned &&
        unknown_controls[0u].generation >
            unknown_controls_before[0u].generation &&
        unknown_controls[1u].generation >
            unknown_controls_before[1u].generation;
    if (!unknown_failure || !unknown_unchanged) {
      std::fprintf(
          stderr,
          "virtual scan transaction unknown status=%u retry=%u writes=%llu "
          "bytes=%llu/%llu version=%llu/%llu generations=%llu/%llu,%llu/%llu "
          "control=%u,%u quarantined=%u phase=%u\n",
          static_cast<unsigned>(unknown_status.reason()),
          static_cast<unsigned>(unknown_retry_status.reason()),
          static_cast<unsigned long long>(unknown_facts.write_count),
          static_cast<unsigned long long>(unknown_facts.write_bytes),
          static_cast<unsigned long long>(unknown_facts_before.write_bytes),
          static_cast<unsigned long long>(unknown_first.publication.generation),
          static_cast<unsigned long long>(
              unknown_publication_before.generation),
          static_cast<unsigned long long>(unknown_generations[0u]),
          static_cast<unsigned long long>(unknown_generations_before[0u]),
          static_cast<unsigned long long>(unknown_generations[1u]),
          static_cast<unsigned long long>(unknown_generations_before[1u]),
          static_cast<unsigned>(unknown_controls[0u].poisoned),
          static_cast<unsigned>(unknown_controls[1u].poisoned),
          static_cast<unsigned>(
              unknown_output_backing->transaction_quarantined()),
          static_cast<unsigned>(unknown_state == nullptr
                                    ? detail::VirtualPipelinePhase::Ready
                                    : unknown_state->phase));
      return 13;
    }
  }
  return 0;
}

[[nodiscard]] int RunPoisonRecovery(ScanFixture &fixture) {
  using namespace rund::compute;
  const auto values = fixture.values;
  auto &inclusive_backing = fixture.inclusive_backing;
  auto &inclusive_program = fixture.inclusive_program;
  auto &inclusive = fixture.inclusive;
  inclusive_backing->fail_next_write_after(sizeof(std::uint32_t),
                                           Reason::BackendFailed);
  const Status poisoned_write = inclusive->run();
  // CPU publishes this backing failure while draining the first epoch. Both
  // internal scan executions belong to one already completed physical epoch.
  const Stats failed_write_stats = inclusive->stats();
  if (fixture.backend == Backend::Cpu &&
      failed_write_stats.pipeline.residency.epoch_count != 1u) {
    std::fprintf(stderr, "virtual scan failed drain epochs=%llu expected=1\n",
                 static_cast<unsigned long long>(
                     failed_write_stats.pipeline.residency.epoch_count));
    return 18;
  }
  auto poisoned_view =
      virtual_buffer<std::uint32_t>(ScanElements, inclusive_backing);
  auto probe_backing =
      std::make_shared<MemoryVirtualBacking>(sizeof(values), sizeof(values));
  auto probe_output =
      virtual_buffer<std::uint32_t>(ScanElements, probe_backing);
  auto poison_probe =
      poisoned_view && probe_output
          ? virtual_pipeline(*inclusive_program, *poisoned_view, *probe_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                Reason::PipelineInvalid);
  const Status probe_status =
      poison_probe ? poison_probe->run() : Status::fail(poison_probe.reason());
  const Status recovered = inclusive->run();
  const bool recovered_match =
      recovered && scan_matches(*inclusive_backing, values, true);
  if (poisoned_write.reason() != Reason::BackendFailed || !poison_probe ||
      probe_status.reason() != Reason::BufferPoisoned || !recovered_match ||
      !inclusive_backing->tail_poisoned()) {
    std::fprintf(stderr,
                 "virtual scan poison write=%u probe=%u recovered=%u match=%u "
                 "write_failures=%llu partial=%llu\n",
                 static_cast<unsigned>(poisoned_write.reason()),
                 static_cast<unsigned>(probe_status.reason()),
                 static_cast<unsigned>(recovered.reason()),
                 static_cast<unsigned>(recovered_match),
                 static_cast<unsigned long long>(
                     inclusive_backing->facts().write_failure_count),
                 static_cast<unsigned long long>(
                     inclusive_backing->facts().partial_write_bytes));
    return 12;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::scan

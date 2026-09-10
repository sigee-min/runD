#include "local.hpp"

#include "../golden.hpp"

#include "src/compute/virtual/state.hpp"

#include <array>
#include <cstdio>
#include <cstring>

namespace rund_node_test_virtual::product::failure {

int CheckDeviceVsmFailure(FailureFixture &fixture) {
  using namespace rund::compute;
  auto &prepared = *fixture.prepared;
  auto &input_backing = *fixture.input_backing;
  auto &output_backing = *fixture.output_backing;
  auto &program = *fixture.program;

  const auto &write_fault_state =
      detail::VirtualPipelineAccess::state(prepared);
  if (write_fault_state == nullptr) {
    return 20;
  }
  const bool prior_write_fault_device_vsm_required =
      write_fault_state->geometry.device_vsm_required;
  // The remainder of this fixture is the explicit DeviceVsm partial-output
  // and recovery contract. Keep the ordinary Persistent read-fault coverage
  // above, then opt this fault block into its established owner semantics.
  write_fault_state->geometry.device_vsm_required = true;
  constexpr std::size_t partial_bytes = 3u * sizeof(std::int32_t);
  const BackingFacts before_write_failure = output_backing.facts();
  output_backing.fail_next_write_after(partial_bytes, Reason::BackendFailed);
  const Status write_failed = prepared.run();
  const BackingFacts after_write_failure = output_backing.facts();
  const Stats failed_stats = prepared.stats();
  if (write_failed.reason() != Reason::BackendFailed ||
      after_write_failure.write_failure_count != 1u ||
      after_write_failure.write_count != before_write_failure.write_count ||
      after_write_failure.partial_write_bytes != partial_bytes ||
      // The failed callback must not publish a partial output page on either
      // Persistent fixed-W or the explicit DeviceVsm route.
      failed_stats.pipeline.residency.failed_page != 0u ||
      failed_stats.pipeline.residency.backing_write_bytes != 0u) {
    std::fprintf(
        stderr,
        "virtual write failure reason=%u failures=%llu writes=%llu "
        "partial=%llu failed_page=%llu bytes=%llu\n",
        static_cast<unsigned>(write_failed.reason()),
        static_cast<unsigned long long>(
            after_write_failure.write_failure_count),
        static_cast<unsigned long long>(after_write_failure.write_count),
        static_cast<unsigned long long>(
            after_write_failure.partial_write_bytes),
        static_cast<unsigned long long>(
            failed_stats.pipeline.residency.failed_page),
        static_cast<unsigned long long>(
            failed_stats.pipeline.residency.backing_write_bytes));
    return 6;
  }

  // Poison belongs to the backing authority, not the first VirtualBuffer
  // wrapper. A fresh typed view over a partially written backing must remain
  // unreadable until the original Pipeline overwrites every logical page.
  auto poisoned_view =
      virtual_buffer<std::int32_t>(LogicalElements, fixture.output_backing);
  auto recovery_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto recovery_output =
      virtual_buffer<std::int32_t>(LogicalElements, recovery_backing);
  auto poison_probe =
      poisoned_view && recovery_output
          ? virtual_pipeline(program, *poisoned_view, *recovery_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  const BackingFacts input_before_insufficient = input_backing.facts();
  const BackingFacts output_before_insufficient = output_backing.facts();
  if (!poison_probe || poison_probe->run().reason() != Reason::BufferPoisoned ||
      prepared.run(0u).reason() != Reason::BufferPoisoned ||
      prepared.run(LogicalElements - 1u).reason() != Reason::BufferPoisoned ||
      input_backing.facts().read_count !=
          input_before_insufficient.read_count ||
      output_backing.facts().write_count !=
          output_before_insufficient.write_count ||
      !prepared.run(LogicalElements)) {
    return 7;
  }

  std::array<std::int32_t, LogicalElements> observed{};
  if (!output_backing.observe(std::as_writable_bytes(std::span{observed})) ||
      !GoldenMatches(observed) || HashValues(observed) != GoldenHash ||
      !input_backing.tail_poisoned() || !output_backing.tail_poisoned()) {
    return 8;
  }
  const BackingFacts input_facts = input_backing.facts();
  const BackingFacts output_facts = output_backing.facts();
  if (input_facts.read_failure_count != 1u ||
      output_facts.write_failure_count != 1u ||
      output_facts.partial_write_bytes != partial_bytes ||
      output_facts.observation_count != 1u) {
    return 9;
  }
  write_fault_state->geometry.device_vsm_required =
      prior_write_fault_device_vsm_required;
  return 0;
}

} // namespace rund_node_test_virtual::product::failure

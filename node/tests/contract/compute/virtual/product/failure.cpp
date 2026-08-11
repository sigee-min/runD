#include "local.hpp"

#include "backing.hpp"
#include "golden.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {

int CheckProductBackingFailure(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-product-backing-retry", PageElements,
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
  SeedInput(seeded);
  if (!input_backing->seed(std::as_bytes(std::span{seeded}))) {
    return 3;
  }
  auto input = virtual_buffer<std::int32_t>(LogicalElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(LogicalElements, output_backing);
  auto prepared =
      input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return 4;
  }

  input_backing->fail_next_read(Reason::BackendFailed);
  const Status read_failed = prepared->run();
  const BackingFacts after_read_failure = input_backing->facts();
  if (read_failed.reason() != Reason::BackendFailed ||
      after_read_failure.read_failure_count != 1u ||
      after_read_failure.read_count != 0u ||
      prepared->stats().pipeline.residency.failed_page != 0u ||
      !prepared->run()) {
    return 5;
  }

  constexpr std::size_t partial_bytes = 3u * sizeof(std::int32_t);
  output_backing->fail_next_write_after(partial_bytes, Reason::BackendFailed);
  const Status write_failed = prepared->run();
  const BackingFacts after_write_failure = output_backing->facts();
  const Stats failed_stats = prepared->stats();
  if (write_failed.reason() != Reason::BackendFailed ||
      after_write_failure.write_failure_count != 1u ||
      after_write_failure.write_count != PageCount ||
      after_write_failure.partial_write_bytes != partial_bytes ||
      // Deferred dirty writeback first touches backing when the next epoch
      // evicts the oldest frame, not while the first epoch is computing.
      failed_stats.pipeline.residency.failed_page != FrameCapacity ||
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
      virtual_buffer<std::int32_t>(LogicalElements, output_backing);
  auto recovery_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto recovery_output =
      virtual_buffer<std::int32_t>(LogicalElements, recovery_backing);
  auto poison_probe =
      poisoned_view && recovery_output
          ? virtual_pipeline(*program, *poisoned_view, *recovery_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  const BackingFacts input_before_insufficient = input_backing->facts();
  const BackingFacts output_before_insufficient = output_backing->facts();
  if (!poison_probe || poison_probe->run().reason() != Reason::BufferPoisoned ||
      prepared->run(0u).reason() != Reason::BufferPoisoned ||
      prepared->run(LogicalElements - 1u).reason() != Reason::BufferPoisoned ||
      input_backing->facts().read_count !=
          input_before_insufficient.read_count ||
      output_backing->facts().write_count !=
          output_before_insufficient.write_count ||
      !prepared->run(LogicalElements)) {
    return 7;
  }

  std::array<std::int32_t, LogicalElements> observed{};
  if (!output_backing->observe(std::as_writable_bytes(std::span{observed})) ||
      !GoldenMatches(observed) || HashValues(observed) != GoldenHash ||
      !input_backing->tail_poisoned() || !output_backing->tail_poisoned()) {
    return 8;
  }
  const BackingFacts input_facts = input_backing->facts();
  const BackingFacts output_facts = output_backing->facts();
  return input_facts.read_failure_count == 1u &&
                 output_facts.write_failure_count == 1u &&
                 output_facts.partial_write_bytes == partial_bytes &&
                 output_facts.observation_count == 1u
             ? 0
             : 9;
}

} // namespace rund_node_test_virtual::product

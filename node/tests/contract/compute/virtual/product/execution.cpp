#include "local.hpp"

#include "backing.hpp"
#include "evidence.hpp"
#include "golden.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"
#include "../../allocation.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {
namespace {

[[nodiscard]] bool
same_counter_shape(const rund::compute::MemoryCounter left,
                   const rund::compute::MemoryCounter right) {
  return left.current == right.current && left.peak == right.peak &&
         left.budget == right.budget;
}

[[nodiscard]] bool same_fixed_memory(const rund::compute::MemoryStats &left,
                                     const rund::compute::MemoryStats &right) {
  return left.backend == right.backend && left.scope == right.scope &&
         same_counter_shape(left.host, right.host) &&
         same_counter_shape(left.frame, right.frame) &&
         same_counter_shape(left.tile, right.tile) &&
         same_counter_shape(left.resident, right.resident) &&
         same_counter_shape(left.staging, right.staging) &&
         same_counter_shape(left.device, right.device) &&
         same_counter_shape(left.transfer, right.transfer);
}

} // namespace

int CheckProductExecution(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-product-execution", PageElements,
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

  const PipelinePlan cold_plan = prepared->plan();
  const void *const input_identity = input_backing->identity();
  const void *const output_identity = output_backing->identity();
  const Status cold = prepared->run();
  if (!cold) {
    const BackingFacts input_facts = input_backing->facts();
    const BackingFacts output_facts = output_backing->facts();
    const Stats failed = prepared->stats();
    std::fprintf(
        stderr,
        "virtual product cold backend=%u reason=%u reads=%llu "
        "writes=%llu loads=%llu epochs=%llu in=%llu out=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(cold.reason()),
        static_cast<unsigned long long>(input_facts.read_count),
        static_cast<unsigned long long>(output_facts.write_count),
        static_cast<unsigned long long>(
            failed.pipeline.residency.page_in_count),
        static_cast<unsigned long long>(failed.pipeline.residency.epoch_count),
        static_cast<unsigned long long>(
            failed.pipeline.residency.backing_read_bytes),
        static_cast<unsigned long long>(
            failed.pipeline.residency.backing_write_bytes));
    return 5;
  }
  const MemoryStats cold_memory = prepared->memory();
  if (input_backing->facts().observation_count != 0u ||
      output_backing->facts().observation_count != 0u) {
    return 6;
  }
  if (prepared->begin_samples().reason() != Reason::Ok ||
      prepared->begin_samples().reason() != Reason::ProfileBusy) {
    return 7;
  }
  if (prepared->profile().reason() != Reason::ProfileBusy) {
    return 7;
  }

  node_compute_allocation::Start();
  bool warm_ok = true;
  for (std::size_t sample = 0u; sample < WarmRuns; ++sample) {
    if (!prepared->run()) {
      warm_ok = false;
      break;
    }
  }
  node_compute_allocation::Stop();
  const std::uint64_t warm_allocations = node_compute_allocation::Count();
  if (!warm_ok) {
    return 8;
  }
  if (prepared->end_samples().reason() != Reason::Ok ||
      prepared->end_samples().reason() != Reason::ProfileInvalid) {
    return 9;
  }

  std::array<std::int32_t, LogicalElements> observed{};
  if (!output_backing->observe(std::as_writable_bytes(std::span{observed}))) {
    return 10;
  }
  const auto profile = prepared->profile();
  if (!profile) {
    return 11;
  }
  const Stats final_run = prepared->stats();
  const MemoryStats final_memory = prepared->memory();
  const PipelinePlan final_plan = prepared->plan();
  const bool profile_matches =
      profile->execution() == final_run && profile->memory() == final_memory;
  const bool same_capacity = cold_plan == final_plan &&
                             input_identity == input_backing->identity() &&
                             output_identity == output_backing->identity() &&
                             same_fixed_memory(cold_memory, final_memory);
  const ProductExecutionEvidence evidence{
      .final_run = final_run,
      .input_cohort = input_backing->facts(),
      .output_cohort = output_backing->facts(),
      .warm_host_allocations = warm_allocations,
      .observed_hash = HashValues(observed),
      .same_capacity = same_capacity,
      .tail_poisoned =
          input_backing->tail_poisoned() && output_backing->tail_poisoned(),
      .profile_matches = profile_matches,
  };
  if (!GoldenMatches(observed) || !ProductExecutionMatches(evidence)) {
    const auto &r = final_run.pipeline.residency;
    std::fprintf(
        stderr,
        "virtual product evidence backend=%u alloc=%llu hash=%llx "
        "out=%llx pages=%llu/%llu epochs=%llu in=%llu outb=%llu "
        "io=%llu/%llu submit=%llu transfer=%llu/%llu samples=%u/%u "
        "claim=%llu/%llu reads=%llu writes=%llu observe=%llu same=%u tail=%u "
        "profile=%u resident=%llu/%llu cold=%llu/%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(warm_allocations),
        static_cast<unsigned long long>(evidence.observed_hash),
        static_cast<unsigned long long>(final_run.output_hash),
        static_cast<unsigned long long>(r.page_in_count),
        static_cast<unsigned long long>(r.page_out_count),
        static_cast<unsigned long long>(r.epoch_count),
        static_cast<unsigned long long>(r.backing_read_bytes),
        static_cast<unsigned long long>(r.backing_write_bytes),
        static_cast<unsigned long long>(final_run.uploaded_bytes),
        static_cast<unsigned long long>(final_run.downloaded_bytes),
        static_cast<unsigned long long>(final_run.command_submits),
        static_cast<unsigned long long>(
            final_run.transfer_submissions.host_to_device),
        static_cast<unsigned long long>(
            final_run.transfer_submissions.device_to_host),
        r.sampled_runs, r.allocation_free_runs,
        static_cast<unsigned long long>(final_run.pipeline.claim_ns),
        static_cast<unsigned long long>(
            final_run.pipeline.claim_conflict_count),
        static_cast<unsigned long long>(evidence.input_cohort.read_count),
        static_cast<unsigned long long>(evidence.output_cohort.write_count),
        static_cast<unsigned long long>(
            evidence.output_cohort.observation_count),
        static_cast<unsigned>(same_capacity),
        static_cast<unsigned>(evidence.tail_poisoned),
        static_cast<unsigned>(profile_matches),
        static_cast<unsigned long long>(final_memory.resident.current),
        static_cast<unsigned long long>(final_memory.resident.peak),
        static_cast<unsigned long long>(cold_memory.resident.current),
        static_cast<unsigned long long>(cold_memory.resident.peak));
    return 12;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product

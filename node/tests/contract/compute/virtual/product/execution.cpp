#include "local.hpp"

#include "backing.hpp"
#include "evidence.hpp"
#include "golden.hpp"
#include "model.hpp"
#include "route.hpp"

#include "../../../target/selection.hpp"
#include "../../allocation.hpp"

#include "src/compute/virtual/run/execution.hpp"
#include "src/compute/virtual/state.hpp"

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
  constexpr std::size_t PersistentBytes = FrameBytes * 2u;
  const ResidencyConfig config =
      backend == Backend::Cpu
          ? ResidencyConfig{}
          : ResidencyConfig{.device_resident_bytes = PersistentBytes,
                            .host_resident_bytes = ResidencyPageBytes * 6u};
  auto prepared =
      input && output
          ? virtual_pipeline(*program, *input, *output, config)
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return 4;
  }
  const auto state = detail::VirtualPipelineAccess::state(*prepared);
  if (state == nullptr || state->pipeline == nullptr ||
      state->pipeline->device == nullptr) {
    return 4;
  }
  ProductRouteObservation route_observation{};
  ProductRouteScope route_scope{*opened, route_observation};
  if (!route_scope) {
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
  ResolveProductRoute(route_observation, backend, true);
  const ProductExecutionEvidence evidence{
      .route_kind = route_observation.kind,
      .owner_mask = route_observation.accepted_owner_mask,
      .accepted_owner_count = route_observation.accepted_owner_count,
      .final_run = final_run,
      .input_cohort = input_backing->facts(),
      .output_cohort = output_backing->facts(),
      .warm_host_allocations = warm_allocations,
      .observed_hash = HashValues(observed),
      .golden_matches = GoldenMatches(observed),
      .same_capacity = same_capacity,
      .tail_poisoned =
          input_backing->tail_poisoned() && output_backing->tail_poisoned(),
      .profile_matches = profile_matches,
  };
  if (!ProductExecutionMatches(evidence)) {
    const auto &r = final_run.pipeline.residency;
    std::fprintf(
        stderr,
        "virtual product evidence backend=%u alloc=%llu hash=%llx "
        "out=%llx pages=%llu/%llu epochs=%llu hits=%llu evict=%llu "
        "prefetch=%llu late=%llu in=%llu outb=%llu io-pages=%llu/%llu "
        "io=%llu/%llu submit=%llu inflight=%llu transfer=%llu/%llu "
        "samples=%u/%u "
        "claim=%llu/%llu reads=%llu writes=%llu observe=%llu same=%u tail=%u "
        "profile=%u resident=%llu/%llu cold=%llu/%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(warm_allocations),
        static_cast<unsigned long long>(evidence.observed_hash),
        static_cast<unsigned long long>(final_run.output_hash),
        static_cast<unsigned long long>(r.page_in_count),
        static_cast<unsigned long long>(r.page_out_count),
        static_cast<unsigned long long>(r.epoch_count),
        static_cast<unsigned long long>(r.cache_hit_count),
        static_cast<unsigned long long>(r.eviction_count),
        static_cast<unsigned long long>(r.prefetch_count),
        static_cast<unsigned long long>(r.late_page_count),
        static_cast<unsigned long long>(r.backing_read_bytes),
        static_cast<unsigned long long>(r.backing_write_bytes),
        static_cast<unsigned long long>(r.page_in_bytes),
        static_cast<unsigned long long>(r.page_out_bytes),
        static_cast<unsigned long long>(final_run.uploaded_bytes),
        static_cast<unsigned long long>(final_run.downloaded_bytes),
        static_cast<unsigned long long>(final_run.command_submits),
        static_cast<unsigned long long>(final_run.command_inflight_peak),
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

  if (backend != Backend::Cpu) {
    constexpr std::uint64_t Q1Elements = FrameCapacity * PageElements;
    constexpr std::uint64_t Q1Bytes = Q1Elements * sizeof(std::int32_t);
    auto fault_input_backing =
        std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
    auto fault_output_backing =
        std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
    auto fault_input =
        virtual_buffer<std::int32_t>(LogicalElements, fault_input_backing);
    auto fault_output =
        virtual_buffer<std::int32_t>(LogicalElements, fault_output_backing);
    auto faulted =
        fault_input && fault_output
            ? virtual_pipeline(*program, *fault_input, *fault_output,
                               ResidencyConfig{})
            : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                  Reason::PipelineInvalid);
    if (!fault_input_backing->seed(std::as_bytes(std::span{seeded})) ||
        !faulted) {
      return 13;
    }
    const BackingFacts before_abandon = fault_output_backing->facts();
    rund::compute::detail::inject_virtual_execution_close_failure_once();
    const Status abandoned = faulted->run(Q1Elements);
    const BackingFacts after_abandon = fault_output_backing->facts();

    // The physical write happened before the injected close contradiction,
    // but no backing generation may be published. A shorter retry must still
    // observe recovery poison; only a complete Q=1 overwrite may clear it.
    auto recovery = virtual_pipeline(*program, *fault_input, *fault_output,
                                     ResidencyConfig{});
    const Status incomplete = recovery ? recovery->run(PageElements)
                                       : Status::fail(Reason::PipelineInvalid);
    const Status retried = recovery ? recovery->run(Q1Elements)
                                    : Status::fail(Reason::PipelineInvalid);
    std::array<std::int32_t, LogicalElements> recovered{};
    const bool observed_recovery = fault_output_backing->observe(
        std::as_writable_bytes(std::span{recovered}));
    bool exact_prefix = observed_recovery;
    for (std::size_t index = 0u; index < Q1Elements && exact_prefix; ++index) {
      exact_prefix = recovered[index] == (seeded[index] + 5) * 3;
    }
    if (abandoned.reason() != Reason::PipelineInvalid ||
        after_abandon.write_count - before_abandon.write_count !=
            FrameCapacity ||
        after_abandon.write_bytes - before_abandon.write_bytes != Q1Bytes ||
        incomplete.reason() != Reason::BufferPoisoned || !retried ||
        !exact_prefix) {
      std::fprintf(
          stderr,
          "virtual execution abandon backend=%u reason=%u writes=%llu/%llu "
          "incomplete=%u retry=%u prefix=%u\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned>(abandoned.reason()),
          static_cast<unsigned long long>(after_abandon.write_count -
                                          before_abandon.write_count),
          static_cast<unsigned long long>(after_abandon.write_bytes -
                                          before_abandon.write_bytes),
          static_cast<unsigned>(incomplete.reason()),
          static_cast<unsigned>(retried.reason()),
          static_cast<unsigned>(exact_prefix));
      return 14;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product

#include "evidence.hpp"

#include "../../../hash/fnv.hpp"
#include "../../pipeline/local.hpp"

#include <limits>

namespace rund::compute::detail {
namespace {

[[nodiscard]] std::uint64_t
virtual_graph_hash(const VirtualPipelineState &state,
                   const std::uint64_t physical) noexcept {
  ::rund::node::hash_detail::Fnv hash{
      ::rund::node::hash_detail::kFnvStandardOffset};
  constexpr char domain[] = "rund.compute.virtual.pipeline.v1";
  hash.Bytes(reinterpret_cast<const std::uint8_t *>(domain),
             sizeof(domain) - 1u);
  hash.Number(physical);
  hash.Number(state.pipeline->plan.residency.logical_bytes);
  hash.Number(state.pipeline->plan.residency.page_bytes);
  hash.Number(state.pipeline->plan.residency.page_count);
  hash.Number(state.pipeline->plan.residency.frame_capacity);
  hash.Number(state.pipeline->plan.residency.identity_hi);
  hash.Number(state.pipeline->plan.residency.identity_lo);
  return hash.Finish();
}

[[nodiscard]] bool allocation_free(const Stats &stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         stats.command_capacity_rejections == 0u &&
         stats.shader_compile_ns == 0u && stats.spirv_compile_ns == 0u &&
         stats.pipeline_create_ns == 0u && stats.descriptor_setup_ns == 0u &&
         stats.pipeline.claim_ns == 0u &&
         stats.pipeline.claim_conflict_count == 0u;
}

void increment(std::uint32_t &value) noexcept {
  if (value != std::numeric_limits<std::uint32_t>::max()) {
    ++value;
  }
}

} // namespace

Stats begin_virtual_run_evidence(const VirtualPipelineState &state,
                                 const std::uint64_t active_count) noexcept {
  const ResidencyStats previous = state.stats.pipeline.residency;
  Stats stats{
      .backend = state.pipeline->device->backend,
      .pipeline =
          PipelineStats{
              .step_count = state.pipeline->logical_step_count,
              .resource_count = state.pipeline->resources.size(),
              .barrier_count = state.pipeline->plan.barrier_count,
              .failed_step_index = PipelineStats::no_failed_step,
              .preparation_evidence = state.stats.pipeline.preparation_evidence,
              .prepared_template_count =
                  state.pipeline->plan.prepared_template_count,
              .prepared_command_count =
                  state.pipeline->plan.prepared_command_count,
          },
  };
  stats.pipeline.residency = ResidencyStats{
      .logical_bytes = state.pipeline->plan.residency.logical_bytes,
      .active_count = active_count,
      .page_bytes = state.pipeline->plan.residency.page_bytes,
      .page_count = state.pipeline->plan.residency.page_count,
      .frame_capacity = state.pipeline->plan.residency.frame_capacity,
      .sampled_runs = previous.sampled_runs,
      .allocation_free_runs = previous.allocation_free_runs,
      .plan_identity_hi = state.pipeline->plan.residency.identity_hi,
      .plan_identity_lo = state.pipeline->plan.residency.identity_lo,
  };
  return stats;
}

Status publish_virtual_run_evidence(VirtualPipelineState &state, Stats stats,
                                    const Status status,
                                    const std::uint64_t failed_page,
                                    const std::uint64_t output_hash,
                                    const bool poison_pipeline) noexcept {
  const std::uint64_t physical_hash =
      stats.graph_hash != 0u ? stats.graph_hash
                             : pipeline_fingerprint(state.pipeline).lo;
  stats.graph_hash = virtual_graph_hash(state, physical_hash);
  stats.output_hash = status ? output_hash : 0u;
  stats.pipeline.residency.failed_page =
      status ? ResidencyStats::no_failed_page : failed_page;
  if (state.samples == VirtualPipelineState::SampleState::Active) {
    increment(stats.pipeline.residency.sampled_runs);
    if (status && allocation_free(stats)) {
      increment(stats.pipeline.residency.allocation_free_runs);
    }
  }
  state.stats = stats;
  state.phase = poison_pipeline ? VirtualPipelinePhase::Poisoned
                                : VirtualPipelinePhase::Ready;
  return status;
}

} // namespace rund::compute::detail

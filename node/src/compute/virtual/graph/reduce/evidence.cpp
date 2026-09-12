#include "evidence.hpp"

#include <rund/counter.hpp>

#include <algorithm>

namespace rund::compute::detail::graph_reduce {

using ::rund::detail::counter::Accumulate;

bool record_input_evidence(
    Stats &stats, const VirtualRunProjection &run, const Backend backend,
    const std::span<const residency::GraphLeasePort> ports,
    const std::span<const residency::CacheBinding> bindings,
    const std::span<const residency::CacheTransition> transitions,
    const std::uint64_t fetched_pages,
    const std::uint64_t backing_bytes) noexcept {
  std::uint64_t execution_fetches = 0u;
  std::uint64_t execution_hits = 0u;
  for (const residency::GraphLeasePort &port : ports) {
    if (port.access != residency::Access::Read ||
        port.first_binding > bindings.size() ||
        port.binding_count > bindings.size() - port.first_binding) {
      if (port.access == residency::Access::Read) {
        return false;
      }
      continue;
    }
    for (std::size_t page = 0u; page < port.binding_count; ++page) {
      const residency::CacheBinding binding =
          bindings[port.first_binding + page];
      execution_fetches += static_cast<std::uint64_t>(binding.fetch);
      execution_hits += static_cast<std::uint64_t>(!binding.fetch);
    }
  }
  if ((fetched_pages == 0u) != (backing_bytes == 0u) ||
      (backend == Backend::Cpu && execution_fetches != fetched_pages)) {
    return false;
  }
  std::uint64_t evictions = 0u;
  for (const residency::CacheTransition &transition : transitions) {
    bool input = false;
    for (const residency::GraphLeasePort &port : ports) {
      input =
          input || (port.access == residency::Access::Read &&
                    transition.frame >= port.region.first &&
                    transition.frame - port.region.first < port.region.count);
    }
    evictions += static_cast<std::uint64_t>(
        transition.kind == residency::TransitionKind::Unmap && input);
  }
  Accumulate(stats.pipeline.residency.cache_hit_count, execution_hits);
  Accumulate(stats.pipeline.residency.page_in_count, execution_fetches);
  Accumulate(stats.pipeline.residency.eviction_count, evictions);
  Accumulate(
      stats.pipeline.residency.page_in_bytes,
      backend == Backend::Cpu ? 0u : execution_fetches * run.input_page_bytes);
  return true;
}


void classify_backing(Stats &stats,
                      const std::span<const residency::PageUse> sources,
                      const std::uint64_t fetched, const bool speculative,
                      const bool accelerator) noexcept {
  if (fetched == 0u) {
    return;
  }
  const bool frozen = std::all_of(sources.begin(), sources.end(),
                                  [](const residency::PageUse use) {
                                    return use.prefetch_epoch < use.ready_epoch;
                                  });
  Accumulate(accelerator && speculative && frozen
                 ? stats.pipeline.residency.prefetch_count
                 : stats.pipeline.residency.late_page_count,
             fetched);
}

} // namespace rund::compute::detail::graph_reduce

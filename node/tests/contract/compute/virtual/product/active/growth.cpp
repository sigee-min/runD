#include "local.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_virtual::product::active {

[[nodiscard]] int CheckGrowth(ActiveFixture &fixture) {
  using namespace rund::compute;
  auto &prepared = *fixture.prepared;
  auto &input_backing = *fixture.input_backing;
  auto &output_backing = *fixture.output_backing;
  constexpr std::array<std::size_t, 4u> active_counts{0u, 7u, 35u,
                                                      LogicalElements};
  for (const std::size_t active_count : active_counts) {
    const BackingFacts input_before = input_backing.facts();
    const BackingFacts output_before = output_backing.facts();
    if (!prepared.run(active_count)) {
      return 8;
    }
    const Stats conditioned = prepared.stats();
    // The preceding 35-element phase materializes pages 0 and 1 completely.
    // Growing to the full prefix must retain both even though the boundary
    // extent changes; only pages whose bytes depend on that boundary miss.
    const RouteKind mode =
        ClassifyMode(fixture.backend, conditioned.pipeline.residency,
                     conditioned.pipeline.residency.page_count);
    const bool device_vsm = mode == RouteKind::DeviceVsm;
    const bool persistent = mode == RouteKind::Persistent;
    const std::uint64_t expected_cross_active_hits = device_vsm   ? 0u
                                                     : persistent ? 2u
                                                                  : 3u;
    const std::uint64_t expected_cross_active_loads = device_vsm ? PageCount
                                                      : persistent
                                                          ? PageCount - 2u
                                                          : PageCount - 3u;
    if (active_count == LogicalElements &&
        (conditioned.pipeline.residency.cache_hit_count !=
             expected_cross_active_hits ||
         conditioned.pipeline.residency.page_in_count !=
             expected_cross_active_loads)) {
      std::fprintf(
          stderr,
          "virtual cross-active hits=%llu loads=%llu evictions=%llu "
          "expected_hits=%llu expected_loads=%llu\n",
          static_cast<unsigned long long>(
              conditioned.pipeline.residency.cache_hit_count),
          static_cast<unsigned long long>(
              conditioned.pipeline.residency.page_in_count),
          static_cast<unsigned long long>(
              conditioned.pipeline.residency.eviction_count),
          static_cast<unsigned long long>(expected_cross_active_hits),
          static_cast<unsigned long long>(expected_cross_active_loads));
      return 9;
    }
    if (!prepared.begin_samples()) {
      return 8;
    }
    node_compute_allocation::Start();
    bool warm_ok = true;
    for (std::size_t sample = 0u; sample < WarmRuns; ++sample) {
      warm_ok = warm_ok && static_cast<bool>(prepared.run(active_count));
    }
    node_compute_allocation::Stop();
    const std::uint64_t allocations = node_compute_allocation::Count();
    if (!warm_ok || !prepared.end_samples()) {
      return 10;
    }
    if (const int evidence = CheckActiveEvidence(
            fixture, active_count, input_before, output_before, allocations);
        evidence != 0) {
      return evidence;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::active

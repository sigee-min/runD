#include "fixed.hpp"

#include <rund/compute/async.hpp>

#include <array>
#include <vector>

namespace rund_node_graph_services {

template <class Cache>
[[nodiscard]] auto asynchronous_fixed_flow(rund::compute::Device &device,
                                           Cache &cache, const char *const root,
                                           const char *const tail) {
  using T = rund::compute::Fixed<16, 16>;
  return rund::compute::on(device, cache)
      .template map<T>(
          root, 4u,
          [](auto value) {
            return rund::compute::quantize<
                T, rund::compute::Rounding::Down, rund::compute::Overflow::Wrap,
                rund::compute::Approximation::Deterministic>(value + value);
          })
      .map(tail, [](auto value) {
        return rund::compute::quantize<
            T, rund::compute::Rounding::Down, rund::compute::Overflow::Wrap,
            rund::compute::Approximation::Deterministic>(value + value);
      });
}

[[nodiscard]] bool check_asynchronous_graph(rund::compute::Device &device,
                                            const Backend backend,
                                            CanonicalReference &reference) {
  using T = rund::compute::Fixed<16, 16>;
  auto cache = rund::compute::program_cache(device, 1u);
  if (!cache) {
    return false;
  }
  auto pending_first =
      asynchronous_fixed_flow(device, *cache, "async-fixed-first",
                              "async-fixed-tail")
          .compile_async();
  auto pending_second =
      asynchronous_fixed_flow(device, *cache, "async-fixed-renamed",
                              "async-fixed-tail-renamed")
          .compile_async();
  if (!pending_first || !pending_second) {
    return false;
  }
  auto first = pending_first->get();
  auto second = pending_second->get();
  const auto cache_stats = cache->stats();
  if (!first || !second || !uses_backend(*first, backend) ||
      !uses_backend(*second, backend) ||
      first->fingerprint() != second->fingerprint() ||
      !SameGraph(first->graph(), second->graph()) || cache_stats.misses != 1u ||
      cache_stats.hits + cache_stats.waits != 1u ||
      cache_stats.ready_entries != 1u || cache_stats.in_flight != 0u) {
    return false;
  }
  const Info &graph = first->graph();
  if (!ValidResourceGraph(graph) || graph.nodes.size() != 2u ||
      graph.barriers.empty() || graph.inputs.size() != 1u ||
      graph.outputs.size() != 1u ||
      graph.nodes[1u].dependencies != std::vector<std::uint32_t>{0u} ||
      !default_fixed_resource_metadata<T>(graph, graph.inputs.front(),
                                          Visibility::Input) ||
      !fixed_resource_metadata<T>(
          graph, graph.outputs.front(), Visibility::Output,
          rund::compute::Rounding::Down, rund::compute::Overflow::Wrap,
          rund::compute::Approximation::Deterministic)) {
    return false;
  }
  constexpr T negative_quarter = T::template ratio<-1, 4>();
  constexpr T zero = T::zero();
  constexpr T quarter = T::template ratio<1, 4>();
  constexpr T half = T::template ratio<1, 2>();
  constexpr T negative_one = T::template ratio<-1, 1>();
  constexpr T one = T::template ratio<1, 1>();
  constexpr T two = T::template ratio<2, 1>();
  const std::array<T, 4u> input{negative_quarter, zero, quarter, half};
  const std::array<T, 4u> expected{negative_one, zero, one, two};
  auto first_job = first->resident(std::span<const T>{input});
  auto second_job = second->resident(std::span<const T>{input});
  ExecutionEvidence first_execution{};
  ExecutionEvidence second_execution{};
  if (!first_job || !second_job ||
      !run_fixed_job(*first_job, backend, std::span<const T>{expected},
                     first_execution) ||
      !run_fixed_job(*second_job, backend, std::span<const T>{expected},
                     second_execution) ||
      first_execution.graph_hash != second_execution.graph_hash ||
      first_execution.output_hash != second_execution.output_hash ||
      cache->stats().misses != 1u || cache->stats().in_flight != 0u) {
    return false;
  }
  return MatchReference(reference, graph, first_execution);
}

[[nodiscard]] bool CheckAsynchronous(rund::compute::Device &device,
                                     const Backend backend,
                                     CanonicalReference &reference) {
  return check_asynchronous_graph(device, backend, reference);
}

} // namespace rund_node_graph_services

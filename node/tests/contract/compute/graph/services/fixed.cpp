#include "fixed.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace rund_node_graph_services {

template <class First, class Second>
[[nodiscard]] bool check_mixed_fixed_external(rund::compute::Device &device,
                                              const Backend backend,
                                              CanonicalReference &reference,
                                              const std::string_view label) {
  auto program = rund::compute::on(device)
                     .template input<First>(4u)
                     .template zip_input<Second>(4u)
                     .map("mixed-fixed-select-second",
                          [](auto, auto second) {
                            return rund::compute::quantize<Second>(second);
                          })
                     .compile();
  if (!program || !uses_backend(*program, backend)) {
    std::fprintf(stderr,
                 "graph services mixed fixed compile backend=%u family=%.*s\n",
                 static_cast<unsigned>(backend), static_cast<int>(label.size()),
                 label.data());
    return false;
  }
  const Info &graph = program->graph();
  if (!ValidResourceGraph(graph) || graph.inputs.size() != 2u ||
      graph.outputs.size() != 1u ||
      !default_fixed_resource_metadata<First>(graph, graph.inputs[0u],
                                              Visibility::Input) ||
      !default_fixed_resource_metadata<Second>(graph, graph.inputs[1u],
                                               Visibility::Input) ||
      !default_fixed_resource_metadata<Second>(graph, graph.outputs[0u],
                                               Visibility::Output)) {
    return false;
  }
  constexpr First first_quarter = First::template ratio<1, 4>();
  constexpr First first_half = First::template ratio<1, 2>();
  constexpr Second second_eighth = Second::template ratio<1, 8>();
  constexpr Second second_quarter = Second::template ratio<1, 4>();
  constexpr Second second_half = Second::template ratio<1, 2>();
  constexpr Second second_three_quarters = Second::template ratio<3, 4>();
  const std::array<First, 4u> first{first_quarter, first_half, first_quarter,
                                    first_half};
  const std::array<Second, 4u> second{second_eighth, second_quarter,
                                      second_half, second_three_quarters};
  auto job = program->resident(std::span<const First>{first},
                               std::span<const Second>{second});
  ExecutionEvidence execution{};
  return job &&
         run_fixed_job(*job, backend, std::span<const Second>{second},
                       execution) &&
         MatchReference(reference, graph, execution);
}

template <class T, class Cache>
[[nodiscard]] auto fixed_policy_identity_flow(rund::compute::Device &device,
                                              Cache &cache,
                                              const char *const name) {
  return rund::compute::on(device, cache)
      .template map<T>(name, 4u, [](auto value) {
        return rund::compute::quantize<
            T, rund::compute::Rounding::Down, rund::compute::Overflow::Wrap,
            rund::compute::Approximation::Deterministic>(
            value +
            rund::compute::fixed(rund::compute::FixedOp::Quarter, value));
      });
}

template <class T, class Cache>
[[nodiscard]] auto fixed_constant_identity_flow(rund::compute::Device &device,
                                                Cache &cache,
                                                const char *const name) {
  return rund::compute::on(device, cache)
      .template map<T>(name, 4u, [](auto anchor) {
        return rund::compute::quantize<T>(rund::compute::fixed_zero(anchor));
      });
}

template <class T>
[[nodiscard]] bool check_fixed_cache_width(
    rund::compute::Device &device, rund::compute::ProgramCache &cache,
    const Backend backend, const std::size_t prior_entries,
    CanonicalReference &policy_reference,
    CanonicalReference &constant_reference,
    rund::compute::graph::Fingerprint &policy_fingerprint,
    rund::compute::graph::Fingerprint &constant_fingerprint) {
  auto policy_first =
      fixed_policy_identity_flow<T>(device, cache, "fixed-cache-policy-first")
          .compile();
  auto policy_renamed =
      fixed_policy_identity_flow<T>(device, cache, "fixed-cache-policy-renamed")
          .compile();
  const auto policy_stats = cache.stats();
  if (!policy_first || !policy_renamed ||
      !uses_backend(*policy_first, backend) ||
      !uses_backend(*policy_renamed, backend) ||
      policy_first->fingerprint() != policy_renamed->fingerprint() ||
      !SameGraph(policy_first->graph(), policy_renamed->graph()) ||
      policy_stats.misses != prior_entries + 1u ||
      policy_stats.hits != prior_entries + 1u ||
      policy_stats.ready_entries != prior_entries + 1u ||
      policy_stats.in_flight != 0u) {
    return false;
  }
  const Info &policy_graph = policy_first->graph();
  if (!ValidResourceGraph(policy_graph) || policy_graph.inputs.size() != 1u ||
      policy_graph.outputs.size() != 1u ||
      !default_fixed_resource_metadata<T>(
          policy_graph, policy_graph.inputs.front(), Visibility::Input) ||
      !fixed_resource_metadata<T>(
          policy_graph, policy_graph.outputs.front(), Visibility::Output,
          rund::compute::Rounding::Down, rund::compute::Overflow::Wrap,
          rund::compute::Approximation::Deterministic)) {
    return false;
  }
  constexpr T negative_quarter = T::template ratio<-1, 4>();
  constexpr T zero = T::zero();
  constexpr T quarter = T::template ratio<1, 4>();
  constexpr T half = T::template ratio<1, 2>();
  constexpr T three_quarters = T::template ratio<3, 4>();
  const std::array<T, 4u> input{negative_quarter, zero, quarter, half};
  const std::array<T, 4u> policy_expected{zero, quarter, half, three_quarters};
  auto policy_job = policy_first->resident(std::span<const T>{input});
  ExecutionEvidence policy_execution{};
  if (!policy_job ||
      !run_fixed_job(*policy_job, backend, std::span<const T>{policy_expected},
                     policy_execution) ||
      !MatchReference(policy_reference, policy_graph, policy_execution)) {
    return false;
  }

  auto constant_first = fixed_constant_identity_flow<T>(
                            device, cache, "fixed-cache-constant-first")
                            .compile();
  auto constant_renamed = fixed_constant_identity_flow<T>(
                              device, cache, "fixed-cache-constant-renamed")
                              .compile();
  const auto constant_stats = cache.stats();
  if (!constant_first || !constant_renamed ||
      !uses_backend(*constant_first, backend) ||
      !uses_backend(*constant_renamed, backend) ||
      constant_first->fingerprint() != constant_renamed->fingerprint() ||
      constant_first->fingerprint() == policy_first->fingerprint() ||
      !SameGraph(constant_first->graph(), constant_renamed->graph()) ||
      constant_stats.misses != prior_entries + 2u ||
      constant_stats.hits != prior_entries + 2u ||
      constant_stats.ready_entries != prior_entries + 2u ||
      constant_stats.in_flight != 0u) {
    return false;
  }
  const Info &constant_graph = constant_first->graph();
  if (!ValidResourceGraph(constant_graph) ||
      constant_graph.inputs.size() != 1u ||
      constant_graph.outputs.size() != 1u ||
      !default_fixed_resource_metadata<T>(
          constant_graph, constant_graph.inputs.front(), Visibility::Input) ||
      !default_fixed_resource_metadata<T>(
          constant_graph, constant_graph.outputs.front(), Visibility::Output)) {
    return false;
  }
  const std::array<T, 4u> zeros{zero, zero, zero, zero};
  auto constant_job = constant_first->resident(std::span<const T>{input});
  ExecutionEvidence constant_execution{};
  if (!constant_job ||
      !run_fixed_job(*constant_job, backend, std::span<const T>{zeros},
                     constant_execution) ||
      !MatchReference(constant_reference, constant_graph, constant_execution)) {
    return false;
  }
  policy_fingerprint = policy_first->fingerprint();
  constant_fingerprint = constant_first->fingerprint();
  return true;
}

[[nodiscard]] bool
check_fixed_cache_identity(rund::compute::Device &device, const Backend backend,
                           CrossBackendReferences &references) {
  using I16F16 = rund::compute::Fixed<16, 16>;
  using I20F44 = rund::compute::Fixed<20, 44>;
  auto cache = rund::compute::program_cache(device, 4u);
  if (!cache) {
    return false;
  }
  rund::compute::graph::Fingerprint policy32{};
  rund::compute::graph::Fingerprint constant32{};
  rund::compute::graph::Fingerprint policy64{};
  rund::compute::graph::Fingerprint constant64{};
  if (!check_fixed_cache_width<I16F16>(
          device, *cache, backend, 0u, references.fixed_policy[0u],
          references.fixed_constant[0u], policy32, constant32) ||
      !check_fixed_cache_width<I20F44>(
          device, *cache, backend, 2u, references.fixed_policy[1u],
          references.fixed_constant[1u], policy64, constant64) ||
      policy32 == policy64 || constant32 == constant64 ||
      policy32 == constant32 || policy64 == constant64) {
    return false;
  }
  const auto stats = cache->stats();
  return stats.misses == 4u && stats.hits == 4u && stats.waits == 0u &&
         stats.ready_entries == 4u && stats.in_flight == 0u;
}

[[nodiscard]] bool
CheckMixedFixed(rund::compute::Device &device, const Backend backend,
                std::array<CanonicalReference, 2u> &references) {
  return check_mixed_fixed_external<rund::compute::Fixed<16, 16>,
                                    rund::compute::Fixed<8, 24>>(
             device, backend, references[0u], "fixed-i16-f16") &&
         check_mixed_fixed_external<rund::compute::Fixed<20, 44>,
                                    rund::compute::Fixed<32, 32>>(
             device, backend, references[1u], "fixed-i20-f44");
}

[[nodiscard]] bool CheckFixedCache(rund::compute::Device &device,
                                   const Backend backend,
                                   CrossBackendReferences &references) {
  return check_fixed_cache_identity(device, backend, references);
}

} // namespace rund_node_graph_services

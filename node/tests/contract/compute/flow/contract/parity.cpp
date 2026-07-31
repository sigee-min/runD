#include "model.hpp"

#include "../../../../../src/compute/open/probe.hpp"

#include <array>
#include <vector>

namespace rund_node_flow_contract {

int CheckParity(const rund::compute::Backend backend,
                const std::uint32_t workers, std::uint64_t &graph_hash,
                std::uint64_t &output_hash) {
  const std::array<std::uint32_t, 4> input{1, 2, 3, 4};
  const std::vector<std::uint32_t> expected{2, 6, 12, 17};
  std::uint64_t plan_open_count = 0u;
  bool plan_lazy = true;
  auto plan = [&] {
    rund::compute::detail::ScopedOpenProbe probe{plan_open_count};
    auto recipe = backend == rund::compute::Backend::Cpu
                      ? rund::compute::on(rund::compute::Target::cpu(workers))
                            .map<std::uint32_t>("choose", input.size(),
                                                [](auto x) {
                                                  return rund::compute::select(
                                                      (x > 1u) && (x < 4u),
                                                      x * 2u, x + 1u);
                                                })
                      : flow_on(backend).map<std::uint32_t>(
                            "choose", input.size(), [](auto x) {
                              return rund::compute::select((x > 1u) && (x < 4u),
                                                           x * 2u, x + 1u);
                            });
    plan_lazy = plan_lazy && plan_open_count == 0u;
    (void)std::move(recipe).scan(rund::compute::Scan::InclusiveSum);
    plan_lazy = plan_lazy && plan_open_count == 0u;
    return std::move(recipe).compile();
  }();
  if (!plan || !plan_lazy || plan_open_count != 1u) {
    return 1;
  }
  auto job = plan->resident(input);
  if (!job || !job->run()) {
    return 2;
  }
  auto planned_output = job->read();
  if (!planned_output || *planned_output != expected) {
    return 3;
  }
  const auto stats = job->stats();
  if (stats.backend != backend || stats.graph_hash == 0 ||
      stats.output_hash == 0) {
    return 4;
  }
  if (graph_hash == 0) {
    graph_hash = stats.graph_hash;
    output_hash = stats.output_hash;
  } else if (graph_hash != stats.graph_hash ||
             output_hash != stats.output_hash) {
    return 5;
  }

  std::uint64_t flow_open_count = 0;
  bool flow_lazy = true;
  rund::compute::Result<std::vector<std::uint32_t>> flow = [&] {
    rund::compute::detail::ScopedOpenProbe probe{flow_open_count};
    auto recipe =
        backend == rund::compute::Backend::Cpu
            ? rund::compute::on(rund::compute::Target::cpu(workers), input)
            : flow_on(backend, input);
    flow_lazy = flow_lazy && flow_open_count == 0;
    auto mapped = std::move(recipe).map("choose", [](auto x) {
      return rund::compute::select((x > 1u) && (x < 4u), x * 2u, x + 1u);
    });
    flow_lazy = flow_lazy && flow_open_count == 0;
    (void)std::move(mapped).scan(rund::compute::Scan::InclusiveSum);
    flow_lazy = flow_lazy && flow_open_count == 0;
    return std::move(mapped).collect();
  }();
  if (!flow || *flow != expected || !flow_lazy || flow_open_count != 1) {
    return 6;
  }
  return 0;
}

[[nodiscard]] int
CheckParityBackends(const std::span<const rund::compute::Backend> backends) {
  std::uint64_t graph_hash = 0;
  std::uint64_t output_hash = 0;
  if (const int result =
          CheckParity(rund::compute::Backend::Cpu, 1u, graph_hash, output_hash);
      result != 0) {
    return 20 + result;
  }
  if (const int result =
          CheckParity(rund::compute::Backend::Cpu, 4u, graph_hash, output_hash);
      result != 0) {
    return 30 + result;
  }
  for (const rund::compute::Backend backend : backends) {
    if (backend == rund::compute::Backend::Cpu) {
      continue;
    }
    if (const int result = CheckParity(backend, 0u, graph_hash, output_hash);
        result != 0) {
      return 40 + result;
    }
  }
  return 0;
}

} // namespace rund_node_flow_contract

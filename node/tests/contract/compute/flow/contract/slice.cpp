#include "local.hpp"

#include <rund/compute/abi/job.hpp>

#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"
#include "src/compute/graph/state.hpp"

#include "../../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace rund_node_flow_contract {
namespace {

template <std::size_t Depth, std::uint64_t First, class Expression>
[[nodiscard]] constexpr auto balanced_expression(Expression value) {
  if constexpr (Depth == 0u) {
    return value ^ First;
  } else {
    constexpr std::uint64_t Width = std::uint64_t{1u} << (Depth - 1u);
    return balanced_expression<Depth - 1u, First>(value) ^
           balanced_expression<Depth - 1u, First + Width>(value);
  }
}

template <std::size_t Index> struct DeepField final {};

[[nodiscard]] int
CheckPointwiseReduceSlicesBackend(const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t kCapacity = 17u;
  constexpr std::uint64_t kActive = 5u;

  auto device = open(rund::node::test_contract::target_for(backend, 1u));
  if (!device) {
    return 1;
  }
  auto cache = program_cache(*device, 2u);
  if (!cache) {
    return 2;
  }
  const auto compile = [&] {
    return on(*device, *cache)
        .map<std::uint64_t>("slice-add", kCapacity,
                            [](auto value) { return value + 3u; })
        .map("slice-scale", [](auto value) { return value * 2u; })
        .reduce(Reduce::Sum)
        .compile();
  };
  auto first = compile();
  auto cached = compile();
  if (!first || !cached) {
    return 3;
  }
  const auto &first_state = detail::FlowAccess::state(*first);
  const auto &cached_state = detail::FlowAccess::state(*cached);
  const ProgramCache::Stats cache_stats = cache->stats();
  if (first_state == nullptr || first_state != cached_state ||
      first_state->canonical_graph == nullptr || cache_stats.misses != 1u ||
      cache_stats.hits != 1u || cache_stats.ready_entries != 1u) {
    return 4;
  }

  auto slices = detail::graph_compile::compile_tiled_graph_slices(first_state);
  if (!slices || slices->stages.size() != 2u ||
      slices->stages[0].program == nullptr ||
      slices->stages[1].program == nullptr ||
      slices->stages[0].program == first_state ||
      slices->stages[1].program == first_state ||
      slices->stages[0].program == slices->stages[1].program ||
      slices->capacity != kCapacity || slices->stages[0].inputs.size() != 1u ||
      slices->stages[0].outputs.size() != 1u ||
      slices->stages[1].inputs != slices->stages[0].outputs ||
      slices->stages[1].outputs.size() != 1u ||
      !slices->stages[1].tile_partial) {
    return 5;
  }
  const detail::GraphState &canonical = *first_state->canonical_graph;
  const auto *const collective =
      std::get_if<detail::GraphPrimitive>(&canonical.steps.back());
  const std::span<const std::uint32_t> collective_inputs =
      collective == nullptr ? std::span<const std::uint32_t>{}
                            : canonical.value_ids.view(collective->inputs);
  if (collective == nullptr || collective_inputs.size() != 1u ||
      slices->stages[0].node + 1u != slices->stages[1].node ||
      slices->stages[1].node + 1u != canonical.steps.size() ||
      slices->input_resources != canonical.inputs ||
      slices->stages[0].outputs.front() != collective_inputs.front() ||
      slices->output_resource != canonical.outputs.front() ||
      slices->stages[0].node >= first_state->graph_info.nodes.size() ||
      slices->stages[1].node >= first_state->graph_info.nodes.size() ||
      first_state->graph_info.nodes[slices->stages[0].node].index !=
          slices->stages[0].node ||
      first_state->graph_info.nodes[slices->stages[1].node].index !=
          slices->stages[1].node ||
      first_state->graph_info.inputs != slices->input_resources ||
      first_state->graph_info.outputs.front() != slices->output_resource) {
    return 14;
  }
  const auto valid_stage = [](const auto &stage,
                              const std::size_t output_size) {
    return stage->canonical_graph != nullptr &&
           stage->input_types.size() == 2u &&
           stage->input_types[0u] == detail::Type::U64 &&
           stage->input_types[1u] == detail::Type::U64 &&
           stage->input_sizes == std::vector<std::size_t>{kCapacity, 1u} &&
           stage->bounded_input_capacities ==
               std::vector<std::size_t>{0u, kCapacity} &&
           stage->output_types.size() == 1u &&
           stage->output_types.front() == detail::Type::U64 &&
           stage->output_sizes == std::vector<std::size_t>{output_size} &&
           stage->graph_info.nodes.size() == 1u;
  };
  if (!valid_stage(slices->stages[0].program, kCapacity) ||
      !valid_stage(slices->stages[1].program, 1u) ||
      !slices->stages[0].program->graph_info.fingerprint ||
      !slices->stages[1].program->graph_info.fingerprint ||
      slices->stages[0].program->graph_info.fingerprint ==
          slices->stages[1].program->graph_info.fingerprint ||
      slices->stages[0].program->graph_info.fingerprint ==
          first_state->graph_info.fingerprint ||
      slices->stages[1].program->graph_info.fingerprint ==
          first_state->graph_info.fingerprint) {
    return 6;
  }

  std::array<std::uint64_t, kCapacity> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<std::uint64_t>(index + 1u);
  }
  const std::array<std::uint64_t, 1u> active{kActive};
  auto prefix_job = detail::make_job(slices->stages[0].program,
                                     std::span<const std::uint64_t>{input},
                                     std::span<const std::uint64_t>{active});
  if (!prefix_job || !detail::run_job(*prefix_job)) {
    return 7;
  }
  auto prefix = detail::read_job<std::uint64_t>(*prefix_job);
  if (!prefix || prefix->size() != kCapacity) {
    return 8;
  }
  for (std::size_t index = 0u; index < kActive; ++index) {
    if ((*prefix)[index] != (input[index] + 3u) * 2u) {
      return 9;
    }
  }

  auto reduce_job = detail::make_job(slices->stages[1].program,
                                     std::span<const std::uint64_t>{*prefix},
                                     std::span<const std::uint64_t>{active});
  if (!reduce_job || !detail::run_job(*reduce_job)) {
    return 10;
  }
  auto reduced = detail::read_job<std::uint64_t>(*reduce_job);
  if (!reduced || *reduced != std::vector<std::uint64_t>{60u}) {
    return 11;
  }

  auto unsliced = on(*device)
                      .map<std::uint64_t>("slice-identity", kCapacity,
                                          [](auto value) { return value; })
                      .reduce(Reduce::Sum)
                      .compile();
  if (!unsliced) {
    return 12;
  }
  auto rejected = detail::graph_compile::compile_tiled_graph_slices(
      detail::FlowAccess::state(*unsliced));
  if (rejected || rejected.reason() != Reason::PrimitiveUnsupported) {
    return 13;
  }

  // Both Maps fit the per-kernel expression envelope independently, while
  // their expression composition exceeds it. They therefore remain two
  // authored kernels, but the tiled compiler must place both in one prepared
  // stage: their internal value is not a VSM materialization or Host epoch.
  auto dag =
      on(*device)
          .input<std::uint64_t>(kCapacity)
          .map(
              "slice-deep-first",
              [](auto value) {
                return record(
                    field<DeepField<0u>>(balanced_expression<7u, 1u>(value)),
                    field<DeepField<1u>>(balanced_expression<7u, 129u>(value)));
              })
          .branch([](auto values) {
            return zip(values.template get<DeepField<0u>>(),
                       values.template get<DeepField<1u>>())
                .map("slice-deep-second",
                     [](auto left, auto right) {
                       return balanced_expression<7u, 257u>(left) ^
                              balanced_expression<7u, 385u>(right);
                     })
                .reduce(Reduce::Sum);
          })
          .compile();
  if (!dag) {
    return 15;
  }
  auto dag_slices = detail::graph_compile::compile_tiled_graph_slices(
      detail::FlowAccess::state(*dag));
  if (!dag_slices) {
    return 16;
  }
  if (dag_slices->stages.size() != 2u ||
      dag_slices->stages.front().program == nullptr ||
      dag_slices->stages.front().program->graph_info.nodes.size() != 2u ||
      dag_slices->resources.size() != 3u ||
      !dag_slices->stages.back().tile_partial) {
    return 17;
  }
  auto fused_job = detail::make_job(dag_slices->stages.front().program,
                                    std::span<const std::uint64_t>{input},
                                    std::span<const std::uint64_t>{active});
  if (!fused_job || !detail::run_job(*fused_job)) {
    return 18;
  }
  const auto fused = detail::read_job<std::uint64_t>(*fused_job);
  if (!fused || fused->size() != kCapacity) {
    return 19;
  }
  for (std::size_t index = 0u; index < kActive; ++index) {
    const std::uint64_t left = balanced_expression<7u, 1u>(input[index]);
    const std::uint64_t right = balanced_expression<7u, 129u>(input[index]);
    const std::uint64_t expected = balanced_expression<7u, 257u>(left) ^
                                   balanced_expression<7u, 385u>(right);
    if ((*fused)[index] != expected) {
      return 20;
    }
  }

  // A later Map that rereads the public input is not an internal value of one
  // linear prefix. Keep that branch visible to the residency planner so its
  // external-input next-use, pin interval, and ready-wavefront edge cannot be
  // erased by prefix fusion.
  auto branched =
      on(*device)
          .input<std::uint64_t>(kCapacity)
          .branch([](auto values) {
            const auto left = values.map("slice-branch-left",
                                         [](auto value) { return value + 1u; });
            const auto right = values.map(
                "slice-branch-right", [](auto value) { return value * 2u; });
            return zip(left, right)
                .map("slice-branch-join",
                     [](auto first, auto second) { return first + second; })
                .reduce(Reduce::Sum);
          })
          .compile();
  if (!branched) {
    return 21;
  }
  const auto branch_state = detail::FlowAccess::state(*branched);
  if (branch_state == nullptr || branch_state->canonical_graph == nullptr) {
    return 21;
  }
  auto branch_slices =
      detail::graph_compile::compile_tiled_graph_slices(branch_state);
  if (!branch_slices || branch_slices->stages.size() != 4u ||
      branch_slices->resources.size() != 5u ||
      branch_slices->stages[0u].inputs.size() != 1u ||
      branch_slices->stages[1u].inputs != branch_slices->stages[0u].inputs ||
      branch_slices->stages[0u].outputs.size() != 1u ||
      branch_slices->stages[1u].outputs.size() != 1u ||
      branch_slices->stages[2u].inputs.size() != 2u ||
      branch_slices->stages[2u].inputs[0u] !=
          branch_slices->stages[0u].outputs[0u] ||
      branch_slices->stages[2u].inputs[1u] !=
          branch_slices->stages[1u].outputs[0u] ||
      branch_slices->stages[2u].outputs != branch_slices->stages[3u].inputs ||
      !branch_slices->stages[3u].tile_partial) {
    return 22;
  }
  for (std::size_t index = 0u; index < branch_slices->stages.size(); ++index) {
    if (branch_slices->stages[index].program == nullptr ||
        branch_slices->stages[index].node != index ||
        branch_slices->stages[index].program->graph_info.nodes.size() != 1u) {
      return 23;
    }
  }
  return 0;
}

} // namespace

[[nodiscard]] int CheckPointwiseReduceSlices(
    const std::span<const rund::compute::Backend> backends) {
  for (std::size_t index = 0u; index < backends.size(); ++index) {
    const int result = CheckPointwiseReduceSlicesBackend(backends[index]);
    if (result != 0) {
      return static_cast<int>(index * 16u) + result;
    }
  }
  return 0;
}

} // namespace rund_node_flow_contract

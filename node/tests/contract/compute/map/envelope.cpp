#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/abi/graph.hpp>

#include "src/compute/expression/state.hpp"
#include "src/compute/flow/state.hpp"
#include "src/compute/graph/state.hpp"
#include "src/compute/map/build.hpp"
#include "src/compute/program/state.hpp"

#include <kernel/program/compute/limit.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <vector>

namespace compute_map_contract {
[[nodiscard]] bool
Envelope(const std::shared_ptr<rund::compute::detail::DeviceState> &device,
         const rund::compute::Backend backend,
         rund::compute::graph::Fingerprint &reference32,
         rund::compute::graph::Fingerprint &reference64) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::size_t WitnessNodes =
      3u * static_cast<std::size_t>(rund::kernel::kMaxComputeNodeCount) / 2u;
  static_assert(graph::NodeCapacity == rund::kernel::kMaxGraphNodeCount);
  static_assert(graph::ValueCapacity == rund::kernel::kMaxGraphValueCount);
  static_assert(graph::BuffersPerNode == rund::kernel::kMaxGraphBuffersPerNode);
  static_assert(graph::OutputCapacity == rund::kernel::kMaxGraphOutputCount);
  static_assert(WitnessNodes > rund::kernel::kMaxComputeNodeCount);
  static_assert(WitnessNodes < rund::kernel::kMaxGraphNodeCount);
  // One stored add occupies Input, Constant, Add, Quantize in a fresh
  // expression. Every exact successor reuses the substituted predecessor and
  // adds Constant, Add, Quantize. The canonical fusion envelope is therefore
  // floor((IR - 4) / 3) + 1 authored Maps per physical Map.
  constexpr std::size_t MapsPerNode =
      1u +
      (static_cast<std::size_t>(rund::kernel::kMaxComputeNodeCount) - 4u) / 3u;
  constexpr std::size_t ExpectedNodes =
      (WitnessNodes + MapsPerNode - 1u) / MapsPerNode;
  constexpr std::size_t InternalResources = ExpectedNodes - 1u;
  constexpr std::size_t ExpectedBarriers = ExpectedNodes - 1u;

  const auto verify = [&]<class T>(graph::Fingerprint &reference) {
    const FixedFormat format = storage_format<T>();
    auto expression = make_expr();
    ExprRef value = detail::input(expression, type<T>(), 0u, format);
    const ExprRef unit = constant(expression, type<T>(), 1u, format);
    value = binary(ExprOp::Add, std::move(value), unit);
    value = quantize_expr(std::move(value), type<T>(), format);
    if (value.node == 0u || !expression->status) {
      return false;
    }

    auto flow = make_flow_on(device, type<T>(), 1u, {}, format);
    for (std::size_t index = 0u; index < WitnessNodes; ++index) {
      flow_map(flow, "program envelope", value);
    }
    if (!flow->status || flow_step_count(flow) != WitnessNodes) {
      return false;
    }
    auto program = compile_flow(flow);
    if (!program) {
      std::fprintf(stderr,
                   "program envelope compile failed backend=%u width=%zu "
                   "reason=%.*s\n",
                   static_cast<unsigned>(backend), sizeof(T) * 8u,
                   static_cast<int>(program.error().size()),
                   program.error().data());
      return false;
    }
    const graph::Info &info = (*program)->graph_info;
    if (info.authored_nodes != WitnessNodes ||
        info.lowered_nodes != ExpectedNodes ||
        info.nodes.size() != info.lowered_nodes ||
        info.resources.size() != ExpectedNodes + 1u ||
        info.barriers.size() != ExpectedBarriers ||
        info.memory.logical_bytes != InternalResources * sizeof(T) ||
        info.memory.physical_bytes != sizeof(T) ||
        info.memory.allocation_count != 1u) {
      std::fprintf(
          stderr,
          "program envelope shape failed backend=%u width=%zu "
          "authored=%llu lowered=%llu nodes=%zu resources=%zu barriers=%zu "
          "logical=%llu "
          "physical=%llu allocations=%llu\n",
          static_cast<unsigned>(backend), sizeof(T) * 8u,
          static_cast<unsigned long long>(info.authored_nodes),
          static_cast<unsigned long long>(info.lowered_nodes),
          info.nodes.size(), info.resources.size(), info.barriers.size(),
          static_cast<unsigned long long>(info.memory.logical_bytes),
          static_cast<unsigned long long>(info.memory.physical_bytes),
          static_cast<unsigned long long>(info.memory.allocation_count));
      return false;
    }
    if (!reference) {
      reference = info.fingerprint;
    } else if (reference != info.fingerprint) {
      return false;
    }

    const std::array<T, 1u> input{T::zero()};
    auto job = make_job(*program, std::span<const T>{input});
    if (!job || !run_job(*job)) {
      return false;
    }
    auto output = read_job<T>(*job);
    const Stats stats = job_stats(*job);
    return output && *output == std::vector<T>{T::from_raw(WitnessNodes)} &&
           stats.backend == backend;
  };

  return verify.template operator()<Fixed<16u, 16u>>(reference32) &&
         verify.template operator()<Fixed<20u, 44u>>(reference64);
}

} // namespace compute_map_contract

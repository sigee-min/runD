#include "../local.hpp"

#include "../../../allocation.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::test_contract::window {
namespace {

template <class Seed, class Fold>
[[nodiscard]] int CheckBuildAllocationRollback(Device &device, const Seed &seed,
                                               const Fold &fold) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> scalar{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> count_value{kOutputMaximum};
  constexpr std::array<std::uint32_t, 1u> witness_value{0u};
  auto outer = device.upload<std::uint32_t>(scalar);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_value);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto window = device.buffer<std::uint32_t>(kOutputMaximum);
  if (!outer || !values || !lanes || !witness || !count || !final || !window) {
    return 1;
  }

  const std::array<ResourceView, 4u> inputs{
      BufferAccess::view(*outer, ResourceAccess::Read),
      BufferAccess::view(*values, ResourceAccess::Read),
      BufferAccess::view(*lanes, ResourceAccess::Read),
      BufferAccess::view(*witness, ResourceAccess::Read),
  };
  const std::array<ResourceView, 1u> finals{
      BufferAccess::view(*final, ResourceAccess::Write),
  };
  const std::array<ResourceView, 1u> windows{
      BufferAccess::view(*window, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  if (build == nullptr) {
    return 2;
  }

  // Keep a non-empty prefix so the failure proves exact rollback rather than
  // merely clearing a newly-created build.
  build->steps.push_back(PipelineBuildStep{});
  build->internals.push_back(PipelineInternal{});
  build->publications.push_back(PipelineBuildTerminalPublication{});
  build->window_controls.push_back(PipelineBuildWindowControl{});
  build->nested_windows.push_back(PipelineBuildNestedWindow{});
  build->binding_count = 3u;

  // Leave exactly one allocation-free internal slot. The first nested
  // internal append must consume that slot and the second must grow the
  // vector, independent of the implementation's vector growth factor.
  build->internals.reserve(build->internals.size() + 1u);
  while (build->internals.size() + 1u < build->internals.capacity()) {
    build->internals.push_back(PipelineInternal{});
  }
  const std::size_t steps = build->steps.size();
  const std::size_t internals = build->internals.size();
  const std::size_t publications = build->publications.size();
  const std::size_t window_controls = build->window_controls.size();
  const std::size_t nested_windows = build->nested_windows.size();
  const std::size_t bindings = build->binding_count;

  // Seven non-empty local route vectors allocate first. The next allocation
  // is the second internal append, after the first append consumed the sole
  // spare slot, so this necessarily exercises partial-mutation rollback.
  node_compute_allocation::FailAfter(7u);
  append_pipeline_window_repeat(
      build, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
      BufferAccess::view(*count, ResourceAccess::Read), inputs, finals, windows,
      kOutputMaximum, kOutputTile, 0u, NoWindowTerminal, 1u);
  node_compute_allocation::ClearFailure();
  if (build->failure != Reason::PipelineCapacity ||
      build->steps.size() != steps || build->internals.size() != internals ||
      build->publications.size() != publications ||
      build->window_controls.size() != window_controls ||
      build->nested_windows.size() != nested_windows ||
      build->binding_count != bindings) {
    return 3;
  }

  build->failure = Reason::Ok;
  append_pipeline_window_repeat(
      build, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
      BufferAccess::view(*count, ResourceAccess::Read), inputs, finals, windows,
      kOutputMaximum, kOutputTile, 0u, NoWindowTerminal, 1u);
  return build->failure == Reason::Ok && build->steps.size() > steps &&
                 build->internals.size() > internals &&
                 build->publications.size() > publications &&
                 build->window_controls.size() == window_controls + 1u &&
                 build->nested_windows.size() == nested_windows + 1u &&
                 build->binding_count > bindings
             ? 0
             : 4;
}

} // namespace

int CheckOutputBuildAllocationRollback(Device &device, const OutputSeed &seed,
                                       const OutputFold &fold) {
  return CheckBuildAllocationRollback(device, seed, fold);
}

} // namespace rund::node::test_contract::window

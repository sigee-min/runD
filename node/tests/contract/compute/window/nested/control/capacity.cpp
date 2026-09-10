#include "../../../pipeline/local.hpp"
#include "../../local.hpp"
#include "../local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/prepared.hpp"
#include "src/compute/cpu/run/state.hpp"
#include "src/compute/memory/cpu.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace rund::node::test_contract::window {

int CheckWorkspaceObservationCapacity(rund::compute::Device &device) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::size_t window_count = 22u;
  constexpr std::size_t rows_per_window = 5u;
  constexpr std::size_t row_count = window_count * rows_per_window;
  static_assert(3u * window_count > PipelineStepCapacity);
  static_assert(row_count < PipelineRouteCapacity);

  auto seed = on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto total, auto ordinal) {
                    (void)total;
                    return ordinal.scan(Scan::InclusiveSum)
                        .map("workspace-capacity-seed",
                             [](auto value) { return value + 1u; });
                  })
                  .compile();
  auto action = on(device)
                    .input<std::uint32_t>(1u)
                    .branch([](auto value) {
                      return value.scan(Scan::InclusiveSum)
                          .map("workspace-capacity-action",
                               [](auto current) { return current + 1u; });
                    })
                    .compile();
  auto fold = on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto outer, auto tile) {
                    auto prefix = tile.scan(Scan::InclusiveSum);
                    return outer.combine(
                        "workspace-capacity-fold", prefix,
                        [](auto left, auto right) { return left + right; });
                  })
                  .compile();
  constexpr std::array<std::uint32_t, 1u> initial{1u};
  constexpr std::array<std::uint32_t, 1u> inactive{0u};
  auto outer = device.upload<std::uint32_t>(initial);
  auto count = device.upload<std::uint32_t>(inactive);
  if (!seed || !action || !fold || !outer || !count) {
    return 1;
  }
  std::vector<rund::compute::Buffer<std::uint32_t>> outputs;
  outputs.reserve(window_count);
  for (std::size_t index = 0u; index < window_count; ++index) {
    auto output = device.buffer<std::uint32_t>(1u);
    if (!output) {
      return 1;
    }
    outputs.push_back(std::move(*output));
  }

  const auto body = tile_repeat<1u>(*seed, *action, *fold);
  auto builder = pipeline(device).profile(PipelineProfile::Steps);
  for (std::size_t index = 0u; index < window_count; ++index) {
    builder.windows<1u, 1u>(body, rund::compute::window(*count), read(*outer),
                            write_final(outputs[index]));
  }
  auto prepared = std::move(builder).prepare();
  const std::shared_ptr<PipelineState> state =
      prepared ? PipelineStateAccess::state(*prepared)
               : std::shared_ptr<PipelineState>{};
  if (!prepared || state == nullptr || state->steps.size() != row_count ||
      !prepared->run()) {
    return 2;
  }

  std::array<const JobWorkspace *, row_count> workspaces{};
  std::size_t workspace_count = 0u;
  for (const PipelineStep &step : state->steps) {
    if (step.job == nullptr || step.job->workspace == nullptr) {
      return 3;
    }
    const JobWorkspace *const workspace = step.job->workspace.get();
    if (std::find(workspaces.begin(), workspaces.begin() + workspace_count,
                  workspace) == workspaces.begin() + workspace_count) {
      workspaces[workspace_count++] = workspace;
    }
  }
  std::array<PipelineStepProfile, row_count> rows{};
  const auto profile = prepared->profile(rows);
  if (workspace_count <= PipelineStepCapacity || !profile ||
      profile->written != row_count || profile->total != row_count ||
      !::rund_node_test_pipeline::ProfileMemoryReconciles(*profile, rows) ||
      profile->memory.host.current ==
          std::numeric_limits<std::uint64_t>::max() ||
      profile->memory.resident.current ==
          std::numeric_limits<std::uint64_t>::max()) {
    return 4;
  }
  return 0;
}

} // namespace rund::node::test_contract::window

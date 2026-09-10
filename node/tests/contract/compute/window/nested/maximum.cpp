#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/prepared.hpp"
#include "src/compute/cpu/run/state.hpp"
#include "src/compute/memory/cpu.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::test_contract::window {
template <class Action, class Fold>
[[nodiscard]] int CheckMaximumPlan(rund::compute::Device &device,
                                   const Action &action, const Fold &fold) {
  using namespace rund::compute;
  constexpr std::size_t tile = 1024u;
  constexpr std::size_t inner = 64u;
  constexpr std::size_t small_maximum = tile;
  constexpr std::size_t medium_maximum = 2u * tile;
  constexpr std::size_t steady_maximum = 3u * tile;
  constexpr std::size_t maximum = 516096u;
  constexpr std::size_t small_outer = CeilDiv(small_maximum, tile);
  constexpr std::size_t medium_outer = CeilDiv(medium_maximum, tile);
  constexpr std::size_t steady_outer = CeilDiv(steady_maximum, tile);
  constexpr std::size_t outer = CeilDiv(maximum, tile);
  static_assert(outer == 504u);
  static_assert(outer * inner > PipelineIterationCapacity);

  const auto planned = [&]<std::size_t Maximum>() -> Result<PipelinePlan> {
    auto seed = [&] {
      if constexpr (Maximum == 1024u) {
        return MakeNestedSeed1024Program(device);
      } else if constexpr (Maximum == 2048u) {
        return MakeNestedSeed2048Program(device);
      } else if constexpr (Maximum == 3072u) {
        return MakeNestedSeed3072Program(device);
      } else {
        return MakeNestedSeed516096Program(device);
      }
    }();
    auto outer_state = device.buffer<std::uint32_t>(1u);
    auto queue = device.buffer<std::uint32_t>(Maximum);
    auto domain = device.buffer<std::uint32_t>(kDomain);
    auto count = device.buffer<std::uint32_t>(1u);
    auto output = device.buffer<std::uint32_t>(1u);
    if (!seed || !outer_state || !queue || !domain || !count || !output) {
      return Result<PipelinePlan>::fail(Reason::PipelineInvalid);
    }
    const auto body = tile_repeat<inner>(*seed, action, fold);
    auto builder = pipeline(device);
    builder.windows<Maximum, tile>(body, rund::compute::window(*count),
                                   read(*outer_state, *queue, *domain),
                                   write_final(*output));
    return builder.plan();
  };

  const auto small = planned.template operator()<small_maximum>();
  const auto medium = planned.template operator()<medium_maximum>();
  const auto steady = planned.template operator()<steady_maximum>();
  const auto large = planned.template operator()<maximum>();
  constexpr std::uint64_t small_schedule_bytes =
      small_outer * sizeof(std::uint32_t);
  constexpr std::uint64_t large_schedule_bytes = outer * sizeof(std::uint32_t);
  constexpr std::uint64_t state_bytes = (outer + 5u) * sizeof(std::uint32_t);
  const Backend selected_backend =
      rund::compute::detail::DeviceAccess::state(device)->backend;
  const bool cpu = selected_backend == Backend::Cpu;
  const bool accelerator = selected_backend != Backend::Cpu;
  const bool plans_ready = small && medium && steady && large;
  const std::uint64_t host_step =
      small && medium &&
              medium->prepared_host_bytes >= small->prepared_host_bytes
          ? medium->prepared_host_bytes - small->prepared_host_bytes
          : 0u;
  const std::uint64_t prepared_delta =
      small && large && large->prepared_bytes >= small->prepared_bytes
          ? large->prepared_bytes - small->prepared_bytes
          : 0u;
  const std::uint64_t steady_route_step =
      medium && steady && steady->prepared_bytes >= medium->prepared_bytes
          ? steady->prepared_bytes - medium->prepared_bytes
          : 0u;
  const auto prepared_components_exact = [](const PipelinePlan &plan) {
    return plan.prepared_bytes ==
           plan.prepared_buffer_bytes + plan.prepared_host_bytes +
               plan.prepared_tile_bytes + plan.prepared_native_bytes;
  };
  const bool cpu_prepared_shape =
      !cpu ||
      (plans_ready && host_step != 0u &&
       large->prepared_host_bytes - small->prepared_host_bytes ==
           (outer - small_outer) * host_step &&
       prepared_delta ==
           large->prepared_host_bytes - small->prepared_host_bytes +
               (large->prepared_tile_bytes - small->prepared_tile_bytes) &&
       small->prepared_buffer_bytes == medium->prepared_buffer_bytes &&
       small->prepared_buffer_bytes == steady->prepared_buffer_bytes &&
       small->prepared_buffer_bytes == large->prepared_buffer_bytes &&
       small->prepared_tile_bytes <= medium->prepared_tile_bytes &&
       medium->prepared_tile_bytes == steady->prepared_tile_bytes &&
       medium->prepared_tile_bytes == large->prepared_tile_bytes &&
       small->prepared_native_bytes == 0u &&
       medium->prepared_native_bytes == 0u &&
       steady->prepared_native_bytes == 0u &&
       large->prepared_native_bytes == 0u);
  const bool monotonic_prepared_shape =
      plans_ready && small->prepared_bytes < medium->prepared_bytes &&
      medium->prepared_bytes < steady->prepared_bytes &&
      steady->prepared_bytes < large->prepared_bytes;
  const bool affine_prepared_shape =
      plans_ready && steady_route_step != 0u &&
      large->prepared_bytes - medium->prepared_bytes ==
          (outer - medium_outer) * steady_route_step;
  const bool backend_structural_prepared_shape =
      !accelerator ||
      (plans_ready &&
       small->prepared_host_bytes < medium->prepared_host_bytes &&
       medium->prepared_host_bytes < steady->prepared_host_bytes &&
       steady->prepared_host_bytes < large->prepared_host_bytes &&
       small->prepared_native_bytes < medium->prepared_native_bytes &&
       medium->prepared_native_bytes < steady->prepared_native_bytes &&
       steady->prepared_native_bytes < large->prepared_native_bytes &&
       small->prepared_buffer_bytes == medium->prepared_buffer_bytes &&
       medium->prepared_buffer_bytes == steady->prepared_buffer_bytes &&
       steady->prepared_buffer_bytes == large->prepared_buffer_bytes &&
       small->prepared_tile_bytes == medium->prepared_tile_bytes &&
       medium->prepared_tile_bytes == steady->prepared_tile_bytes &&
       steady->prepared_tile_bytes == large->prepared_tile_bytes);
  const bool compact_route_prepared_shape =
      monotonic_prepared_shape && backend_structural_prepared_shape &&
      (accelerator || affine_prepared_shape);
  if (!plans_ready || !cpu_prepared_shape || !compact_route_prepared_shape ||
      !prepared_components_exact(*small) ||
      !prepared_components_exact(*medium) ||
      !prepared_components_exact(*steady) ||
      !prepared_components_exact(*large) ||
      small->state_bytes < small_schedule_bytes ||
      large->state_bytes < large_schedule_bytes ||
      small->state_bytes - small_schedule_bytes !=
          large->state_bytes - large_schedule_bytes ||
      small->transient_bytes != large->transient_bytes ||
      small->scratch_bytes != large->scratch_bytes ||
      small->scratch_count != large->scratch_count ||
      small->allocation_count != large->allocation_count ||
      small->node_count != large->node_count ||
      small->resource_count != large->resource_count ||
      large->state_bytes != state_bytes || large->outer_window_count != outer ||
      large->tile_capacity != tile || large->inner_iteration_count != inner ||
      small->prepared_template_count != small_outer + 2u + 3u ||
      medium->prepared_template_count != medium_outer + 2u + 3u ||
      steady->prepared_template_count != steady_outer + 2u + 3u ||
      large->prepared_template_count != outer + 2u + 3u ||
      large->prepared_command_count != outer * (inner + 2u) ||
      large->peak_bytes != small->peak_bytes +
                               (large->state_bytes - small->state_bytes) +
                               prepared_delta ||
      large->logical_bytes <= large->physical_bytes) {
    if (small && large) {
      std::fprintf(
          stderr,
          "nested maximum plan outer=%llu tile=%llu inner=%llu "
          "state=%llu/%llu normalized=%llu/%llu "
          "transient=%llu/%llu prepared=%llu/%llu scratch=%llu:%llu/"
          "%llu:%llu allocations=%llu/%llu nodes=%llu/%llu "
          "resources=%llu/%llu templates=%llu commands=%llu "
          "logical/physical=%llu/%llu\n",
          static_cast<unsigned long long>(large->outer_window_count),
          static_cast<unsigned long long>(large->tile_capacity),
          static_cast<unsigned long long>(large->inner_iteration_count),
          static_cast<unsigned long long>(small->state_bytes),
          static_cast<unsigned long long>(large->state_bytes),
          static_cast<unsigned long long>(small->state_bytes -
                                          small_schedule_bytes),
          static_cast<unsigned long long>(large->state_bytes -
                                          large_schedule_bytes),
          static_cast<unsigned long long>(small->transient_bytes),
          static_cast<unsigned long long>(large->transient_bytes),
          static_cast<unsigned long long>(small->prepared_bytes),
          static_cast<unsigned long long>(large->prepared_bytes),
          static_cast<unsigned long long>(small->scratch_bytes),
          static_cast<unsigned long long>(small->scratch_count),
          static_cast<unsigned long long>(large->scratch_bytes),
          static_cast<unsigned long long>(large->scratch_count),
          static_cast<unsigned long long>(small->allocation_count),
          static_cast<unsigned long long>(large->allocation_count),
          static_cast<unsigned long long>(small->node_count),
          static_cast<unsigned long long>(large->node_count),
          static_cast<unsigned long long>(small->resource_count),
          static_cast<unsigned long long>(large->resource_count),
          static_cast<unsigned long long>(large->prepared_template_count),
          static_cast<unsigned long long>(large->prepared_command_count),
          static_cast<unsigned long long>(large->logical_bytes),
          static_cast<unsigned long long>(large->physical_bytes));
      if (medium) {
        std::fprintf(
            stderr,
            "nested maximum prepared buffer=%llu/%llu/%llu "
            "host=%llu/%llu/%llu tile=%llu/%llu/%llu "
            "native=%llu/%llu/%llu host_step=%llu delta=%llu\n",
            static_cast<unsigned long long>(small->prepared_buffer_bytes),
            static_cast<unsigned long long>(medium->prepared_buffer_bytes),
            static_cast<unsigned long long>(large->prepared_buffer_bytes),
            static_cast<unsigned long long>(small->prepared_host_bytes),
            static_cast<unsigned long long>(medium->prepared_host_bytes),
            static_cast<unsigned long long>(large->prepared_host_bytes),
            static_cast<unsigned long long>(small->prepared_tile_bytes),
            static_cast<unsigned long long>(medium->prepared_tile_bytes),
            static_cast<unsigned long long>(large->prepared_tile_bytes),
            static_cast<unsigned long long>(small->prepared_native_bytes),
            static_cast<unsigned long long>(medium->prepared_native_bytes),
            static_cast<unsigned long long>(large->prepared_native_bytes),
            static_cast<unsigned long long>(host_step),
            static_cast<unsigned long long>(prepared_delta));
      }
    }
    return 1;
  }
  return 0;
}

int CheckNestedMaximumPlan(rund::compute::Device &device,
                           const NestedAction &action, const NestedFold &fold) {
  return CheckMaximumPlan(device, action, fold);
}

} // namespace rund::node::test_contract::window

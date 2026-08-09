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
template <class Seed, class Fold>
[[nodiscard]] int CheckRetainedReuse(rund::compute::Device &device,
                                     const Seed &seed, const Fold &fold) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{kMaximum};
  auto action = MakeNestedMemoryActionProgram(device);
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{kQueue});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output_one = device.buffer<std::uint32_t>(1u);
  auto output_many = device.buffer<std::uint32_t>(1u);
  auto output_short = device.buffer<std::uint32_t>(1u);
  if (!action || !outer || !queue || !domain || !count || !output_one ||
      !output_many || !output_short) {
    return 1;
  }

  const auto make_builder =
      [&]<std::size_t Inner>(rund::compute::Buffer<std::uint32_t> &output) {
        const auto body = tile_repeat<Inner>(seed, *action, fold);
        auto builder = pipeline(device);
        builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                         read(*outer, *queue, *domain),
                                         write_final(output));
        return builder;
      };

  auto one_builder = make_builder.template operator()<1u>(*output_one);
  auto two_builder = make_builder.template operator()<2u>(*output_short);
  auto three_builder = make_builder.template operator()<3u>(*output_short);
  auto many_builder = make_builder.template operator()<64u>(*output_many);
  const auto one_plan = one_builder.plan();
  const auto two_plan = two_builder.plan();
  const auto three_plan = three_builder.plan();
  const auto many_plan = many_builder.plan();
  const Backend selected_backend =
      rund::compute::detail::DeviceAccess::state(device)->backend;
  const bool cpu = selected_backend == Backend::Cpu;
  const bool accelerator = selected_backend != Backend::Cpu;
  const auto seed_storage = rund::compute::detail::plan_cpu_graph_storage(
      rund::compute::detail::ProgramAccess::state(seed));
  const auto action_storage = rund::compute::detail::plan_cpu_graph_storage(
      rund::compute::detail::ProgramAccess::state(*action));
  const auto fold_storage = rund::compute::detail::plan_cpu_graph_storage(
      rund::compute::detail::ProgramAccess::state(fold));
  rund::compute::detail::CpuExecutionStoragePlan storage_execution{};
  const bool storage_ready =
      seed_storage && action_storage && fold_storage &&
      rund::compute::detail::merge_cpu_execution_storage_plan(
          storage_execution, seed_storage->execution) &&
      rund::compute::detail::merge_cpu_execution_storage_plan(
          storage_execution, action_storage->execution) &&
      rund::compute::detail::merge_cpu_execution_storage_plan(
          storage_execution, fold_storage->execution);
  const rund::compute::detail::CpuStorageBytes execution_payload =
      storage_ready ? rund::compute::detail::cpu_execution_storage_payload(
                          storage_execution)
                    : rund::compute::detail::CpuStorageBytes{};
  const std::uint64_t storage_host =
      storage_ready
          ? seed_storage->private_total.host +
                action_storage->private_total.host +
                fold_storage->private_total.host + execution_payload.host
          : 0u;
  const std::uint64_t storage_tile =
      storage_ready
          ? seed_storage->private_total.tile +
                action_storage->private_total.tile +
                fold_storage->private_total.tile + execution_payload.tile
          : 0u;
  const std::uint64_t action_delta =
      kOuter * 63u * action->graph().memory.logical_bytes;
  const std::uint64_t phase_live = std::max({seed.graph().memory.live_bytes,
                                             action->graph().memory.live_bytes,
                                             fold.graph().memory.live_bytes});
  const auto expected_logical = [&](const PipelinePlan &plan,
                                    const std::uint64_t inner) {
    const std::uint64_t infrastructure = plan.state_bytes + plan.prepared_bytes;
    return infrastructure +
           kOuter * (seed.graph().memory.logical_bytes +
                     inner * action->graph().memory.logical_bytes +
                     fold.graph().memory.logical_bytes);
  };
  const auto prepared_components_exact = [](const PipelinePlan &plan) {
    return plan.prepared_bytes ==
           plan.prepared_buffer_bytes + plan.prepared_host_bytes +
               plan.prepared_tile_bytes + plan.prepared_native_bytes;
  };
  const bool plans_ready = one_plan && two_plan && three_plan && many_plan;
  const std::uint64_t prepared_delta =
      one_plan && many_plan &&
              many_plan->prepared_bytes >= one_plan->prepared_bytes
          ? many_plan->prepared_bytes - one_plan->prepared_bytes
          : 0u;
  const std::uint64_t host_step =
      two_plan && three_plan &&
              three_plan->prepared_host_bytes >= two_plan->prepared_host_bytes
          ? three_plan->prepared_host_bytes - two_plan->prepared_host_bytes
          : 0u;
  const std::uint64_t route_step =
      two_plan && three_plan &&
              three_plan->prepared_bytes >= two_plan->prepared_bytes
          ? three_plan->prepared_bytes - two_plan->prepared_bytes
          : 0u;
  const bool monotonic_prepared_shape =
      plans_ready && one_plan->prepared_bytes < two_plan->prepared_bytes &&
      two_plan->prepared_bytes < three_plan->prepared_bytes &&
      three_plan->prepared_bytes < many_plan->prepared_bytes;
  const bool affine_prepared_shape =
      plans_ready && route_step != 0u &&
      many_plan->prepared_bytes - two_plan->prepared_bytes == 62u * route_step;
  // Native backend reservations are deliberately structural rather than a
  // public constant-per-route coefficient. Metal ICB tails use calibrated
  // bit-ceil size classes, and each backend owns its exact command/descriptor
  // equation. Keep this cross-backend contract on shared component and shape
  // invariants instead of mirroring either private planner.
  const bool backend_structural_prepared_shape =
      !accelerator ||
      (plans_ready &&
       one_plan->prepared_host_bytes < two_plan->prepared_host_bytes &&
       two_plan->prepared_host_bytes < three_plan->prepared_host_bytes &&
       three_plan->prepared_host_bytes < many_plan->prepared_host_bytes &&
       one_plan->prepared_native_bytes < two_plan->prepared_native_bytes &&
       two_plan->prepared_native_bytes < three_plan->prepared_native_bytes &&
       three_plan->prepared_native_bytes < many_plan->prepared_native_bytes &&
       one_plan->prepared_buffer_bytes == many_plan->prepared_buffer_bytes &&
       one_plan->prepared_tile_bytes == many_plan->prepared_tile_bytes);
  const bool compact_route_prepared_shape =
      monotonic_prepared_shape && backend_structural_prepared_shape &&
      (accelerator || affine_prepared_shape);
  const bool cpu_prepared_shape =
      !cpu ||
      (plans_ready && seed_storage && action_storage && fold_storage &&
       storage_host != 0u && storage_tile != 0u && prepared_delta != 0u &&
       host_step != 0u && one_plan->prepared_host_bytes >= storage_host &&
       two_plan->prepared_host_bytes > one_plan->prepared_host_bytes &&
       three_plan->prepared_host_bytes > two_plan->prepared_host_bytes &&
       many_plan->prepared_host_bytes > three_plan->prepared_host_bytes &&
       many_plan->prepared_host_bytes - two_plan->prepared_host_bytes ==
           62u * host_step &&
       one_plan->prepared_buffer_bytes == many_plan->prepared_buffer_bytes &&
       one_plan->prepared_tile_bytes == storage_tile &&
       two_plan->prepared_tile_bytes == storage_tile &&
       three_plan->prepared_tile_bytes == storage_tile &&
       many_plan->prepared_tile_bytes == storage_tile &&
       one_plan->prepared_native_bytes == 0u &&
       many_plan->prepared_native_bytes == 0u);
  if (!plans_ready || action->graph().memory.logical_bytes == 0u ||
      !prepared_components_exact(*one_plan) ||
      !prepared_components_exact(*two_plan) ||
      !prepared_components_exact(*three_plan) ||
      !prepared_components_exact(*many_plan) || !compact_route_prepared_shape ||
      !cpu_prepared_shape || one_plan->state_bytes != many_plan->state_bytes ||
      one_plan->transient_bytes != many_plan->transient_bytes ||
      one_plan->scratch_bytes != many_plan->scratch_bytes ||
      one_plan->scratch_count != many_plan->scratch_count ||
      one_plan->logical_bytes != expected_logical(*one_plan, 1u) ||
      many_plan->logical_bytes != expected_logical(*many_plan, 64u) ||
      one_plan->logical_bytes + action_delta + prepared_delta !=
          many_plan->logical_bytes ||
      one_plan->live_bytes !=
          one_plan->state_bytes + one_plan->prepared_bytes + phase_live ||
      one_plan->live_bytes + prepared_delta != many_plan->live_bytes ||
      one_plan->physical_bytes != one_plan->state_bytes +
                                      one_plan->prepared_bytes +
                                      one_plan->transient_bytes ||
      one_plan->physical_bytes + prepared_delta != many_plan->physical_bytes ||
      one_plan->peak_bytes + prepared_delta != many_plan->peak_bytes ||
      one_plan->allocation_count != many_plan->allocation_count ||
      one_plan->resource_count != many_plan->resource_count ||
      one_plan->prepared_template_count != kOuter + 1u + 3u ||
      two_plan->prepared_template_count != kOuter + 2u + 3u ||
      three_plan->prepared_template_count != kOuter + 2u + 3u ||
      many_plan->prepared_template_count != kOuter + 2u + 3u ||
      one_plan->prepared_command_count != kOuter * (1u + 2u) ||
      two_plan->prepared_command_count != kOuter * (2u + 2u) ||
      three_plan->prepared_command_count != kOuter * (3u + 2u) ||
      many_plan->prepared_command_count != kOuter * (64u + 2u)) {
    if (one_plan && many_plan) {
      std::fprintf(
          stderr,
          "nested reuse plan state=%llu/%llu transient=%llu/%llu "
          "prepared=%llu/%llu scratch=%llu:%llu/%llu:%llu "
          "logical=%llu/%llu action=%llu delta=%llu live=%llu/%llu "
          "peak=%llu/%llu physical=%llu/%llu allocations=%llu/%llu "
          "templates=%llu/%llu commands=%llu/%llu "
          "host=%llu/%llu/%llu/%llu native=%llu/%llu/%llu/%llu "
          "route_step=%llu host_step=%llu\n",
          static_cast<unsigned long long>(one_plan->state_bytes),
          static_cast<unsigned long long>(many_plan->state_bytes),
          static_cast<unsigned long long>(one_plan->transient_bytes),
          static_cast<unsigned long long>(many_plan->transient_bytes),
          static_cast<unsigned long long>(one_plan->prepared_bytes),
          static_cast<unsigned long long>(many_plan->prepared_bytes),
          static_cast<unsigned long long>(one_plan->scratch_bytes),
          static_cast<unsigned long long>(one_plan->scratch_count),
          static_cast<unsigned long long>(many_plan->scratch_bytes),
          static_cast<unsigned long long>(many_plan->scratch_count),
          static_cast<unsigned long long>(one_plan->logical_bytes),
          static_cast<unsigned long long>(many_plan->logical_bytes),
          static_cast<unsigned long long>(action->graph().memory.logical_bytes),
          static_cast<unsigned long long>(action_delta),
          static_cast<unsigned long long>(one_plan->live_bytes),
          static_cast<unsigned long long>(many_plan->live_bytes),
          static_cast<unsigned long long>(one_plan->peak_bytes),
          static_cast<unsigned long long>(many_plan->peak_bytes),
          static_cast<unsigned long long>(one_plan->physical_bytes),
          static_cast<unsigned long long>(many_plan->physical_bytes),
          static_cast<unsigned long long>(one_plan->allocation_count),
          static_cast<unsigned long long>(many_plan->allocation_count),
          static_cast<unsigned long long>(one_plan->prepared_template_count),
          static_cast<unsigned long long>(many_plan->prepared_template_count),
          static_cast<unsigned long long>(one_plan->prepared_command_count),
          static_cast<unsigned long long>(many_plan->prepared_command_count),
          static_cast<unsigned long long>(one_plan->prepared_host_bytes),
          static_cast<unsigned long long>(two_plan->prepared_host_bytes),
          static_cast<unsigned long long>(three_plan->prepared_host_bytes),
          static_cast<unsigned long long>(many_plan->prepared_host_bytes),
          static_cast<unsigned long long>(one_plan->prepared_native_bytes),
          static_cast<unsigned long long>(two_plan->prepared_native_bytes),
          static_cast<unsigned long long>(three_plan->prepared_native_bytes),
          static_cast<unsigned long long>(many_plan->prepared_native_bytes),
          static_cast<unsigned long long>(route_step),
          static_cast<unsigned long long>(host_step));
    }
    return 2;
  }

  auto one = std::move(one_builder)
                 .budget(MemoryBudget{.bytes = one_plan->peak_bytes})
                 .prepare();
  auto many = std::move(many_builder)
                  .budget(MemoryBudget{.bytes = many_plan->peak_bytes})
                  .prepare();
  const graph::Fingerprint action_fingerprint = action->fingerprint();
  if (!one || !many || !action_fingerprint ||
      one->fingerprint() == many->fingerprint() ||
      ActionOwnerCount(*one, action_fingerprint) != 1u ||
      ActionOwnerCount(*many, action_fingerprint) != 2u) {
    std::fprintf(
        stderr,
        "nested reuse prepare one=%u/%u many=%u/%u fingerprints=%u/%u "
        "owners=%llu/%llu\n",
        static_cast<unsigned>(one.ok()), static_cast<unsigned>(one.reason()),
        static_cast<unsigned>(many.ok()), static_cast<unsigned>(many.reason()),
        static_cast<unsigned>(one ? static_cast<bool>(one->fingerprint())
                                  : false),
        static_cast<unsigned>(many ? static_cast<bool>(many->fingerprint())
                                   : false),
        static_cast<unsigned long long>(
            one ? ActionOwnerCount(*one, action_fingerprint) : 0u),
        static_cast<unsigned long long>(
            many ? ActionOwnerCount(*many, action_fingerprint) : 0u));
    return 3;
  }

  auto short_builder = make_builder.template operator()<64u>(*output_short);
  auto rejected = std::move(short_builder)
                      .budget(MemoryBudget{.bytes = many_plan->peak_bytes - 1u})
                      .prepare();
  if (rejected || rejected.reason() != Reason::PipelineMemoryBudget) {
    std::fprintf(stderr, "nested short budget status=%u reason=%u peak=%llu\n",
                 static_cast<unsigned>(rejected.ok()),
                 static_cast<unsigned>(rejected.reason()),
                 static_cast<unsigned long long>(many_plan->peak_bytes));
    return 4;
  }
  return 0;
}

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

[[nodiscard]] int
CheckNestedAggregateStats(rund::compute::Device &device,
                          const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t first_maximum = 5u;
  constexpr std::size_t first_tile = 3u;
  constexpr std::size_t first_inner = 2u;
  constexpr std::size_t first_outer = CeilDiv(first_maximum, first_tile);
  constexpr std::size_t second_maximum = 8u;
  constexpr std::size_t second_tile = 3u;
  constexpr std::size_t second_inner = 4u;
  constexpr std::size_t second_outer = CeilDiv(second_maximum, second_tile);
  constexpr std::array<std::uint32_t, 1u> first_initial{100u};
  constexpr std::array<std::uint32_t, 1u> second_initial{200u};
  constexpr std::array<std::uint32_t, 1u> first_count_values{0u};
  constexpr std::array<std::uint32_t, 1u> second_count_values{4u};
  constexpr std::uint64_t executed_outer = 2u;
  constexpr std::uint64_t skipped_outer =
      first_outer + (second_outer - executed_outer);
  constexpr std::uint64_t executed_inner = executed_outer * second_inner;
  constexpr std::uint64_t skipped_inner =
      first_outer * first_inner +
      (second_outer - executed_outer) * second_inner;
  constexpr std::uint32_t second_expected =
      second_initial[0u] + (1u + second_inner) + (2u + second_inner);

  auto seed = MakeNestedTerminalSeedProgram(device);
  auto action = on(device)
                    .map<std::uint32_t>("nested-window-aggregate-action", 1u,
                                        [](auto value) { return value + 1u; })
                    .compile();
  auto fold = MakeNestedFailureFoldProgram(device, false);
  auto first_outer_seed = device.upload<std::uint32_t>(first_initial);
  auto second_outer_seed = device.upload<std::uint32_t>(second_initial);
  auto first_count = device.upload<std::uint32_t>(first_count_values);
  auto second_count = device.upload<std::uint32_t>(second_count_values);
  auto first_output = device.buffer<std::uint32_t>(1u);
  auto second_output = device.buffer<std::uint32_t>(1u);
  if (!seed || !action || !fold || !first_outer_seed || !second_outer_seed ||
      !first_count || !second_count || !first_output || !second_output) {
    return 1;
  }

  const auto first_body = tile_repeat<first_inner>(*seed, *action, *fold);
  const auto second_body = tile_repeat<second_inner>(*seed, *action, *fold);
  auto builder = pipeline(device);
  builder
      .windows<first_maximum, first_tile>(
          first_body, rund::compute::window(*first_count),
          read(*first_outer_seed), write_final(*first_output))
      .windows<second_maximum, second_tile>(
          second_body, rund::compute::window(*second_count),
          read(*second_outer_seed), write_final(*second_output));
  auto prepared = std::move(builder).prepare();
  std::array<std::uint32_t, 1u> first_actual{};
  std::array<std::uint32_t, 1u> second_actual{};
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  if (!prepared || !ran || !prepared->read(*first_output, first_actual) ||
      !prepared->read(*second_output, second_actual) ||
      first_actual != first_initial || second_actual[0u] != second_expected ||
      stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.executed_outer_window_count != executed_outer ||
      stats.pipeline.skipped_outer_window_count != skipped_outer ||
      stats.pipeline.executed_inner_iteration_count != executed_inner ||
      stats.pipeline.skipped_inner_iteration_count != skipped_inner ||
      stats.control.iteration_count != executed_outer ||
      stats.control.skipped_iteration_count != skipped_outer ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "nested aggregate backend=%u prepared=%u status=%u reason=%u "
        "outputs=%u/%u:%u/%u outer=%llu/%llu:%llu/%llu "
        "inner=%llu/%llu:%llu/%llu control=%llu/%llu submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        first_actual[0u], first_initial[0u], second_actual[0u], second_expected,
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(executed_outer),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(skipped_outer),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(executed_inner),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(skipped_inner),
        static_cast<unsigned long long>(stats.control.iteration_count),
        static_cast<unsigned long long>(stats.control.skipped_iteration_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 2;
  }
  return 0;
}

int CheckNestedRetainedReuse(rund::compute::Device &device,
                             const NestedSeed &seed, const NestedFold &fold) {
  return CheckRetainedReuse(device, seed, fold);
}
int CheckNestedMaximumPlan(rund::compute::Device &device,
                           const NestedAction &action, const NestedFold &fold) {
  return CheckMaximumPlan(device, action, fold);
}

} // namespace rund::node::test_contract::window

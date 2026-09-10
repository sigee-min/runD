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

int CheckNestedRetainedReuse(rund::compute::Device &device,
                             const NestedSeed &seed, const NestedFold &fold) {
  return CheckRetainedReuse(device, seed, fold);
}

} // namespace rund::node::test_contract::window

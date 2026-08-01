#include "local.hpp"

#include <cstdio>
#include <utility>

namespace rund::node::test_contract::window {
namespace {

template <class Value>
[[nodiscard]] int CheckPlanFormat(Device &device) {
  using namespace rund::compute;
  constexpr std::size_t tile = 8192u;
  constexpr std::size_t maximum = 516096u;
  constexpr std::size_t windows = CeilDiv(maximum, tile);
  static_assert(windows == 63u);
  static_assert(ResidentWindow<maximum, tile>::window_count == windows);

  auto single_body = Fold<Value, tile, tile>(device);
  auto large_body = Fold<Value, maximum, tile>(device);
  constexpr std::array<Value, 1u> seed{Value::from_raw(0)};
  constexpr std::array<std::uint32_t, 1u> single_count{
      static_cast<std::uint32_t>(tile)};
  constexpr std::array<std::uint32_t, 1u> large_count{
      static_cast<std::uint32_t>(maximum)};
  auto single_seed = device.upload<Value>(std::span<const Value>{seed});
  auto single_input = device.buffer<Value>(tile);
  auto single_output = device.buffer<Value>(1u);
  auto single_total = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{single_count});
  auto large_seed = device.upload<Value>(std::span<const Value>{seed});
  auto large_input = device.buffer<Value>(maximum);
  auto large_output = device.buffer<Value>(1u);
  auto large_total =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{large_count});
  if (!single_body || !large_body || !single_seed || !single_input ||
      !single_output || !single_total || !large_seed || !large_input ||
      !large_output || !large_total) {
    return 1;
  }

  auto single = pipeline(device);
  single.template windows<tile, tile>(
      *single_body, rund::compute::window(*single_total),
      read(*single_seed, *single_input), write_final(*single_output));
  const auto single_plan = single.plan();

  auto large = pipeline(device);
  large.template windows<maximum, tile>(
      *large_body, rund::compute::window(*large_total),
      read(*large_seed, *large_input), write_final(*large_output));
  const auto large_plan = large.plan();
  if (!single_plan || !large_plan || single_plan->transient_bytes == 0u ||
      single_plan->total_bytes !=
          single_plan->persistent_bytes + single_plan->peak_bytes ||
      large_plan->total_bytes !=
          large_plan->persistent_bytes + large_plan->peak_bytes ||
      large_plan->transient_bytes != single_plan->transient_bytes ||
      large_plan->allocation_count != single_plan->allocation_count ||
      large_plan->reuse_count <= single_plan->reuse_count ||
      single_plan->publish_count != 1u || large_plan->publish_count != 1u ||
      single_plan->publish_bytes != sizeof(Value) ||
      large_plan->publish_bytes != sizeof(Value) ||
      single_plan->persistent_bytes <
          (tile + 2u) * sizeof(Value) + sizeof(std::uint32_t) ||
      large_plan->persistent_bytes <
          (maximum + 2u) * sizeof(Value) + sizeof(std::uint32_t) ||
      large_plan->peak_iteration >= windows || large_plan->peak_step != 0u) {
    const auto dump = [](const char *const name, const PipelinePlan &plan) {
      std::fprintf(
          stderr,
          "window plan %s persistent=%llu state=%llu transient=%llu "
          "prepared=%llu view=%llu peak=%llu total=%llu "
          "alloc=%llu reuse=%llu publish=%llu/%llu step=%llu iteration=%llu\n",
          name, static_cast<unsigned long long>(plan.persistent_bytes),
          static_cast<unsigned long long>(plan.state_bytes),
          static_cast<unsigned long long>(plan.transient_bytes),
          static_cast<unsigned long long>(plan.prepared_bytes),
          static_cast<unsigned long long>(plan.view_bytes),
          static_cast<unsigned long long>(plan.peak_bytes),
          static_cast<unsigned long long>(plan.total_bytes),
          static_cast<unsigned long long>(plan.allocation_count),
          static_cast<unsigned long long>(plan.reuse_count),
          static_cast<unsigned long long>(plan.publish_count),
          static_cast<unsigned long long>(plan.publish_bytes),
          static_cast<unsigned long long>(plan.peak_step),
          static_cast<unsigned long long>(plan.peak_iteration));
    };
    if (single_plan) {
      dump("single", *single_plan);
    } else {
      std::fprintf(stderr, "window plan single rejected reason=%u\n",
                   static_cast<unsigned>(single_plan.reason()));
    }
    if (large_plan) {
      dump("large", *large_plan);
    } else {
      std::fprintf(stderr, "window plan large rejected reason=%u\n",
                   static_cast<unsigned>(large_plan.reason()));
    }
    return 2;
  }

  const MemoryStats before = device.memory();
  if (large_plan->peak_bytes == 0u) {
    return 3;
  }
  auto rejected =
      std::move(large)
          .budget(MemoryBudget{.bytes = large_plan->peak_bytes - 1u})
          .prepare();
  const MemoryStats after = device.memory();
  if (rejected || rejected.reason() != Reason::PipelineMemoryBudget ||
      !NoAllocation(before, after)) {
    const auto dump = [](const char *const name, const MemoryCounter &left,
                         const MemoryCounter &right) {
      if (!Same(left, right)) {
        std::fprintf(stderr,
                     "window plan memory %s before=%llu/%llu/%llu/%llu/%llu "
                     "after=%llu/%llu/%llu/%llu/%llu\n",
                     name, static_cast<unsigned long long>(left.current),
                     static_cast<unsigned long long>(left.peak),
                     static_cast<unsigned long long>(left.cumulative),
                     static_cast<unsigned long long>(left.reused),
                     static_cast<unsigned long long>(left.budget),
                     static_cast<unsigned long long>(right.current),
                     static_cast<unsigned long long>(right.peak),
                     static_cast<unsigned long long>(right.cumulative),
                     static_cast<unsigned long long>(right.reused),
                     static_cast<unsigned long long>(right.budget));
      }
    };
    std::fprintf(stderr, "window plan rejection ok=%u reason=%u\n",
                 static_cast<unsigned>(rejected.ok()),
                 static_cast<unsigned>(rejected.reason()));
    dump("host", before.host, after.host);
    dump("frame", before.frame, after.frame);
    dump("tile", before.tile, after.tile);
    dump("resident", before.resident, after.resident);
    dump("staging", before.staging, after.staging);
    dump("device", before.device, after.device);
    dump("transfer", before.transfer, after.transfer);
    return 4;
  }

  auto prepared = std::move(single)
                      .budget(MemoryBudget{.bytes = single_plan->peak_bytes})
                      .prepare();
  if (!prepared || prepared->plan() != *single_plan) {
    std::fprintf(stderr,
                 "window plan prepare status=%u reason=%u budget=%llu "
                 "peak=%llu\n",
                 static_cast<unsigned>(prepared.ok()),
                 static_cast<unsigned>(prepared.reason()),
                 static_cast<unsigned long long>(single_plan->peak_bytes),
                 static_cast<unsigned long long>(
                     prepared ? prepared->plan().peak_bytes : 0u));
    return 5;
  }
  return 0;
}

} // namespace

[[nodiscard]] int CheckPlan(Device &device) {
  const int fixed32 = CheckPlanFormat<Fixed<16, 16>>(device);
  if (fixed32 != 0) {
    return fixed32;
  }
  const int fixed64 = CheckPlanFormat<Fixed<20, 44>>(device);
  return fixed64 == 0 ? 0 : 10 + fixed64;
}

} // namespace rund::node::test_contract::window

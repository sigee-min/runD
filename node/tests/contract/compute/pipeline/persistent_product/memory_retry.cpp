#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "fixture.hpp"

#include "src/compute/device/state.hpp"
#include "src/compute/virtual/run/sliding/internal.hpp"
#include "src/compute/virtual/state.hpp"

#include <cstdio>

namespace rund_node_test_persistent_product {
namespace {

namespace sliding = rund::compute::detail::sliding_product_detail;

struct RenewalCompetition final {
  rund::storage::Reservation competitor{};
};

[[nodiscard]] bool
same_live_memory(const rund::storage::Report &left,
                 const rund::storage::Report &right) noexcept {
  return left.capacity_bytes == right.capacity_bytes &&
         left.physical_bytes == right.physical_bytes &&
         left.allocated_bytes == right.allocated_bytes &&
         left.reserved_bytes == right.reserved_bytes &&
         left.available_bytes == right.available_bytes;
}

[[nodiscard]] bool
same_memory_shape(const rund::storage::Report &left,
                  const rund::storage::Report &right) noexcept {
  return left.capacity_bytes == right.capacity_bytes &&
         left.physical_bytes == right.physical_bytes;
}

[[nodiscard]] bool
reserve_competitor(rund::compute::detail::DeviceState &device,
                   const rund::storage::Report &baseline,
                   RenewalCompetition &competition) noexcept {
  if (baseline.available_bytes == 0u) {
    return false;
  }
  competition.competitor =
      device.pipeline_memory_budget.reserve(baseline.available_bytes);
  return static_cast<bool>(competition.competitor);
}

[[nodiscard]] bool
matches_blocked_renewal(const RenewalCompetition &competition,
                        const rund::storage::Report &baseline,
                        const rund::compute::Status &blocked,
                        const sliding::SlidingProductOwner &owner,
                        const rund::storage::Report &contended) noexcept {
  return static_cast<bool>(competition.competitor) && contended && !blocked &&
         blocked.reason() ==
             rund::compute::Reason::DevicePipelineMemoryCapacity &&
         owner.memory == nullptr && !owner.capacity &&
         same_memory_shape(contended, baseline) &&
         contended.allocated_bytes == baseline.allocated_bytes &&
         contended.reserved_bytes ==
             baseline.reserved_bytes + baseline.available_bytes &&
         contended.available_bytes == 0u &&
         contended.reservation_count == baseline.reservation_count + 1u &&
         contended.commit_count == baseline.commit_count &&
         contended.refund_count == baseline.refund_count &&
         contended.rejection_count == baseline.rejection_count;
}

[[nodiscard]] bool
matches_recovered_renewal(const rund::storage::Report &baseline,
                          const rund::compute::Status &recovered,
                          const sliding::SlidingProductOwner &owner,
                          const rund::storage::Report &renewed) noexcept {
  return recovered && owner.memory == nullptr && owner.capacity &&
         !owner.capacity.committed() &&
         owner.capacity.max_allocated_bytes() == baseline.available_bytes &&
         renewed && same_memory_shape(renewed, baseline) &&
         renewed.allocated_bytes == baseline.allocated_bytes &&
         renewed.reserved_bytes ==
             baseline.reserved_bytes + baseline.available_bytes &&
         renewed.available_bytes == 0u &&
         renewed.reservation_count == baseline.reservation_count + 2u &&
         renewed.commit_count == baseline.commit_count &&
         renewed.refund_count == baseline.refund_count + 1u &&
         renewed.rejection_count == baseline.rejection_count;
}

[[nodiscard]] bool
matches_restored_memory(const rund::storage::Report &baseline,
                        const rund::storage::Report &after) noexcept {
  return after && same_live_memory(after, baseline) &&
         after.reservation_count == baseline.reservation_count + 2u &&
         after.commit_count == baseline.commit_count &&
         after.refund_count == baseline.refund_count + 2u &&
         after.rejection_count == baseline.rejection_count;
}

void report_failure(const RenewalCompetition &competition,
                    const rund::storage::Report &baseline,
                    const rund::compute::Status &blocked,
                    const rund::storage::Report &contended,
                    const rund::compute::Status &recovered,
                    const rund::storage::Report &renewed,
                    const rund::storage::Report &after) noexcept {
  std::fprintf(stderr,
               "persistent memory retry competitor=%d statuses=%u/%u "
               "baseline=%llu/%llu/%llu/%llu/%llu/%llu "
               "contended=%llu/%llu/%llu/%llu/%llu/%llu "
               "renewed=%llu/%llu/%llu/%llu/%llu/%llu "
               "after=%llu/%llu/%llu/%llu/%llu/%llu\n",
               competition.competitor ? 1 : 0,
               static_cast<unsigned>(blocked.reason()),
               static_cast<unsigned>(recovered.reason()),
               static_cast<unsigned long long>(baseline.allocated_bytes),
               static_cast<unsigned long long>(baseline.reserved_bytes),
               static_cast<unsigned long long>(baseline.reservation_count),
               static_cast<unsigned long long>(baseline.commit_count),
               static_cast<unsigned long long>(baseline.refund_count),
               static_cast<unsigned long long>(baseline.rejection_count),
               static_cast<unsigned long long>(contended.allocated_bytes),
               static_cast<unsigned long long>(contended.reserved_bytes),
               static_cast<unsigned long long>(contended.reservation_count),
               static_cast<unsigned long long>(contended.commit_count),
               static_cast<unsigned long long>(contended.refund_count),
               static_cast<unsigned long long>(contended.rejection_count),
               static_cast<unsigned long long>(renewed.allocated_bytes),
               static_cast<unsigned long long>(renewed.reserved_bytes),
               static_cast<unsigned long long>(renewed.reservation_count),
               static_cast<unsigned long long>(renewed.commit_count),
               static_cast<unsigned long long>(renewed.refund_count),
               static_cast<unsigned long long>(renewed.rejection_count),
               static_cast<unsigned long long>(after.allocated_bytes),
               static_cast<unsigned long long>(after.reserved_bytes),
               static_cast<unsigned long long>(after.reservation_count),
               static_cast<unsigned long long>(after.commit_count),
               static_cast<unsigned long long>(after.refund_count),
               static_cast<unsigned long long>(after.rejection_count));
}

} // namespace

bool CheckPersistentMemoryRetry(const rund::compute::Backend backend,
                                bool &unavailable) noexcept {
  PreparedProduct prepared{};
  if (!PrepareProduct(backend, 5u, prepared, unavailable)) {
    return unavailable;
  }
  const auto &device = prepared.state->pipeline->device;
  if (device == nullptr) {
    return false;
  }
  const rund::storage::Report baseline =
      device->pipeline_memory_budget.report();
  if (!baseline || baseline.available_bytes == 0u) {
    return false;
  }

  sliding::SlidingProductOwner owner{};
  const auto run = std::make_shared<sliding::SlidingProductRun>();
  RenewalCompetition competition{};
  if (!reserve_competitor(*device, baseline, competition)) {
    return false;
  }
  const rund::compute::Status blocked =
      sliding::renew_persistent_memory(*prepared.state, owner, *run);
  const rund::storage::Report contended =
      device->pipeline_memory_budget.report();
  const bool blocked_exact =
      matches_blocked_renewal(competition, baseline, blocked, owner, contended);
  const bool released = static_cast<bool>(competition.competitor.refund());
  const rund::compute::Status recovered =
      sliding::renew_persistent_memory(*prepared.state, owner, *run);
  const rund::storage::Report renewed = device->pipeline_memory_budget.report();
  const bool recovered_exact =
      matches_recovered_renewal(baseline, recovered, owner, renewed);
  const bool refunded = static_cast<bool>(owner.capacity.refund());
  const rund::storage::Report after = device->pipeline_memory_budget.report();
  const bool restored = matches_restored_memory(baseline, after);
  if (!blocked_exact || !released || !recovered_exact || !refunded ||
      !restored) {
    report_failure(competition, baseline, blocked, contended, recovered,
                   renewed, after);
    return false;
  }
  return true;
}

} // namespace rund_node_test_persistent_product

#endif

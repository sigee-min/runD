#include "test/assert.hpp"

#include "task/coroutine/allocation.hpp"

#include <rund/storage.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using rund::ReasonCode;
using rund::storage::Budget;
using rund::storage::Reservation;
using rund::storage::Usage;

static_assert(std::is_copy_constructible_v<Budget>);
static_assert(std::is_copy_assignable_v<Budget>);
static_assert(!std::is_copy_constructible_v<Reservation>);
static_assert(!std::is_copy_assignable_v<Reservation>);
static_assert(std::is_nothrow_move_constructible_v<Reservation>);
static_assert(std::is_nothrow_move_assignable_v<Reservation>);

int BasicHierarchyAndStateContract() {
  const Budget missing{};
  TEST_ASSERT(!missing);
  TEST_ASSERT(missing.code() == ReasonCode::StorageBudgetInvalid);
  TEST_ASSERT(!missing.report());
  TEST_ASSERT(!Budget{0u});

  runtime_task_allocation::FailNext();
  const Budget allocation_failed{100u};
  TEST_ASSERT(!allocation_failed);
  TEST_ASSERT(allocation_failed.code() ==
              ReasonCode::StorageBudgetAllocationFailed);

  Budget root{100u};
  TEST_ASSERT(root);
  runtime_task_allocation::FailNext();
  const Budget child_allocation_failed = root.child(60u);
  TEST_ASSERT(!child_allocation_failed);
  TEST_ASSERT(child_allocation_failed.code() ==
              ReasonCode::StorageBudgetAllocationFailed);

  Budget child = root.child(60u);
  Budget sibling = root.child(100u);
  Budget grandchild = child.child(50u);
  TEST_ASSERT(child);
  TEST_ASSERT(sibling);
  TEST_ASSERT(grandchild);
  TEST_ASSERT(!child.child(61u));
  TEST_ASSERT(root.report().available_bytes == 100u);
  TEST_ASSERT(child.report().available_bytes == 60u);

  Reservation held = child.reserve(40u);
  TEST_ASSERT(held);
  TEST_ASSERT(!held.committed());
  TEST_ASSERT(held.max_allocated_bytes() == 40u);
  TEST_ASSERT(child.report().reserved_bytes == 40u);
  TEST_ASSERT(root.report().reserved_bytes == 40u);
  TEST_ASSERT(root.report().available_bytes == 60u);

  Reservation rejected = sibling.reserve(70u);
  TEST_ASSERT(!rejected);
  TEST_ASSERT(rejected.code() == ReasonCode::StorageCapacityExceeded);
  TEST_ASSERT(root.report().rejection_count == 1u);
  TEST_ASSERT(sibling.report().rejection_count == 1u);

  const auto committed =
      held.commit(Usage{.physical_bytes = 30u, .allocated_bytes = 35u});
  TEST_ASSERT(committed);
  TEST_ASSERT(held.committed());
  TEST_ASSERT(held.usage().physical_bytes == 30u);
  TEST_ASSERT(held.usage().allocated_bytes == 35u);
  const auto child_committed = child.report();
  TEST_ASSERT(child_committed.reserved_bytes == 0u);
  TEST_ASSERT(child_committed.allocated_bytes == 35u);
  TEST_ASSERT(child_committed.physical_bytes == 30u);
  TEST_ASSERT(child_committed.available_bytes == 25u);
  TEST_ASSERT(child_committed.peak_reserved_bytes == 40u);
  TEST_ASSERT(child_committed.peak_allocated_bytes == 35u);
  TEST_ASSERT(child_committed.peak_physical_bytes == 30u);
  TEST_ASSERT(child_committed.peak_used_bytes == 40u);
  TEST_ASSERT(child_committed.reservation_count == 1u);
  TEST_ASSERT(child_committed.commit_count == 1u);

  const auto refunded = held.refund();
  TEST_ASSERT(refunded);
  TEST_ASSERT(!held);
  TEST_ASSERT(!held.refund());
  TEST_ASSERT(child.report().available_bytes == 60u);
  TEST_ASSERT(root.report().available_bytes == 100u);
  TEST_ASSERT(child.report().refund_count == 1u);
  TEST_ASSERT(root.report().refund_count == 1u);

  {
    Reservation zero = child.reserve(0u);
    TEST_ASSERT(zero);
    TEST_ASSERT(zero.max_allocated_bytes() == 0u);
    TEST_ASSERT(
        zero.commit(Usage{.physical_bytes = 7u, .allocated_bytes = 0u}));
    TEST_ASSERT(child.report().allocated_bytes == 0u);
    TEST_ASSERT(child.report().physical_bytes == 7u);
    TEST_ASSERT(child.report().available_bytes == 60u);
  }
  TEST_ASSERT(child.report().physical_bytes == 0u);
  TEST_ASSERT(root.report().physical_bytes == 0u);

  {
    Reservation pending = grandchild.reserve(20u);
    TEST_ASSERT(pending);
    TEST_ASSERT(grandchild.report().reserved_bytes == 20u);
    TEST_ASSERT(child.report().reserved_bytes == 20u);
    TEST_ASSERT(root.report().reserved_bytes == 20u);
  }
  TEST_ASSERT(grandchild.report().reserved_bytes == 0u);
  TEST_ASSERT(child.report().reserved_bytes == 0u);
  TEST_ASSERT(root.report().reserved_bytes == 0u);
  TEST_ASSERT(grandchild.report().refund_count == 1u);
  TEST_ASSERT(child.report().refund_count == 3u);
  TEST_ASSERT(root.report().refund_count == 3u);

  {
    Reservation first = child.reserve(10u);
    TEST_ASSERT(first);
    Reservation second = std::move(first);
    TEST_ASSERT(!first);
    TEST_ASSERT(second);
    TEST_ASSERT(
        second.commit(Usage{.physical_bytes = 7u, .allocated_bytes = 8u}));
    Reservation third{};
    third = std::move(second);
    TEST_ASSERT(!second);
    TEST_ASSERT(third.committed());
  }
  TEST_ASSERT(child.report().allocated_bytes == 0u);
  TEST_ASSERT(child.report().physical_bytes == 0u);
  TEST_ASSERT(root.report().allocated_bytes == 0u);
  TEST_ASSERT(root.report().physical_bytes == 0u);
  return 0;
}

int CommitValidationContract() {
  Budget root{64u};
  Budget child = root.child(32u);
  Reservation reservation = child.reserve(16u);
  TEST_ASSERT(reservation);

  const auto allocated_too_large =
      reservation.commit(Usage{.physical_bytes = 12u, .allocated_bytes = 17u});
  TEST_ASSERT(!allocated_too_large);
  TEST_ASSERT(allocated_too_large.code() == ReasonCode::StorageCommitInvalid);
  TEST_ASSERT(child.report().reserved_bytes == 16u);

  TEST_ASSERT(
      reservation.commit(Usage{.physical_bytes = 20u, .allocated_bytes = 12u}));
  TEST_ASSERT(
      !reservation.commit(Usage{.physical_bytes = 1u, .allocated_bytes = 1u}));
  TEST_ASSERT(child.report().commit_count == 1u);
  TEST_ASSERT(child.report().allocated_bytes == 12u);
  TEST_ASSERT(child.report().physical_bytes == 20u);
  TEST_ASSERT(child.report().rejection_count == 1u);
  TEST_ASSERT(root.report().rejection_count == 1u);

  Budget physical{1u};
  Reservation maximum = physical.reserve(0u);
  TEST_ASSERT(maximum);
  TEST_ASSERT(maximum.commit(
      Usage{.physical_bytes = std::numeric_limits<std::uint64_t>::max(),
            .allocated_bytes = 0u}));
  Reservation overflow = physical.reserve(0u);
  TEST_ASSERT(overflow);
  const auto overflowed =
      overflow.commit(Usage{.physical_bytes = 1u, .allocated_bytes = 0u});
  TEST_ASSERT(!overflowed);
  TEST_ASSERT(overflowed.code() == ReasonCode::StorageCommitInvalid);
  TEST_ASSERT(physical.report().physical_bytes ==
              std::numeric_limits<std::uint64_t>::max());
  TEST_ASSERT(physical.report().rejection_count == 1u);

  Budget aggregate{1u};
  Budget producer = aggregate.child(1u);
  Reservation aggregate_usage = aggregate.reserve(0u);
  TEST_ASSERT(aggregate_usage);
  TEST_ASSERT(aggregate_usage.commit(
      Usage{.physical_bytes = std::numeric_limits<std::uint64_t>::max(),
            .allocated_bytes = 0u}));
  Reservation producer_usage = producer.reserve(0u);
  TEST_ASSERT(producer_usage);
  const auto ancestor_overflow =
      producer_usage.commit(Usage{.physical_bytes = 1u, .allocated_bytes = 0u});
  TEST_ASSERT(!ancestor_overflow);
  TEST_ASSERT(ancestor_overflow.code() == ReasonCode::StorageCommitInvalid);
  TEST_ASSERT(producer.report().physical_bytes == 0u);
  TEST_ASSERT(producer.report().reserved_bytes == 0u);
  TEST_ASSERT(producer.report().rejection_count == 1u);
  TEST_ASSERT(aggregate.report().physical_bytes ==
              std::numeric_limits<std::uint64_t>::max());
  TEST_ASSERT(aggregate.report().rejection_count == 1u);
  return 0;
}

int PartitionContract() {
  Budget budget{100u};
  Reservation remainder = budget.reserve(80u);
  TEST_ASSERT(remainder);

  Reservation publication = remainder.partition(30u);
  TEST_ASSERT(publication);
  TEST_ASSERT(publication.max_allocated_bytes() == 30u);
  TEST_ASSERT(remainder.max_allocated_bytes() == 50u);
  const auto split = budget.report();
  TEST_ASSERT(split.reserved_bytes == 80u);
  TEST_ASSERT(split.available_bytes == 20u);
  TEST_ASSERT(split.reservation_count == 1u);

  Reservation zero = remainder.partition(0u);
  TEST_ASSERT(zero);
  TEST_ASSERT(zero.max_allocated_bytes() == 0u);
  TEST_ASSERT(remainder.max_allocated_bytes() == 50u);
  TEST_ASSERT(zero.commit(Usage{}));

  Reservation oversized = remainder.partition(51u);
  TEST_ASSERT(!oversized);
  TEST_ASSERT(oversized.code() == ReasonCode::StorageReservationInvalid);
  TEST_ASSERT(remainder.max_allocated_bytes() == 50u);
  TEST_ASSERT(budget.report().reserved_bytes == 80u);
  TEST_ASSERT(budget.report().rejection_count == 1u);

  TEST_ASSERT(
      publication.commit(Usage{.physical_bytes = 20u, .allocated_bytes = 30u}));
  TEST_ASSERT(
      remainder.commit(Usage{.physical_bytes = 40u, .allocated_bytes = 45u}));
  const auto committed = budget.report();
  TEST_ASSERT(committed.reserved_bytes == 0u);
  TEST_ASSERT(committed.allocated_bytes == 75u);
  TEST_ASSERT(committed.physical_bytes == 60u);
  TEST_ASSERT(committed.available_bytes == 25u);
  TEST_ASSERT(committed.commit_count == 3u);
  TEST_ASSERT(committed.reservation_count == 1u);
  TEST_ASSERT(!publication.partition(1u));

  TEST_ASSERT(publication.refund());
  TEST_ASSERT(remainder.refund());
  const auto released = budget.report();
  TEST_ASSERT(released.allocated_bytes == 0u);
  TEST_ASSERT(released.physical_bytes == 0u);
  TEST_ASSERT(released.available_bytes == 100u);
  return 0;
}

int ConcurrentAdmissionContract() {
  constexpr std::uint64_t kCapacity = 32u;
  constexpr std::uint64_t kContenders = 96u;
  Budget root{kCapacity};
  Budget child = root.child(kCapacity);
  std::atomic<std::uint64_t> attempted{0u};
  std::atomic<std::uint64_t> accepted{0u};
  std::atomic<bool> release{false};
  std::vector<std::thread> workers{};
  workers.reserve(kContenders);
  for (std::uint64_t index = 0u; index < kContenders; ++index) {
    workers.emplace_back([child, &attempted, &accepted, &release] {
      Reservation reservation = child.reserve(1u);
      if (reservation && reservation.commit(Usage{.physical_bytes = 1u,
                                                  .allocated_bytes = 1u})) {
        accepted.fetch_add(1u, std::memory_order_relaxed);
      }
      attempted.fetch_add(1u, std::memory_order_release);
      while (!release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    });
  }
  while (attempted.load(std::memory_order_acquire) != kContenders) {
    std::this_thread::yield();
  }

  const auto child_full = child.report();
  const auto root_full = root.report();
  TEST_ASSERT(accepted.load(std::memory_order_relaxed) == kCapacity);
  TEST_ASSERT(child_full.allocated_bytes == kCapacity);
  TEST_ASSERT(child_full.physical_bytes == kCapacity);
  TEST_ASSERT(child_full.available_bytes == 0u);
  TEST_ASSERT(child_full.reservation_count == kCapacity);
  TEST_ASSERT(child_full.commit_count == kCapacity);
  TEST_ASSERT(child_full.rejection_count == kContenders - kCapacity);
  TEST_ASSERT(root_full.allocated_bytes == child_full.allocated_bytes);
  TEST_ASSERT(root_full.rejection_count == child_full.rejection_count);

  release.store(true, std::memory_order_release);
  for (std::thread &worker : workers) {
    worker.join();
  }
  const auto child_empty = child.report();
  TEST_ASSERT(child_empty.allocated_bytes == 0u);
  TEST_ASSERT(child_empty.physical_bytes == 0u);
  TEST_ASSERT(child_empty.available_bytes == kCapacity);
  TEST_ASSERT(child_empty.refund_count == kCapacity);
  return 0;
}

} // namespace

int RunRuntimeStorageContract() {
  if (const int result = BasicHierarchyAndStateContract(); result != 0) {
    return result;
  }
  if (const int result = CommitValidationContract(); result != 0) {
    return result;
  }
  if (const int result = PartitionContract(); result != 0) {
    return result;
  }
  return ConcurrentAdmissionContract();
}

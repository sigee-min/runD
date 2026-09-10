#include "../internal.hpp"

#include <limits>
#include <new>

namespace rund::compute::detail::sliding_product_detail {
namespace {

[[nodiscard]] bool add_bytes(const std::uint64_t left,
                             const std::uint64_t right,
                             std::uint64_t &result) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] constexpr std::uint64_t common_retained_bytes() noexcept {
  // These are the exact logical fixed owners allocated by preparation.  The
  // allocator control-block/capacity envelope remains allocator overhead and
  // is not relabelled as an exact process-heap byte count.
  return sizeof(SlidingProductOwner) + sizeof(SlidingProductRun) +
         sizeof(residency::execution::Plan);
}

} // namespace

Status reserve_persistent_capacity(VirtualPipelineState &state,
                                   storage::Reservation &capacity) noexcept {
  capacity = {};
  if (state.pipeline == nullptr || state.pipeline->device == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const storage::Report report =
      state.pipeline->device->pipeline_memory_budget.report();
  if (!report || report.available_bytes == 0u) {
    return Status::fail(Reason::DevicePipelineMemoryCapacity);
  }
  capacity = state.pipeline->device->pipeline_memory_budget.reserve(
      report.available_bytes);
  return capacity ? Status::success()
                  : Status::fail(Reason::DevicePipelineMemoryCapacity);
}

Status renew_persistent_memory(VirtualPipelineState &state,
                               SlidingProductOwner &owner,
                               SlidingProductRun &run) noexcept {
  static_cast<void>(run);
  if (owner.memory == nullptr) {
    return owner.capacity ? Status::success()
                          : reserve_persistent_capacity(state, owner.capacity);
  }
  // A Known terminal is quiescent before the next Authority lease. Keep the
  // committed native lowering and its exact memory charge; only the request
  // credentials are rebound by prepare_persistent_lowering().
  return Status::success();
}

Status commit_persistent_memory(
    SlidingProductOwner &owner,
    node::accel::detail::PreparedResidencyPersistentSlidingPreparation
        &&candidate,
    node::accel::detail::PreparedResidencyPersistentSlidingPreparation
        &published) noexcept {
  published = {};
  std::uint64_t retained = 0u;
  std::uint64_t admitted = 0u;
  const bool valid =
      owner.capacity && !owner.capacity.committed() &&
      owner.memory == nullptr && candidate &&
      add_bytes(common_retained_bytes(),
                candidate.backend.capability.retained_bytes, retained) &&
      add_bytes(retained, candidate.backend.capability.transient_bytes,
                admitted) &&
      admitted <= owner.capacity.max_allocated_bytes();
  if (!valid) {
    candidate = {};
    static_cast<void>(owner.capacity.refund());
    return Status::fail(Reason::DevicePipelineMemoryCapacity);
  }

  storage::Reservation exact = owner.capacity.partition(admitted);
  const storage::Status refunded = owner.capacity.refund();
  if (!exact || !refunded) {
    candidate = {};
    return Status::fail(Reason::PipelineInvalid);
  }
  const storage::Status committed = exact.commit(storage::Usage{
      .physical_bytes = 0u,
      .allocated_bytes = admitted,
  });
  if (!committed) {
    candidate = {};
    return Status::fail(Reason::PipelineInvalid);
  }
  try {
    owner.memory = std::make_shared<storage::Reservation>(std::move(exact));
  } catch (const std::bad_alloc &) {
    candidate = {};
    return Status::fail(Reason::PipelineMemoryBudget);
  }
  published = std::move(candidate);
  return Status::success();
}

} // namespace rund::compute::detail::sliding_product_detail

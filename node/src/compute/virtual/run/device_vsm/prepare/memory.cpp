#include "../internal.hpp"

#include <kernel/core/checked.hpp>

#include <new>

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

[[nodiscard]] bool add_bytes(const std::uint64_t value,
                             std::uint64_t &total) noexcept {
  return kernel::checked::add(total, value, total);
}

} // namespace

Status reserve_device_vsm_capacity(VirtualPipelineState &state,
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

Status commit_device_vsm_memory(DeviceVsmProductOwner &owner) noexcept {
  std::uint64_t admitted = sizeof(DeviceVsmProductOwner);
  bool valid = owner.capacity && !owner.capacity.committed() &&
               owner.memory == nullptr && owner.proof != nullptr &&
               owner.preparation && owner.input_count != 0u &&
               owner.input_count <= owner.input_staging.size() &&
               add_bytes(sizeof(node::accel::detail::DeviceVsmProof), admitted);
  for (std::size_t index = 0u; valid && index < owner.input_count; ++index) {
    valid = add_bytes(owner.input_staging[index].size(), admitted);
  }
  valid = valid && add_bytes(owner.output_staging.size(), admitted) &&
          add_bytes(owner.preparation.capability.retained_bytes, admitted) &&
          add_bytes(owner.preparation.capability.transient_bytes, admitted) &&
          admitted <= owner.capacity.max_allocated_bytes();
  if (!valid) {
    static_cast<void>(owner.capacity.refund());
    return Status::fail(Reason::DevicePipelineMemoryCapacity);
  }
  storage::Reservation exact = owner.capacity.partition(admitted);
  const storage::Status refunded = owner.capacity.refund();
  if (!exact || !refunded) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const storage::Status committed = exact.commit(storage::Usage{
      .physical_bytes = 0u,
      .allocated_bytes = admitted,
  });
  if (!committed) {
    return Status::fail(Reason::PipelineInvalid);
  }
  try {
    owner.memory = std::make_shared<storage::Reservation>(std::move(exact));
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineMemoryBudget);
  }
  return Status::success();
}

} // namespace rund::compute::detail::device_vsm_product_detail

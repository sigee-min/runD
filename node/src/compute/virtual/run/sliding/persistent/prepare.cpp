#include "../../../../../accel/kernel/prepared/interface/api.hpp"
#include "../internal.hpp"

#include <string_view>

namespace rund::compute::detail::sliding_product_detail {

Status
prepare_persistent_lowering(SlidingProductRun &state,
                            SlidingProductOwner &owner, residency::Pool &pool,
                            const residency::ExecutionLease lease) noexcept {
  if (state.persistent_preparation.backend.lowering != nullptr) {
    auto next = make_persistent_request(state, owner, lease);
    if (!pool.authority().sliding().validate_execution_sliding_lease(owner.plan,
                                                                     lease) ||
        !persistent_sliding_request_valid(
            state.persistent_preparation.backend.capability, next)) {
      discard_staged_roles(owner);
      return Status::fail(Reason::PipelineInvalid);
    }
    if (next.ticket == nullptr || next.ticket->quarantined ||
        !node::accel::detail::persistent_sliding_stage_ticket(
            next.ticket, state.persistent_preparation.backend.capability,
            next)) {
      discard_staged_roles(owner);
      return Status::fail(Reason::PipelineBusy);
    }
    return Status::success();
  }
  const auto request = make_persistent_request(state, owner, lease);
  auto candidate = node::accel::detail::PrepareKernelPipelinePersistentSliding(
      request,
      std::span<
          const node::accel::detail::PreparedResidencyPersistentSlidingRole>{
          owner.persistent_roles.data(), owner.role_count});
  if (!candidate) {
    // The common prepared seam preserves the backend's cold-preparation
    // reason. Project only the explicit capability-unavailable key to the
    // clean-decline boundary; capacity, memory, command, encoding, and other
    // backend-native failures remain terminal.
    const rund::AccelCheck failure = candidate.backend.capability.check;
    candidate = {};
    static_cast<void>(owner.capacity.refund());
    const std::string_view reason =
        failure.reason == nullptr ? std::string_view{} : failure.reason;
    return Status::fail(project_reason(reason, Reason::PipelineInvalid));
  }
  if (candidate.backend.capability.mode != owner.mode) {
    candidate = {};
    static_cast<void>(owner.capacity.refund());
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!node::accel::detail::persistent_sliding_product_capable(
          candidate.backend.capability)) {
    candidate = {};
    static_cast<void>(owner.capacity.refund());
    return Status::fail(Reason::PipelineInvalid);
  }
  return commit_persistent_memory(owner, std::move(candidate),
                                  state.persistent_preparation);
}

} // namespace rund::compute::detail::sliding_product_detail

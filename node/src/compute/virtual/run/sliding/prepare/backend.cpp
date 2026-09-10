#include "../../../../../accel/kernel/prepared/interface/api.hpp"
#include "../../../../backend.hpp"
#include "../internal.hpp"

#include <string_view>

namespace rund::compute::detail::sliding_product_detail {

Status verify_persistent_backend(const SlidingProductOwner &owner) noexcept {
  const auto *const roles = owner.pending_valid
                                ? owner.pending.persistent_roles.data()
                                : owner.persistent_roles.data();
  const std::size_t count =
      owner.pending_valid ? owner.pending.role_count : owner.role_count;
  const std::span<
      const node::accel::detail::PreparedResidencyPersistentSlidingRole>
      role_span{roles, count};
  const auto capability = node::accel::detail::
      QueryPreparedKernelPipelinePersistentSlidingCapability(
          role_span, owner.plan.epoch_count(),
          node::accel::detail::ResidencySlidingMemory::HostCoherent,
          owner.mode);
  if (capability.mode != owner.mode ||
      capability.memory !=
          node::accel::detail::ResidencySlidingMemory::HostCoherent) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!capability.check.ok) {
    const std::string_view reason = capability.check.reason == nullptr
                                        ? std::string_view{}
                                        : capability.check.reason;
    return Status::fail(project_reason(reason, Reason::PipelineInvalid));
  }
  return node::accel::detail::persistent_sliding_product_capable(capability)
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail::sliding_product_detail

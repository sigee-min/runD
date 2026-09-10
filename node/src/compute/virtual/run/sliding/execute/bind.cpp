#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

residency::execution::Sliding
bind_controller(residency::Pool &pool, const SlidingProductOwner &owner,
                const residency::ExecutionLease lease) noexcept {
  if (owner.run == nullptr) {
    return {};
  }
  const auto host_inputs = owner.plan.host_input_regions();
  const auto host_outputs = owner.plan.host_output_regions();
  const std::uint32_t input_capacity =
      std::min(host_inputs[0u].count, host_inputs[1u].count);
  const std::uint32_t output_capacity =
      std::min(host_outputs[0u].count, host_outputs[1u].count);
  auto sliding = pool.authority().sliding();
  residency::execution::Sliding joined = owner.run->sliding;
  if (!joined) {
    joined = residency::execution::Sliding::create_bound(
        residency::execution::SlidingInvocation::direct(owner.plan_owner),
        lease.token, lease.generation, input_capacity, output_capacity);
  }
  if (!joined || !sliding.bind_execution_sliding(joined)) {
    static_cast<void>(
        sliding.abandon_execution_sliding(owner.plan, lease));
    return {};
  }
  owner.run->sliding = joined;
  return joined;
}

} // namespace rund::compute::detail::sliding_product_detail

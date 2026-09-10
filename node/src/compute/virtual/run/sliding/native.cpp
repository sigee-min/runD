#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

ProjectionResult
select_native(SlidingProductRun &state, SlidingProductWork &work,
              const std::uint64_t turn, const std::uint8_t slot,
              node::accel::detail::PreparedResidencySlidingSelection
                  &selection) noexcept {
  if (!work.native && !state.pool->authority().sliding().issue_execution_sliding_native(
                          state.cold->plan, state.sliding, work.projection,
                          work.uses, work.native)) {
    return ProjectionResult::Pending;
  }
  work.terminaled = false;
  const std::size_t count = work.native.input_frames().size();
  if (count == 0u || count > selection.locals.size()) {
    return ProjectionResult::Failed;
  }
  for (std::size_t local = 0u; local < count; ++local) {
    selection.locals[local] = static_cast<std::uint32_t>(local);
  }
  selection.local_count = count;
  selection.read_mask = work.native.active_mask();
  selection.write_mask = work.native.active_mask();
  selection.descriptor_generation = turn + 1u;
  const auto &role = state.cold->roles[slot];
  selection.control_generation =
      role.first_control_generation +
      static_cast<std::uint32_t>(turn) * role.control_generation_stride;
  return ProjectionResult::Ready;
}

} // namespace rund::compute::detail::sliding_product_detail

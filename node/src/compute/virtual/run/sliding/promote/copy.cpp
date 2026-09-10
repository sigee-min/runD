#include "../internal.hpp"

#include "assemble.hpp"

#include <cstring>

namespace rund::compute::detail::sliding_product_detail {

Status copy_promote_frames(
    SlidingProductRun &state, const std::uint64_t coordinate,
    const residency::execution::SlidingPromote &promote) noexcept {
  if (promote.assembles_window()) {
    return assemble_window_frames(state, coordinate, promote);
  }
  const std::size_t bank = coordinate % residency::execution::BankCapacity;
  const BufferWriteView target =
      residency_input_view(*state.cold->pipelines[bank], *state.run);
  const auto hosts = promote.host_frames();
  const auto devices = promote.device_input_frames();
  const residency::FrameRegion region =
      state.cold->plan.device_input_regions()[bank];
  for (std::size_t local = 0u; local < hosts.size(); ++local) {
    if ((promote.transfer_mask() & (std::uint32_t{1u} << local)) == 0u) {
      continue;
    }
    const std::byte *const source =
        virtual_host_input_frame(*state.run, hosts[local]);
    if (!target || source == nullptr || devices[local] < region.first ||
        devices[local] - region.first >= region.count) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t offset =
        static_cast<std::size_t>(devices[local] - region.first) *
        static_cast<std::size_t>(state.run->input_page_bytes);
    if (offset > target.bytes ||
        state.run->input_page_bytes > target.bytes - offset) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::memcpy(target.data + offset, source,
                static_cast<std::size_t>(state.run->input_page_bytes));
  }
  return Status::success();
}

} // namespace rund::compute::detail::sliding_product_detail

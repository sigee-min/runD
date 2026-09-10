#include "../internal.hpp"

#include <cstring>

namespace rund::compute::detail::sliding_product_detail {

Status
copy_drain_frame(SlidingProductRun &state, const BufferReadView source,
                 const residency::FrameRegion device_region,
                 const residency::execution::SlidingDrain &drain) noexcept {
  std::byte *const target =
      virtual_host_output_frame(*state.run, drain.host_frame());
  if (!source || target == nullptr ||
      drain.device_frame() < device_region.first ||
      drain.device_frame() - device_region.first >= device_region.count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t offset =
      static_cast<std::size_t>(drain.device_frame() - device_region.first) *
      static_cast<std::size_t>(state.run->output_page_bytes);
  if (offset > source.bytes ||
      state.run->output_page_bytes > source.bytes - offset) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::memcpy(target, source.data + offset,
              static_cast<std::size_t>(state.run->output_page_bytes));
  return Status::success();
}

} // namespace rund::compute::detail::sliding_product_detail

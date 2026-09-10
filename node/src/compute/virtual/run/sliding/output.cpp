#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

OutputServiceResult service_output(SlidingProductRun &state,
                                   SlidingProductWork &work,
                                   Status &failure) noexcept {
  failure = Status::success();
  const std::uint64_t coordinate = work.projection.coordinate.ordinal;
  const std::size_t bank = coordinate % residency::execution::BankCapacity;
  const BufferReadView source =
      residency_output_view(*state.cold->pipelines[bank]);
  const residency::FrameRegion device_region =
      state.cold->plan.device_output_regions()[bank];
  for (std::size_t output = 0u; output < work.output_count; ++output) {
    const std::size_t use = work.projection.fetch_count + output;
    const OutputServiceResult drained =
        service_drain(state, work, output, use, source, device_region, failure);
    if (drained != OutputServiceResult::Ready) {
      return drained;
    }
    const OutputServiceResult persisted =
        service_persist(state, work, output, use, failure);
    if (persisted != OutputServiceResult::Ready) {
      return persisted;
    }
  }
  return OutputServiceResult::Ready;
}

} // namespace rund::compute::detail::sliding_product_detail

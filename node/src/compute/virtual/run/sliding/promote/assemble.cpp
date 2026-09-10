#include "assemble.hpp"

#include "../fetch/fill.hpp"

#include <cstring>

namespace rund::compute::detail::sliding_product_detail {

Status assemble_window_frames(
    SlidingProductRun &state, const std::uint64_t coordinate,
    const residency::execution::SlidingPromote &promote) noexcept {
  if (state.run == nullptr || state.cold == nullptr ||
      !promote.assembles_window()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t bank = coordinate % residency::execution::BankCapacity;
  const BufferWriteView target =
      residency_input_view(*state.cold->pipelines[bank], *state.run);
  const residency::FrameRegion region =
      state.cold->plan.device_input_regions()[bank];
  const auto hosts = promote.host_frames();
  const auto devices = promote.device_input_frames();
  const auto targets = promote.targets();
  const auto slices = promote.slices();
  std::array<std::uint64_t, residency::execution::UseCapacity> copied{};
  if (!target || devices.size() != targets.size() ||
      targets.size() > copied.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (const residency::execution::WindowFootprintSlice slice : slices) {
    if (slice.source >= hosts.size() || slice.target >= targets.size() ||
        slice.bytes == 0u ||
        slice.source_offset > state.run->input_payload_bytes ||
        slice.bytes > state.run->input_payload_bytes - slice.source_offset ||
        slice.target_offset > targets[slice.target].frame_bytes ||
        slice.bytes > targets[slice.target].frame_bytes - slice.target_offset ||
        slice.bytes >
            std::numeric_limits<std::uint64_t>::max() - copied[slice.target]) {
      return Status::fail(Reason::PipelineInvalid);
    }
    copied[slice.target] += slice.bytes;
    if ((promote.transfer_mask() & (std::uint32_t{1u} << slice.target)) == 0u) {
      continue;
    }
    const std::byte *const source =
        virtual_host_input_frame(*state.run, hosts[slice.source]);
    const std::uint32_t device = devices[slice.target];
    if (source == nullptr || device < region.first ||
        device - region.first >= region.count) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::uint64_t local = device - region.first;
    std::uint64_t target_offset = 0u;
    if (!kernel::checked::mul(local, state.run->input_page_bytes,
                              target_offset) ||
        !kernel::checked::add(target_offset, slice.target_offset,
                              target_offset)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (target_offset > target.bytes ||
        slice.bytes > target.bytes - target_offset) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::memcpy(target.data + static_cast<std::size_t>(target_offset),
                source + static_cast<std::size_t>(slice.source_offset),
                static_cast<std::size_t>(slice.bytes));
  }
  for (std::size_t local = 0u; local < targets.size(); ++local) {
    if (copied[local] != targets[local].bytes ||
        targets[local].frame_bytes != state.run->input_page_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if ((promote.transfer_mask() & (std::uint32_t{1u} << local)) == 0u) {
      continue;
    }
    const std::uint32_t device = devices[local];
    if (device < region.first || device - region.first >= region.count) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t offset = static_cast<std::size_t>(
        (device - region.first) * state.run->input_page_bytes);
    if (offset > target.bytes ||
        state.run->input_page_bytes > target.bytes - offset ||
        !fill_fetch_frame(target.data + offset, targets[local])) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail::sliding_product_detail

#pragma once

#include "internal.hpp"

namespace rund::compute::detail::residency::physical::fetch {

template <typename State>
[[nodiscard]] bool
validate_projection(const execution::Plan &plan, State &state,
                    const execution::SlidingProjection &projection,
                    const std::span<PageUse> uses, const std::size_t use,
                    const std::size_t frame_count,
                    Projection &result) noexcept {
  execution::SlidingProjection expected{};
  std::uint64_t expected_bytes = 0u;
  execution::FetchSource source{};
  execution::FetchReuseSource reuse{};
  execution::Node input{};
  execution::WindowFootprintProjection footprint{};
  const bool window = plan.has_window_footprint();
  if (!state.exact(projection, uses, expected) ||
      !state.invocation.fetch_use(projection, uses, use) ||
      !state.invocation.use_bytes(projection, uses, use, expected_bytes) ||
      ((!window &&
        !plan.project(execution::NodeId{.epoch = projection.coordinate.ordinal,
                                        .phase = execution::Phase::Input},
                      input)) ||
       (window &&
        !plan.window_footprint(projection.coordinate.ordinal, footprint))) ||
      use >= (window ? footprint.source_count : input.input_count) ||
      use >= state.coordinate_input_count ||
      uses[use].key.resource != plan.input_resource() ||
      uses[use].key.page != (window ? footprint.sources[use].key.page
                                    : input.input[use].key.page) ||
      !(window ? plan.canonical_input_source(uses[use].key.page, source)
               : plan.input_source(uses[use].key.page, source)) ||
      source.key !=
          (window ? footprint.sources[use].key : input.input[use].key) ||
      !source.materializes_frame()) {
    return false;
  }
  const std::size_t bank = state.host_bank(projection.coordinate);
  if (bank >= execution::BankCapacity) {
    return false;
  }
  const FrameRegion region = plan.host_input_regions()[bank];
  if (region.tier != FrameTier::Host || region.role != FrameRole::Input ||
      state.input_stride == 0u || state.input_stride > region.count ||
      region.first > frame_count ||
      state.input_stride > frame_count - region.first) {
    return false;
  }
  const auto ring = state.input_ring(projection.coordinate);
  if (std::any_of(ring.begin(), ring.end(), [&](const auto &cell) {
        return cell.state != execution::InputState::Free &&
               cell.coordinate == projection.coordinate && cell.use == use;
      })) {
    return false;
  }
  const execution::FetchReuseSource canonical_reuse =
      window ? execution::FetchReuseSource{
                   .key = source.key,
                   .source_offset = source.target_offset,
                   .target_offset = source.target_offset,
                   .bytes = source.bytes,
                   .read_offset = source.offset + source.bytes,
                   .read_target_offset = source.target_offset + source.bytes,
                   .read_bytes = 0u,
               }
             : execution::FetchReuseSource{};
  result = Projection{
      .source = source,
      .reuse = window ? canonical_reuse
                      : (plan.input_reuse(input.input[use].key.page, reuse)
                             ? reuse
                             : execution::FetchReuseSource{}),
      .demand = window ? footprint.sources[use] : input.input[use],
      .region = region,
      .host_regions = plan.host_input_regions(),
      .expected_bytes = expected_bytes};
  return true;
}

} // namespace rund::compute::detail::residency::physical::fetch

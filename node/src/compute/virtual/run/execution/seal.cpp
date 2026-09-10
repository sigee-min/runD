#include "../execution.hpp"

#include "../cache.hpp"
#include "fill.hpp"

#include "../../../pipeline/state.hpp"

#include <limits>

namespace rund::compute::detail {
namespace {

[[nodiscard]] residency::GraphMaterialization
materialization(const std::uint32_t resource, residency::CacheKey key,
                const std::uint64_t page_bytes, const std::uint64_t page_count,
                const std::uint64_t boundary_extent) noexcept {
  key.page = 0u;
  key.extent = 0u;
  return residency::GraphMaterialization{
      .resource = resource,
      .key = key,
      .page_bytes = page_bytes,
      .page_count = page_count,
      .boundary_extent = boundary_extent,
  };
}

[[nodiscard]] residency::CacheKey canonical_input_cache_key(
    const VirtualRunProjection &run, const std::uint64_t backing,
    const std::uint64_t version, const std::uint64_t page) noexcept {
  const bool boundary = run.input_payload_bytes != 0u &&
                        page + 1u == run.active.stream.page_count() &&
                        run.input_visible_bytes % run.input_payload_bytes != 0u;
  return residency::CacheKey{
      .backing = backing,
      .version = version,
      .extent = boundary ? run.cache_extent : 0u,
      .materialization_hi = run.canonical_input_identity_hi,
      .materialization_lo = run.canonical_input_identity_lo,
      .page = page,
  };
}

} // namespace

residency::execution::SealResult
seal_virtual_execution(const VirtualPipelineState &state,
                       const VirtualRunProjection &run) noexcept {
  using namespace residency;
  using namespace residency::execution;
  const std::shared_ptr<PipelineState> &pipeline = state.pipeline;
  SealResult invalid{};
  if (pipeline == nullptr || pipeline->device == nullptr ||
      pipeline->residency_pool == nullptr ||
      pipeline->device->backend == Backend::Cpu || run.reduction || run.scan ||
      run.graph_reduction ||
      (state.geometry.route != VirtualRoute::Pointwise &&
       state.geometry.route != VirtualRoute::Window) ||
      run.active.stream.page_count() == 0u || run.frame_capacity == 0u ||
      run.frame_capacity > UseCapacity ||
      run.frame_capacity > std::numeric_limits<std::uint32_t>::max() ||
      pipeline->residency_input == pipeline->residency_output ||
      pipeline->residency_input == std::numeric_limits<std::uint32_t>::max() ||
      pipeline->residency_output == std::numeric_limits<std::uint32_t>::max() ||
      pipeline->residency_input >= std::numeric_limits<std::uint32_t>::max() ||
      pipeline->residency_output >= std::numeric_limits<std::uint32_t>::max()) {
    return invalid;
  }
  const std::uint64_t page_count = run.active.stream.page_count();
  const std::uint64_t last_page = page_count - 1u;
  const CacheKey last_input =
      virtual_cache_key(run, run.input_backing, run.input_version, last_page);
  const CacheKey last_output = virtual_output_cache_key(
      run, run.output_backing, run.output_version, last_page);
  CacheKey first_input =
      virtual_cache_key(run, run.input_backing, run.input_version, 0u);
  CacheKey first_output =
      virtual_output_cache_key(run, run.output_backing, run.output_version, 0u);
  const bool canonical_window = state.geometry.route == VirtualRoute::Window;
  const CacheKey first_canonical_input =
      canonical_window ? canonical_input_cache_key(run, run.input_backing,
                                                   run.input_version, 0u)
                       : CacheKey{};
  const CacheKey last_canonical_input =
      canonical_window ? canonical_input_cache_key(run, run.input_backing,
                                                   run.input_version, last_page)
                       : CacheKey{};
  const std::uint32_t input_resource =
      static_cast<std::uint32_t>(pipeline->residency_input + 1u);
  const std::uint32_t output_resource =
      static_cast<std::uint32_t>(pipeline->residency_output + 1u);
  const std::uint32_t capacity = static_cast<std::uint32_t>(run.frame_capacity);
  const std::uint32_t host_capacity = run.host_frame_capacity;
  const std::uint32_t output_capacity = run.host_output_frame_capacity;
  if (input_resource == 0u || output_resource == 0u ||
      input_resource == output_resource || run.input_page_bytes == 0u ||
      run.output_page_bytes == 0u || run.input_payload_bytes == 0u ||
      run.output_payload_bytes == 0u || run.active.input_bytes == 0u ||
      run.active.output_bytes == 0u || host_capacity < capacity ||
      output_capacity < capacity ||
      run.input_prefix_bytes > run.input_page_bytes ||
      run.input_payload_bytes > run.input_page_bytes - run.input_prefix_bytes) {
    return invalid;
  }
  if (run.first_host_input_frame >
          std::numeric_limits<std::uint32_t>::max() - host_capacity ||
      run.first_host_output_frame >
          std::numeric_limits<std::uint32_t>::max() - output_capacity) {
    return invalid;
  }
  VirtualFetchFill fill{};
  if (!project_virtual_fetch_fill(state, run, fill)) {
    return invalid;
  }

  Request request{
      .page_count = page_count,
      .frame_capacity = capacity,
      .prefetch_distance = run.active.stream.prefetch_distance(),
      .input =
          Materialization{
              .cache = materialization(input_resource, first_input,
                                       run.input_page_bytes, page_count,
                                       last_input.extent),
              .access = Access::Read,
              .logical_bytes = run.active.input_bytes,
              .payload_bytes = run.input_payload_bytes,
              .read_prefix_bytes = run.input_prefix_bytes,
              .target_prefix_bytes = run.input_prefix_bytes,
              .read_suffix_bytes = run.input_page_bytes -
                                   run.input_prefix_bytes -
                                   run.input_payload_bytes,
              .fill = fill.policy,
              .fill_element_bytes = fill.element_bytes,
              .fill_value = fill.value,
              .next_use_base = page_count,
              .next_use_stride = 1u,
          },
      .canonical_input =
          canonical_window
              ? materialization(input_resource, first_canonical_input,
                                run.input_payload_bytes, page_count,
                                last_canonical_input.extent)
              : GraphMaterialization{},
      .output =
          Materialization{
              .cache = materialization(output_resource, first_output,
                                       run.output_page_bytes, page_count,
                                       last_output.extent),
              .access = Access::Write,
              .logical_bytes = run.active.output_bytes,
              .payload_bytes = run.output_payload_bytes,
          },
      .publication =
          Publication{
              .backing = run.output_backing,
              .version = run.output_version,
              .extent = DirtyExtent{.bytes = run.active.output_bytes},
          },
      .host_input = {FrameRegion{.tier = FrameTier::Host,
                                 .role = FrameRole::Input,
                                 .first = run.first_host_input_frame,
                                 .count = host_capacity},
                     FrameRegion{
                         .tier = FrameTier::Host,
                         .role = FrameRole::Input,
                         .first = static_cast<std::uint32_t>(
                             run.first_host_input_frame + host_capacity),
                         .count = host_capacity}},
      .device_input = run.input_regions,
      .device_output = run.output_regions,
      .host_output = {FrameRegion{.tier = FrameTier::Host,
                                  .role = FrameRole::Output,
                                  .first = run.first_host_output_frame,
                                  .count = output_capacity},
                      FrameRegion{
                          .tier = FrameTier::Host,
                          .role = FrameRole::Output,
                          .first = static_cast<std::uint32_t>(
                              run.first_host_output_frame + output_capacity),
                          .count = output_capacity}},
  };
  return seal(request);
}

} // namespace rund::compute::detail

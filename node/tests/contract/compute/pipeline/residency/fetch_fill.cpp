#include "local.hpp"

#include "src/compute/device/residency/execution/plan.hpp"
#include "src/compute/virtual/run/sliding/fetch/fill.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace rund_node_test_pipeline_residency {
namespace {

namespace execution = rund::compute::detail::residency::execution;
namespace residency = rund::compute::detail::residency;
namespace sliding = rund::compute::detail::sliding_product_detail;

[[nodiscard]] execution::Request request(const execution::FetchFill fill,
                                         const std::uint64_t value) noexcept {
  const auto materialization = [](const std::uint32_t resource,
                                  const std::uint64_t backing) noexcept {
    return residency::GraphMaterialization{
        .resource = resource,
        .key = {.backing = backing,
                .version = 1u,
                .domain = residency::CacheDomain::Backing},
        .page_bytes = 24u,
        .page_count = 3u,
        .boundary_extent = 11u,
    };
  };
  return execution::Request{
      .page_count = 3u,
      .frame_capacity = 1u,
      .input =
          execution::Materialization{
              .cache = materialization(1u, 101u),
              .access = residency::Access::Read,
              .logical_bytes = 44u,
              .payload_bytes = 16u,
              .read_prefix_bytes = 4u,
              .target_prefix_bytes = 4u,
              .read_suffix_bytes = 4u,
              .fill = fill,
              .fill_element_bytes = 4u,
              .fill_value = value,
              .next_use_base = 3u,
              .next_use_stride = 1u,
          },
      .output =
          execution::Materialization{
              .cache = materialization(2u, 202u),
              .access = residency::Access::Write,
              .logical_bytes = 44u,
              .payload_bytes = 16u,
          },
      .publication = {.backing = 202u, .version = 1u, .extent = {.bytes = 44u}},
      .host_input = {residency::FrameRegion{.tier = residency::FrameTier::Host,
                                            .role = residency::FrameRole::Input,
                                            .count = 1u},
                     residency::FrameRegion{.tier = residency::FrameTier::Host,
                                            .role = residency::FrameRole::Input,
                                            .first = 1u,
                                            .count = 1u}},
      .device_input =
          {residency::FrameRegion{.tier = residency::FrameTier::Device,
                                  .role = residency::FrameRole::Input,
                                  .count = 1u},
           residency::FrameRegion{.tier = residency::FrameTier::Device,
                                  .role = residency::FrameRole::Input,
                                  .first = 1u,
                                  .count = 1u}},
      .device_output =
          {residency::FrameRegion{.tier = residency::FrameTier::Device,
                                  .role = residency::FrameRole::Output,
                                  .first = 10u,
                                  .count = 1u},
           residency::FrameRegion{.tier = residency::FrameTier::Device,
                                  .role = residency::FrameRole::Output,
                                  .first = 11u,
                                  .count = 1u}},
      .host_output = {residency::FrameRegion{.tier = residency::FrameTier::Host,
                                             .role =
                                                 residency::FrameRole::Output,
                                             .first = 10u,
                                             .count = 1u},
                      residency::FrameRegion{.tier = residency::FrameTier::Host,
                                             .role =
                                                 residency::FrameRole::Output,
                                             .first = 11u,
                                             .count = 1u}},
  };
}

[[nodiscard]] bool
materialize(const execution::Plan &plan, const std::uint64_t page,
            const std::array<std::uint32_t, 11u> &backing,
            const std::array<std::uint32_t, 6u> &expected) noexcept {
  execution::FetchSource source{};
  std::array<std::uint32_t, 6u> frame{};
  if (!plan.input_source(page, source) || !source.materializes_frame() ||
      source.offset > sizeof(backing) ||
      source.bytes > sizeof(backing) - source.offset ||
      source.target_offset > sizeof(frame) ||
      source.bytes > sizeof(frame) - source.target_offset) {
    return false;
  }
  std::memcpy(
      reinterpret_cast<std::byte *>(frame.data()) + source.target_offset,
      reinterpret_cast<const std::byte *>(backing.data()) + source.offset,
      static_cast<std::size_t>(source.bytes));
  return sliding::fill_fetch_frame(reinterpret_cast<std::byte *>(frame.data()),
                                   source) &&
         frame == expected;
}

[[nodiscard]] bool check_recipe(const execution::FetchFill fill,
                                const std::uint64_t value,
                                const std::array<std::uint32_t, 6u> &first,
                                const std::array<std::uint32_t, 6u> &last,
                                std::uint64_t &identity) noexcept {
  const execution::SealResult sealed = execution::seal(request(fill, value));
  const std::array<std::uint32_t, 11u> backing{1u, 2u, 3u, 4u,  5u, 6u,
                                               7u, 8u, 9u, 10u, 11u};
  const std::array<std::uint32_t, 6u> middle{4u, 5u, 6u, 7u, 8u, 9u};
  identity = sealed.plan.identity();
  return sealed && sealed.plan.input_sources_materializable() &&
         materialize(sealed.plan, 0u, backing, first) &&
         materialize(sealed.plan, 1u, backing, middle) &&
         materialize(sealed.plan, 2u, backing, last);
}

} // namespace

int CheckFetchFill() {
  std::uint64_t clamp_identity = 0u;
  std::uint64_t clip_sum_identity = 0u;
  std::uint64_t clip_min_identity = 0u;
  const std::uint32_t maximum = std::numeric_limits<std::uint32_t>::max();
  if (!check_recipe(execution::FetchFill::RepeatBoundary, 0u,
                    {1u, 1u, 2u, 3u, 4u, 5u}, {8u, 9u, 10u, 11u, 11u, 11u},
                    clamp_identity)) {
    return 1;
  }
  if (!check_recipe(execution::FetchFill::ConstantBoundary, 0u,
                    {0u, 1u, 2u, 3u, 4u, 5u}, {8u, 9u, 10u, 11u, 0u, 0u},
                    clip_sum_identity)) {
    return 2;
  }
  if (!check_recipe(execution::FetchFill::ConstantBoundary, maximum,
                    {maximum, 1u, 2u, 3u, 4u, 5u},
                    {8u, 9u, 10u, 11u, maximum, maximum}, clip_min_identity)) {
    return 3;
  }
  const execution::SealResult reuse_plan =
      execution::seal(request(execution::FetchFill::RepeatBoundary, 0u));
  execution::FetchReuseSource first_reuse{};
  execution::FetchReuseSource middle_reuse{};
  execution::FetchReuseSource tail_reuse{};
  if (!reuse_plan || reuse_plan.plan.input_reuse(0u, first_reuse) ||
      !reuse_plan.plan.input_reuse(1u, middle_reuse) ||
      !reuse_plan.plan.input_reuse(2u, tail_reuse) ||
      middle_reuse != execution::FetchReuseSource{.key = middle_reuse.key,
                                                  .source_offset = 16u,
                                                  .target_offset = 0u,
                                                  .bytes = 8u,
                                                  .read_offset = 20u,
                                                  .read_target_offset = 8u,
                                                  .read_bytes = 16u} ||
      tail_reuse != execution::FetchReuseSource{.key = tail_reuse.key,
                                                .source_offset = 16u,
                                                .target_offset = 0u,
                                                .bytes = 8u,
                                                .read_offset = 36u,
                                                .read_target_offset = 8u,
                                                .read_bytes = 8u} ||
      middle_reuse.key.page != 0u || tail_reuse.key.page != 1u) {
    return 4;
  }
  execution::Request malformed =
      request(execution::FetchFill::ConstantBoundary, std::uint64_t{1u} << 32u);
  return clamp_identity == 0u || clip_sum_identity == 0u ||
                 clip_min_identity == 0u ||
                 clamp_identity == clip_sum_identity ||
                 clamp_identity == clip_min_identity ||
                 clip_sum_identity == clip_min_identity ||
                 execution::seal(malformed)
             ? 5
             : 0;
}

} // namespace rund_node_test_pipeline_residency

#include "local.hpp"

#include <array>

namespace rund_node_test_pipeline_residency::execution_test {

[[nodiscard]] residency::CacheKey InputKey() {
  return residency::CacheKey{
      .backing = 11u, .version = 3u, .materialization_hi = 7u};
}

[[nodiscard]] residency::CacheKey OutputKey() {
  return residency::CacheKey{
      .backing = 12u, .version = 4u, .materialization_hi = 8u};
}

[[nodiscard]] residency::FrameRegion region(
    const residency::FrameTier tier, const residency::FrameRole role,
    const std::uint32_t first) {
  return residency::FrameRegion{
      .tier = tier, .role = role, .first = first, .count = 3u};
}

[[nodiscard]] execution::Request MakeRequest() {
  const residency::CacheKey input_key = InputKey();
  const residency::CacheKey output_key = OutputKey();
  return execution::Request{
      .page_count = 10u,
      .frame_capacity = 3u,
      .prefetch_distance = 2u,
      .input =
          execution::Materialization{
              .cache =
                  residency::GraphMaterialization{
                      .resource = 1u,
                      .key = input_key,
                      .page_bytes = 128u,
                      .page_count = 10u,
                      .boundary_extent = 24'575u,
                  },
              .access = residency::Access::Read,
              .logical_bytes = 1216u,
              .payload_bytes = 128u,
              .next_use_base = 10u,
              .next_use_stride = 1u,
          },
      .output =
          execution::Materialization{
              .cache =
                  residency::GraphMaterialization{
                      .resource = 2u,
                      .key = output_key,
                      .page_bytes = 96u,
                      .page_count = 10u,
                      .boundary_extent = 24'575u,
                  },
              .access = residency::Access::Write,
              .logical_bytes = 770u,
              .payload_bytes = 80u,
              .dirty_origin = 4096u,
          },
      .publication =
          execution::Publication{
              .backing = 12u,
              .version = 4u,
              .generation = 9u,
              .capability = 17u,
              .extent = residency::DirtyExtent{.offset = 4096u, .bytes = 770u},
          },
      .host_input = {region(residency::FrameTier::Host,
                            residency::FrameRole::Input, 0u),
                     region(residency::FrameTier::Host,
                            residency::FrameRole::Input, 3u)},
      .device_input = {region(residency::FrameTier::Device,
                              residency::FrameRole::Input, 6u),
                       region(residency::FrameTier::Device,
                              residency::FrameRole::Input, 9u)},
      .device_output = {region(residency::FrameTier::Device,
                               residency::FrameRole::Output, 12u),
                        region(residency::FrameTier::Device,
                               residency::FrameRole::Output, 15u)},
      .host_output = {region(residency::FrameTier::Host,
                             residency::FrameRole::Output, 18u),
                      region(residency::FrameTier::Host,
                             residency::FrameRole::Output, 21u)},
  };
}

[[nodiscard]] execution::Request MakeFootprintRequest() {
  execution::Request request{};
  request.page_count = 5u;
  request.frame_capacity = 2u;
  request.prefetch_distance = 2u;
  request.input = execution::Materialization{
      .cache =
          residency::GraphMaterialization{
              .resource = 1u,
              .key =
                  residency::CacheKey{
                      .backing = 71u,
                      .version = 3u,
                      .materialization_hi = 0x455850414e444544ull,
                      .materialization_lo = 0x57494e444f575f31ull},
              .page_bytes = 64u,
              .page_count = 5u,
              .boundary_extent = 57u,
          },
      .access = residency::Access::Read,
      .logical_bytes = 228u,
      .payload_bytes = 48u,
      .read_prefix_bytes = 8u,
      .target_prefix_bytes = 8u,
      .read_suffix_bytes = 8u,
      .fill = execution::FetchFill::ConstantBoundary,
      .fill_element_bytes = 4u,
      .next_use_base = 5u,
      .next_use_stride = 1u,
  };
  request.canonical_input = residency::GraphMaterialization{
      .resource = 1u,
      .key = residency::CacheKey{.backing = 71u,
                                 .version = 3u,
                                 .materialization_hi = 0x43414e4f4e494341ull,
                                 .materialization_lo = 0x4c5f504147455f31ull},
      .page_bytes = 48u,
      .page_count = 5u,
      .boundary_extent = 57u,
  };
  request.output = execution::Materialization{
      .cache =
          residency::GraphMaterialization{
              .resource = 2u,
              .key =
                  residency::CacheKey{
                      .backing = 72u,
                      .version = 4u,
                      .materialization_hi = 0x57494e444f575f4full,
                      .materialization_lo = 0x55545055545f3031ull},
              .page_bytes = 48u,
              .page_count = 5u,
              .boundary_extent = 57u,
          },
      .access = residency::Access::Write,
      .logical_bytes = 228u,
      .payload_bytes = 48u,
  };
  request.publication = execution::Publication{
      .backing = 72u,
      .version = 4u,
      .extent = residency::DirtyExtent{.bytes = 228u},
  };
  request.host_input = {
      residency::FrameRegion{.tier = residency::FrameTier::Host,
                             .role = residency::FrameRole::Input,
                             .first = 0u,
                             .count = 4u},
      residency::FrameRegion{.tier = residency::FrameTier::Host,
                             .role = residency::FrameRole::Input,
                             .first = 4u,
                             .count = 4u},
  };
  request.device_input = {
      residency::FrameRegion{.tier = residency::FrameTier::Device,
                             .role = residency::FrameRole::Input,
                             .first = 8u,
                             .count = 2u},
      residency::FrameRegion{.tier = residency::FrameTier::Device,
                             .role = residency::FrameRole::Input,
                             .first = 10u,
                             .count = 2u},
  };
  request.device_output = {
      residency::FrameRegion{.tier = residency::FrameTier::Device,
                             .role = residency::FrameRole::Output,
                             .first = 12u,
                             .count = 2u},
      residency::FrameRegion{.tier = residency::FrameTier::Device,
                             .role = residency::FrameRole::Output,
                             .first = 14u,
                             .count = 2u},
  };
  request.host_output = {
      residency::FrameRegion{.tier = residency::FrameTier::Host,
                             .role = residency::FrameRole::Output,
                             .first = 16u,
                             .count = 2u},
      residency::FrameRegion{.tier = residency::FrameTier::Host,
                             .role = residency::FrameRole::Output,
                             .first = 18u,
                             .count = 2u},
  };
  return request;
}

[[nodiscard]] execution::Request MakeOneRequest(
    const execution::Request &request) {
  execution::Request one = request;
  one.page_count = 3u;
  one.input.cache.page_count = 3u;
  one.input.logical_bytes = 384u;
  one.input.cache.boundary_extent = 0u;
  one.output.cache.page_count = 3u;
  one.output.logical_bytes = 240u;
  one.output.cache.boundary_extent = 0u;
  one.publication.extent.bytes = 240u;
  return one;
}

[[nodiscard]] std::array<
    std::pair<residency::FrameTier, residency::FrameRole>, 8u>
FrameOwners() {
  return {{
      {residency::FrameTier::Host, residency::FrameRole::Input},
      {residency::FrameTier::Host, residency::FrameRole::Input},
      {residency::FrameTier::Device, residency::FrameRole::Input},
      {residency::FrameTier::Device, residency::FrameRole::Input},
      {residency::FrameTier::Device, residency::FrameRole::Output},
      {residency::FrameTier::Device, residency::FrameRole::Output},
      {residency::FrameTier::Host, residency::FrameRole::Output},
      {residency::FrameTier::Host, residency::FrameRole::Output},
  }};
}

[[nodiscard]] bool dependency(const execution::Plan &plan,
                              const execution::NodeId node,
                              const execution::NodeId expected) {
  std::array<execution::NodeId, execution::PredecessorCapacity> storage{};
  std::size_t count = 0u;
  if (!plan.predecessors(node, storage, count)) {
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    if (storage[index] == expected) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool RegisterOwners(residency::Authority &authority,
                                  const std::uint32_t count) {
  const auto owners = FrameOwners();
  for (std::size_t index = 0u; index < owners.size(); ++index) {
    std::uint32_t first_frame = 0u;
    if (!authority.register_frames(owners[index].first, owners[index].second,
                                   count, first_frame) ||
        first_frame != index * count) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool SeedCachedInput(residency::Authority &authority,
                                   const residency::CacheKey key) {
  const residency::CacheUse cached{
      .key = key,
      .access = residency::Access::Read,
      .next_use = 1u,
  };
  const residency::AuthorityResult seeded = authority.begin(
      std::span<const residency::CacheUse>{&cached, 1u},
      residency::FrameTier::Host, residency::FrameRole::Input, 0u, 3u);
  std::array<std::uint8_t, 1u> resident{};
  const std::array<residency::CacheKey, 1u> cached_keys{key};
  return seeded && authority.activate(seeded.lease.token) &&
         authority.complete(seeded.lease.token, true) &&
         authority.probe(cached_keys, resident, 0u, 3u) && resident[0] == 1u;
}

} // namespace rund_node_test_pipeline_residency::execution_test

#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckAuthorityGraph() {
  using namespace rund::compute::detail::residency;
  const auto never = std::numeric_limits<std::uint64_t>::max();
  // Generic Graph admission is one multi-resource transaction. The common
  // local is selected across all ports (not just an anchor), a failed final
  // port mutates nothing, and the last Transient read retires atomically with
  // publication of the next output.
  Authority graph_epoch;
  std::uint32_t graph_input_a_base = 99u;
  std::uint32_t graph_input_b_base = 99u;
  std::uint32_t graph_internal_base = 99u;
  std::uint32_t graph_output_base = 99u;
  if (!graph_epoch.register_frames(FrameTier::Device, FrameRole::Input, 2u,
                                   graph_input_a_base) ||
      !graph_epoch.register_frames(FrameTier::Device, FrameRole::Input, 2u,
                                   graph_input_b_base) ||
      !graph_epoch.register_frames(FrameTier::Device,
                                   FrameRole::Intermediate, 2u,
                                   graph_internal_base) ||
      !graph_epoch.register_frames(FrameTier::Device, FrameRole::Output, 2u,
                                   graph_output_base)) {
    return 64;
  }
  const FrameRegion graph_input_a_region{.tier = FrameTier::Device,
                                         .role = FrameRole::Input,
                                         .first = graph_input_a_base,
                                         .count = 2u};
  const FrameRegion graph_input_b_region{.tier = FrameTier::Device,
                                         .role = FrameRole::Input,
                                         .first = graph_input_b_base,
                                         .count = 2u};
  const FrameRegion graph_internal_region{.tier = FrameTier::Device,
                                          .role = FrameRole::Intermediate,
                                          .first = graph_internal_base,
                                          .count = 2u};
  const FrameRegion graph_output_region{.tier = FrameTier::Device,
                                        .role = FrameRole::Output,
                                        .first = graph_output_base,
                                        .count = 2u};
  const std::array graph_b_seed{
      CacheUse{.key = {.backing = 82u,
                       .version = 1u,
                       .materialization_hi = 22u,
                       .page = 9u},
               .access = Access::Read,
               .next_use = never},
      CacheUse{.key = {.backing = 82u,
                       .version = 1u,
                       .materialization_hi = 22u,
                       .page = 0u},
               .access = Access::Read,
               .next_use = 0u},
  };
  const AuthorityResult seeded_b = graph_epoch.begin(
      graph_b_seed, graph_input_b_region.tier, graph_input_b_region.role,
      graph_input_b_region.first, graph_input_b_region.count);
  if (!seeded_b || !graph_epoch.complete(seeded_b.lease.token, true)) {
    return 65;
  }
  const GraphMaterialization graph_a_materialization{
      .resource = 1u,
      .key = {.backing = 81u, .version = 1u, .materialization_hi = 21u},
      .page_bytes = 64u,
      .page_count = 1u,
  };
  const GraphMaterialization graph_b_materialization{
      .resource = 2u,
      .key = {.backing = 82u, .version = 1u, .materialization_hi = 22u},
      .page_bytes = 64u,
      .page_count = 1u,
  };
  const GraphMaterialization graph_internal_materialization{
      .resource = 3u,
      .key = {.backing = 83u,
              .version = 1u,
              .materialization_hi = 23u,
              .domain = CacheDomain::Transient},
      .page_bytes = 64u,
      .page_count = 1u,
  };
  const GraphMaterialization graph_output_materialization{
      .resource = 4u,
      .key = {.backing = 84u,
              .version = 1u,
              .materialization_hi = 24u,
              .domain = CacheDomain::Transient},
      .page_bytes = 64u,
      .page_count = 1u,
  };
  const std::array<PageUse, 3u> graph_stage0_uses{
      PageUse{.key = {.resource = 1u, .page = 0u},
              .access = Access::Read,
              .next_use = 2u,
              .pin = {.first_epoch = 0u, .last_epoch = 0u}},
      PageUse{.key = {.resource = 2u, .page = 0u},
              .access = Access::Read,
              .next_use = 2u,
              .pin = {.first_epoch = 0u, .last_epoch = 0u}},
      PageUse{.key = {.resource = 3u, .page = 0u},
              .access = Access::Write,
              .dirty = {.bytes = 64u},
              .next_use = 1u,
              .pin = {.first_epoch = 0u, .last_epoch = 1u}},
  };
  const std::array<GraphPortRequest, 3u> graph_stage0_ports{
      GraphPortRequest{.program_port = 0u,
                       .first_use = 0u,
                       .use_count = 1u,
                       .materialization = graph_a_materialization,
                       .region = graph_input_a_region,
                       .cache_regions = {graph_input_a_region},
                       .cache_region_count = 1u},
      GraphPortRequest{.program_port = 1u,
                       .first_use = 1u,
                       .use_count = 1u,
                       .materialization = graph_b_materialization,
                       .region = graph_input_b_region,
                       .cache_regions = {graph_input_b_region},
                       .cache_region_count = 1u},
      GraphPortRequest{.program_port = 0u,
                       .first_use = 2u,
                       .use_count = 1u,
                       .materialization = graph_internal_materialization,
                       .region = graph_internal_region,
                       .cache_regions = {graph_internal_region},
                       .cache_region_count = 1u},
  };
  const AuthorityResult graph_stage0 = graph_epoch.begin_graph_epoch(
      graph_stage0_uses, graph_stage0_ports, 0u, 0u);
  if (!graph_stage0 || graph_stage0.lease.ports.size() != 3u ||
      graph_stage0.lease.bindings.size() != 3u ||
      graph_stage0.lease.bindings[0].frame != graph_input_a_base + 1u ||
      graph_stage0.lease.bindings[1].frame != graph_input_b_base + 1u ||
      graph_stage0.lease.bindings[2].frame != graph_internal_base + 1u ||
      graph_epoch.complete(graph_stage0.lease.token, true) ||
      !graph_epoch.activate(graph_stage0.lease.token) ||
      !graph_epoch.complete(graph_stage0.lease.token, true)) {
    return 66;
  }
  auto overlapping_ports = graph_stage0_ports;
  overlapping_ports.back().region = graph_input_a_region;
  if (graph_epoch
          .begin_graph_epoch(graph_stage0_uses, overlapping_ports, 0u, 0u)
          .failure != AuthorityFailure::Invalid) {
    return 67;
  }
  auto duplicate_program_port = graph_stage0_ports;
  duplicate_program_port[1].program_port = 0u;
  if (graph_epoch
          .begin_graph_epoch(graph_stage0_uses, duplicate_program_port, 0u,
                             0u)
          .failure != AuthorityFailure::Invalid) {
    return 68;
  }
  std::array<PageUse, 2u> graph_stage1_uses{
      PageUse{.key = {.resource = 3u, .page = 0u},
              .access = Access::Read,
              .next_use = never,
              .pin = {.first_epoch = 0u, .last_epoch = 1u},
              .prefetch_epoch = 1u,
              .ready_epoch = 1u},
      PageUse{.key = {.resource = 4u, .page = 0u},
              .access = Access::Write,
              .dirty = {.bytes = 64u},
              .next_use = never,
              .pin = {.first_epoch = 1u, .last_epoch = 1u},
              .prefetch_epoch = 1u,
              .ready_epoch = 1u},
  };
  const std::array<GraphPortRequest, 2u> graph_stage1_ports{
      GraphPortRequest{.program_port = 0u,
                       .first_use = 0u,
                       .use_count = 1u,
                       .materialization = graph_internal_materialization,
                       .region = graph_internal_region,
                       .cache_regions = {graph_internal_region},
                       .cache_region_count = 1u},
      GraphPortRequest{.program_port = 0u,
                       .first_use = 1u,
                       .use_count = 1u,
                       .materialization = graph_output_materialization,
                       .region = graph_output_region,
                       .cache_regions = {graph_output_region},
                       .cache_region_count = 1u},
  };
  const AuthorityResult failed_stage1 = graph_epoch.begin_graph_epoch(
      graph_stage1_uses, graph_stage1_ports, 0u, 1u);
  if (!failed_stage1 || !graph_epoch.activate(failed_stage1.lease.token) ||
      !graph_epoch.complete(failed_stage1.lease.token, false, true)) {
    return 69;
  }
  CacheKey graph_internal_key{};
  CacheKey graph_output_key{};
  if (!project_graph_cache_key(graph_internal_materialization,
                               graph_stage1_uses[0].key,
                               graph_internal_key) ||
      !project_graph_cache_key(graph_output_materialization,
                               graph_stage1_uses[1].key, graph_output_key)) {
    return 70;
  }
  std::array<std::uint8_t, 1u> graph_resident{};
  const std::array graph_internal_keys{graph_internal_key};
  if (!graph_epoch.probe(graph_internal_keys, graph_resident,
                         graph_internal_region.first,
                         graph_internal_region.count) ||
      graph_resident[0] != 1u) {
    return 71;
  }
  const AuthorityResult graph_stage1 = graph_epoch.begin_graph_epoch(
      graph_stage1_uses, graph_stage1_ports, 0u, 1u);
  if (!graph_stage1 || !graph_epoch.activate(graph_stage1.lease.token) ||
      !graph_epoch.complete(graph_stage1.lease.token, true) ||
      !graph_epoch.probe(graph_internal_keys, graph_resident,
                         graph_internal_region.first,
                         graph_internal_region.count) ||
      graph_resident[0] != 0u) {
    return 72;
  }
  const std::array graph_output_keys{graph_output_key};
  const AuthorityResult graph_output_retire = graph_epoch.begin_discard(
      graph_output_keys, graph_output_region.first, graph_output_region.count);
  const std::array graph_epoch_regions{graph_input_a_region,
                                       graph_input_b_region,
                                       graph_internal_region,
                                       graph_output_region};
  if (!graph_output_retire ||
      !graph_epoch.discard(graph_output_retire.lease.token) ||
      !graph_epoch.release_frames(graph_epoch_regions)) {
    return 73;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency

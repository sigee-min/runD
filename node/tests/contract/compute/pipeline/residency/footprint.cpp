#include "local.hpp"

#include "src/compute/pipeline/residency/footprint.hpp"

namespace rund_node_test_pipeline_residency {

int CheckFootprints() {
  using namespace rund::compute::detail::residency;
  FootprintEpoch window{.node = 4u, .tile = 3u};
  if (!ProjectWindow(
          WindowFootprint{
              .input_resource = 1u,
              .output_resource = 2u,
              .input_elements = 64u,
              .output_first = 16u,
              .output_count = 16u,
              .window_size = 5u,
              .stride = 1u,
              .pad_left = 2u,
              .element_bytes = 4u,
          },
          32u, window)) {
    return 1;
  }
  // Output [16,31] needs the [14,33] halo: input pages 1..4 and output
  // pages 2..3. This helper owns byte geometry only; the recurrent planner is
  // the sole future-use/pin/identity authority.
  if (window.uses.size() != 6u) {
    return 2;
  }
  const auto &window_uses = window.uses;
  if (window_uses[0].key != PageKey{.resource = 1u, .page = 1u} ||
      window_uses[1].key != PageKey{.resource = 1u, .page = 2u} ||
      window_uses[2].key != PageKey{.resource = 1u, .page = 3u} ||
      window_uses[3].key != PageKey{.resource = 1u, .page = 4u} ||
      window_uses[4].key != PageKey{.resource = 2u, .page = 2u} ||
      window_uses[5].key != PageKey{.resource = 2u, .page = 3u} ||
      window_uses[0].dirty != DirtyRange{} ||
      window_uses[4].dirty != DirtyRange{.offset = 0u, .bytes = 32u} ||
      window_uses[5].dirty != DirtyRange{.offset = 0u, .bytes = 32u}) {
    return 3;
  }

  FootprintEpoch adjacent{.node = 4u, .tile = 4u};
  if (!ProjectWindow(
          WindowFootprint{
              .input_resource = 1u,
              .output_resource = 2u,
              .input_elements = 64u,
              .output_first = 32u,
              .output_count = 16u,
              .window_size = 5u,
              .stride = 1u,
              .pad_left = 2u,
              .element_bytes = 4u,
          },
          32u, adjacent) ||
      adjacent.uses.size() != 6u ||
      adjacent.uses[0].key != PageKey{.resource = 1u, .page = 3u} ||
      adjacent.uses[1].key != PageKey{.resource = 1u, .page = 4u} ||
      adjacent.uses[2].key != PageKey{.resource = 1u, .page = 5u} ||
      adjacent.uses[3].key != PageKey{.resource = 1u, .page = 6u}) {
    return 7;
  }
  // The halo overlap is the same canonical input pages 3 and 4, not two
  // halo-expanded backing identities. Authority may therefore retain/pin the
  // pages once while each kernel tile constructs its own shared halo.
  if (window_uses[2].key != adjacent.uses[0].key ||
      window_uses[3].key != adjacent.uses[1].key) {
    return 8;
  }

  std::vector<PageDemand> partial;
  if (!ProjectRange(ByteRange{.resource = 4u,
                              .access = Access::Write,
                              .offset = 28u,
                              .bytes = 8u},
                    32u, partial) ||
      partial.size() != 2u ||
      partial[0].dirty != DirtyRange{.offset = 28u, .bytes = 4u} ||
      partial[1].dirty != DirtyRange{.offset = 0u, .bytes = 4u}) {
    return 4;
  }

  std::vector<FootprintEpoch> scan;
  if (!ProjectScan(
          ScanFootprint{
              .input_resource = 1u,
              .output_resource = 2u,
              .partial_resource = 3u,
              .element_count = 32u,
              .tile_elements = 16u,
              .element_bytes = 4u,
          },
          64u, scan) ||
      scan.size() != 5u || scan[0].node != 0u || scan[2].node != 1u ||
      scan[3].node != 2u) {
    return 5;
  }
  return scan[0].uses.size() == 3u && scan[2].uses.size() == 1u &&
                 scan[4].uses.size() == 2u
             ? 0
             : 6;
}

} // namespace rund_node_test_pipeline_residency

#include "local.hpp"

#include "../../allocation.hpp"
#include "src/compute/pipeline/residency/planner.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace rund_node_test_pipeline_residency {

int CheckGraphProjection() {
  using rund::compute::detail::Type;
  using namespace rund::compute::detail::residency;
  // All permutations through four frames, both origins, every active tail.
  // Reverse the authored descriptor order to exercise canonicalization too.
  for (std::uint32_t frames = 1u; frames <= 4u; ++frames) {
    std::array<std::uint32_t, 4u> permutation{0u, 1u, 2u, 3u};
    do {
      for (const auto origin : {GraphPageOrigin::Begin, GraphPageOrigin::End}) {
        TiledGraphPlanInput input{
            .page_count = frames * 2u,
            .requested_frames = frames,
            .max_frames = frames,
            .prefetch_distance = 1u,
            .resources = {{.resource = 1u,
                           .type = Type::U64,
                           .page_bytes = 64u,
                           .logical_bytes = frames * 128u,
                           .kind = GraphResourceKind::ExternalInput,
                           .persistence = ResourcePersistence::Backing},
                          {.resource = 2u,
                           .type = Type::U64,
                           .page_bytes = 64u,
                           .logical_bytes = frames * 128u,
                           .kind = GraphResourceKind::ExternalOutput,
                           .persistence = ResourcePersistence::Transient}},
            .stages = {{.node = 0u,
                        .ports = {{.resource = 1u, .access = Access::Read},
                                  {.resource = 2u, .access = Access::Write}}}}};
        for (std::uint32_t target = frames; target != 0u; --target) {
          input.resources[0].remaps.push_back(
              {.source_local = permutation[target - 1u],
               .target_local = target - 1u,
               .source_origin = origin});
        }
        const auto plan = PlanResidency(input);
        if (!plan) {
          return 1;
        }
        for (std::uint32_t tail = 1u; tail <= frames; ++tail) {
          const std::uint64_t pages = frames + tail;
          const std::array<std::uint64_t, 2u> bytes{pages * 64u, pages * 64u - 8u};
          TiledGraphInvocation active{};
          if (!plan.plan.tiled_graph().active(pages, bytes, active)) {
            return 2;
          }
          std::array<PageUse, 8u> uses{};
          Epoch epoch{};
          bool expected_valid = true;
          for (std::uint32_t target = 0u; target < tail; ++target) {
            expected_valid = expected_valid && permutation[target] < tail;
          }
          node_compute_allocation::Start();
          const bool projected = active.project(
              1u, 0u, std::span{uses}.first(tail * 2u), epoch);
          node_compute_allocation::Stop();
          if (projected != expected_valid ||
              node_compute_allocation::Count() != 0u) {
            return 3;
          }
          if (!projected) {
            continue;
          }
          for (std::uint32_t target = 0u; target < tail; ++target) {
            const std::uint64_t source =
                origin == GraphPageOrigin::Begin
                    ? permutation[target]
                    : tail - 1u - permutation[target];
            const PageUse read{.key = {.resource = 1u, .page = frames + source},
                               .access = Access::Read,
                               .next_use = 3u,
                               .pin = {1u, 1u},
                               .prefetch_epoch = 0u,
                               .ready_epoch = 1u};
            const PageUse write{
                .key = {.resource = 2u, .page = frames + target},
                .access = Access::Write,
                .dirty = {.bytes = target + 1u == tail ? 56u : 64u},
                .pin = {1u, 1u},
                .prefetch_epoch = 1u,
                .ready_epoch = 1u};
            if (uses[target] != read || uses[tail + target] != write) {
              return 4;
            }
          }
        }
      }
    } while (std::next_permutation(permutation.begin(),
                                   permutation.begin() + frames));
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency

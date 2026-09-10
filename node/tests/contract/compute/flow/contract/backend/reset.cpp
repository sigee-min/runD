#include "../model.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

namespace rund_node_flow_contract {

[[nodiscard]] int CheckResetProjection(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const Target target = backend == Backend::Cpu
                            ? Target::cpu(2u)
                            : rund::node::test_contract::target_for(backend);
  constexpr std::uint32_t Convex = 3u;
  constexpr std::uint32_t Mesh = 2u;
  constexpr std::uint64_t Width = Convex + Mesh;
  constexpr std::size_t OutputCount = 2u * Width;
  constexpr std::array<std::uint64_t, 2u * Convex> convex{11u,  0u, 0u,
                                                          101u, 0u, 0u};
  constexpr std::array<std::uint64_t, 2u * Mesh> mesh{17u, 0u, 107u, 0u};
  const std::vector<std::uint64_t> expected{11u,  0u, 0u, 17u,  0u,
                                            101u, 0u, 0u, 107u, 0u};

  auto program =
      on(target)
          .input<std::uint64_t>(convex.size())
          .zip_input<std::uint64_t>(mesh.size())
          .branch([](auto left, auto right) {
            const auto left_ordinal =
                left.indices()
                    .map("reset-projection",
                         [](auto index) { return mask(index == index); })
                    .scan(Scan::ExclusiveSum);
            const auto right_ordinal =
                right.indices()
                    .map("reset-projection",
                         [](auto index) { return mask(index == index); })
                    .scan(Scan::ExclusiveSum);
            const auto left_target = left_ordinal.map(
                "reset-projection",
                capture(
                    [](auto index, auto source, auto output) {
                      const auto field = index / source;
                      return field * output + index - field * source;
                    },
                    Convex, static_cast<std::uint32_t>(Width)));
            const auto right_target = right_ordinal.map(
                "reset-projection",
                capture(
                    [](auto index, auto source, auto output, auto offset) {
                      const auto field = index / source;
                      return field * output + offset + index - field * source;
                    },
                    Mesh, static_cast<std::uint32_t>(Width), Convex));
            const auto left_segment =
                left.scatter(left_target, {.count = OutputCount});
            const auto right_segment =
                right.scatter(right_target, {.count = OutputCount});
            return zip(left_segment, right_segment, left_segment.indices())
                .map("reset-projection",
                     capture(
                         [](auto left_value, auto right_value, auto index,
                            auto width, auto left_count) {
                           const auto field = index / width;
                           return select(index - field * width < left_count,
                                         left_value, right_value);
                         },
                         Width, static_cast<std::uint64_t>(Convex)));
          })
          .compile();
  if (!program || program->graph().memory.reset_count != 2u) {
    return 1;
  }
  auto job = program->resident(std::span{convex}, std::span{mesh});
  if (!job || !job->run()) {
    return 2;
  }
  const auto first = job->read();
  const auto first_stats = job->stats();
  const bool fusion_required = backend != Backend::Cpu;
  if (!first || *first != expected ||
      (fusion_required && first_stats.fusions == 0u) || !job->run()) {
    return 3;
  }
  const auto warm = job->read();
  return warm && *warm == expected &&
                 (!fusion_required || job->stats().fusions > 0u)
             ? 0
             : 4;
}

} // namespace rund_node_flow_contract

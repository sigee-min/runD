#include "local.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace rund_node_test_flow_primitives {

[[nodiscard]] int CheckPoolSurface() {
  using namespace rund::compute;
  const auto rejected = [](const PoolSpec options,
                           const std::string_view reason) {
    auto program =
        Target()
            .input<std::int32_t>(5u)
            .branch([=](auto values) { return values.pool(options); })
            .compile();
    return !program && program.error() == reason;
  };
  if (!rejected(PoolSpec{.op = static_cast<Window>(0xffu)},
                "compute_window_op_unsupported") ||
      !rejected(PoolSpec{.edge = static_cast<WindowEdge>(0xffu)},
                "compute_window_edge_unsupported") ||
      !rejected(PoolSpec{.tail = static_cast<PoolTail>(0xffu)},
                "compute_graph_primitive_invalid") ||
      !rejected(PoolSpec{.width = 0u}, "compute_window_zero") ||
      !rejected(PoolSpec{.stride = 0u}, "compute_window_zero") ||
      !rejected(PoolSpec{.width = 6u}, "compute_graph_shape_mismatch")) {
    return 1;
  }
  const auto rejected_window = [](const std::size_t count,
                                  const WindowSpec options,
                                  const std::string_view reason) {
    auto program =
        Target()
            .input<std::int32_t>(count)
            .branch([=](auto values) { return values.window(options); })
            .compile();
    return !program && program.error() == reason;
  };
  if (!rejected_window(5u, WindowSpec{.radius = 0u},
                       "compute_window_radius_invalid") ||
      !rejected_window(5u, WindowSpec{.radius = 6u},
                       "compute_window_radius_invalid")) {
    return 2;
  }
  if constexpr (std::numeric_limits<std::size_t>::max() >
                std::numeric_limits<std::uint32_t>::max()) {
    constexpr std::size_t oversized =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) +
        1u;
    auto overflow = Target()
                        .map<std::int32_t>("window-count-overflow", oversized,
                                           [](auto value) { return value; })
                        .filter([](auto value) { return value != 0; })
                        .branch([](auto values) {
                          return values.window(WindowSpec{.radius = 1u});
                        })
                        .compile();
    if (overflow || overflow.error() != "compute_window_count_overflow") {
      return 2;
    }
  }
  auto empty_pool = Target()
                        .input<std::int32_t>(0u)
                        .branch([](auto values) {
                          return values.pool(PoolSpec{
                              .op = static_cast<Window>(0xffu), .width = 1u});
                        })
                        .compile();
  auto empty_window =
      Target()
          .input<std::int32_t>(0u)
          .branch([](auto values) {
            return values.window(
                WindowSpec{.op = static_cast<Window>(0xffu), .radius = 1u});
          })
          .compile();
  if (empty_pool || empty_window ||
      empty_pool.error() != "compute_window_op_unsupported" ||
      empty_window.error() != "compute_window_op_unsupported") {
    return 3;
  }

  const auto q_matches = [](const std::size_t count, const std::size_t width,
                            const std::size_t stride,
                            const std::size_t drop_count,
                            const std::size_t keep_count) {
    auto program =
        Target()
            .input<std::int32_t>(count)
            .branch([=](auto values) {
              return outputs(values.pool(PoolSpec{.width = width,
                                                  .stride = stride,
                                                  .tail = PoolTail::Drop}),
                             values.pool(PoolSpec{.width = width,
                                                  .stride = stride,
                                                  .tail = PoolTail::Keep}));
            })
            .compile();
    return program && program->template output_size<0u>() == drop_count &&
           program->template output_size<1u>() == keep_count;
  };
  if (!q_matches(5u, 5u, 1u, 1u, 5u) ||
      !q_matches(5u, 3u, 7u, 1u, 1u) ||
      !q_matches(10u, 4u, 3u, 3u, 4u) ||
      !q_matches(10u, 4u, 2u, 4u, 5u)) {
    return 4;
  }

  constexpr PoolSpec keep_clip{
      .op = Window::Sum,
      .width = 3u,
      .stride = 2u,
      .edge = WindowEdge::Clip,
      .tail = PoolTail::Keep,
  };
  constexpr PoolSpec drop_clip{
      .op = Window::Sum,
      .width = keep_clip.width,
      .stride = keep_clip.stride,
      .edge = keep_clip.edge,
      .tail = PoolTail::Drop,
  };
  auto program =
      Target()
          .input<std::int32_t>(5u)
          .branch([=](auto values) {
            return outputs(values.pool(keep_clip), values.pool(drop_clip));
          })
          .compile();
  const std::array<std::uint64_t, 2u> expected_counts{3u, 2u};
  if (!program || program->template output_size<0u>() != 3u ||
      program->template output_size<1u>() != 2u ||
      program->graph().nodes.size() != expected_counts.size()) {
    return 5;
  }
  for (std::size_t index = 0u; index < expected_counts.size(); ++index) {
    const graph::Node &node = program->graph().nodes[index];
    if (node.operation != graph::Operation::Window ||
        node.elements != expected_counts[index] || node.accesses.size() != 2u ||
        node.accesses[0u].element_count != 5u ||
        node.accesses[1u].element_count != expected_counts[index]) {
      return 5;
    }
  }
  const std::array<std::int32_t, 5u> input{1, 2, 3, 4, 5};
  auto output = program->run(input);
  return output &&
                 std::get<0>(*output) == std::vector<std::int32_t>{6, 12, 5} &&
                 std::get<1>(*output) == std::vector<std::int32_t>{6, 12}
             ? 0
             : 6;
}

} // namespace rund_node_test_flow_primitives

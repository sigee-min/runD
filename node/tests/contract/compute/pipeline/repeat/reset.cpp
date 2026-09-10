#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/job/local.hpp"

#include <cstdio>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckResetRepeat(rund::compute::Device &device,
                                   const Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> seed{1, 3, 5, 7};
  // The second logical invocation reaches an empty workset after one body and
  // skips the next controlled body. It reuses the first invocation's external
  // output owners, so the empty result proves that earlier active bytes cannot
  // survive a skipped warm-history path.
  auto controlled =
      on(device)
          .input<std::int32_t>(seed.size())
          .branch([](auto values) {
            auto active = values.filter(
                [](auto value) { return value > std::int32_t{0}; });
            return active.template unroll<2u>(
                [](auto work) {
                  return work.map("repeat-reset-step", [](auto value) {
                    return value - std::int32_t{1};
                  });
                },
                [](auto value) { return value <= std::int32_t{0}; });
          })
          .compile();
  constexpr std::array<std::int32_t, 4u> live_input{4, 3, 0, 0};
  constexpr std::array<std::int32_t, 4u> empty_input{1, 0, 0, 0};
  constexpr std::array<std::int32_t, 4u> poisoned_values{91, 92, 93, 94};
  constexpr std::array<std::uint32_t, 1u> poisoned_count{95u};
  auto live_source = device.upload<std::int32_t>(live_input);
  auto empty_source = device.upload<std::int32_t>(empty_input);
  auto controlled_values = device.upload<std::int32_t>(poisoned_values);
  auto controlled_count = device.upload<std::uint32_t>(poisoned_count);
  if (!controlled || !live_source || !empty_source || !controlled_values ||
      !controlled_count) {
    return 8;
  }
  auto live = pipeline(device)
                  .then(*controlled, read(*live_source),
                        write(*controlled_values, *controlled_count))
                  .prepare();
  const auto controlled_state = detail::ProgramAccess::state(*controlled);
  if (controlled_state == nullptr ||
      controlled_state->graph_info.memory.reset_count != 2u ||
      controlled_state->graph_info.memory.reset_bytes !=
          2u * seed.size() * sizeof(std::int32_t)) {
    return 8;
  }
  std::array<std::size_t, 2u> reset_resources{};
  std::size_t reset_count = 0u;
  for (std::size_t index = 0u;
       index < controlled_state->graph_info.resources.size(); ++index) {
    if (!controlled_state->graph_info.resources[index].requires_reset()) {
      continue;
    }
    if (reset_count >= reset_resources.size()) {
      return 8;
    }
    reset_resources[reset_count++] = index;
  }
  if (reset_count != reset_resources.size()) {
    return 8;
  }
  const graph::Resource first_reset =
      controlled_state->graph_info.resources[reset_resources[0u]];
  const graph::Resource second_reset =
      controlled_state->graph_info.resources[reset_resources[1u]];
  const detail::GraphValueRoute first_route =
      controlled_state->graph_value_routes[reset_resources[0u]];
  const detail::GraphValueRoute second_route =
      controlled_state->graph_value_routes[reset_resources[1u]];
  if (first_route.source != detail::GraphBindSource::Internal ||
      second_route.source != detail::GraphBindSource::Internal ||
      first_route.index != second_route.index ||
      first_route.offset_bytes != second_route.offset_bytes ||
      first_reset.last_use >= second_reset.reset_node) {
    return 8;
  }
  std::array<std::int32_t, 4u> controlled_output{};
  std::array<std::uint32_t, 1u> controlled_size{};
  if (!live || !live->run() ||
      !ReadExact(*live, *controlled_values, controlled_output) ||
      !ReadExact(*live, *controlled_count, controlled_size) ||
      controlled_size[0u] != 2u ||
      controlled_output != std::array<std::int32_t, 4u>{2, 1, 0, 0}) {
    std::fprintf(
        stderr,
        "repeat live backend=%u prepare=%u count=%u values=%d,%d,%d,%d\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(live.reason()),
        controlled_size[0u], controlled_output[0u], controlled_output[1u],
        controlled_output[2u], controlled_output[3u]);
    return 8;
  }
  auto empty = pipeline(device)
                   .then(*controlled, read(*empty_source),
                         write(*controlled_values, *controlled_count))
                   .prepare();
  if (!empty || !empty->run() ||
      !ReadExact(*empty, *controlled_values, controlled_output) ||
      !ReadExact(*empty, *controlled_count, controlled_size) ||
      controlled_size[0u] != 0u ||
      controlled_output != std::array<std::int32_t, 4u>{0, 0, 0, 0} ||
      empty->stats().control.iteration_count != 1u ||
      empty->stats().control.skipped_iteration_count != 1u ||
      empty->stats().reset_bytes == 0u || empty->stats().reset_commands == 0u) {
    std::fprintf(
        stderr,
        "repeat reset backend=%u prepare=%u count=%u values=%d,%d,%d,%d "
        "iterations=%llu skipped=%llu reset_bytes=%llu reset_commands=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(empty.reason()),
        controlled_size[0u], controlled_output[0u], controlled_output[1u],
        controlled_output[2u], controlled_output[3u],
        static_cast<unsigned long long>(
            empty ? empty->stats().control.iteration_count : 0u),
        static_cast<unsigned long long>(
            empty ? empty->stats().control.skipped_iteration_count : 0u),
        static_cast<unsigned long long>(empty ? empty->stats().reset_bytes
                                              : 0u),
        static_cast<unsigned long long>(empty ? empty->stats().reset_commands
                                              : 0u));
    return 8;
  }
  return 0;
}

} // namespace rund_node_test_pipeline

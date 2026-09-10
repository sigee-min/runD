#include "gather.hpp"
#include "local.hpp"
#include "scatter.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_pipeline::view {

[[nodiscard]] int CheckPoolReset(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 8u> source_values{0, 1, 2, 3, 4, 5, 6, 7};
  auto source = Upload(device, source_values);
  auto even = on(device)
                  .map<std::int32_t>("pipeline-even-view", 4u,
                                     [](auto value) { return value + 1; })
                  .compile();
  if (!source || !even) {
    return 17;
  }
  auto even_input = source->view(0u, 4u, 2u);
  if (!even_input) {
    return 17;
  }
  std::array<std::int32_t, 8u> observed{};
  // Return an exact-size nonzero allocation at a logical ownership boundary.
  // Pooling backends reacquire that storage; every backend must still publish
  // a zeroed Buffer before the partial strided write touches only even lanes.
  auto poison =
      on(device)
          .map<std::int32_t>("pipeline-view-pool-poison", source_values.size(),
                             [](auto value) { return value + 101; })
          .compile();
  if (!poison) {
    return 17;
  }
  {
    auto poison_target = device.buffer<std::int32_t>(source_values.size());
    if (!poison_target) {
      return 18;
    }
    auto poisoned = pipeline(device)
                        .then(*poison, read(*source), write(*poison_target))
                        .prepare();
    if (!poisoned || !poisoned->run()) {
      return 19;
    }
  }
  auto reused_target = device.buffer<std::int32_t>(source_values.size());
  if (!reused_target) {
    return 20;
  }
  auto reused_output = reused_target->view(0u, 4u, 2u);
  if (!reused_output) {
    return 21;
  }
  auto reused = pipeline(device)
                    .then(*even, read(*even_input), write(*reused_output))
                    .prepare();
  if (!reused || !reused->run() ||
      !ReadExact(*reused, *reused_target, observed) ||
      observed != std::array<std::int32_t, 8u>{1, 0, 3, 0, 5, 0, 7, 0}) {
    return 22;
  }

  return 0;
}

template <class T>
[[nodiscard]] bool CheckStridedReset(rund::compute::Device &device,
                                     const Backend backend) {
  using namespace rund::compute;
  auto scatter =
      on(device)
          .map<T>("pipeline-view-reset", 1u, [](auto value) { return value; })
          .scatter(1u, {.count = 2u})
          .compile();
  constexpr std::array<T, 4u> history{101u, 202u, 303u, 404u};
  constexpr std::array<T, 1u> first_value{7u};
  constexpr std::array<std::uint32_t, 1u> first_index{0u};
  constexpr std::array<T, 1u> second_value{9u};
  constexpr std::array<std::uint32_t, 1u> second_index{1u};
  auto target = device.upload<T>(history);
  auto first_values = device.upload<T>(first_value);
  auto first_indices = device.upload<std::uint32_t>(first_index);
  auto second_values = device.upload<T>(second_value);
  auto second_indices = device.upload<std::uint32_t>(second_index);
  if (!scatter || !target || !first_values || !first_indices ||
      !second_values || !second_indices ||
      scatter->graph().memory.reset_bytes != 2u * sizeof(T) ||
      scatter->graph().memory.reset_count != 1u) {
    return false;
  }
  auto view = target->view(0u, 2u, 2u);
  if (!view) {
    return false;
  }
  auto first =
      pipeline(device)
          .then(*scatter, read(*first_values, *first_indices), write(*view))
          .prepare();
  std::array<T, 4u> observed{};
  if (!first || !first->run() || !ReadExact(*first, *target, observed) ||
      observed != std::array<T, 4u>{7u, 202u, 0u, 404u}) {
    return false;
  }
  auto second =
      pipeline(device)
          .then(*scatter, read(*second_values, *second_indices), write(*view))
          .prepare();
  if (!second || !second->run() || !ReadExact(*second, *target, observed) ||
      observed != std::array<T, 4u>{0u, 202u, 9u, 404u} ||
      second->stats().reset_bytes != 2u * sizeof(T) ||
      second->stats().reset_commands != 1u) {
    std::fprintf(
        stderr,
        "pipeline reset backend=%u width=%zu bytes=%llu commands=%llu\n",
        static_cast<unsigned>(backend), sizeof(T),
        static_cast<unsigned long long>(second ? second->stats().reset_bytes
                                               : 0u),
        static_cast<unsigned long long>(second ? second->stats().reset_commands
                                               : 0u));
    return false;
  }
  return true;
}

[[nodiscard]] bool CheckScatterOffset(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 5u> values{99, 5, 7, 11, 13};
  constexpr std::array<std::uint32_t, 5u> indices{99u, 0u, 1u, 0u, 1u};
  constexpr std::array<std::uint32_t, 3u> counts{99u, 4u, 99u};
  auto value_buffer = Upload(device, values);
  auto index_buffer = Upload(device, indices);
  auto count_buffer = Upload(device, counts);
  auto output_buffer = device.buffer<std::int32_t>(4u);
  auto exact = on(device)
                   .input<std::int32_t>(4u)
                   .zip_input<std::uint32_t>(4u)
                   .branch([](auto input, auto targets) {
                     return input.scatter_reduce(targets, 2u, Reduce::Sum);
                   })
                   .compile();
  auto bounded = on(device)
                     .input<Bounded<std::int32_t>>(4u)
                     .branch([](auto input) {
                       auto targets = input.indices().map(
                           "pipeline-scatter-target", [](auto ordinal) {
                             return ordinal & std::uint32_t{1};
                           });
                       return input.scatter_reduce(targets, 2u, Reduce::Sum);
                     })
                     .compile();
  if (!value_buffer || !index_buffer || !count_buffer || !output_buffer ||
      !exact || !bounded) {
    return false;
  }
  auto value_view = value_buffer->view(1u, 4u);
  auto index_view = index_buffer->view(1u, 4u);
  auto count_view = count_buffer->view(1u, 1u);
  auto output_view = output_buffer->view(1u, 2u);
  if (!value_view || !index_view || !count_view || !output_view) {
    return false;
  }
  auto exact_run =
      pipeline(device)
          .then(*exact, read(*value_view, *index_view), write(*output_view))
          .prepare();
  std::array<std::int32_t, 4u> observed{};
  if (!exact_run || !exact_run->run() ||
      !ReadExact(*exact_run, *output_buffer, observed) ||
      observed != std::array<std::int32_t, 4u>{0, 16, 20, 0}) {
    return false;
  }
  auto bounded_run =
      pipeline(device)
          .then(*bounded, read(*value_view, *count_view), write(*output_view))
          .prepare();
  return bounded_run && bounded_run->run() &&
         ReadExact(*bounded_run, *output_buffer, observed) &&
         observed == std::array<std::int32_t, 4u>{0, 16, 20, 0} &&
         CheckGatherPreflight<std::uint32_t>(device) &&
         CheckGatherPreflight<std::uint64_t>(device) &&
         CheckScatterCohortTelemetry<std::uint32_t>(device) &&
         CheckScatterCohortTelemetry<std::uint64_t>(device);
}

[[nodiscard]] bool CheckStridedReset(rund::compute::Device &device,
                                     const rund::compute::Backend backend) {
  return CheckStridedReset<std::uint32_t>(device, backend) &&
         CheckStridedReset<std::uint64_t>(device, backend);
}

} // namespace rund_node_test_pipeline::view

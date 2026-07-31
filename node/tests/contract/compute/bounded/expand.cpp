#include "model.hpp"

#include "../../target/selection.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <ranges>
#include <vector>

namespace rund_node_bounded_contract {

template <class T>
int CheckFilteredExpand(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<T, 4u> input{T{1}, T{2}, T{3}, T{4}};
  auto output = on(rund::node::test_contract::target_for(backend), input)
                    .filter([](auto value) { return value > T{1}; })
                    .expand(
                        MaxItems{3u}, [](auto value) { return value - T{1}; },
                        [](auto value, auto index) { return value + index; })
                    .collect();
  return output && *output == std::vector<T>{T{2}, T{3}, T{4}, T{4}, T{5}, T{6}}
             ? 0
             : 1;
}

template <class T> int CheckExpandLaws(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<T, 3u> input{T{2}, T{1}, T{3}};
  const std::vector<T> expected{T{20}, T{21}, T{10}, T{30}, T{31}, T{32}};
  auto output =
      on(rund::node::test_contract::target_for(backend), input)
          .expand(
              MaxItems{3u}, [](auto value) { return value; },
              [](auto value, auto index) { return value * T{10} + index; })
          .collect();
  if (!output || *output != expected) {
    return 1;
  }

  const std::array<T, 2u> zeros{T{0}, T{0}};
  auto zero_output =
      on(rund::node::test_contract::target_for(backend), zeros)
          .expand(
              MaxItems{3u}, [](auto value) { return value; },
              [](auto value, auto index) { return value + index; })
          .collect();
  if (!zero_output || !zero_output->empty()) {
    return 2;
  }

  const std::array<T, 0u> empty{};
  auto empty_output =
      on(rund::node::test_contract::target_for(backend), empty)
          .expand(
              MaxItems{3u}, [](auto value) { return value; },
              [](auto value, auto index) { return value + index; })
          .collect();
  if (!empty_output || !empty_output->empty()) {
    return 3;
  }

  if constexpr (!std::same_as<T, std::uint64_t>) {
    const std::array<T, 2u> overflow{T{1}, T{4}};
    auto rejected =
        on(rund::node::test_contract::target_for(backend), overflow)
            .expand(
                MaxItems{3u}, [](auto value) { return value; },
                [](auto value, auto index) { return value + index; })
            .collect();
    if (rejected || rejected.error() != "compute_bounded_count_invalid") {
      return 4;
    }
  }
  if constexpr (std::is_signed_v<T>) {
    const std::array<T, 2u> negative{T{1}, T{-1}};
    auto negative_result =
        on(rund::node::test_contract::target_for(backend), negative)
            .expand(
                MaxItems{3u}, [](auto value) { return value; },
                [](auto value, auto index) { return value + index; })
            .collect();
    if (negative_result ||
        negative_result.error() != "compute_bounded_count_invalid") {
      return 5;
    }
  }
  return 0;
}

template <class T>
int CheckExpandReuse(const rund::compute::Backend backend,
                     rund::compute::Stats *const evidence = nullptr) {
  using namespace rund::compute;
  const std::array<T, 3u> input{T{2}, T{1}, T{3}};
  const std::vector<T> expected{T{20}, T{21}, T{10}, T{30}, T{31}, T{32}};
  auto program =
      on(rund::node::test_contract::target_for(backend))
          .template map<T>("expand-reuse", input.size(),
                           [](auto value) { return value; })
          .expand(
              MaxItems{3u}, [](auto value) { return value; },
              [](auto value, auto index) { return value * T{10} + index; })
          .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return 2;
  }
  const Stats cold = job->stats();
  if (!job->run()) {
    return 3;
  }
  const Stats warm = job->stats();
  if (warm.graph_hash != cold.graph_hash || warm.pipeline_compiles != 0u ||
      warm.buffer_allocations != 0u || warm.uploaded_bytes != 0u ||
      warm.download_events != 0u) {
    return 4;
  }
  auto output = job->read();
  if (!output || *output != expected) {
    return 5;
  }
  const std::array<T, 3u> overflow{T{4}, T{1}, T{3}};
  if (!job->write(overflow)) {
    return 6;
  }
  const Status rejected = job->run();
  if (rejected || rejected.error() != "compute_bounded_count_invalid") {
    return 7;
  }
  if (!job->write(input) || !job->run()) {
    return 8;
  }
  output = job->read();
  if (!output || *output != expected) {
    return 9;
  }
  if (evidence != nullptr) {
    *evidence = job->stats();
  }
  return 0;
}

[[nodiscard]] int CheckExpandBackend(const rund::compute::Backend backend,
                                     rund::compute::Stats *const stats) {
  const std::array filtered{CheckFilteredExpand<std::int32_t>(backend),
                            CheckFilteredExpand<std::uint32_t>(backend),
                            CheckFilteredExpand<std::int64_t>(backend),
                            CheckFilteredExpand<std::uint64_t>(backend)};
  if (std::ranges::any_of(filtered,
                          [](const int result) { return result != 0; })) {
    return 280 + 10 * static_cast<int>(backend);
  }
  const std::array laws{CheckExpandLaws<std::int32_t>(backend),
                        CheckExpandLaws<std::uint32_t>(backend),
                        CheckExpandLaws<std::int64_t>(backend),
                        CheckExpandLaws<std::uint64_t>(backend)};
  if (std::ranges::any_of(laws, [](const int result) { return result != 0; })) {
    return 320 + 10 * static_cast<int>(backend);
  }
  if (CheckExpandReuse<std::int32_t>(backend, stats) != 0 ||
      CheckExpandReuse<std::uint64_t>(backend) != 0) {
    return 360 + 10 * static_cast<int>(backend);
  }
  return 0;
}

} // namespace rund_node_bounded_contract

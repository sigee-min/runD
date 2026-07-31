#include "local.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_bounded_contract {

[[nodiscard]] int CheckParity() {
  using namespace rund::compute;
  const std::array<std::int64_t, 4> input{1, 2, 3, 4};
  std::uint64_t graph_hash = 0u;
  std::uint64_t output_hash = 0u;
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    auto program = on(rund::node::test_contract::target_for(backend))
                       .map<std::int64_t>("bounded-parity", input.size(),
                                          [](auto value) { return value; })
                       .filter([](auto value) { return value > 2; })
                       .scan(Scan::InclusiveSum)
                       .compile();
    if (!program) {
      return 1;
    }
    auto job = program->resident(input);
    if (!job || !job->run()) {
      return 2;
    }
    auto output = job->read();
    if (!output || *output != std::vector<std::int64_t>{3, 7}) {
      return 3;
    }
    const Stats stats = job->stats();
    if (stats.graph_hash == 0u || stats.output_hash == 0u) {
      return 4;
    }
    if (graph_hash == 0u) {
      graph_hash = stats.graph_hash;
      output_hash = stats.output_hash;
    } else if (stats.graph_hash != graph_hash ||
               stats.output_hash != output_hash) {
      return 5;
    }
  }
  const std::array<std::int64_t, 5> unordered{5, 1, 4, 2, 3};
  graph_hash = 0u;
  output_hash = 0u;
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    auto program =
        on(rund::node::test_contract::target_for(backend))
            .map<std::int64_t>("bounded-sort-parity", unordered.size(),
                               [](auto value) { return value; })
            .filter([](auto value) { return value > 1; })
            .sort()
            .compile();
    if (!program) {
      return 6;
    }
    auto job = program->resident(unordered);
    if (!job || !job->run()) {
      return 7;
    }
    auto output = job->read();
    if (!output || *output != std::vector<std::int64_t>{2, 3, 4, 5}) {
      return 8;
    }
    const Stats stats = job->stats();
    if (graph_hash == 0u) {
      graph_hash = stats.graph_hash;
      output_hash = stats.output_hash;
    } else if (stats.graph_hash != graph_hash ||
               stats.output_hash != output_hash) {
      return 9;
    }
  }
  graph_hash = 0u;
  output_hash = 0u;
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    auto program =
        on(rund::node::test_contract::target_for(backend))
            .map<std::int64_t>("bounded-argsort-parity", unordered.size(),
                               [](auto value) { return value; })
            .filter([](auto value) { return value > 1; })
            .argsort()
            .compile();
    if (!program) {
      return 10;
    }
    auto job = program->resident(unordered);
    if (!job || !job->run()) {
      return 11;
    }
    auto output = job->read();
    if (!output || *output != std::vector<std::uint32_t>{2u, 3u, 1u, 0u}) {
      return 12;
    }
    const Stats stats = job->stats();
    if (graph_hash == 0u) {
      graph_hash = stats.graph_hash;
      output_hash = stats.output_hash;
    } else if (stats.graph_hash != graph_hash ||
               stats.output_hash != output_hash) {
      return 13;
    }
  }
  return 0;
}

} // namespace rund_node_bounded_contract

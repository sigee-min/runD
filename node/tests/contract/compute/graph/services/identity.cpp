#include "model.hpp"

#include <array>
#include <vector>

namespace rund_node_graph_services {

[[nodiscard]] int CheckIdentity(rund::compute::Device *const device,
                                rund::compute::ProgramCache *const cache) {
  auto first = rund::compute::on(*device, *cache)
                   .map<std::int32_t>("diagnostic-first", 4u,
                                      [](auto value) { return value + 1; })
                   .map("twice", [](auto value) { return value * 2; })
                   .compile();
  auto second = rund::compute::on(*device, *cache)
                    .map<std::int32_t>("renamed-only", 4u,
                                       [](auto value) { return value + 1; })
                    .map("also-renamed", [](auto value) { return value * 2; })
                    .compile();
  if (!first || !second || first->fingerprint() != second->fingerprint()) {
    return 3;
  }
  const auto shared = cache->stats();
  if (shared.misses != 1u || shared.hits != 1u || shared.ready_entries != 1u ||
      shared.in_flight != 0u) {
    return 4;
  }

  auto expression_order_cache = rund::compute::program_cache(*device, 2u);
  if (!expression_order_cache) {
    return 44;
  }
  auto expression_plus_first =
      rund::compute::on(*device, *expression_order_cache)
          .map<std::int32_t>("expression-plus-first", 4u,
                             [](auto value) {
                               const auto plus = value + 1;
                               const auto times = value * 2;
                               return plus + times;
                             })
          .compile();
  auto expression_times_first =
      rund::compute::on(*device, *expression_order_cache)
          .map<std::int32_t>("expression-times-first", 4u,
                             [](auto value) {
                               const auto times = value * 2;
                               const auto plus = value + 1;
                               return plus + times;
                             })
          .compile();
  const auto expression_order_stats = expression_order_cache->stats();
  if (!expression_plus_first || !expression_times_first ||
      expression_plus_first->fingerprint() !=
          expression_times_first->fingerprint() ||
      expression_order_stats.misses != 1u ||
      expression_order_stats.hits != 1u || expression_order_stats.waits != 0u ||
      expression_order_stats.ready_entries != 1u) {
    return 45;
  }
  const std::array<std::int32_t, 4u> ordered_input{1, 2, 3, 4};
  auto expression_order_output = expression_times_first->run(ordered_input);
  if (!expression_order_output ||
      *expression_order_output != std::vector<std::int32_t>{4, 7, 10, 13}) {
    return 46;
  }

  auto flow_order_cache = rund::compute::program_cache(*device, 2u);
  if (!flow_order_cache) {
    return 47;
  }
  auto flow_plus_first =
      rund::compute::on(*device, *flow_order_cache)
          .map<std::int32_t>("flow-source-a", 4u,
                             [](auto value) { return value; })
          .branch([](auto values) {
            const auto plus =
                values.map("flow-plus-a", [](auto value) { return value + 1; });
            const auto times = values.map("flow-times-a",
                                          [](auto value) { return value * 2; });
            return rund::compute::outputs(plus, times);
          })
          .compile();
  auto flow_times_first =
      rund::compute::on(*device, *flow_order_cache)
          .map<std::int32_t>("flow-source-b", 4u,
                             [](auto value) { return value; })
          .branch([](auto values) {
            const auto times = values.map("flow-times-b",
                                          [](auto value) { return value * 2; });
            const auto plus =
                values.map("flow-plus-b", [](auto value) { return value + 1; });
            return rund::compute::outputs(plus, times);
          })
          .compile();
  const auto flow_order_stats = flow_order_cache->stats();
  if (!flow_plus_first || !flow_times_first ||
      flow_plus_first->fingerprint() != flow_times_first->fingerprint() ||
      flow_order_stats.misses != 1u || flow_order_stats.hits != 1u ||
      flow_order_stats.waits != 0u || flow_order_stats.ready_entries != 1u) {
    return 48;
  }
  auto flow_order_job = flow_times_first->resident(ordered_input);
  if (!flow_order_job || !flow_order_job->run()) {
    return 49;
  }
  auto flow_plus_output = flow_order_job->template read<0u>();
  auto flow_times_output = flow_order_job->template read<1u>();
  if (!flow_plus_output || !flow_times_output ||
      *flow_plus_output != std::vector<std::int32_t>{2, 3, 4, 5} ||
      *flow_times_output != std::vector<std::int32_t>{2, 4, 6, 8}) {
    return 50;
  }

  const Info graph = first->graph();
  if (!ValidResourceGraph(graph)) {
    return 5;
  }

  const std::array<std::int32_t, 4> low{1, 2, 3, 4};
  const std::array<std::int32_t, 4> high{10, 20, 30, 40};
  auto low_result = first->run(std::span<const std::int32_t>{low});
  auto high_result = first->run(std::span<const std::int32_t>{high});
  if (!low_result || !high_result ||
      *low_result != std::vector<std::int32_t>{4, 6, 8, 10} ||
      *high_result != std::vector<std::int32_t>{22, 42, 62, 82} ||
      *low_result == *high_result || cache->stats().misses != 1u) {
    return 6;
  }

  auto changed = build(*device, *cache, "semantic-change", 2);
  if (!changed || changed->fingerprint() == first->fingerprint() ||
      cache->stats().misses != 2u) {
    return 7;
  }
  auto resized = build(*device, *cache, "interface-bound-change", 2, 5u);
  if (!resized || resized->fingerprint() == changed->fingerprint() ||
      cache->stats().misses != 3u) {
    return 29;
  }
  return 0;
}

} // namespace rund_node_graph_services

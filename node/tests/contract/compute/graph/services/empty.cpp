#include "model.hpp"

#include <array>
#include <span>
#include <vector>

namespace rund_node_graph_services {

[[nodiscard]] int CheckEmpty(rund::compute::Device *const device) {
  using FixedValue = rund::compute::Fixed<16, 16>;
  using AlternateFormat = rund::compute::Fixed<17, 15>;
  auto empty_cache = rund::compute::program_cache(*device, 2u);
  if (!empty_cache) {
    return 21;
  }
  auto empty_first =
      rund::compute::on(*device, *empty_cache)
          .map<std::int32_t>("empty-first", 0u,
                             [](auto value) { return value + 1; })
          .compile();
  auto empty_second =
      rund::compute::on(*device, *empty_cache)
          .map<std::int32_t>("empty-renamed", 0u,
                             [](auto value) { return value + 1; })
          .compile();
  if (!empty_first || !empty_second ||
      empty_first->fingerprint() != empty_second->fingerprint() ||
      !empty_first->fingerprint() || empty_first->graph().nodes.size() != 0u ||
      empty_first->graph().resources.size() != 2u ||
      empty_first->graph().inputs != std::vector<std::uint32_t>{1u} ||
      empty_cache->stats().misses != 1u || empty_cache->stats().hits != 1u) {
    return 22;
  }
  const std::span<const std::int32_t> empty_input;
  auto empty_output = empty_first->run(empty_input);
  if (!empty_output || !empty_output->empty()) {
    return 23;
  }
  auto empty_semantic_cache = rund::compute::program_cache(*device, 2u);
  if (!empty_semantic_cache) {
    return 66;
  }
  auto empty_plus = rund::compute::on(*device, *empty_semantic_cache)
                        .map<std::int32_t>("empty-semantic-plus", 0u,
                                           [](auto value) { return value + 1; })
                        .compile();
  auto empty_times =
      rund::compute::on(*device, *empty_semantic_cache)
          .map<std::int32_t>("empty-semantic-times", 0u,
                             [](auto value) { return value * 2; })
          .compile();
  const auto empty_semantic_stats = empty_semantic_cache->stats();
  if (!empty_plus || !empty_times ||
      empty_plus->fingerprint() == empty_times->fingerprint() ||
      !empty_plus->graph().nodes.empty() ||
      !empty_times->graph().nodes.empty() ||
      empty_semantic_stats.misses != 2u || empty_semantic_stats.hits != 0u ||
      empty_semantic_stats.ready_entries != 2u) {
    return 67;
  }
  auto empty_scan_cache = rund::compute::program_cache(*device, 2u);
  if (!empty_scan_cache) {
    return 68;
  }
  const auto build_empty_scan = [&](const char *const name,
                                    const rund::compute::Scan operation) {
    return rund::compute::on(*device, *empty_scan_cache)
        .map<std::uint32_t>(name, 0u, [](auto value) { return value; })
        .scan(operation)
        .compile();
  };
  auto empty_inclusive = build_empty_scan("empty-scan-inclusive",
                                          rund::compute::Scan::InclusiveSum);
  auto empty_exclusive = build_empty_scan("empty-scan-exclusive",
                                          rund::compute::Scan::ExclusiveSum);
  const auto empty_scan_stats = empty_scan_cache->stats();
  if (!empty_inclusive || !empty_exclusive ||
      empty_inclusive->fingerprint() == empty_exclusive->fingerprint() ||
      !empty_inclusive->graph().nodes.empty() ||
      !empty_exclusive->graph().nodes.empty() ||
      empty_scan_stats.misses != 2u || empty_scan_stats.hits != 0u) {
    return 69;
  }
  auto empty_primitive_cache = rund::compute::program_cache(*device, 2u);
  if (!empty_primitive_cache) {
    return 70;
  }
  const auto build_empty_window = [&](const char *const name,
                                      const std::uint32_t radius) {
    return rund::compute::on(*device, *empty_primitive_cache)
        .map<std::uint32_t>(name, 0u, [](auto value) { return value; })
        .window({.op = rund::compute::Window::Sum, .radius = radius})
        .compile();
  };
  auto empty_radius_one = build_empty_window("empty-window-one", 1u);
  auto empty_radius_two = build_empty_window("empty-window-two", 2u);
  const auto empty_primitive_stats = empty_primitive_cache->stats();
  if (!empty_radius_one || !empty_radius_two ||
      empty_radius_one->fingerprint() == empty_radius_two->fingerprint() ||
      !empty_radius_one->graph().nodes.empty() ||
      !empty_radius_two->graph().nodes.empty() ||
      empty_primitive_stats.misses != 2u || empty_primitive_stats.hits != 0u) {
    return 71;
  }
  auto empty_compact_cache = rund::compute::program_cache(*device, 2u);
  if (!empty_compact_cache) {
    return 60;
  }
  const auto build_empty_compact = [&](const char *const name) {
    return rund::compute::on(*device, *empty_compact_cache)
        .map<std::uint32_t>(name, 0u, [](auto value) { return value; })
        .compact()
        .compile();
  };
  auto empty_compact_first = build_empty_compact("empty-compact-first");
  auto empty_compact_second = build_empty_compact("empty-compact-renamed");
  const auto empty_compact_stats = empty_compact_cache->stats();
  if (!empty_compact_first || !empty_compact_second ||
      !empty_compact_first->fingerprint() ||
      empty_compact_first->fingerprint() !=
          empty_compact_second->fingerprint() ||
      empty_compact_stats.misses != 1u || empty_compact_stats.hits != 1u ||
      empty_compact_stats.ready_entries != 1u) {
    return 61;
  }
  const auto empty_compact_graph = empty_compact_first->graph();
  if (!empty_compact_graph.nodes.empty() ||
      empty_compact_graph.outputs.size() != 2u ||
      empty_compact_graph.resources.empty()) {
    return 62;
  }
  const auto value_output = empty_compact_graph.outputs[0u];
  const auto count_output = empty_compact_graph.outputs[1u];
  if (value_output == 0u || count_output == 0u ||
      value_output > empty_compact_graph.resources.size() ||
      count_output > empty_compact_graph.resources.size()) {
    return 63;
  }
  const auto &empty_values = empty_compact_graph.resources[value_output - 1u];
  const auto &empty_count = empty_compact_graph.resources[count_output - 1u];
  if (empty_values.elements != 0u || empty_values.bytes != 0u ||
      empty_count.elements != 1u ||
      empty_count.element_bytes != sizeof(std::uint32_t) ||
      empty_count.bytes != sizeof(std::uint32_t)) {
    return 64;
  }
  const std::span<const std::uint32_t> empty_flags;
  auto empty_compacted = empty_compact_first->run(empty_flags);
  if (!empty_compacted || !empty_compacted->empty()) {
    return 65;
  }
  auto empty_sort_cache = rund::compute::program_cache(*device, 3u);
  if (!empty_sort_cache) {
    return 72;
  }
  const auto build_empty_sort = [&](const char *const name) {
    return rund::compute::on(*device, *empty_sort_cache)
        .map<std::int32_t>(name, 0u, [](auto value) { return value; })
        .sort()
        .compile();
  };
  auto empty_sort_first = build_empty_sort("empty-sort-first");
  auto empty_sort_second = build_empty_sort("empty-sort-renamed");
  auto empty_argsort = rund::compute::on(*device, *empty_sort_cache)
                           .map<std::int32_t>("empty-argsort", 0u,
                                              [](auto value) { return value; })
                           .argsort()
                           .compile();
  const auto empty_sort_stats = empty_sort_cache->stats();
  if (!empty_sort_first || !empty_sort_second || !empty_argsort ||
      empty_sort_first->fingerprint() != empty_sort_second->fingerprint() ||
      empty_sort_first->fingerprint() == empty_argsort->fingerprint() ||
      !empty_sort_first->graph().nodes.empty() ||
      !empty_argsort->graph().nodes.empty() || empty_sort_stats.misses != 2u ||
      empty_sort_stats.hits != 1u || empty_sort_stats.ready_entries != 2u) {
    return 73;
  }
  const std::span<const std::int32_t> empty_sort_input;
  auto empty_sorted = empty_sort_first->run(empty_sort_input);
  auto empty_indices = empty_argsort->run(empty_sort_input);
  if (!empty_sorted || !empty_indices || !empty_sorted->empty() ||
      !empty_indices->empty()) {
    return 74;
  }
  auto empty_pruned =
      rund::compute::on(*device, *empty_cache)
          .map<std::int32_t>("empty-pruned", 0u,
                             [](auto value) {
                               const auto unused_expression = value * 7;
                               (void)unused_expression;
                               return value + 1;
                             })
          .branch([](auto selected) {
            const auto unused_branch = selected.map(
                "empty-dead-branch", [](auto value) { return value * 9; });
            (void)unused_branch;
            return selected;
          })
          .compile();
  if (!empty_pruned ||
      empty_pruned->fingerprint() != empty_first->fingerprint() ||
      empty_pruned->graph().resources.size() != 2u ||
      !empty_pruned->graph().nodes.empty() ||
      empty_cache->stats().misses != 1u || empty_cache->stats().hits != 2u) {
    return 33;
  }
  auto empty_order_cache = rund::compute::program_cache(*device, 2u);
  if (!empty_order_cache) {
    return 51;
  }
  auto empty_plus_first =
      rund::compute::on(*device, *empty_order_cache)
          .map<std::int32_t>("empty-flow-source-a", 0u,
                             [](auto value) { return value; })
          .branch([](auto values) {
            const auto plus = values.map("empty-flow-plus-a",
                                         [](auto value) { return value + 1; });
            const auto times = values.map("empty-flow-times-a",
                                          [](auto value) { return value * 2; });
            return rund::compute::outputs(plus, times);
          })
          .compile();
  auto empty_times_first =
      rund::compute::on(*device, *empty_order_cache)
          .map<std::int32_t>("empty-flow-source-b", 0u,
                             [](auto value) { return value; })
          .branch([](auto values) {
            const auto times = values.map("empty-flow-times-b",
                                          [](auto value) { return value * 2; });
            const auto plus = values.map("empty-flow-plus-b",
                                         [](auto value) { return value + 1; });
            return rund::compute::outputs(plus, times);
          })
          .compile();
  const auto empty_order_stats = empty_order_cache->stats();
  if (!empty_plus_first || !empty_times_first ||
      empty_plus_first->fingerprint() != empty_times_first->fingerprint() ||
      !empty_plus_first->graph().nodes.empty() ||
      empty_plus_first->graph().outputs.size() != 2u ||
      empty_order_stats.misses != 1u || empty_order_stats.hits != 1u ||
      empty_order_stats.waits != 0u || empty_order_stats.ready_entries != 1u) {
    return 52;
  }
  using EmptyAlternate = rund::compute::Fixed<18, 14>;
  auto empty_interface_cache = rund::compute::program_cache(*device, 4u);
  if (!empty_interface_cache) {
    return 41;
  }
  const auto empty_unused_fixed_input = [&]<class Side>() {
    return rund::compute::on(*device, *empty_interface_cache)
        .map<FixedValue>(
            "empty-interface-root", 0u,
            [](auto value) { return rund::compute::quantize<Side>(value); })
        .combine("empty-interface-unused", 0u,
                 [](auto left, auto right) {
                   (void)right;
                   return left;
                 })
        .map("empty-interface-storage",
             [](auto value) {
               return rund::compute::quantize<FixedValue>(value);
             })
        .compile();
  };
  auto empty_alternate =
      empty_unused_fixed_input.template operator()<AlternateFormat>();
  auto empty_other =
      empty_unused_fixed_input.template operator()<EmptyAlternate>();
  if (!empty_alternate || !empty_other ||
      empty_alternate->fingerprint() == empty_other->fingerprint() ||
      empty_interface_cache->stats().misses != 2u ||
      empty_alternate->graph().inputs != std::vector<std::uint32_t>{1u, 2u} ||
      empty_other->graph().inputs != std::vector<std::uint32_t>{1u, 2u}) {
    return 42;
  }
  const auto alternate_graph = empty_alternate->graph();
  const auto other_graph = empty_other->graph();
  const auto &alternate_side =
      alternate_graph.resources[alternate_graph.inputs[1u] - 1u];
  const auto &other_side = other_graph.resources[other_graph.inputs[1u] - 1u];
  if (alternate_side.integer_bits != 17u ||
      alternate_side.fraction_bits != 15u || other_side.integer_bits != 18u ||
      other_side.fraction_bits != 14u ||
      alternate_side.rounding != rund::compute::Rounding::NearestEven ||
      alternate_side.overflow != rund::compute::Overflow::Saturate ||
      alternate_side.approximation != rund::compute::Approximation::Exact) {
    return 43;
  }
  return 0;
}

} // namespace rund_node_graph_services

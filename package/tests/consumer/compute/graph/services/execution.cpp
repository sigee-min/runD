#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/async.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <type_traits>
#include <vector>

namespace package_compute::graph_services {

int CheckExecution() {
  auto device = rund::compute::open(
      rund::compute::Target::cpu(2u),
      rund::compute::Compile{.workers = 2u, .capacity = 4u});
  if (!device) {
    return device.exit_code();
  }
  auto cache = rund::compute::program_cache(*device, 2u);
  if (!cache) {
    std::fprintf(stderr, "package program cache failed: %.*s\n",
                 static_cast<int>(cache.error().size()), cache.error().data());
    return cache.exit_code();
  }

  auto pending =
      rund::compute::on(*device, *cache)
          .map<std::int32_t>("package-async", 4u,
                             [](auto value) { return value + value; })
          .scan(rund::compute::Scan::InclusiveSum)
          .map("package-middle", [](auto value) { return value + value; })
          .scan(rund::compute::Scan::InclusiveSum)
          .compile_async();
  if (!pending) {
    return pending.exit_code();
  }
  auto first = pending->get();
  auto second =
      rund::compute::on(*device, *cache)
          .map<std::int32_t>("diagnostic-name-only", 4u,
                             [](auto value) { return value + value; })
          .scan(rund::compute::Scan::InclusiveSum)
          .map("diagnostic-middle", [](auto value) { return value + value; })
          .scan(rund::compute::Scan::InclusiveSum)
          .compile();
  const auto reused = cache->stats();
  static_assert(std::is_same_v<decltype(reused),
                               const rund::compute::ProgramCache::Stats>);
  if (!first) {
    return first.exit_code();
  }
  if (!second) {
    return second.exit_code();
  }
  const auto &memory = first->graph().memory;
  bool destructive = false;
  for (const auto &resource : first->graph().resources) {
    destructive = destructive || (resource.source != 0u &&
                                  resource.alias_offset_bytes == 256u);
  }
  if (!first->fingerprint() || first->fingerprint() != second->fingerprint() ||
      !ValidateGraph(first->graph()) || memory.logical_bytes != 48u ||
      memory.live_bytes != 32u || memory.physical_bytes != 272u ||
      memory.allocation_count != 1u || !destructive || reused.misses != 1u ||
      reused.hits != 1u) {
    return 2;
  }
  const std::array<std::int32_t, 4u> low{1, 2, 3, 4};
  const std::array<std::int32_t, 4u> high{10, 20, 30, 40};
  auto low_output = first->run(std::span<const std::int32_t>{low});
  auto high_output = first->run(std::span<const std::int32_t>{high});
  const auto executed = cache->stats();
  if (!low_output) {
    return low_output.exit_code();
  }
  if (!high_output) {
    return high_output.exit_code();
  }
  if (*low_output != std::vector<std::int32_t>{4, 16, 40, 80} ||
      *high_output != std::vector<std::int32_t>{40, 160, 400, 800} ||
      *low_output == *high_output || executed.misses != reused.misses ||
      executed.hits != reused.hits) {
    return 2;
  }

  auto bounded = rund::compute::on(*device)
                     .map<std::int32_t>("package-bounded", 4u,
                                        [](auto value) { return value; })
                     .filter([](auto value) { return value > 1; })
                     .filter([](auto value) { return value > 2; })
                     .compile();
  if (!bounded || !ValidateGraph(bounded->graph())) {
    return bounded ? 2 : bounded.exit_code();
  }
  bool lineage = false;
  for (const auto &resource : bounded->graph().resources) {
    lineage = lineage ||
              (resource.active != 0u &&
               bounded->graph().resources[resource.active - 1u].parent != 0u);
  }
  auto bounded_output = bounded->run(std::span<const std::int32_t>{low});
  if (!lineage || !bounded_output ||
      *bounded_output != std::vector<std::int32_t>{3, 4}) {
    return bounded_output ? 2 : bounded_output.exit_code();
  }

  std::array<std::int32_t, 64u> values{};
  std::array<std::uint32_t, 64u> previous{};
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = static_cast<std::int32_t>(index);
  }
  auto indexed = rund::compute::on(*device)
                     .input<std::int32_t>(values.size())
                     .zip_input<std::uint32_t>(previous.size())
                     .branch([](auto input, auto indices) {
                       auto staged =
                           input.map("package-source",
                                     [](auto value) { return value + 100; });
                       return rund::compute::zip(staged, staged.gather(indices))
                           .map("package-indexed",
                                [](auto direct, auto gathered) {
                                  return direct - gathered;
                                })
                           .reduce(rund::compute::Reduce::Sum);
                     })
                     .compile();
  if (!indexed || !ValidateGraph(indexed->graph())) {
    return indexed ? 2 : indexed.exit_code();
  }
  bool indexed_output = false;
  for (const auto &node : indexed->graph().nodes) {
    if (node.operation != rund::compute::graph::Operation::Map ||
        node.accesses.size() != 4u) {
      continue;
    }
    const auto output = std::find_if(
        node.accesses.begin(), node.accesses.end(), [](const auto &access) {
          return access.mode == rund::compute::resource::AccessMode::Write;
        });
    if (output != node.accesses.end()) {
      indexed_output =
          indexed->graph().resources[output->resource - 1u].source == 0u;
    }
  }
  auto indexed_result = indexed->run(values, previous);
  if (!indexed_output || !indexed_result ||
      *indexed_result != std::vector<std::int32_t>{2016}) {
    return indexed_result ? 2 : indexed_result.exit_code();
  }

  return 0;
}

} // namespace package_compute::graph_services

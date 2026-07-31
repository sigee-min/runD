#pragma once

#include <rund/compute.hpp>
#include <rund/compute/async.hpp>
#include <rund/session.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace package_compute {

inline int SessionCompile() {
  rund::Session session{};
  rund::SessionConfig config{};
  config.workers = 2u;
  config.compile = {.workers = 1u, .capacity = 2u};
  const auto opened = session.open(config);
  if (!opened) {
    return opened.exit_code();
  }
  auto device = rund::compute::open(session, rund::compute::Target::cpu());
  const int operation = [&]() -> int {
    if (!device) {
      return device.exit_code();
    }
    if (device->compile().workers != 1u || device->compile().capacity != 2u) {
      return 2;
    }
    auto pending = rund::compute::on(*device)
                       .map<std::int32_t>("session-package-async", 4u,
                                          [](auto value) { return value + 1; })
                       .compile_async();
    if (!pending) {
      return pending.exit_code();
    }
    const auto compiled = pending->get();
    return compiled ? 0 : compiled.exit_code();
  }();
  const auto closed = session.close();
  if (operation != 0 && operation != 2) {
    return operation;
  }
  if (!closed) {
    return closed.exit_code();
  }
  if (operation != 0) {
    return operation;
  }
  auto stopped = rund::compute::on(*device)
                     .map<std::int32_t>("session-package-stopped", 4u,
                                        [](auto value) { return value + 1; })
                     .compile_async();
  return !stopped && stopped.reason() ==
                         rund::compute::Reason::AsyncCompileUnavailable
             ? 0
             : 2;
}

inline int ResourcePlan() {
  using namespace rund::compute::resource;
  const std::array<Resource, 4u> resources{
      Resource{
          .id = 1u, .bytes = 64u, .alias_group = 7u, .alias_offset_bytes = 0u},
      Resource{
          .id = 2u, .bytes = 64u, .alias_group = 7u, .alias_offset_bytes = 32u},
      Resource{
          .id = 3u, .bytes = 32u, .alias_group = 7u, .alias_offset_bytes = 96u},
      Resource{
          .id = 4u, .bytes = 0u, .alias_group = 8u, .alias_offset_bytes = 0u},
  };
  const std::array<Access, 5u> accesses{
      Access{.node = 0u,
             .resource = 1u,
             .mode = AccessMode::Write,
             .offset_bytes = 16u,
             .size_bytes = 32u},
      Access{.node = 1u,
             .resource = 2u,
             .mode = AccessMode::Read,
             .offset_bytes = 0u,
             .size_bytes = 16u},
      Access{.node = 2u,
             .resource = 2u,
             .mode = AccessMode::Write,
             .offset_bytes = 32u,
             .size_bytes = 16u},
      Access{.node = 3u,
             .resource = 1u,
             .mode = AccessMode::Read,
             .offset_bytes = 0u,
             .size_bytes = 8u},
      Access{.node = 4u,
             .resource = 1u,
             .mode = AccessMode::Write,
             .offset_bytes = 16u,
             .size_bytes = 8u},
  };
  const auto plan = analyze(resources, accesses, 5u);
  if (!plan) {
    return plan.exit_code();
  }
  if (plan->lifetimes.size() != resources.size() ||
      plan->lifetimes[0u].first_use != 0u ||
      plan->lifetimes[0u].last_use != 4u ||
      plan->lifetimes[1u].first_use != 1u ||
      plan->lifetimes[1u].last_use != 2u ||
      plan->lifetimes[2u].first_use != NoNode ||
      plan->lifetimes[2u].last_use != NoNode ||
      plan->lifetimes[3u].first_use != NoNode ||
      plan->lifetimes[3u].last_use != NoNode ||
      plan->dependencies !=
          std::vector<Dependency>{{.before_node = 0u, .after_node = 1u},
                                  {.before_node = 0u, .after_node = 4u}} ||
      plan->barriers.size() != 2u) {
    return 2;
  }
  const Barrier &alias = plan->barriers.front();
  if (alias.alias_group != 7u || alias.before_resource != 1u ||
      alias.after_resource != 2u || alias.offset_bytes != 32u ||
      alias.size_bytes != 16u || alias.before_node != 0u ||
      alias.after_node != 1u || alias.before != AccessMode::Write ||
      alias.after != AccessMode::Read) {
    return 2;
  }
  return 0;
}

inline bool Count(const rund::compute::graph::Info &graph,
                  const std::uint32_t id) {
  if (id == 0u || id > graph.resources.size()) {
    return false;
  }
  const auto &value = graph.resources[id - 1u];
  return value.elements == 1u &&
         (value.type == rund::compute::graph::Value::U32 ||
          value.type == rund::compute::graph::Value::U64);
}

inline bool SameStorage(const rund::compute::graph::Resource &left,
                        const rund::compute::graph::Resource &right) {
  return left.type == right.type && left.integer_bits == right.integer_bits &&
         left.fraction_bits == right.fraction_bits &&
         left.rounding == right.rounding && left.overflow == right.overflow &&
         left.approximation == right.approximation &&
         left.elements == right.elements &&
         left.element_bytes == right.element_bytes && left.bytes == right.bytes;
}

inline bool Destructive(const rund::compute::graph::Resource &source,
                        const rund::compute::graph::Resource &target) {
  return target.source == source.id && source.last_use == target.first_use &&
         source.alias_offset_bytes == target.alias_offset_bytes &&
         SameStorage(source, target);
}

inline bool ValidGraph(const rund::compute::graph::Info &graph) {
  using rund::compute::graph::Visibility;
  using rund::compute::resource::AccessMode;
  if (!graph.fingerprint || graph.resources.empty() || graph.nodes.empty() ||
      graph.inputs.empty() || graph.outputs.empty()) {
    return false;
  }
  std::vector<std::uint64_t> extents(graph.resources.size());
  std::vector<bool> groups(graph.resources.size());
  std::uint64_t logical = 0u;
  for (std::size_t index = 0u; index < graph.resources.size(); ++index) {
    const auto &resource = graph.resources[index];
    if (resource.id != index + 1u || resource.element_bytes == 0u ||
        resource.elements > std::numeric_limits<std::uint64_t>::max() /
                                resource.element_bytes ||
        resource.bytes != resource.elements * resource.element_bytes ||
        resource.alias_group == 0u ||
        resource.alias_group > graph.resources.size() ||
        (resource.active != 0u && !Count(graph, resource.active)) ||
        (resource.parent != 0u &&
         (!Count(graph, resource.id) || !Count(graph, resource.parent))) ||
        (resource.source != 0u && (resource.source > graph.resources.size() ||
                                   resource.source == resource.id)) ||
        (resource.first_use == rund::compute::resource::NoNode) !=
            (resource.last_use == rund::compute::resource::NoNode) ||
        (resource.first_use != rund::compute::resource::NoNode &&
         resource.first_use > resource.last_use)) {
      return false;
    }
    std::uint32_t ancestor = resource.parent;
    for (std::size_t depth = 0u; ancestor != 0u; ++depth) {
      if (depth >= graph.resources.size() || ancestor == resource.id) {
        return false;
      }
      ancestor = graph.resources[ancestor - 1u].parent;
    }
    if (resource.visibility != Visibility::Internal) {
      if (resource.alias_group != resource.id ||
          resource.alias_offset_bytes != 0u || resource.requires_reset()) {
        return false;
      }
      continue;
    }
    if (resource.bytes > std::numeric_limits<std::uint64_t>::max() - logical ||
        resource.alias_offset_bytes % 256u != 0u ||
        resource.alias_offset_bytes >
            std::numeric_limits<std::uint64_t>::max() - resource.bytes) {
      return false;
    }
    logical += resource.bytes;
    if (resource.first_use == rund::compute::resource::NoNode) {
      if (resource.alias_group != resource.id ||
          resource.alias_offset_bytes != 0u || resource.requires_reset()) {
        return false;
      }
      continue;
    }
    const auto &representative = graph.resources[resource.alias_group - 1u];
    if (representative.visibility != Visibility::Internal ||
        representative.alias_group != resource.alias_group ||
        representative.first_use == rund::compute::resource::NoNode ||
        representative.reset_node != resource.reset_node) {
      return false;
    }
    const std::size_t group = resource.alias_group - 1u;
    groups[group] = true;
    extents[group] =
        std::max(extents[group], resource.alias_offset_bytes + resource.bytes);
  }
  for (std::size_t left = 0u; left < graph.resources.size(); ++left) {
    for (std::size_t right = left + 1u; right < graph.resources.size();
         ++right) {
      const auto &first = graph.resources[left];
      const auto &second = graph.resources[right];
      if (first.alias_group != second.alias_group) {
        continue;
      }
      if (first.visibility != Visibility::Internal ||
          second.visibility != Visibility::Internal ||
          first.first_use == rund::compute::resource::NoNode ||
          second.first_use == rund::compute::resource::NoNode) {
        return false;
      }
      const bool live_overlap = !(first.last_use < second.first_use ||
                                  second.last_use < first.first_use);
      const bool byte_overlap =
          first.alias_offset_bytes < second.alias_offset_bytes + second.bytes &&
          second.alias_offset_bytes < first.alias_offset_bytes + first.bytes;
      const bool proved_alias =
          Destructive(first, second) || Destructive(second, first);
      if ((live_overlap || first.requires_reset() || second.requires_reset()) &&
          byte_overlap && !proved_alias) {
        return false;
      }
    }
  }
  std::uint64_t physical = 0u;
  std::uint64_t allocations = 0u;
  for (std::size_t group = 0u; group < groups.size(); ++group) {
    if (!groups[group]) {
      continue;
    }
    if (extents[group] > std::numeric_limits<std::uint64_t>::max() - physical) {
      return false;
    }
    physical += extents[group];
    ++allocations;
  }
  std::uint64_t live = 0u;
  for (std::uint32_t node = 0u; node < graph.nodes.size(); ++node) {
    std::uint64_t current = 0u;
    for (const auto &resource : graph.resources) {
      if (resource.visibility != Visibility::Internal ||
          resource.first_use == rund::compute::resource::NoNode ||
          node < resource.first_use || node > resource.last_use) {
        continue;
      }
      if (resource.bytes >
          std::numeric_limits<std::uint64_t>::max() - current) {
        return false;
      }
      current += resource.bytes;
    }
    live = std::max(live, current);
  }
  if (graph.memory.logical_bytes != logical ||
      graph.memory.live_bytes != live ||
      graph.memory.physical_bytes != physical ||
      graph.memory.allocation_count != allocations) {
    return false;
  }
  for (const auto input : graph.inputs) {
    if (input == 0u || input > graph.resources.size() ||
        graph.resources[input - 1u].visibility != Visibility::Input) {
      return false;
    }
  }
  for (const auto output : graph.outputs) {
    if (output == 0u || output > graph.resources.size() ||
        graph.resources[output - 1u].visibility != Visibility::Output) {
      return false;
    }
  }
  std::uint64_t reads = 0u;
  for (const auto &node : graph.nodes) {
    if (node.index >= graph.nodes.size() || node.accesses.empty()) {
      return false;
    }
    for (const auto dependency : node.dependencies) {
      if (dependency >= node.index) {
        return false;
      }
    }
    for (const auto &access : node.accesses) {
      if (access.resource == 0u || access.resource > graph.resources.size()) {
        return false;
      }
      const auto &resource = graph.resources[access.resource - 1u];
      if (access.offset_bytes != 0u || access.size_bytes != resource.bytes ||
          access.element_bytes != resource.element_bytes ||
          access.element_count != resource.elements ||
          access.stride_bytes != resource.element_bytes ||
          resource.first_use > node.index || resource.last_use < node.index) {
        return false;
      }
      if (access.mode == AccessMode::Read) {
        const std::uint64_t bytes = access.element_bytes * access.element_count;
        reads = bytes > std::numeric_limits<std::uint64_t>::max() - reads
                    ? std::numeric_limits<std::uint64_t>::max()
                    : reads + bytes;
      }
    }
  }
  if (graph.read_bytes != reads) {
    return false;
  }
  std::vector<bool> boundary(graph.nodes.size());
  for (const auto &barrier : graph.barriers) {
    if (barrier.alias_group == 0u || barrier.before_resource == 0u ||
        barrier.after_resource == 0u || barrier.size_bytes == 0u ||
        barrier.before_node >= barrier.after_node ||
        barrier.after_node >= graph.nodes.size() ||
        boundary[barrier.after_node] ||
        (barrier.before == AccessMode::Read &&
         barrier.after == AccessMode::Read)) {
      return false;
    }
    boundary[barrier.after_node] = true;
  }
  return true;
}

inline int Services() {
  if (const int resources = ResourcePlan(); resources != 0) {
    return resources;
  }
  if (const int service = SessionCompile(); service != 0) {
    return service;
  }
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
      !ValidGraph(first->graph()) || memory.logical_bytes != 48u ||
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
  if (!bounded || !ValidGraph(bounded->graph())) {
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
  if (!indexed || !ValidGraph(indexed->graph())) {
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

} // namespace package_compute

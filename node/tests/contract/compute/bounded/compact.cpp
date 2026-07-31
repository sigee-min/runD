#include "model.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace rund_node_bounded_contract {

int CheckCompactCapacityBackend(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<std::uint32_t, 0u> empty{};
  auto empty_result = on(rund::node::test_contract::target_for(backend), empty)
                          .compact()
                          .collect();
  if (!empty_result || !empty_result->empty()) {
    return 1;
  }
  auto empty_device = open(rund::node::test_contract::target_for(backend));
  if (!empty_device) {
    return 15;
  }
  auto empty_cache = program_cache(*empty_device, 2u);
  if (!empty_cache) {
    return 16;
  }
  const auto build_empty = [&](const char *const name) {
    return on(*empty_device, *empty_cache)
        .map<std::uint32_t>(name, 0u, [](auto value) { return value; })
        .compact()
        .compile();
  };
  auto empty_first = build_empty("compact-empty-first");
  auto empty_second = build_empty("compact-empty-renamed");
  const auto empty_cache_stats = empty_cache->stats();
  if (!empty_first || !empty_second || !empty_first->fingerprint() ||
      empty_first->fingerprint() != empty_second->fingerprint() ||
      empty_cache_stats.misses != 1u || empty_cache_stats.hits != 1u ||
      empty_cache_stats.ready_entries != 1u) {
    return 17;
  }
  const graph::Info empty_graph = empty_first->graph();
  if (!empty_graph.nodes.empty() || empty_graph.outputs.size() != 2u ||
      empty_graph.resources.empty()) {
    return 18;
  }
  const std::uint32_t value_output = empty_graph.outputs[0u];
  const std::uint32_t count_output = empty_graph.outputs[1u];
  if (value_output == 0u || count_output == 0u ||
      value_output > empty_graph.resources.size() ||
      count_output > empty_graph.resources.size()) {
    return 19;
  }
  const graph::Resource &empty_values =
      empty_graph.resources[value_output - 1u];
  const graph::Resource &empty_count = empty_graph.resources[count_output - 1u];
  if (empty_values.elements != 0u || empty_values.bytes != 0u ||
      empty_count.elements != 1u ||
      empty_count.element_bytes != sizeof(std::uint32_t) ||
      empty_count.bytes != sizeof(std::uint32_t)) {
    return 20;
  }
  auto empty_job = empty_first->resident(empty);
  if (!empty_job || !empty_job->run()) {
    return 21;
  }
  const std::uint64_t empty_transfer_before =
      empty_job->memory().transfer.cumulative;
  auto empty_read = empty_job->read();
  const std::uint64_t empty_transfer_after =
      empty_job->memory().transfer.cumulative;
  const Stats empty_stats = empty_job->stats();
  if (!empty_read || !empty_read->empty() ||
      empty_stats.download_events != 0u ||
      empty_transfer_after != empty_transfer_before ||
      empty_stats.output_hash != EmptyCompactHash()) {
    std::fprintf(
        stderr,
        "empty compact hash/read backend=%u read=%u size=%zu "
        "download_events=%llu transfer=%llu/%llu hash=%llu expected=%llu\n",
        static_cast<unsigned>(backend), empty_read ? 1u : 0u,
        empty_read ? empty_read->size() : 0u,
        static_cast<unsigned long long>(empty_stats.download_events),
        static_cast<unsigned long long>(empty_transfer_before),
        static_cast<unsigned long long>(empty_transfer_after),
        static_cast<unsigned long long>(empty_stats.output_hash),
        static_cast<unsigned long long>(EmptyCompactHash()));
    return 22;
  }

  const std::array<std::uint32_t, 5u> sparse{1u, 0u, 1u, 0u, 0u};
  const std::vector<std::uint32_t> expected{0u, 2u};
  auto exact = on(rund::node::test_contract::target_for(backend), sparse)
                   .compact({.capacity = 2u})
                   .collect();
  auto underfilled = on(rund::node::test_contract::target_for(backend), sparse)
                         .compact({.capacity = 4u})
                         .collect();
  auto default_capacity =
      on(rund::node::test_contract::target_for(backend), sparse)
          .compact()
          .collect();
  if (!exact || !underfilled || !default_capacity || *exact != expected ||
      *underfilled != expected || *default_capacity != expected) {
    if (!exact || !underfilled || !default_capacity) {
      const auto error = !exact         ? exact.error()
                         : !underfilled ? underfilled.error()
                                        : default_capacity.error();
      std::fprintf(stderr, "bounded compact backend=%u: %.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(error.size()), error.data());
    }
    return 2;
  }

  const std::array<std::uint32_t, 5u> none{};
  auto none_program =
      on(rund::node::test_contract::target_for(backend))
          .map<std::uint32_t>("compact-zero-logical", none.size(),
                              [](auto value) { return value; })
          .compact({.capacity = 4u})
          .compile();
  if (!none_program) {
    return 3;
  }
  auto none_job = none_program->resident(none);
  if (!none_job || !none_job->run()) {
    return 4;
  }
  const std::uint64_t none_transfer_before =
      none_job->memory().transfer.cumulative;
  auto none_result = none_job->read();
  const std::uint64_t none_transfer_after =
      none_job->memory().transfer.cumulative;
  if (!none_result || !none_result->empty() ||
      none_transfer_after - none_transfer_before != sizeof(std::uint32_t)) {
    return 5;
  }

  auto downstream =
      on(rund::node::test_contract::target_for(backend), sparse)
          .branch([](auto flags) {
            auto compact = flags.compact({.capacity = 4u});
            return outputs(compact.map("compact-map",
                                       [](auto index) { return index + 10u; }),
                           compact.reduce(Reduce::Sum),
                           compact
                               .map("compact-sort-input",
                                    [](auto index) { return index ^ 2u; })
                               .sort());
          })
          .collect();
  if (!downstream ||
      std::get<0>(*downstream) != std::vector<std::uint32_t>{10u, 12u} ||
      std::get<1>(*downstream) != 2u ||
      std::get<2>(*downstream) != std::vector<std::uint32_t>{0u, 2u}) {
    return 6;
  }

  auto repeated =
      on(rund::node::test_contract::target_for(backend))
          .map<std::uint32_t>("compact-repeated-tail", sparse.size(),
                              [](auto value) { return value; })
          .compact()
          .compile();
  const std::array<std::uint32_t, 5u> full{1u, 1u, 1u, 1u, 1u};
  if (!repeated) {
    return 7;
  }
  auto repeated_job = repeated->resident(full);
  if (!repeated_job || !repeated_job->run()) {
    return 8;
  }
  auto full_result = repeated_job->read();
  if (!full_result ||
      *full_result != std::vector<std::uint32_t>{0u, 1u, 2u, 3u, 4u} ||
      !repeated_job->write(sparse) || !repeated_job->run()) {
    return 9;
  }
  const std::uint64_t transfer_before =
      repeated_job->memory().transfer.cumulative;
  auto sparse_result = repeated_job->read();
  const std::uint64_t transfer_after =
      repeated_job->memory().transfer.cumulative;
  constexpr std::uint64_t expected_read_bytes = sizeof(std::uint32_t) * 3u;
  if (!sparse_result || *sparse_result != expected ||
      transfer_after - transfer_before != expected_read_bytes) {
    return 10;
  }

  const std::array<std::uint32_t, 5u> overflow{1u, 0u, 1u, 1u, 0u};
  auto rejected = on(rund::node::test_contract::target_for(backend), overflow)
                      .compact({.capacity = 2u})
                      .collect();
  if (rejected || rejected.error() != "compute_compact_capacity_insufficient") {
    return 11;
  }
  auto recovery =
      on(rund::node::test_contract::target_for(backend))
          .map<std::uint32_t>("compact-capacity-recovery", overflow.size(),
                              [](auto value) { return value; })
          .compact({.capacity = 2u})
          .compile();
  if (!recovery) {
    return 12;
  }
  auto recovery_job = recovery->resident(sparse);
  if (!recovery_job || !recovery_job->run()) {
    return 13;
  }
  auto initial = recovery_job->read();
  if (!initial || *initial != expected || !recovery_job->write(overflow)) {
    return 14;
  }
  const Status overflow_status = recovery_job->run();
  auto stale = recovery_job->read();
  if (overflow_status ||
      overflow_status.error() != "compute_compact_capacity_insufficient" ||
      stale || stale.error() != "compute_compact_capacity_insufficient" ||
      !recovery_job->write(sparse) || !recovery_job->run()) {
    return 15;
  }
  auto recovered = recovery_job->read();
  if (!recovered || *recovered != expected) {
    return 16;
  }
  auto count_only =
      on(rund::node::test_contract::target_for(backend))
          .map<std::uint32_t>("compact-count-overflow", overflow.size(),
                              [](auto value) { return value; })
          .compact({.capacity = 2u})
          .count()
          .compile();
  if (!count_only) {
    return 17;
  }
  auto count_job = count_only->resident(overflow);
  if (!count_job) {
    return 18;
  }
  const Status count_status = count_job->run();
  return !count_status &&
                 count_status.error() == "compute_compact_capacity_insufficient"
             ? 0
             : 19;
}

} // namespace rund_node_bounded_contract

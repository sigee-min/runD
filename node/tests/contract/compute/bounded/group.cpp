#include "local.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_bounded_contract {

int CheckGroupRewriteBackend(const rund::compute::Backend backend,
                             std::uint64_t &reference_graph,
                             std::uint64_t &reference_many,
                             std::uint64_t &reference_fewer) {
  using namespace rund::compute;
  const std::array<std::uint32_t, 5u> many{4u, 3u, 2u, 1u, 0u};
  const std::array<std::uint32_t, 5u> fewer{7u, 7u, 7u, 7u, 7u};
  const auto many_expected =
      std::tuple{std::vector<std::uint32_t>{0u, 1u, 2u, 3u, 4u},
                 std::vector<std::uint32_t>{1u, 1u, 1u, 1u, 1u}};
  const auto fewer_expected = std::tuple{std::vector<std::uint32_t>{7u},
                                         std::vector<std::uint32_t>{5u}};
  auto program = on(rund::node::test_contract::target_for(backend))
                     .map<std::uint32_t>("bounded-group-rewrite", many.size(),
                                         [](auto value) { return value; })
                     .group_by([](auto value) { return value; })
                     .aggregate([](auto group) {
                       return outputs(group.key(), group.count());
                     })
                     .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(many);
  if (!job || !job->run()) {
    return 2;
  }
  auto many_result = job->read_all();
  const Stats many_stats = job->stats();
  if (!many_result || *many_result != many_expected) {
    std::fprintf(stderr, "bounded group initial backend=%u result=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(static_cast<bool>(many_result)));
    return 3;
  }
  const Status write = job->write(fewer);
  if (!write) {
    std::fprintf(stderr, "bounded group write backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(write.error().size()), write.error().data());
    return 3;
  }
  const Status rerun = job->run();
  if (!rerun) {
    std::fprintf(stderr, "bounded group rerun backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(rerun.error().size()), rerun.error().data());
    return 3;
  }
  auto fewer_result = job->read_all();
  const Stats fewer_stats = job->stats();
  if (!fewer_result || *fewer_result != fewer_expected ||
      many_stats.graph_hash == 0u || many_stats.output_hash == 0u ||
      fewer_stats.output_hash == 0u ||
      many_stats.graph_hash != fewer_stats.graph_hash) {
    return 4;
  }
  if (reference_graph == 0u) {
    reference_graph = many_stats.graph_hash;
    reference_many = many_stats.output_hash;
    reference_fewer = fewer_stats.output_hash;
    return 0;
  }
  return many_stats.graph_hash == reference_graph &&
                 many_stats.output_hash == reference_many &&
                 fewer_stats.output_hash == reference_fewer
             ? 0
             : 5;
}

} // namespace rund_node_bounded_contract

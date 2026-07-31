#include "model.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace rund_node_backend_contract {

[[nodiscard]] int CheckMapParity() {
  constexpr std::size_t kCount = 4097u;
  std::vector<std::int32_t> input(kCount);
  std::vector<std::int32_t> expected(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    input[index] = static_cast<std::int32_t>(index % 97u);
    expected[index] = input[index] * 2 + 5;
  }
  std::uint64_t graph_hash = 0;
  std::uint64_t output_hash = 0;

  for (const auto backend :
       rund::node::test_contract::selected_compute_backends()) {
    const auto target =
        backend == rund::compute::Backend::Cpu
            ? rund::compute::on(rund::compute::Target::cpu(1u))
            : rund::compute::on(rund::node::test_contract::target_for(backend));
    auto program =
        target
            .map<std::int32_t>("backend-parity", input.size(),
                               [](auto value) { return value * 2 + 5; })
            .compile();
    if (!program || !UsesBackend(*program, backend)) {
      std::fprintf(stderr, "compute backend %u compile failed: %.*s\n",
                   static_cast<unsigned>(backend),
                   program ? 0 : static_cast<int>(program.error().size()),
                   program ? "" : program.error().data());
      return 3;
    }
    auto job = program->resident(std::span<const std::int32_t>{input});
    if (!job || !job->run()) {
      std::fprintf(stderr, "compute backend %u run failed: %.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(job.error().size()), job.error().data());
      return 5;
    }
    auto output = job->read();
    if (!output || *output != expected) {
      return 6;
    }
    const auto stats = job->stats();
    if (stats.graph_hash == 0 || stats.output_hash == 0) {
      return 7;
    }
    if (stats.dispatches == 0u ||
        (backend != rund::compute::Backend::Cpu && stats.dispatches != 1u)) {
      return 9;
    }
    if (backend != rund::compute::Backend::Cpu) {
      auto repeat_program =
          rund::compute::on(rund::node::test_contract::target_for(backend))
              .map<std::int32_t>("backend-parity", input.size(),
                                 [](auto value) { return value * 2 + 5; })
              .compile();
      if (!repeat_program) {
        return 11;
      }
      auto repeat_job =
          repeat_program->resident(std::span<const std::int32_t>{input});
      if (!repeat_job || !repeat_job->run()) {
        return 12;
      }
      auto repeat_output = repeat_job->read();
      if (!repeat_output || *repeat_output != expected) {
        return 13;
      }
      const auto repeat_stats = repeat_job->stats();
      if (repeat_stats.dispatches != stats.dispatches ||
          repeat_stats.graph_hash != stats.graph_hash ||
          repeat_stats.output_hash != stats.output_hash) {
        return 14;
      }
    }
    if (graph_hash == 0) {
      graph_hash = stats.graph_hash;
      output_hash = stats.output_hash;
    } else if (stats.graph_hash != graph_hash ||
               stats.output_hash != output_hash) {
      return 8;
    }
  }
  return 0;
}

} // namespace rund_node_backend_contract

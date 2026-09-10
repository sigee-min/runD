#include "local.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace rund_node_test_flow_primitives {

[[nodiscard]] int CheckBoundedCollectiveRepeat() {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 4u> input{3u, 0u, 1u, 2u};
  auto program =
      on(Target::cpu(2u))
          .input<std::uint32_t>(input.size())
          .branch([](auto values) {
            auto active = values.filter([](auto value) { return value != 0u; });
            return active.template unroll<2u>(
                [](auto work) { return work.sort(); },
                [](auto value) { return value == std::uint32_t{99}; });
          })
          .compile();
  if (!program) {
    std::fprintf(stderr, "bounded collective repeat reason=%.*s\n",
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job)
    return 2;
  const Status ran = job->run();
  if (!ran)
    return 3;
  const Stats stats = job->stats();
  auto output = job->read();
  if (!output || *output != std::vector<std::uint32_t>{1u, 2u, 3u} ||
      stats.backend != Backend::Cpu || stats.control.iteration_count != 2u ||
      stats.control.skipped_iteration_count != 0u) {
    return 4;
  }

  auto scan_program =
      on(rund::compute::Target::cpu(2u))
          .input<std::uint32_t>(input.size())
          .branch([](auto values) {
            auto active = values.filter([](auto value) { return value != 0u; });
            return active.template unroll<2u>(
                [](auto work) { return work.scan(Scan::InclusiveSum); },
                [](auto value) { return value == std::uint32_t{99}; });
          })
          .compile();
  if (!scan_program)
    return 5;
  auto scan_job = scan_program->resident(input);
  if (!scan_job || !scan_job->run())
    return 6;
  const Stats scan_stats = scan_job->stats();
  auto scan_output = scan_job->read();
  return scan_output &&
                 *scan_output == std::vector<std::uint32_t>{3u, 7u, 13u} &&
                 scan_stats.control.iteration_count == 2u &&
                 scan_stats.control.skipped_iteration_count == 0u
             ? 0
             : 7;
}

[[nodiscard]] int CheckBoundedBoundaryConsumers() {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> values{5, 7, 11, 13};
  constexpr std::array<std::uint32_t, 1u> overflow_count{5u};
  constexpr std::array<std::uint32_t, 1u> full_count{values.size()};

  auto scan =
      on(Target::cpu(2u))
          .input<Bounded<std::int32_t>>(values.size())
          .branch([](auto input) { return input.scan(Scan::InclusiveSum); })
          .compile();
  if (!scan)
    return 1;
  const auto scan_fingerprint = scan->graph().fingerprint;
  const std::size_t scan_nodes = scan->graph().nodes.size();
  const std::size_t scan_resources = scan->graph().resources.size();
  auto rejected_scan = scan->run(values, overflow_count);
  if (rejected_scan || rejected_scan.error() != "compute_workset_overflow" ||
      scan->graph().fingerprint != scan_fingerprint ||
      scan->graph().nodes.size() != scan_nodes ||
      scan->graph().resources.size() != scan_resources) {
    return 2;
  }
  auto scanned = scan->run(values, full_count);
  if (!scanned || *scanned != std::vector<std::int32_t>{5, 12, 23, 36}) {
    return 3;
  }
  auto scan_job = scan->resident(values, full_count);
  if (!scan_job || !scan_job->run())
    return 4;
  const Status rejected_scan_write = scan_job->write(values, overflow_count);
  auto retained_scan = scan_job->read();
  if (rejected_scan_write ||
      rejected_scan_write.error() != "compute_workset_overflow" ||
      !retained_scan ||
      *retained_scan != std::vector<std::int32_t>{5, 12, 23, 36} ||
      scan->graph().fingerprint != scan_fingerprint ||
      scan->graph().nodes.size() != scan_nodes ||
      scan->graph().resources.size() != scan_resources ||
      !scan_job->write(values, full_count) || !scan_job->run()) {
    return 5;
  }

  constexpr std::array<std::uint32_t, 4u> targets{0u, 1u, 0u, 1u};
  auto scatter_reduce =
      on(Target::cpu(2u))
          .input<Bounded<std::uint32_t>>(targets.size())
          .branch([](auto input) {
            return input.scatter_reduce(input, 2u, Reduce::Sum);
          })
          .compile();
  if (!scatter_reduce)
    return 6;
  const auto scatter_fingerprint = scatter_reduce->graph().fingerprint;
  const std::size_t scatter_nodes = scatter_reduce->graph().nodes.size();
  const std::size_t scatter_resources =
      scatter_reduce->graph().resources.size();
  auto rejected_scatter = scatter_reduce->run(targets, overflow_count);
  if (rejected_scatter ||
      rejected_scatter.error() != "compute_workset_overflow" ||
      scatter_reduce->graph().fingerprint != scatter_fingerprint ||
      scatter_reduce->graph().nodes.size() != scatter_nodes ||
      scatter_reduce->graph().resources.size() != scatter_resources) {
    return 7;
  }
  auto scattered = scatter_reduce->run(targets, full_count);
  return scattered && *scattered == std::vector<std::uint32_t>{0u, 2u} ? 0 : 8;
}

} // namespace rund_node_test_flow_primitives

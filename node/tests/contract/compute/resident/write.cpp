#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include "../../target/selection.hpp"

#include "../../../../src/compute/job/state.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

template <class Job>
concept AcceptsWrongWrite =
    requires(Job &job, std::array<std::uint32_t, 4> input) {
      job.write(std::span<const std::uint32_t>{input});
    };

[[nodiscard]] bool CheckStatusRecovery(const rund::compute::Target target,
                                       const rund::compute::Backend backend) {
  using rund::compute::Reduce;
  using rund::compute::Scan;
  constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
  const std::array<std::uint32_t, 4u> overflow{maximum, 1u, 0u, 0u};
  const std::array<std::uint32_t, 4u> valid{1u, 2u, 3u, 4u};

  auto scan = rund::compute::on(target)
                  .map<std::uint32_t>("status-scan", overflow.size(),
                                      [](auto value) { return value; })
                  .scan(Scan::InclusiveSum)
                  .compile();
  if (!scan) {
    return false;
  }
  auto scan_job = scan->resident(overflow);
  if (!scan_job) {
    return false;
  }
  const auto scan_failure = scan_job->run();
  if (scan_failure || scan_failure.error() != "compute_scan_sum_overflow" ||
      !scan_job->write(valid) || !scan_job->run()) {
    std::fprintf(stderr, "status recovery scan backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(scan_failure.error().size()),
                 scan_failure.error().data());
    return false;
  }
  auto scanned = scan_job->read();
  if (!scanned || *scanned != std::vector<std::uint32_t>{1u, 3u, 6u, 10u}) {
    return false;
  }

  auto reduce = rund::compute::on(target)
                    .map<std::uint32_t>("status-reduce", overflow.size(),
                                        [](auto value) { return value; })
                    .reduce(Reduce::Sum)
                    .compile();
  if (!reduce) {
    return false;
  }
  auto reduce_job = reduce->resident(overflow);
  if (!reduce_job) {
    return false;
  }
  const auto reduce_failure = reduce_job->run();
  if (reduce_failure ||
      reduce_failure.error() != "compute_reduce_sum_overflow" ||
      !reduce_job->write(valid) || !reduce_job->run()) {
    std::fprintf(stderr, "status recovery reduce backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(reduce_failure.error().size()),
                 reduce_failure.error().data());
    return false;
  }
  auto reduced = reduce_job->read();
  if (!reduced || *reduced != std::vector<std::uint32_t>{10u}) {
    return false;
  }

  const std::array<std::uint32_t, 2u> invalid_indices{4u, 0u};
  const std::array<std::uint32_t, 2u> valid_indices{1u, 3u};
  auto gather = rund::compute::on(target)
                    .map<std::uint32_t>("status-gather", valid.size(),
                                        [](auto value) { return value; })
                    .gather(valid_indices.size())
                    .compile();
  if (!gather) {
    return false;
  }
  auto gather_job = gather->resident(valid, invalid_indices);
  if (!gather_job) {
    return false;
  }
  const auto gather_failure = gather_job->run();
  if (gather_failure ||
      gather_failure.error() != "compute_gather_index_out_of_range" ||
      !gather_job->write(valid, valid_indices) || !gather_job->run()) {
    std::fprintf(stderr, "status recovery gather backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(gather_failure.error().size()),
                 gather_failure.error().data());
    return false;
  }
  auto gathered = gather_job->read();
  return gathered && *gathered == std::vector<std::uint32_t>{2u, 4u};
}

int RunComputeResidentWriteContract() {
  using Job = rund::compute::Job<std::int32_t(std::int32_t)>;
  static_assert(!AcceptsWrongWrite<Job>);

  const std::array<std::int32_t, 4> first{1, 2, 3, 4};
  const std::array<std::int32_t, 4> second{5, 6, 7, 8};
  auto program = rund::compute::on(rund::compute::Target::cpu(2u))
                     .map<std::int32_t>("write", first.size(),
                                        [](auto value) { return value * 2; })
                     .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(first);
  if (!job || job->memory().transfer.current != 0u ||
      job->memory().transfer.cumulative !=
          first.size() * sizeof(std::int32_t) ||
      !job->run()) {
    return 2;
  }
  auto initial = job->read();
  const auto initial_stats = job->stats();
  if (!initial || *initial != std::vector<std::int32_t>{2, 4, 6, 8}) {
    return 3;
  }

  if (!job->write(second)) {
    return 4;
  }
  const auto write = job->write_stats();
  if (write.bytes != second.size() * sizeof(std::int32_t) ||
      write.copies != 1u || write.uploads != 0u) {
    return 5;
  }
  auto stale = job->read();
  if (stale || stale.error() != "compute_resident_not_run") {
    return 6;
  }

  if (!job->run()) {
    return 7;
  }
  const auto warm = job->stats();
  auto updated = job->read();
  if (!updated || *updated != std::vector<std::int32_t>{10, 12, 14, 16} ||
      warm.graph_hash != initial_stats.graph_hash ||
      warm.output_hash == initial_stats.output_hash ||
      warm.pipeline_compiles != 0u || warm.buffer_allocations != 0u ||
      warm.uploaded_bytes != 0u || warm.download_events != 0u) {
    return 8;
  }

  const std::span<const std::int32_t> short_input{second.data(), 3u};
  const auto rejected = job->write(short_input);
  if (rejected || rejected.error() != "compute_shape_mismatch") {
    return 9;
  }

  const std::array<std::uint32_t, 4> values{3u, 1u, 4u, 2u};
  const std::array<std::uint32_t, 2> indices{3u, 0u};
  const std::array<std::uint32_t, 4> next_values{8u, 7u, 6u, 5u};
  const std::array<std::uint32_t, 2> next_indices{1u, 2u};
  auto gathered = rund::compute::on(rund::compute::Target::cpu(2u))
                      .map<std::uint32_t>("copy", values.size(),
                                          [](auto value) { return value; })
                      .gather(indices.size())
                      .compile();
  if (!gathered) {
    return 10;
  }
  auto gathered_job = gathered->resident(values, indices);
  if (!gathered_job || gathered_job->memory().transfer.current != 0u ||
      gathered_job->memory().transfer.cumulative !=
          values.size() * sizeof(std::uint32_t) +
              indices.size() * sizeof(std::uint32_t) ||
      !gathered_job->run()) {
    return 11;
  }
  auto first_gather = gathered_job->read();
  if (!first_gather || *first_gather != std::vector<std::uint32_t>{2u, 3u}) {
    return 12;
  }
  const auto graph = gathered_job->stats().graph_hash;

  const auto state = rund::compute::detail::JobAccess::state(*gathered_job);
  auto second_write_buffer = state->write_inputs[1];
  state->write_inputs[1].reset();
  const auto transfer_failed = gathered_job->write(next_values, next_indices);
  state->write_inputs[1] = std::move(second_write_buffer);
  if (transfer_failed) {
    return 15;
  }
  auto preserved = gathered_job->read();
  if (!preserved || *preserved != std::vector<std::uint32_t>{2u, 3u} ||
      !gathered_job->run()) {
    return 16;
  }
  auto rerun = gathered_job->read();
  if (!rerun || *rerun != std::vector<std::uint32_t>{2u, 3u}) {
    return 17;
  }

  if (!gathered_job->write(next_values, next_indices) || !gathered_job->run()) {
    return 13;
  }
  auto next_gather = gathered_job->read();
  if (!next_gather || *next_gather != std::vector<std::uint32_t>{7u, 6u} ||
      gathered_job->stats().graph_hash != graph ||
      gathered_job->write_stats().copies != 2u) {
    return 14;
  }

  for (const auto backend :
       rund::node::test_contract::selected_accelerators()) {
    const auto target = rund::node::test_contract::target_for(backend);
    auto device = rund::compute::open(target);
    if (!device) {
      return 18;
    }
    auto accelerated =
        rund::compute::on(target)
            .map<std::int32_t>("write", first.size(),
                               [](auto value) { return value * 2; })
            .compile();
    if (!accelerated) {
      return 19;
    }
    auto accelerated_job = accelerated->resident(first);
    if (!accelerated_job) {
      std::fprintf(stderr, "resident write backend=%u prepare reason=%.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(accelerated_job.error().size()),
                   accelerated_job.error().data());
      return 20;
    }
    const auto prepared_memory = accelerated_job->memory();
    if (prepared_memory.transfer.current != 0u ||
        prepared_memory.transfer.cumulative !=
            first.size() * sizeof(std::int32_t)) {
      std::fprintf(
          stderr,
          "resident write backend=%u transfer current=%llu "
          "cumulative=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(prepared_memory.transfer.current),
          static_cast<unsigned long long>(prepared_memory.transfer.cumulative));
      return 20;
    }
    const auto initial_run = accelerated_job->run();
    if (!initial_run) {
      std::fprintf(stderr, "resident write backend=%u run reason=%.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(initial_run.error().size()),
                   initial_run.error().data());
      return 20;
    }
    auto accelerated_initial = accelerated_job->read();
    if (!accelerated_initial ||
        *accelerated_initial != std::vector<std::int32_t>{2, 4, 6, 8}) {
      return 30;
    }
    if (!accelerated_job->write(second)) {
      return 20;
    }
    const auto transfer = accelerated_job->write_stats();
    if (transfer.bytes != second.size() * sizeof(std::int32_t) ||
        transfer.uploads != 1u || transfer.copies != 0u ||
        !accelerated_job->run()) {
      return 21;
    }
    const auto warm_stats = accelerated_job->stats();
    auto accelerated_output = accelerated_job->read();
    if (!accelerated_output ||
        *accelerated_output != std::vector<std::int32_t>{10, 12, 14, 16} ||
        warm_stats.pipeline_compiles != 0u ||
        warm_stats.buffer_allocations != 0u ||
        warm_stats.download_events != 0u) {
      return 22;
    }

    const std::array<std::uint32_t, 4> scan_first{1u, 2u, 3u, 4u};
    const std::array<std::uint32_t, 4> scan_second{5u, 4u, 3u, 2u};
    auto scan_program =
        rund::compute::on(target)
            .map<std::uint32_t>("isolated-scan", scan_first.size(),
                                [](auto value) { return value + 1u; })
            .scan(rund::compute::Scan::InclusiveSum)
            .compile();
    if (!scan_program) {
      return 23;
    }
    auto first_scan_job = scan_program->resident(scan_first);
    auto second_scan_job = scan_program->resident(scan_second);
    if (!first_scan_job || !second_scan_job) {
      return 24;
    }
    const auto first_scan_state =
        rund::compute::detail::JobAccess::state(*first_scan_job);
    const auto second_scan_state =
        rund::compute::detail::JobAccess::state(*second_scan_job);
    bool has_internal = false;
    if (first_scan_state->graph_buffers.size() != 1u ||
        first_scan_state->graph_buffers.size() !=
            second_scan_state->graph_buffers.size()) {
      return 25;
    }
    for (std::size_t index = 0u; index < first_scan_state->graph_buffers.size();
         ++index) {
      const auto &left = first_scan_state->graph_buffers[index];
      const auto &right = second_scan_state->graph_buffers[index];
      if ((left == nullptr) != (right == nullptr)) {
        return 26;
      }
      if (left != nullptr) {
        has_internal = true;
        if (left == right) {
          return 27;
        }
      }
    }
    if (!has_internal || !first_scan_job->run() || !second_scan_job->run()) {
      return 28;
    }
    const auto first_scan_output = first_scan_job->read();
    const auto second_scan_output = second_scan_job->read();
    if (!first_scan_output || !second_scan_output ||
        *first_scan_output != std::vector<std::uint32_t>{2u, 5u, 9u, 14u} ||
        *second_scan_output != std::vector<std::uint32_t>{6u, 11u, 15u, 18u}) {
      return 29;
    }
    if (!CheckStatusRecovery(target, backend)) {
      return 31;
    }
  }
  return 0;
}

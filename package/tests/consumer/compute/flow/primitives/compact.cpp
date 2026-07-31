#include "model.hpp"

namespace package_compute {

int Compact() {
  auto compacted = rund::compute::on(rund::compute::Target::cpu(), flags)
                       .compact({.capacity = 4u})
                       .collect();
  if (!compacted) {
    return compacted.exit_code();
  }
  if (*compacted != std::vector<std::uint32_t>{0u, 2u}) {
    return FlowMismatch(__LINE__);
  }
  auto mapped_compact =
      rund::compute::on(rund::compute::Target::cpu(), flags)
          .compact({.capacity = 4u})
          .map("consumer-compact-map", [](auto index) { return index + 10u; })
          .collect();
  if (!mapped_compact) {
    return mapped_compact.exit_code();
  }
  if (*mapped_compact != std::vector<std::uint32_t>{10u, 12u}) {
    return FlowMismatch(__LINE__);
  }
  const std::array<std::uint32_t, 0u> empty_flags{};
  auto empty_compact =
      rund::compute::on(rund::compute::Target::cpu(), empty_flags)
          .compact()
          .collect();
  if (!empty_compact) {
    return empty_compact.exit_code();
  }
  if (!empty_compact->empty()) {
    return FlowMismatch(__LINE__);
  }
  const std::array<std::uint32_t, 5u> compact_overflow{1u, 0u, 1u, 1u, 0u};
  auto rejected_compact =
      rund::compute::on(rund::compute::Target::cpu(), compact_overflow)
          .compact({.capacity = 2u})
          .collect();
  if (rejected_compact ||
      rejected_compact.code() != rund::compute::Code::Capacity ||
      rejected_compact.error() != "compute_compact_capacity_insufficient") {
    return FlowMismatch(__LINE__);
  }
  const std::array<std::uint32_t, 4u> compact_full{1u, 1u, 1u, 1u};
  auto compact_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::uint32_t>("installed-compact-resident", flags.size(),
                              [](auto value) { return value; })
          .compact({.capacity = 4u})
          .compile();
  static_assert(std::same_as<decltype(compact_program),
                             rund::compute::Result<InstalledCompactProgram>>);
  if (!compact_program) {
    return compact_program.exit_code();
  }
  auto compact_job = compact_program->resident(compact_full);
  if (!compact_job) {
    return compact_job.exit_code();
  }
  const auto compact_full_run = compact_job->run();
  if (!compact_full_run) {
    return compact_full_run.exit_code();
  }
  auto compact_full_result = compact_job->read();
  if (!compact_full_result) {
    return compact_full_result.exit_code();
  }
  if (*compact_full_result != std::vector<std::uint32_t>{0u, 1u, 2u, 3u}) {
    return FlowMismatch(__LINE__);
  }
  const auto compact_write = compact_job->write(flags);
  if (!compact_write) {
    return compact_write.exit_code();
  }
  const auto compact_sparse_run = compact_job->run();
  if (!compact_sparse_run) {
    return compact_sparse_run.exit_code();
  }
  const std::uint64_t compact_transfer_before =
      compact_job->memory().transfer.cumulative;
  auto compact_sparse_result = compact_job->read();
  const std::uint64_t compact_transfer_after =
      compact_job->memory().transfer.cumulative;
  if (!compact_sparse_result) {
    return compact_sparse_result.exit_code();
  }
  if (*compact_sparse_result != std::vector<std::uint32_t>{0u, 2u} ||
      compact_transfer_after - compact_transfer_before !=
          sizeof(std::uint32_t) * 3u) {
    return FlowMismatch(__LINE__);
  }
  return 0;
}

} // namespace package_compute

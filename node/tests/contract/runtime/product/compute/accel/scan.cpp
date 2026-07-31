#include "local.hpp"

#include "../../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <vector>

namespace rund::node::test_contract {

int CheckComputeAccelScanConcurrency(::rund::Session &session,
                                     const compute::Target target) {
  const std::array<std::uint32_t, 8> first{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
  const std::array<std::uint32_t, 8> second{8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u};
  auto program =
      compute::on(target)
          .map<std::uint32_t>("node-host-concurrent-scan", first.size(),
                              [](auto value) { return value + 1u; })
          .scan(compute::Scan::InclusiveSum)
          .compile();
  if (!program) {
    return 4;
  }
  auto first_job = program->resident(first);
  auto second_job = program->resident(second);
  if (!first_job || !second_job) {
    const auto first_reason =
        first_job ? std::string_view{"ok"} : first_job.error();
    const auto second_reason =
        second_job ? std::string_view{"ok"} : second_job.error();
    std::fprintf(stderr, "scan resident first=%.*s second=%.*s\n",
                 static_cast<int>(first_reason.size()), first_reason.data(),
                 static_cast<int>(second_reason.size()), second_reason.data());
    return 5;
  }
  auto first_task = session.compute(*first_job).submit();
  auto second_task = session.compute(*second_job).submit();
  if (!first_task.wait() || !second_task.wait()) {
    return 6;
  }
  const auto first_output = first_job->read();
  const auto second_output = second_job->read();
  if (!first_output || !second_output ||
      *first_output !=
          std::vector<std::uint32_t>{2u, 5u, 9u, 14u, 20u, 27u, 35u, 44u} ||
      *second_output !=
          std::vector<std::uint32_t>{9u, 17u, 24u, 30u, 35u, 39u, 42u, 44u}) {
    return 7;
  }

  constexpr std::string_view kOverflow = "compute_scan_sum_overflow";
  const std::array<std::uint32_t, 2u> overflow{
      std::numeric_limits<std::uint32_t>::max(), 1u};
  const std::array<std::uint32_t, 2u> valid{1u, 2u};
  auto recovery_program =
      compute::on(target)
          .map<std::uint32_t>("node-host-scan-recovery", overflow.size(),
                              [](auto value) { return value; })
          .scan(compute::Scan::InclusiveSum)
          .compile();
  if (!recovery_program) {
    return 8;
  }
  auto recovery = recovery_program->resident(overflow);
  if (!recovery) {
    return 9;
  }

  const compute::Status sync_overflow = recovery->run();
  if (sync_overflow || sync_overflow.error() != kOverflow) {
    return 10;
  }
  if (!recovery->write(valid) || !recovery->run()) {
    return 11;
  }
  auto recovered = recovery->read();
  if (!recovered || *recovered != std::vector<std::uint32_t>{1u, 3u}) {
    return 12;
  }

  if (!recovery->write(overflow)) {
    return 13;
  }
  const auto async_overflow = session.compute(*recovery).submit().wait();
  if (async_overflow || async_overflow.error() != kOverflow) {
    return 14;
  }
  if (!recovery->write(valid) || !session.compute(*recovery).submit().wait()) {
    return 15;
  }
  recovered = recovery->read();
  if (!recovered || *recovered != std::vector<std::uint32_t>{1u, 3u}) {
    return 16;
  }

  auto scan_only_program =
      compute::on(target)
          .input<std::uint32_t>(first.size())
          .branch([](auto values) {
            return values.scan(compute::Scan::InclusiveSum);
          })
          .compile();
  if (!scan_only_program) {
    return 17;
  }
  auto scan_only_first = scan_only_program->resident(first);
  auto scan_only_second = scan_only_program->resident(second);
  if (!scan_only_first || !scan_only_second) {
    return 18;
  }

  auto scan_only_first_task = session.compute(*scan_only_first).submit();
  auto scan_only_second_task = session.compute(*scan_only_second).submit();
  if (!scan_only_first_task.wait() || !scan_only_second_task.wait()) {
    return 19;
  }
  auto scan_only_first_output = scan_only_first->read();
  auto scan_only_second_output = scan_only_second->read();
  if (!scan_only_first_output || !scan_only_second_output ||
      *scan_only_first_output !=
          std::vector<std::uint32_t>{1u, 3u, 6u, 10u, 15u, 21u, 28u, 36u} ||
      *scan_only_second_output !=
          std::vector<std::uint32_t>{8u, 15u, 21u, 26u, 30u, 33u, 35u, 36u}) {
    return 20;
  }

  if (!scan_only_first->write(second) || !scan_only_second->write(first)) {
    return 21;
  }
  scan_only_first_task = session.compute(*scan_only_first).submit();
  scan_only_second_task = session.compute(*scan_only_second).submit();
  if (!scan_only_first_task.wait() || !scan_only_second_task.wait()) {
    return 22;
  }
  scan_only_first_output = scan_only_first->read();
  scan_only_second_output = scan_only_second->read();
  if (!scan_only_first_output || !scan_only_second_output ||
      *scan_only_first_output !=
          std::vector<std::uint32_t>{8u, 15u, 21u, 26u, 30u, 33u, 35u, 36u} ||
      *scan_only_second_output !=
          std::vector<std::uint32_t>{1u, 3u, 6u, 10u, 15u, 21u, 28u, 36u}) {
    return 23;
  }
  return 0;
}

} // namespace rund::node::test_contract

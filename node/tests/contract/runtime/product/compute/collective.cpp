#include "../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace rund::node::test_contract {

int CheckComputeCollectives(::rund::Session &session) {
  constexpr std::size_t count = 64u * 1024u + 17u;
  std::vector<std::uint32_t> input(count);
  std::uint32_t sum = 0u;
  for (std::size_t index = 0u; index < count; ++index) {
    input[index] = static_cast<std::uint32_t>(index % 7u);
    sum += input[index];
  }
  auto scan_program = compute::on(compute::Target::cpu(2u))
                          .map<std::uint32_t>("node-host-scan", count,
                                              [](auto value) { return value; })
                          .scan(compute::Scan::InclusiveSum)
                          .compile();
  if (!scan_program) {
    return 2;
  }
  auto scan_job = scan_program->resident(std::span<const std::uint32_t>{input});
  if (!scan_job || !session.compute(*scan_job).submit().wait()) {
    return 3;
  }
  auto scan_output = scan_job->read();
  if (!scan_output || scan_output->size() != count) {
    return 4;
  }
  std::uint32_t prefix = 0u;
  for (std::size_t index = 0u; index < count; ++index) {
    prefix += input[index];
    if ((*scan_output)[index] != prefix) {
      return 5;
    }
  }

  auto reduce_program =
      compute::on(compute::Target::cpu(2u))
          .map<std::uint32_t>("node-host-reduce", count,
                              [](auto value) { return value; })
          .reduce(compute::Reduce::Sum)
          .compile();
  if (!reduce_program) {
    return 6;
  }
  auto reduce_job =
      reduce_program->resident(std::span<const std::uint32_t>{input});
  if (!reduce_job || !session.compute(*reduce_job).submit().wait()) {
    return 7;
  }
  auto reduce_output = reduce_job->read();
  if (!reduce_output || reduce_output->size() != 1u ||
      (*reduce_output)[0] != sum) {
    return 8;
  }
  const compute::Stats scan_stats = scan_job->stats();
  const compute::Stats reduce_stats = reduce_job->stats();
  return scan_stats.participating_workers >= 2u &&
                 reduce_stats.participating_workers >= 2u
             ? 0
             : 9;
}

} // namespace rund::node::test_contract

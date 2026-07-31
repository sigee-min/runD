#include "../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <cstdint>
#include <vector>

namespace rund::node::test_contract {

int CheckComputeProgramConcurrency(::rund::Session &session) {
  constexpr std::size_t count = 1u << 18u;
  std::vector<std::int32_t> first(count, 7);
  std::vector<std::int32_t> second(count, -11);
  auto program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-concurrent", count,
                             [](auto value) { return value * 3 + 5; })
          .compile();
  if (!program) {
    return 1;
  }
  auto first_job = program->resident(first);
  auto second_job = program->resident(second);
  if (!first_job || !second_job) {
    return 2;
  }

  auto first_task = session.compute(*first_job).submit();
  auto second_task = session.compute(*second_job).submit();
  if (!first_task.wait() || !second_task.wait()) {
    return 3;
  }
  auto first_output = first_job->read();
  auto second_output = second_job->read();
  if (!first_output || !second_output) {
    return 4;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    if ((*first_output)[index] != 26 || (*second_output)[index] != -28) {
      return 5;
    }
  }
  return 0;
}

} // namespace rund::node::test_contract

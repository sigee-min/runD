#include "../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>

namespace rund::node::test_contract {

int CheckComputeCancelEpochs();

int CheckComputeCancelRace(::rund::Session &session) {
  constexpr std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-cancel-race", input.size(),
                             [](auto value) { return value * 3 + 1; })
          .compile();
  if (!program) {
    return 1;
  }
  for (std::uint32_t iteration = 0u; iteration < 256u; ++iteration) {
    auto job = program->resident(input);
    if (!job) {
      return 2;
    }
    auto task = session.compute(*job).submit();
    const compute::Status cancelled = task.cancel();
    const compute::Completion result = task.wait();
    if (cancelled &&
        (result || result.error() != std::string_view{"compute_cancelled"})) {
      return 3;
    }
    if (!cancelled &&
        cancelled.error() != std::string_view{"compute_already_completed"}) {
      return 4;
    }
  }
  const int epochs = CheckComputeCancelEpochs();
  if (epochs != 0) {
    std::fprintf(stderr, "compute cancellation epoch contract failed: %d\n",
                 epochs);
  }
  return epochs;
}

} // namespace rund::node::test_contract

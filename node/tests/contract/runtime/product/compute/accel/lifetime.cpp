#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <cstdint>

namespace rund::node::test_contract {

int CheckComputeAccelLifetime(::rund::Session &session,
                              const compute::Target target) {
  constexpr std::array<std::int32_t, 8> input{-3, 0, 4, 9, 12, -8, 7, 21};
  compute::Submission retained{};
  {
    auto program =
        compute::on(target)
            .map<std::int32_t>("node-host-accel-retained", input.size(),
                               [](auto value) { return value * 2 + 5; })
            .compile();
    if (!program) {
      return 1;
    }
    auto job = program->resident(input);
    if (!job) {
      return 2;
    }
    retained = session.compute(*job).submit();
  }
  return retained.wait() ? 0 : 3;
}

} // namespace rund::node::test_contract

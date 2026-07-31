#pragma once

#include <array>
#include <cstdint>

namespace rund {
class Session;
}

namespace rund::node::test_contract {

int CheckComputeJobGate(::rund::Session &session,
                        const std::array<std::int32_t, 4> &input);
int CheckComputeTaskCapacity();
int CheckComputeProgramConcurrency(::rund::Session &session);
int CheckComputeCollectives(::rund::Session &session);
int CheckComputeAwait(::rund::Session &session);
int CheckComputeCancelRace(::rund::Session &session);

} // namespace rund::node::test_contract

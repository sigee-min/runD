#pragma once
#include <cstdint>
#include <rund/compute/ops.hpp>
#include <rund/compute/program.hpp>
#include <rund/compute/target.hpp>
namespace rund {
class Session;
}
namespace rund::node::test_contract {

int CheckComputeAccelBackend(compute::Target target);
int CheckComputeAccelConcurrency(
    ::rund::Session &session, compute::Target target,
    compute::Program<std::int32_t(std::int32_t)> &program,
    std::span<const std::int32_t> first, std::span<const std::int32_t> second);
int CheckVulkanCommandCapacity(
    compute::Program<std::int32_t(std::int32_t)> &program,
    std::span<const std::int32_t> input);
int CheckComputeAccelScanConcurrency(::rund::Session &session,
                                     compute::Target target);
int CheckComputeAccelLifetime(::rund::Session &session, compute::Target target);

} // namespace rund::node::test_contract

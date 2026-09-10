#pragma once

#include "src/accel/kernel/residency/service_free_direct.hpp"

#include <cstdint>

namespace rund_node_test_pipeline_residency::service_free_direct_test {

namespace accel = rund::node::accel::detail;

struct FakeState final {
  accel::ServiceFreeDirectCapability capability{};
  std::uint64_t submit_count{};
  std::uint64_t final_count{};
};

[[nodiscard]] rund::AccelCheck
FakeSubmit(const accel::ServiceFreeDirectRequest &) noexcept;

[[nodiscard]] int CheckAuthority();

} // namespace rund_node_test_pipeline_residency::service_free_direct_test

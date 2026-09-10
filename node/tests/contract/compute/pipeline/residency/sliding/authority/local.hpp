#pragma once

#include "../local.hpp"

namespace rund_node_test_pipeline_residency::sliding_authority {

[[nodiscard]] int CheckPhysicalForecast();
[[nodiscard]] int CheckInactiveBank();
[[nodiscard]] int CheckHaloReuse();
[[nodiscard]] int CheckOutputReservation();
[[nodiscard]] int CheckMalformedReceipts();
[[nodiscard]] int CheckCacheAuthority();

} // namespace rund_node_test_pipeline_residency::sliding_authority

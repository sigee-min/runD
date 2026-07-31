#pragma once

namespace rund_node_backend_contract {

[[nodiscard]] bool CheckStatsProjection();
[[nodiscard]] int CheckMapParity();
[[nodiscard]] bool CheckDomains();
[[nodiscard]] bool CheckPrimitiveDomains();
[[nodiscard]] bool CheckHistogram();
[[nodiscard]] bool CheckSegments();
[[nodiscard]] bool CheckMovementDomains();
[[nodiscard]] bool CheckPartialWriteReset();
[[nodiscard]] bool CheckExactMovement();
[[nodiscard]] bool CheckMatrices();
[[nodiscard]] bool CheckArgsorts();

} // namespace rund_node_backend_contract

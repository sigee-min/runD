#pragma once

#include "src/accel/kernel/reset/model.hpp"

namespace node_accel_contract::reset_contract {

using Range = rund::node::accel::detail::reset::Range;

[[nodiscard]] bool Accepted(Range range) noexcept;
[[nodiscard]] bool Rejected(Range range) noexcept;

[[nodiscard]] bool CheckWidthAndLayout();
[[nodiscard]] bool CheckInvalidRanges();
[[nodiscard]] bool CheckVulkanExecution();
[[nodiscard]] bool CheckUniqueProjection();
[[nodiscard]] bool CheckLifetimeOverlap();
[[nodiscard]] bool CheckSealedPlans();
[[nodiscard]] bool CheckProjectionMatrix();
[[nodiscard]] bool CheckProjectionAmbiguity();

} // namespace node_accel_contract::reset_contract

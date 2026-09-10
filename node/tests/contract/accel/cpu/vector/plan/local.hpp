#pragma once

#include <kernel/program/compute/lowering/parse.hpp>

#include "../../local.hpp"

#include <cstddef>

namespace node_accel_contract::vector_plan {

struct IndependentSlotLowerBound final {
  std::size_t peak = 0u;
  std::size_t once_count = 0u;
  bool ok = false;
};

[[nodiscard]] IndependentSlotLowerBound
IndependentPhysicalSlotLowerBound(const rund::kernel::ComputeIR &ir);

} // namespace node_accel_contract::vector_plan

namespace node_accel_contract {

[[nodiscard]] bool CheckExecutorSelectorsAndFixedScratch();
[[nodiscard]] bool CheckIntegerScratch();
[[nodiscard]] bool CheckCommitDemand();
[[nodiscard]] bool CheckStablePrefixFailure();
[[nodiscard]] bool CheckStableEdgeAuthority();

} // namespace node_accel_contract

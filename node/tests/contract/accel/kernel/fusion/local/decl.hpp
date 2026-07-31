#pragma once

#include <accel/context/value.hpp>

#include "model.hpp"

namespace node_accel_contract::fusion {

[[nodiscard]] bool RunFusedChainCase(const rund::AccelContext &context,
                                     const rund::compute_dsl::ComputeOp &op,
                                     const Inputs &inputs);
[[nodiscard]] bool RunExtraReadCase(
    const rund::AccelContext &context, const rund::compute_dsl::ComputeOp &op,
    const rund::compute_dsl::ComputeOp &two_read_op, const Inputs &inputs);
[[nodiscard]] bool RunLongChainCase(
    const rund::AccelContext &context,
    const rund::compute_dsl::ComputeOp &op, const Inputs &inputs);
[[nodiscard]] bool RunCapacityCase(
    const rund::AccelContext &context,
    const rund::compute_dsl::ComputeOp &op);
[[nodiscard]] bool RunConflictCase(const rund::AccelContext &context,
                                   const rund::compute_dsl::ComputeOp &op,
                                   const Inputs &inputs);
[[nodiscard]] bool
RunTwoReadCase(const rund::AccelContext &context,
               const rund::compute_dsl::ComputeOp &two_read_op,
               const Inputs &inputs);
[[nodiscard]] bool RunVisibilityCase(const rund::AccelContext &context,
                                     bool internal_intermediate);
[[nodiscard]] bool RunRegionCase(const rund::AccelContext &context);

} // namespace node_accel_contract::fusion

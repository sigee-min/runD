#pragma once

#include <accel/kernel/value.hpp>

#include "model.hpp"

#include "src/accel/context/internal.hpp"

namespace node_accel_contract::fusion {

[[nodiscard]] inline bool StepArtifactIsChecked(
    const rund::node::accel::detail::KernelExecutionStep &step,
    const rund::AccelKernel &kernel) noexcept {
  return step.artifact.ok && step.artifact.key.api == kernel.frozen_caps.api &&
         step.artifact.key.scalar == kernel.scalar &&
         (step.artifact.key.op_hash_hi != 0u ||
          step.artifact.key.op_hash_lo != 0u) &&
         step.artifact.key.op_hash_hi ==
             step.artifact.key.canonical_ir_hash_hi &&
         step.artifact.key.op_hash_lo ==
             step.artifact.key.canonical_ir_hash_lo &&
         !step.artifact.source_text.empty() &&
         step.artifact.canonical_ir_bytes.empty() && !step.cpu_input.ok;
}

} // namespace node_accel_contract::fusion

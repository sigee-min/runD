#pragma once

#include <accel/device.hpp>

#include <node/accel/pick.hpp>

#include "bindings.hpp"

#include <string_view>

namespace node_accel_contract {

[[nodiscard]] bool VulkanBackendRejectsSelfConsistentForgedSource() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(vulkan::RequiredPolicy());
  if (!pick.check.ok) {
    return vulkan::FailureReasonIsPrecise(pick);
  }

  vulkan::ForgedArtifactWork work{};
  const rund::compute_dsl::ComputeOp valid_op =
      vulkan::MakeValidForgeTarget(work);
  const rund::compute_dsl::ComputeOp forged_op = vulkan::MakeForgedBody(work);
  const vulkan::ForgedArtifactLowering lowering =
      vulkan::BuildForgedArtifactLowering(pick, valid_op, forged_op);
  if (!lowering.ok) {
    return false;
  }

  vulkan::ForgedArtifactBindings bindings{};
  vulkan::PrepareForgedArtifactBindings(bindings, work, lowering);
  const bool accepted = pick.backend.execute(
      pick.backend.context, lowering.plan, lowering.artifact, &bindings.window,
      1u, bindings.bindings);
  if (accepted || pick.backend.last_error == nullptr) {
    return false;
  }
  return std::string_view{pick.backend.last_error(pick.backend.context)} ==
         "compute_artifact_mismatch";
}

} // namespace node_accel_contract

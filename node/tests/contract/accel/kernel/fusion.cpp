#include <accel/api.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <kernel/program/compute/lowering/entry.hpp>

#include "fusion/local.hpp"
#include "fusion/reject/conflict.hpp"
#include "fusion/success/chain/capacity.hpp"
#include "fusion/success/chain/long.hpp"
#include "fusion/success/chain/run.hpp"
#include "fusion/success/output.hpp"
#include "src/accel/graph/step.hpp"
#include "test/assert.hpp"
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include <array>
#include <cstdio>
#include <vector>

namespace node_accel_contract {

[[nodiscard]] bool OriginalDispatchScalarCoversMultipleWindows(
    const rund::compute_dsl::ComputeOp &op) {
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Metal);
  if (!artifact.ok) {
    return false;
  }
  std::vector<rund::node::accel::detail::GraphCompileNode> nodes{};
  nodes.reserve(2u);
  for (std::size_t index = 0u; index < 2u; ++index) {
    nodes.push_back(rund::node::accel::detail::GraphCompileNode{
        .ir = &op.ir(),
        .artifact = artifact,
        .map_metadata = artifact.metadata,
        .element_count = 8u,
    });
  }
  const rund::kernel::ComputeCaps caps{
      .api = rund::kernel::ComputeApi::Metal,
      .device_bytes = 1024u * 1024u,
      .staging_bytes = 1024u * 1024u,
      .max_window_tiles = 3u,
      .subgroup_width = 1u,
      .ok = true,
      .reason = "ok",
  };
  const rund::node::accel::detail::FrozenDispatchCount count =
      rund::node::accel::detail::BuildOriginalDispatchCount(nodes, caps);
  return count.ok && count.count == 6u;
}

bool AccelGraphKernelFusionContract() {
  const rund::compute_dsl::ComputeOp op = fusion::BuildFixedLane32Op();
  if (!op.ok() || !OriginalDispatchScalarCoversMultipleWindows(op)) {
    std::fprintf(stderr, "fusion multi-window scalar failed\n");
    return false;
  }

  rund::AccelDevice pick =
      rund::node::accel::PickAccel(fusion::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    if (!fusion::PickUnavailableReasonIsPrecise(pick, rund::AccelApi::Metal)) {
      return false;
    }
    pick = rund::node::accel::PickAccel(fusion::Policy(rund::AccelApi::Vulkan));
  }
  if (!pick.check.ok) {
    return fusion::PickUnavailableReasonIsPrecise(pick, rund::AccelApi::Vulkan);
  }

  const rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  const rund::compute_dsl::ComputeOp two_read_op =
      fusion::BuildTwoReadFixedLane32Op();
  if (!two_read_op.ok()) {
    return false;
  }

  const fusion::Inputs inputs = fusion::BuildInputs();
  if (!fusion::RunFusedChainCase(context, op, inputs)) {
    std::fprintf(stderr, "fusion fused-chain failed\n");
    return false;
  }
  if (!fusion::RunExtraReadCase(context, op, two_read_op, inputs)) {
    std::fprintf(stderr, "fusion extra-read failed\n");
    return false;
  }
  if (!fusion::RunLongChainCase(context, op, inputs)) {
    std::fprintf(stderr, "fusion long-chain failed\n");
    return false;
  }
  if (!fusion::RunCapacityCase(context, op)) {
    std::fprintf(stderr, "fusion capacity failed\n");
    return false;
  }
  if (!fusion::RunConflictCase(context, op, inputs)) {
    std::fprintf(stderr, "fusion conflict failed\n");
    return false;
  }
  if (!fusion::RunTwoReadCase(context, two_read_op, inputs)) {
    std::fprintf(stderr, "fusion two-read failed\n");
    return false;
  }
  return true;
}

} // namespace node_accel_contract

int RunAccelKernelFusionContract() {
  namespace fusion = node_accel_contract::fusion;
  TEST_ASSERT(node_accel_contract::AccelGraphKernelFusionContract());
  rund::AccelDevice pick =
      rund::node::accel::PickAccel(fusion::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    TEST_ASSERT(
        fusion::PickUnavailableReasonIsPrecise(pick, rund::AccelApi::Metal));
    pick = rund::node::accel::PickAccel(fusion::Policy(rund::AccelApi::Vulkan));
  }
  if (!pick.check.ok) {
    TEST_ASSERT(
        fusion::PickUnavailableReasonIsPrecise(pick, rund::AccelApi::Vulkan));
    return 0;
  }
  const rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  TEST_ASSERT(context.check.ok);
  TEST_ASSERT(fusion::RunVisibilityCase(context, false));
  TEST_ASSERT(fusion::RunVisibilityCase(context, true));
  TEST_ASSERT(fusion::RunRegionCase(context));
  constexpr std::array output_apis{rund::AccelApi::Cpu, rund::AccelApi::Metal,
                                   rund::AccelApi::Vulkan};
  for (const rund::AccelApi api : output_apis) {
    const rund::AccelDevice output_pick =
        rund::node::accel::PickAccel(fusion::Policy(api));
    if (!output_pick.check.ok) {
      TEST_ASSERT(api != rund::AccelApi::Cpu);
      TEST_ASSERT(fusion::PickUnavailableReasonIsPrecise(output_pick, api));
      continue;
    }
    const rund::AccelContext output_context =
        rund::node::accel::OpenAccel(output_pick);
    TEST_ASSERT(output_context.check.ok);
    TEST_ASSERT(fusion::TerminalOutputsPreserved(output_context));
  }
  return 0;
}
#include <kernel/program/compute/lowering/entry.hpp>

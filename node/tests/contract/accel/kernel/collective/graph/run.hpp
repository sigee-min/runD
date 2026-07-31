#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/graph/node.hpp>

#include "reject.hpp"
#include "test/assert.hpp"

namespace node_accel_contract::collective {

[[nodiscard]] bool GraphContract() {
  rund::AccelDevice pick = graph_case::PickGraphBackend();
  if (!pick.check.ok) {
    return PickUnavailableReasonIsPrecise(pick, rund::AccelApi::Vulkan);
  }

  graph_case::State state = graph_case::MakeState(pick);
  if (!graph_case::StateOk(state)) {
    return false;
  }
  TEST_ASSERT(graph_case::ReservedGraphKindSlotsRejectAtCompile(state));
  TEST_ASSERT(graph_case::ScanGraphProducesPlan(state));
  TEST_ASSERT(graph_case::RepresentativeCollectivesMatchCpu(pick));
  TEST_ASSERT(graph_case::InvalidCollectiveNodesReject(state));
  return true;
}

} // namespace node_accel_contract::collective

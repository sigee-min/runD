#pragma once

#include "src/accel/context/internal/execution.hpp"
#include "src/accel/kernel/backend/execute.hpp"
#include "src/accel/kernel/recurrence.hpp"
#include "src/accel/kernel/recurrence/plan.hpp"
#include "src/accel/kernel/recurrence/source.hpp"
#include "src/accel/kernel/step/map/stride.hpp"
#include "src/accel/metal/pipeline/guard.hpp"

#include <kernel/program/compute/lowering/text.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace node_accel_contract {

using rund::kernel::ComputeApi;
using rund::kernel::ComputeBindingAccess;
using rund::kernel::ComputeDomain;
using rund::kernel::ComputeScalar;
using rund::kernel::LoweringArtifactKind;
using rund::node::accel::detail::BackendBatchEntry;
using rund::node::accel::detail::BackendRecurrence;
using rund::node::accel::detail::BackendRun;
using rund::node::accel::detail::BackendWindow;
using rund::node::accel::detail::BackendWindowPhase;
using rund::node::accel::detail::BoundStep;
using rund::node::accel::detail::BuildMapRecurrence;
using rund::node::accel::detail::BuildNestedAggregate;
using rund::node::accel::detail::BuildNestedMapRecurrence;
using rund::node::accel::detail::KernelExecutionStep;
using rund::node::accel::detail::MapRecurrence;
using rund::node::accel::detail::MapRecurrencePreparationPlan;
using rund::node::accel::detail::MapRecurrenceState;
using rund::node::accel::detail::NestedAggregateState;
using rund::node::accel::detail::NestedTemplateGeometry;
using rund::node::accel::detail::NestedTemplatePhase;
using rund::node::accel::detail::NestedTemplateRouteProjection;
using rund::node::accel::detail::NestedTemplateShape;
using rund::node::accel::detail::PlannedStep;
using rund::node::accel::detail::ProveNestedTemplateGeometry;
using rund::node::accel::detail::ProveNestedTemplateShape;
using rund::node::accel::detail::RunBinds;
using rund::node::accel::detail::SameMapRecurrenceTemplate;
using rund::node::accel::detail::StepBinds;

struct Occurrence final {
  KernelExecutionStep step;
  rund::node::accel::detail::KernelExecution execution;
  PlannedStep planned;
  RunBinds refs;
  StepBinds binds;
  BoundStep bound;
  BackendRun run;
  std::shared_ptr<void> prepared;
  BackendBatchEntry entry;
};

struct Fixture final {
  std::array<std::shared_ptr<void>, 4u> owners;
  std::array<Occurrence, 2u> occurrences;
  std::array<BackendBatchEntry, 2u> entries;
  std::array<std::uint8_t, 2u> barriers;

  Fixture(ComputeApi api, ComputeScalar scalar,
          bool uniform_invariant = false, bool writes_history = false);
};

[[nodiscard]] bool MapRecurrenceNestedMarkerContract();
[[nodiscard]] bool MapRecurrenceGeometryContract();
[[nodiscard]] bool MapRecurrenceHistoryContract();
[[nodiscard]] bool MapRecurrencePreparationContract();
[[nodiscard]] bool MapRecurrenceSourceMaterializationContract();

} // namespace node_accel_contract

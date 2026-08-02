#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../backend/run/local.hpp"
#include "cpu/local.hpp"
#include "cpu/segmented/reduce/backend.hpp"
#include "cpu/segmented/scan/backend.hpp"
#include "run/local.hpp"
#include "scan/inclusive/run.hpp"
#include "scan/stream/run.hpp"
#include "test/assert.hpp"

#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include "src/accel/kernel/backend/run.hpp"
#include "src/accel/kernel/schedule.hpp"
#include "src/accel/kernel/storage.hpp"
#include "src/accel/primitive/shape.hpp"
#include "src/accel/sequence/input/window.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>

namespace node_accel_contract {

[[nodiscard]] bool BackendRunsScanSortCompact(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsInclusiveScan(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsGather(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsHistogram(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsPartition(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsReduce(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsScatter(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsStencil(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsSegmentedScan(const rund::AccelDevice &pick);
[[nodiscard]] bool AvailableBackendsRunSegmentedScan();
[[nodiscard]] bool BackendRunsSegmentedReduce(const rund::AccelDevice &pick);
[[nodiscard]] bool AvailableBackendsRunSegmentedReduce();
[[nodiscard]] bool MapRecurrenceSourceContract();
[[nodiscard]] bool BackendParameterModelsMatchSources();
[[nodiscard]] bool ResetModelContract();
[[nodiscard]] bool AuthorityContract();
[[nodiscard]] bool MetalTemplateMemoryContract();
[[nodiscard]] bool MetalIcbCalibrationContract();

} // namespace node_accel_contract

namespace {

struct ThrowingValue final {
  static bool fail;
  std::size_t value = 0u;

  ThrowingValue() {
    if (fail) {
      throw std::bad_alloc{};
    }
  }
};

bool ThrowingValue::fail = false;

[[nodiscard]] bool IndexedStorageResizeIsTransactional() {
  using Storage =
      rund::node::accel::detail::InlineIndexedStorage<ThrowingValue, 4u>;
  Storage storage{};
  storage.resize(4u);
  ThrowingValue *first = storage.get(0u);
  if (first == nullptr) {
    return false;
  }
  first->value = 7u;

  bool rejected = false;
  ThrowingValue::fail = true;
  try {
    storage.resize(5u);
  } catch (const std::bad_alloc &) {
    rejected = true;
  }
  ThrowingValue::fail = false;
  first = storage.get(0u);
  return rejected && storage.valid() && storage.size() == 4u &&
         storage.data() != nullptr && first != nullptr && first->value == 7u;
}

[[nodiscard]] bool ProgramHazardsHaveOneFrontierAuthority() {
  using rund::kernel::BufferRole;
  using rund::node::accel::detail::BuildScheduledStepOrder;
  using rund::node::accel::detail::KernelExecutionStep;

  std::array<KernelExecutionStep, 7u> steps{};
  for (std::size_t index = 0u; index < steps.size(); ++index) {
    steps[index].graph_binding_indices_ok =
        steps[index].graph_binding_indices.push_back(index);
  }
  const std::array roles{
      BufferRole::Write, BufferRole::Read,  BufferRole::Read, BufferRole::Read,
      BufferRole::Write, BufferRole::Write, BufferRole::Write};
  const std::array<std::uint64_t, 7u> aliases{0u, 1u, 0u, 0u, 0u, 1u, 1u};
  const std::array<std::uint8_t, 7u> required{};
  const auto plan = BuildScheduledStepOrder(steps, roles, aliases, required);
  if (!plan.ok || plan.size() != steps.size()) {
    return false;
  }
  constexpr std::array expected{false, false, true, false, true, false, true};
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    if (plan.at(index) != index ||
        plan.barrier_before(index) != expected[index]) {
      return false;
    }
  }
  auto invalid_aliases = aliases;
  invalid_aliases.back() = invalid_aliases.size();
  return !BuildScheduledStepOrder(steps, roles, invalid_aliases, required).ok;
}

[[nodiscard]] bool SourcePartitionHasOneAuthority() {
  using rund::node::accel::detail::KernelExecutionStep;
  using rund::node::accel::detail::SourceRange;
  using rund::node::accel::detail::SourceStep;
  using rund::node::accel::detail::ValidSourcePartition;

  std::array<KernelExecutionStep, 3u> steps{};
  steps[0u].source =
      SourceRange{.begin = SourceStep{0u}, .end = SourceStep{2u}};
  steps[1u].source =
      SourceRange{.begin = SourceStep{2u}, .end = SourceStep{3u}};
  steps[2u].source =
      SourceRange{.begin = SourceStep{3u}, .end = SourceStep{6u}};
  if (!ValidSourcePartition(steps, 6u) || ValidSourcePartition(steps, 5u) ||
      ValidSourcePartition(steps,
                           static_cast<std::uint64_t>(
                               std::numeric_limits<std::uint32_t>::max())) ||
      ValidSourcePartition(std::span<const KernelExecutionStep>{}, 0u)) {
    return false;
  }

  steps[1u].source.begin = SourceStep{3u};
  if (ValidSourcePartition(steps, 6u)) {
    return false;
  }
  steps[1u].source.begin = SourceStep{1u};
  if (ValidSourcePartition(steps, 6u)) {
    return false;
  }
  steps[1u].source.begin = SourceStep{2u};
  steps[1u].source.end = SourceStep{2u};
  return !ValidSourcePartition(steps, 6u);
}

[[nodiscard]] bool FailureNodeIsFirst() {
  using rund::node::accel::detail::BoundStep;
  using rund::node::accel::detail::KernelExecutionStep;
  using rund::node::accel::detail::NoNode;
  using rund::node::accel::detail::RecordNode;
  using rund::node::accel::detail::SourceRange;
  using rund::node::accel::detail::SourceStep;

  KernelExecutionStep first{};
  first.source = SourceRange{.begin = SourceStep{7u}, .end = SourceStep{8u}};
  KernelExecutionStep later{};
  later.source = SourceRange{.begin = SourceStep{11u}, .end = SourceStep{12u}};
  const BoundStep first_bound{.step = &first};
  const BoundStep later_bound{.step = &later};
  std::uint32_t failure = NoNode;
  RecordNode(nullptr, first_bound);
  RecordNode(&failure, first_bound);
  RecordNode(&failure, later_bound);
  return failure == 7u;
}

[[nodiscard]] bool RangesUseOffsets() {
  using rund::kernel::ResidentBufferRef;
  using rund::node::accel::detail::ResidentOverlap;
  const ResidentBufferRef first{.id = 7u,
                                .bytes = 1024u,
                                .offset_bytes = 0u,
                                .element_bytes = 4u,
                                .stride_bytes = 4u,
                                .count = 16u};
  ResidentBufferRef disjoint = first;
  disjoint.offset_bytes = 256u;
  ResidentBufferRef overlap = first;
  overlap.offset_bytes = 32u;
  ResidentBufferRef separate = first;
  separate.id = 8u;
  return !ResidentOverlap(first, disjoint) && ResidentOverlap(first, overlap) &&
         !ResidentOverlap(first, separate);
}

[[nodiscard]] bool InputWindowPlansRejectMixedAddressing() {
  using rund::kernel::ComputeDispatchWindow;
  using rund::kernel::ExecutionMetadata;
  using rund::kernel::ReadRoute;
  using rund::node::accel::detail::FreezeInputWindowPlans;
  using rund::node::accel::detail::InputWindow;
  using rund::node::accel::detail::InputWindowPlan;

  ExecutionMetadata uniform{};
  uniform.uniform_read_mask = 0x1u;
  uniform.read_count = 1u;
  uniform.ok = true;
  uniform.reason = "ok";
  std::array<InputWindowPlan, 1u> uniform_plan{};
  constexpr ComputeDispatchWindow window{.begin_sequence = 4u,
                                         .tile_count = 4u};
  const auto same_window = [](const ComputeDispatchWindow left,
                              const ComputeDispatchWindow right) {
    return left.begin_sequence == right.begin_sequence &&
           left.tile_count == right.tile_count;
  };
  if (!FreezeInputWindowPlans(uniform, 8u, uniform_plan) ||
      !uniform_plan[0u].base_anchored() || uniform_plan[0u].windowed() ||
      uniform_plan[0u].required_count != 1u ||
      !same_window(
          InputWindow(uniform_plan[0u], window),
          ComputeDispatchWindow{.begin_sequence = 0u, .tile_count = 1u})) {
    return false;
  }

  ExecutionMetadata indexed{};
  indexed.read_routes.push_back(
      ReadRoute{.source = 0u, .index = 1u, .count = 7u});
  indexed.read_count = 2u;
  indexed.ok = true;
  indexed.reason = "ok";
  std::array<InputWindowPlan, 2u> indexed_plans{};
  if (!FreezeInputWindowPlans(indexed, 8u, indexed_plans) ||
      !indexed_plans[0u].base_anchored() ||
      indexed_plans[0u].required_count != 7u ||
      indexed_plans[1u].base_anchored() || !indexed_plans[1u].windowed() ||
      indexed_plans[1u].required_count != 8u ||
      !same_window(
          InputWindow(indexed_plans[0u], window),
          ComputeDispatchWindow{.begin_sequence = 0u, .tile_count = 7u}) ||
      !same_window(InputWindow(indexed_plans[1u], window), window)) {
    return false;
  }

  ExecutionMetadata direct_uniform = uniform;
  direct_uniform.direct_read_mask = 0x1u;
  if (FreezeInputWindowPlans(direct_uniform, 8u, uniform_plan)) {
    return false;
  }

  ExecutionMetadata direct_source = indexed;
  direct_source.direct_read_mask = 0x1u;
  return !FreezeInputWindowPlans(direct_source, 8u, indexed_plans);
}

} // namespace

int RunAccelKernelCoreContract() {
  TEST_ASSERT(IndexedStorageResizeIsTransactional());
  TEST_ASSERT(ProgramHazardsHaveOneFrontierAuthority());
  TEST_ASSERT(SourcePartitionHasOneAuthority());
  TEST_ASSERT(FailureNodeIsFirst());
  TEST_ASSERT(RangesUseOffsets());
  TEST_ASSERT(InputWindowPlansRejectMixedAddressing());
  TEST_ASSERT(node_accel_contract::ResetModelContract());
  TEST_ASSERT(node_accel_contract::AuthorityContract());
  TEST_ASSERT(node_accel_contract::MetalTemplateMemoryContract());
  TEST_ASSERT(node_accel_contract::MetalIcbCalibrationContract());
  TEST_ASSERT(node_accel_contract::BackendParameterModelsMatchSources());
  const rund::AccelDevice cpu_pick = rund::node::accel::PickAccel(
      node_accel_contract::cpu_context::CpuPolicy());
  TEST_ASSERT(cpu_pick.check.ok);
  TEST_ASSERT(cpu_pick.api == rund::AccelApi::Cpu);
  TEST_ASSERT(
      node_accel_contract::cpu_context::CpuContextMatchesAvailableRealBackend(
          cpu_pick));
  const rund::AccelContext cpu_context = rund::node::accel::OpenAccel(cpu_pick);
  TEST_ASSERT(cpu_context.check.ok);
  TEST_ASSERT(node_accel_contract::kernel_case::IndexedWriteBoundaryMatches(
      cpu_context));
  TEST_ASSERT(node_accel_contract::BackendRunsInclusiveScan(cpu_pick));
  TEST_ASSERT(node_accel_contract::BackendRunsScanSortCompact(cpu_pick));
  TEST_ASSERT(node_accel_contract::AccelGraphKernelCompileContract());
  if (const int rc =
          node_accel_contract::RunAccelKernelCollectiveSurfaceContract();
      rc != 0) {
    return rc;
  }
  TEST_ASSERT(node_accel_contract::AccelGraphKernelResidentRunContract());
  TEST_ASSERT(
      node_accel_contract::KernelBindingIndicesUseInlineStorageUntilOverflow());
  TEST_ASSERT(node_accel_contract::AvailableBackendsRunSegmentedScan());
  TEST_ASSERT(node_accel_contract::AvailableBackendsRunSegmentedReduce());
  TEST_ASSERT(node_accel_contract::MapRecurrenceSourceContract());
  TEST_ASSERT(node_accel_contract::BackendRunsSegmentedScan(cpu_pick));
  TEST_ASSERT(node_accel_contract::BackendRunsSegmentedReduce(cpu_pick));
  TEST_ASSERT(node_accel_contract::BackendRunsGather(cpu_pick));
  TEST_ASSERT(node_accel_contract::BackendRunsHistogram(cpu_pick));
  TEST_ASSERT(node_accel_contract::BackendRunsPartition(cpu_pick));
  TEST_ASSERT(node_accel_contract::BackendRunsReduce(cpu_pick));
  TEST_ASSERT(node_accel_contract::BackendRunsScatter(cpu_pick));
  TEST_ASSERT(node_accel_contract::BackendRunsStencil(cpu_pick));
  return 0;
}

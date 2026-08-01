#pragma once

#include <accel/context/buffer/descriptor.hpp>
#include <accel/graph/visibility.hpp>

#include "../../graph/operation.hpp"
#include "admission.hpp"
#include "bindings.hpp"

#include <node/accel/context.hpp>

#include <cstdint>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/admission.hpp>
#include <limits>
#include <span>

namespace rund::node::accel::detail {

struct SourceStep final {
  static constexpr std::uint32_t none =
      std::numeric_limits<std::uint32_t>::max();

  std::uint32_t index{none};

  [[nodiscard]] constexpr bool valid() const noexcept { return index != none; }
};

struct ExecStep final {
  static constexpr std::uint32_t none =
      std::numeric_limits<std::uint32_t>::max();

  std::uint32_t index{none};

  [[nodiscard]] constexpr bool valid() const noexcept { return index != none; }
};

struct SourceRange final {
  SourceStep begin{};
  SourceStep end{};

  [[nodiscard]] constexpr bool valid() const noexcept {
    return begin.valid() && end.valid() && begin.index < end.index;
  }

  [[nodiscard]] constexpr bool contains(const SourceStep step) const noexcept {
    return valid() && step.valid() && begin.index <= step.index &&
           step.index < end.index;
  }
};

struct ResetPlan final {
  std::uint64_t binding{};
  ExecStep step{};
  ExecStep last{};
};

// Fixed-width semantic summary derived once from the admitted canonical Map
// IR. Source backends discard ParsedIR and canonical bytes after graph-token
// creation; recurrence classification consumes this normalized value instead
// of retaining or reparsing either representation.
enum class MapSemanticKind : std::uint8_t {
  Unknown,
  AddWrapU32Pair,
  AddWrapU32Immediate,
  ResidentWindowBaseU32,
  ResidentWindowItemsU32,
  ResidentWindowCountU32,
};

struct MapSemantic final {
  MapSemanticKind kind{MapSemanticKind::Unknown};
  std::uint32_t immediate{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t windows{};
  // Cold ParsedIR proof that every node is total for all admitted lane values.
  // Recurrence lowering consumes this bit after canonical IR has been dropped;
  // source text is never reparsed as semantic authority.
  bool recurrence_total{};
};

static_assert(sizeof(MapSemantic) == 24u);

static_assert(sizeof(SourceStep) == 4u);
static_assert(sizeof(ExecStep) == 4u);
static_assert(sizeof(ResetPlan) == 16u);

struct KernelExecutionStep {
  Operation operation{};
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::compute_lowering_detail::ComputeInputAdmission cpu_input{};
  MapSemantic map_semantic{};
  KernelBindingIndices graph_binding_indices{};
  bool graph_binding_indices_ok = false;
  std::uint64_t primitive_hash_hi = 0u;
  std::uint64_t primitive_hash_lo = 0u;
  std::uint64_t element_count = 0u;
  rund::kernel::GraphControl control{};
  SourceRange source{};

  [[nodiscard]] rund::kernel::NodeKind kind() const noexcept {
    return operation.kind();
  }
};

static_assert(sizeof(KernelExecutionStep) <= 1280u,
              "retained execution step exceeded its footprint budget");

[[nodiscard]] inline bool
ValidSourcePartition(const std::span<const KernelExecutionStep> steps,
                     const std::uint64_t original_count) noexcept {
  if (steps.empty() || original_count == 0u ||
      original_count >= SourceStep::none) {
    return false;
  }
  std::uint32_t cursor = 0u;
  for (const KernelExecutionStep &step : steps) {
    if (!step.source.valid() || step.source.begin.index != cursor) {
      return false;
    }
    cursor = step.source.end.index;
  }
  return cursor == original_count;
}

struct KernelExecution {
  KernelAdmission admission{};
  ContextAdmission context_admission{};
  std::span<const rund::kernel::BufferRole> graph_roles{};
  std::span<const rund::AccelBufferDesc> graph_shapes{};
  std::span<const rund::GraphBufferVisibility> graph_visibilities{};
  std::span<const std::uint64_t> graph_alias_representatives{};
  std::span<const ResetPlan> resets{};
  std::span<const KernelExecutionStep> steps{};
  std::span<const std::uint8_t> required_barriers{};
  // Compile records only dispatches removed by fusion. Warm execution adds
  // this delta to its one authoritative final dispatch count, so non-Map work
  // never needs a duplicate compile-time dispatch-count table.
  std::uint64_t removed_dispatch_count = 0u;
  std::uint64_t original_operation_count = 0u;
  std::uint64_t fused_operation_count = 0u;
  std::uint64_t fusion_rejection_count = 0u;
  const char *fusion_reason = "compute_fusion_invalid";
};

} // namespace rund::node::accel::detail

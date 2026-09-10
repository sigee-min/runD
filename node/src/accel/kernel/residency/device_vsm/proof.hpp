#pragma once

#include "geometry.hpp"
#include "graph_resident.hpp"
#include "identity.hpp"
#include "resident.hpp"

#include "../../../range_aggregate/model/candidate.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/window/model.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

enum class DeviceVsmTopology : std::uint8_t {
  Pointwise = 1u,
  Window = 2u,
  GraphMapReduce = 3u,
  Scan = 4u,
  Reduce = 5u,
  GraphPointwise = 6u,
  GraphResident = 7u,
};

inline constexpr std::uint64_t DeviceVsmWindowParameterBytes = 64u;

enum class DeviceVsmWindowMapKind : std::uint8_t {
  None,
  AddWrapU32Immediate,
  MulWrapU32Immediate,
  CanonicalTotalU32,
};

struct DeviceVsmWindowMap final {
  DeviceVsmWindowMapKind kind{DeviceVsmWindowMapKind::None};
  std::uint32_t immediate{};
  std::uint64_t source_hi{};
  std::uint64_t source_lo{};
  std::uint64_t canonical_hi{};
  std::uint64_t canonical_lo{};

  [[nodiscard]] constexpr bool active() const noexcept {
    return kind != DeviceVsmWindowMapKind::None;
  }

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmWindowMap &) const noexcept = default;
};

struct DeviceVsmWindowFusion final {
  DeviceVsmWindowMap before{};
  DeviceVsmWindowMap before_second{};
  DeviceVsmWindowMap before_third{};
  DeviceVsmWindowMap after{};
  DeviceVsmWindowMap after_second{};
  DeviceVsmWindowMap after_third{};
  std::uint32_t stage_count{1u};

  [[nodiscard]] constexpr bool active() const noexcept {
    return before.active() || before_second.active() || before_third.active() ||
           after.active() || after_second.active() || after_third.active();
  }

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmWindowFusion &) const noexcept = default;
};

struct DeviceVsmWindowProof final {
  rund::kernel::WindowPlan semantic{};
  DeviceVsmWindowFusion fusion{};
  DeviceVsmWindowRingPlan ring{};
  DeviceVsmWindowFootprintAuthority footprint{};
  std::uint64_t range_source_hi{};
  std::uint64_t range_source_lo{};
  std::uint64_t range_execution_hi{};
  std::uint64_t range_execution_lo{};
  std::uint32_t workgroup_width{};
  std::uint32_t shared_radius_capacity{};
  RangePath range_path{RangePath::Direct};
  std::uint8_t range_stage_count{1u};
  bool shared_halo{};
  bool mutates_input{};
};

inline constexpr std::size_t DeviceVsmGraphStageCapacity = 8u;
inline constexpr std::size_t DeviceVsmGraphStageInputCapacity = 8u;

enum class DeviceVsmGraphValueSourceKind : std::uint8_t {
  Invalid,
  ExternalInput,
  StageOutput,
};

struct DeviceVsmGraphValueSource final {
  DeviceVsmGraphValueSourceKind kind{DeviceVsmGraphValueSourceKind::Invalid};
  std::uint8_t index{};

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmGraphValueSource &) const noexcept = default;
};

struct DeviceVsmGraphPointwiseStageTopology final {
  std::array<DeviceVsmGraphValueSource, DeviceVsmGraphStageInputCapacity>
      inputs{};
  std::uint8_t input_count{};

  [[nodiscard]] constexpr bool operator==(
      const DeviceVsmGraphPointwiseStageTopology &) const noexcept = default;
};

// Exact stage-port dataflow. External inputs are indexed in public Program
// order; internal inputs name the unique earlier stage whose sole output owns
// that resource. This topology is independent of the ready-wavefront masks:
// the former authenticates values, while the latter also carries physical
// alias/release dependencies.
struct DeviceVsmGraphPointwiseTopology final {
  std::array<DeviceVsmGraphPointwiseStageTopology, DeviceVsmGraphStageCapacity>
      stages{};
  std::uint8_t stage_count{};
  std::uint8_t external_input_count{};

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmGraphPointwiseTopology &) const noexcept = default;
};

// Exact planner-sealed ready topology consumed by the one-dispatch Graph
// controller. Masks use physical stage ordinals; same-batch dependencies are
// satisfied by the current completed mask, while prior masks authenticate the
// preceding batch's Dispatch/Release-complete facts independently.
struct DeviceVsmGraphWavefrontProof final {
  std::array<std::uint8_t, DeviceVsmGraphStageCapacity> same_dispatch{};
  std::array<std::uint8_t, DeviceVsmGraphStageCapacity> same_release{};
  std::array<std::uint8_t, DeviceVsmGraphStageCapacity> prior_dispatch{};
  std::array<std::uint8_t, DeviceVsmGraphStageCapacity> prior_release{};
  // Exact lowered payload owners inside the planner-sealed stage table.
  // Canonical intermediate Maps may be fused into `map_stage`, but their
  // payload cannot execute before the ready traversal selects this owner.
  std::uint32_t map_stage{};
  std::uint32_t collective_stage{};
  std::uint32_t stage_count{};
  std::uint32_t frame_capacity{};
  std::uint32_t batch_count{};

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmGraphWavefrontProof &) const noexcept = default;
};

// Exact executable Graph surface currently sealed by the Virtual planner:
// one pure U64 Map prefix followed by one pure U64 Reduce. One workgroup owns
// the complete page recurrence, so no cross-workgroup reduction or Host epoch
// edge exists. The prepared collective plan is retained as semantic authority.
struct DeviceVsmGraphMapReduceProof final {
  rund::kernel::ReducePlan semantic{};
  DeviceVsmGraphWavefrontProof wavefront{};
  std::uint32_t workgroup_width{};
};

// Exact two-stage U64 Graph Map chain. Both canonical prepared stages remain
// strongly owned, while the generated kernel substitutes the first stage's
// value directly into the second stage's sole Read. The planner wavefront is
// still executed and authenticated; its map owner contains the fused pair and
// its terminal owner publishes the completed batch without an intermediate
// backing or Host epoch edge.
struct DeviceVsmGraphPointwiseProof final {
  DeviceVsmGraphWavefrontProof wavefront{};
  DeviceVsmGraphPointwiseTopology topology{};
  // Planner-sealed external-input page binding.  The request may carry a
  // staged value, but this copy is the sole post-admission authority.
  DeviceVsmPageMap page_map{};
  std::uint32_t workgroup_width{};
  std::uint32_t stage_count{};
};

// One device-owned whole-run unsigned Scan. The prepared page Scan remains
// the semantic authority; this proof changes only the physical schedule from
// Host-updated page carries to one aggregate GPU recurrence.
enum class DeviceVsmScanMapKind : std::uint8_t {
  None,
  AddWrapU64Immediate,
  CanonicalTotalU64,
};

struct DeviceVsmScanMap final {
  DeviceVsmScanMapKind kind{DeviceVsmScanMapKind::None};
  std::uint64_t immediate{};

  [[nodiscard]] constexpr bool active() const noexcept {
    return kind != DeviceVsmScanMapKind::None;
  }

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmScanMap &) const noexcept = default;
};

struct DeviceVsmScanProof final {
  rund::kernel::ScanPlan semantic{};
  // An exact total element-local Map may be fused ahead of Scan. The
  // canonical prepared Map remains strongly owned by semantic_owner; this
  // summary only freezes the small execution choice needed by the generated
  // whole-run Scan source. CanonicalTotalU64 is authenticated by the exact
  // retained artifact, ParsedIR-derived source, and copied parameter bytes;
  // the enum is never a substitute for that authority.
  DeviceVsmScanMap map{};
  std::uint32_t workgroup_width{};
  std::uint32_t stage_count{1u};
};

// One device-owned whole-run unsigned Reduce. The prepared page Reduce remains
// the semantic authority. Sum publishes only after a full-width overflow
// precheck; CountNonzero is bounded by the admitted logical element count;
// Min/Max consume the same exact nonempty extent through identity trees.
struct DeviceVsmReduceProof final {
  rund::kernel::ReducePlan semantic{};
  std::uint32_t workgroup_width{};
};

// One immutable page-coordinate recurrence. Unlike ServiceFreeDirectProof,
// `page_count` counts distinct Virtual backing pages rather than repeated
// applications to one resident state.
struct DeviceVsmProof final {
  DeviceVsmIdentity identity{};
  DeviceVsmTopology topology{DeviceVsmTopology::Pointwise};
  DeviceVsmWindowProof window{};
  DeviceVsmGraphMapReduceProof graph_map_reduce{};
  DeviceVsmGraphPointwiseProof graph_pointwise{};
  DeviceVsmGraphResidentProof graph_resident{};
  // GraphResident reuses the canonical planner wavefront directly; it is
  // stored once here rather than mirrored in the resident proof rows.
  DeviceVsmGraphWavefrontProof graph_wavefront{};
  DeviceVsmScanProof scan{};
  DeviceVsmReduceProof reduce{};
  std::shared_ptr<const void> semantic_owner{};
  const rund::kernel::LoweringArtifact *artifact{};
  rund::kernel::ComputePlan plan{};
  // Canonical input-major/output-major resident set. The fixed capacity is
  // independent of Q and removes the former 1-in/1-out physical authority
  // duplication from common, Authority, and both native backends.
  DeviceVsmResidentSet residents{};
  const rund::kernel::ComputeDispatchWindow *windows{};
  const std::byte *parameters{};
  DeviceVsmPageGeometry geometry{};
  std::uint64_t output_bytes{};
  std::uint64_t window_count{};
  std::uint64_t parameter_bytes{};
  std::uint8_t width{};
  bool fixed_common_storage{};
};

} // namespace rund::node::accel::detail

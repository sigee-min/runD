#pragma once

#include "base.hpp"

#include <rund/compute/graph/info.hpp>
#include <rund/storage.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

namespace rund::compute::detail {

struct BufferState;

struct DeviceState;

// Compute owns the exact physical identity of one sealed Pipeline View.
// Scheduling, private-Job construction, and publication may project this
// record into their own descriptors, but may not reconstruct it from authored
// bindings.
struct PipelinePublicationViewIdentity final {
  std::uint64_t backing_bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t count{};
  std::uint64_t stride_bytes{};
  std::uint64_t element_bytes{};
  std::uint32_t resource_ordinal{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t usage{};

  [[nodiscard]] constexpr bool
  operator==(const PipelinePublicationViewIdentity &) const noexcept = default;
};

// One constructor-closed physical Pipeline View. Type/FixedFormat retain the
// semantic contract that byte geometry alone cannot distinguish. Element and
// byte coordinates are sealed together once so downstream adapters perform no
// independent arithmetic.
struct PipelinePublicationViewPlan final {
  PipelinePublicationViewIdentity identity{};
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t offset{};
  std::size_t stride{1u};
  std::uint64_t alignment{};

  [[nodiscard]] constexpr bool
  operator==(const PipelinePublicationViewPlan &) const noexcept = default;
};

// Publication targets are caller-owned and may be absent from every Program
// binding. Their owner is held only by the canonical resolved-resource table;
// this record contains publication-specific View meaning only.
struct PipelinePublicationTargetPlan final {
  PipelinePublicationViewPlan view{};
};

enum class PipelinePublicationKind : std::uint8_t {
  Terminal,
  Window,
};

struct PipelineTerminalPublicationPlan final {
  std::array<PipelinePublicationViewPlan, 3u> sources{};
  PipelinePublicationTargetPlan target{};
  std::uint32_t state{std::numeric_limits<std::uint32_t>::max()};
  PipelinePhysicalOutputOrdinal output{};
};

struct PipelineWindowPublicationPlan final {
  PipelinePublicationViewPlan source{};
  PipelinePublicationTargetPlan target{};
  std::uint32_t state{std::numeric_limits<std::uint32_t>::max()};
  PipelinePhysicalOutputOrdinal output{};
};

using PipelinePublicationPlan = std::variant<PipelineTerminalPublicationPlan,
                                             PipelineWindowPublicationPlan>;

[[nodiscard]] inline constexpr PipelinePublicationKind
pipeline_publication_kind(const PipelinePublicationPlan &publication) noexcept {
  return std::holds_alternative<PipelineWindowPublicationPlan>(publication)
             ? PipelinePublicationKind::Window
             : PipelinePublicationKind::Terminal;
}

[[nodiscard]] inline const PipelinePublicationTargetPlan &
pipeline_publication_target(const PipelinePublicationPlan &publication) {
  return std::visit(
      [](const auto &typed) -> const PipelinePublicationTargetPlan & {
        return typed.target;
      },
      publication);
}

[[nodiscard]] inline PipelinePublicationTargetPlan &
pipeline_publication_target(PipelinePublicationPlan &publication) {
  return std::visit(
      [](auto &typed) -> PipelinePublicationTargetPlan & {
        return typed.target;
      },
      publication);
}

struct PipelineStatePair final {
  std::shared_ptr<BufferState> first;
  std::shared_ptr<BufferState> second;
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t count{};
  std::size_t bytes{};
};

struct PipelineSnapshotField final {
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t count{};
  std::size_t offset{};
  std::size_t bytes{};
  // Produced by the copy/download that materializes this field.  Snapshot
  // identity consumes the retained leaf hash instead of walking the complete
  // payload a second time immediately after that copy.
  std::uint64_t payload_hash{};
};

struct StateSnapshotState final {
  graph::Fingerprint fingerprint{};
  std::vector<PipelineSnapshotField> fields;
  // Snapshot construction overwrites the complete payload exactly once.
  // Keep raw owned storage so allocation does not first zero every byte.
  std::unique_ptr<std::byte[]> bytes;
  std::size_t byte_count{};
  std::uint64_t generation{};
  std::uint64_t hash{};
};

struct SnapshotStorageState final {
  mutable std::mutex gate;
  std::array<StateSnapshotState, 2u> banks;
  std::size_t byte_capacity{};
  std::size_t field_capacity{};
  std::uint8_t active{};
  bool valid{};
};

// Resident checkpoint authority shared by Pipeline and LatestDeviceState.
// Frozen owners and schema survive Pipeline destruction, while the publication
// selector changes only in the terminal Device-claim critical section.
struct PipelinePublicationState final {
  std::shared_ptr<DeviceState> device;
  // Declared before publication resources so reverse destruction releases the
  // aggregate Device charge only after every resident owner is gone.
  storage::Reservation publication_memory;
  std::vector<PipelineStatePair> state_pairs;
  graph::Fingerprint fingerprint{};
  mutable std::mutex gate;
  std::uint64_t generation{};
  // Changes whenever this authority publishes different resident payload,
  // including a restore whose generation/parity happen to stay unchanged.
  std::uint64_t payload_epoch{};
  std::uint8_t parity{};
  bool device_lost{};
  // At most one Pipeline may execute against shared pair owners. The attempt
  // reservation freezes selector identity without holding this mutex across
  // an asynchronous backend submission.
  bool attempt_active{};
};

} // namespace rund::compute::detail

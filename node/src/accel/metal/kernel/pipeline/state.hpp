#pragma once

#include "../trace.hpp"
#include "abi.hpp"
#include "capture.hpp"
#include "prepare/spatial_window/proof.hpp"
#include "residency/model.hpp"

#include <accel/check.hpp>

#include "../../../kernel/submission.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

static_assert(sizeof(MetalPipelineStatusBindingRecord) == 96u);

struct MetalPublish final {
  std::array<std::shared_ptr<void>, 3u> sources;
  std::shared_ptr<void> target;
  std::shared_ptr<void> count;
  MetalPublishParams params{};
};

struct MetalWindow final {
  std::shared_ptr<void> resident;
  std::array<std::shared_ptr<void>, 3u> terminals;
  MetalWindowParams params{};
  std::uint32_t entry{};
};

struct MetalPipelineTelemetryRecord final {
  MetalPipelineTelemetrySource source{};
  std::shared_ptr<void> owner;
};

// Cold finalization freezes the only two host inputs needed to submit a warm
// tick. `resources` is one bulk-residency slice; `chunks` is one compact fixed
// record slice. Neither descriptors nor logical-step tables are reachable
// from this structure.
struct MetalWarmSubmission final {
  const id<MTLResource> *resources{};
  NSUInteger resource_count{};
  const MetalIcbChunk *chunks{};
  NSUInteger chunk_count{};

  [[nodiscard]] bool
  owns(const std::vector<id<MTLResource>> &residency,
       const std::vector<MetalIcbChunk> &command_chunks) const noexcept {
    const id<MTLResource> *const expected =
        residency.empty() ? nullptr : residency.data();
    const MetalIcbChunk *const expected_chunks =
        command_chunks.empty() ? nullptr : command_chunks.data();
    return resources == expected && resource_count == residency.size() &&
           chunks == expected_chunks && chunk_count == command_chunks.size();
  }

  // Cold-only integrity gate. Warm execution uses owns() for its constant-time
  // retained-slice check and validates each compact record while issuing it;
  // it never performs a second summation pass over the chunk slice.
  [[nodiscard]] bool matches(const std::vector<id<MTLResource>> &residency,
                             const std::vector<MetalIcbChunk> &command_chunks,
                             const NSUInteger command_count) const noexcept {
    if (!owns(residency, command_chunks)) {
      return false;
    }
    std::uint64_t observed_commands = 0u;
    for (std::size_t index = 0u; index < command_chunks.size(); ++index) {
      const MetalIcbChunk &chunk = command_chunks[index];
      if (!chunk.valid() || (index == 0u && chunk.barrier_before()) ||
          observed_commands >
              std::numeric_limits<std::uint64_t>::max() - chunk.command_count) {
        return false;
      }
      observed_commands += chunk.command_count;
    }
    return observed_commands == command_count;
  }
};

// One cold-frozen slice of the retained ICB command stream for a declared
// physical Pipeline step. Authority selects these rows by exact frame-local
// ordinal; the row contains no residency or replacement policy.
struct MetalResidencyStepRange final {
  std::uint64_t begin{};
  std::uint64_t count{};
  std::uint64_t dispatch_count{};
  std::uint64_t control_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};

  [[nodiscard]] bool valid() const noexcept { return count != 0u; }
};

struct MetalSequence final {
  ~MetalSequence();

  MetalAdapter *adapter{};
  std::vector<id<MTLResource>> residency;
  std::vector<id<MTLComputePipelineState>> pipelines;
  std::vector<MetalIcbChunk> command_chunks;
  std::vector<MetalResidencyStepRange> residency_steps;
  std::vector<std::uint64_t> trace_commands;
  std::vector<MetalPipelineTelemetryRecord> telemetry;
  std::vector<PreparedPipelineStepEvidence> step_evidence;
  std::shared_ptr<void> recurrence;
  std::vector<std::shared_ptr<void>> transducers;
  id<MTLBuffer> parameters = nil;
  id<MTLBuffer> raw_status = nil;
  id<MTLBuffer> control = nil;
  id<MTLBuffer> states = nil;
  id<MTLBuffer> guard_zero = nil;
  id<MTLBuffer> step_control = nil;
  MetalDispatchTrace trace{};
  MetalResidencySubmission residency_submission{};
  MetalResidencySlidingGate residency_sliding{};
  MetalResidencyWindow residency_window{};
  // A whole-run Schedule is cold-lowered independently from the reusable W4
  // owner. The weak link lets signal/abort resolve the one active lowering
  // without giving Pipeline state ownership of caller callbacks.
  std::weak_ptr<void> residency_schedule{};
  PreparedPipelineMemoryMeter *memory_meter{};
  MetalWarmSubmission warm{};
  NSUInteger command_count = 0u;
  MetalResidencyStepRange residency_prefix{};
  MetalResidencyStepRange residency_suffix{};
  std::uint32_t control_command_count{};
  std::uint32_t control_generation_stride{1u};
  std::uint32_t state_count{};
  std::uint64_t retained_bytes{};
  std::uint64_t dispatch_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
  std::uint64_t instrumentation_byte_count{};
  bool uses_status_arena{};
  bool profile_steps{};
  bool direct_aggregate{};
  bool residency_selectable{};
  MetalSpatialWindowProof spatial_window{};
  bool persistent_spatial_window_selectable{};
  submission::State<MetalSequence> submission{};

  [[nodiscard]] bool spatial_window_proof_valid() const noexcept;
};

[[nodiscard]] bool ValidMetalSequence(const MetalSequence *sequence) noexcept;
[[nodiscard]] bool EmptyMetalSequence(const MetalSequence &sequence) noexcept;

#endif

} // namespace rund::node::accel::detail

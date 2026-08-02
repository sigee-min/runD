#pragma once

#include "capture.hpp"

#include <accel/check.hpp>

#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/submission.hpp"
#include "../../runtime/map/resources.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

constexpr std::uint32_t kMetalPipelineReductionWidth = 128u;
static_assert(kMetalPipelineReductionWidth == 128u);

struct MetalPipelineResetMeta final {
  std::uint32_t raw_offset{};
  std::uint32_t reset{};
};

struct MetalPipelineStatusSourceMeta final {
  std::uint32_t encoding{};
  std::uint32_t declared_step{};
  std::array<std::uint32_t, 4u> policies{};
  std::uint32_t limit_low{};
  std::uint32_t limit_high{};
  std::uint32_t raw_offset{};
  std::uint32_t telemetry{};
  std::uint32_t indirect_dispatch_count{};
  std::uint32_t work_item_count_low{};
  std::uint32_t work_item_count_high{};
  std::uint32_t failed_outer_window{PreparedPipelineNoStep};
  std::uint32_t failed_inner_iteration{PreparedPipelineNoStep};
  std::uint32_t failed_nested_phase{};
};

struct MetalPipelineStatusEntryMeta final {
  std::uint32_t source{};
  std::uint32_t raw{};
};

struct MetalPipelineStatusParams final {
  std::uint32_t reset_range_count{};
  std::uint32_t status_count{};
  std::uint32_t reset_word_count{};
  std::uint32_t declared_step_count{};
  std::uint32_t invalid_reason{};
  std::uint32_t generation_stride{};
  std::uint32_t source_count{};
  std::uint32_t phase{};
  // A nested status fold closes its ResidentState in the same dispatch that
  // selects the first failure. Non-nested folds retain the invalid sentinel.
  std::uint32_t window_state{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t window_stop{};
  // A failing Fold has already completed its Action body. The reducer records
  // that proved work before closing the resident route; Seed/Action use zero.
  std::uint32_t window_inner_advance{};
  // The prepared open dispatch resets this exact ResidentState prefix before
  // any recurrence-owned command can observe it.
  std::uint32_t state_count{};
};

struct MetalPipelineTelemetryParams final {
  std::uint32_t kind{};
  std::uint32_t primary_word_count{};
  std::uint32_t count_source{};
  std::uint32_t predicate_source{};
  std::uint32_t has_count{};
  std::uint32_t has_predicate{};
  std::uint32_t iteration{};
  std::uint32_t count_word_offset{};
  std::uint32_t predicate_word_offset{};
  std::uint32_t indirect_dispatch_count{};
  std::uint32_t declared_step_count{};
  std::uint32_t declared_step{};
  std::uint64_t capacity{};
  std::uint64_t predicate_expected{};
  std::uint64_t work_item_count{};
};

static_assert(sizeof(MetalPipelineStatusSourceMeta) == 64u);
static_assert(sizeof(MetalPipelineStatusEntryMeta) == 8u);
static_assert(sizeof(MetalPipelineStatusParams) == 48u);
static_assert(sizeof(MetalPipelineResetMeta) == 8u);
static_assert(sizeof(MetalPipelineStatusBindingRecord) == 96u);
static_assert(sizeof(MetalPipelineTelemetryParams) == 72u);

struct MetalPublishParams final {
  std::uint64_t count{};
  std::array<std::uint64_t, 3u> source_offset_words{};
  std::array<std::uint64_t, 3u> source_stride_words{};
  std::uint64_t target_offset_words{};
  std::uint64_t target_stride_words{};
  std::uint32_t element_words{};
  std::uint32_t declared_step_count{};
  std::uint32_t state{};
  std::uint32_t final{};
  std::uint32_t stop{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t outer{};
  std::uint32_t kind{};
  std::uint64_t count_offset_words{};
};

static_assert(sizeof(MetalPublishParams) == 120u);

struct MetalPublish final {
  std::array<std::shared_ptr<void>, 3u> sources;
  std::shared_ptr<void> target;
  std::shared_ptr<void> count;
  MetalPublishParams params{};
};

struct MetalWindowParams final {
  std::uint64_t count_offset_words{};
  std::array<std::uint64_t, 3u> terminal_offset_words{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t iteration{};
  std::uint32_t expected{};
  std::uint32_t state{};
  std::uint32_t has_terminal{};
  std::uint32_t phase{};
  std::uint32_t declared_step{};
  std::uint32_t overflow_reason{};
  std::uint32_t inner_bound{1u};
  std::uint32_t inner_advance{};
};

static_assert(sizeof(MetalWindowParams) == 80u);

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

struct MetalSequence final {
  MetalAdapter *adapter{};
  std::vector<id<MTLResource>> residency;
  std::vector<id<MTLComputePipelineState>> pipelines;
  std::vector<MetalIcbChunk> command_chunks;
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
  MetalWarmSubmission warm{};
  NSUInteger command_count = 0u;
  std::uint32_t control_command_count{};
  std::uint32_t state_count{};
  std::uint64_t retained_bytes{};
  std::uint64_t dispatch_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
  std::uint64_t instrumentation_byte_count{};
  bool uses_status_arena{};
  bool profile_steps{};
  bool direct_aggregate{};
  submission::State<MetalSequence> submission{};
};

[[nodiscard]] bool ValidMetalSequence(const MetalSequence *sequence) noexcept;
[[nodiscard]] bool EmptyMetalSequence(const MetalSequence &sequence) noexcept;

#endif

} // namespace rund::node::accel::detail

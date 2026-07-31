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
};

struct MetalPipelineStatusBindingRecord final {
  MetalPipelineStatusBinding binding{};
  std::uint32_t raw_offset{};
  std::uint32_t raw_count{};
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

static_assert(sizeof(MetalPipelineStatusSourceMeta) == 52u);
static_assert(sizeof(MetalPipelineStatusEntryMeta) == 8u);
static_assert(sizeof(MetalPipelineStatusParams) == 32u);
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
};

static_assert(sizeof(MetalPublishParams) == 96u);

struct MetalPublish final {
  std::array<std::shared_ptr<void>, 3u> sources;
  std::shared_ptr<void> target;
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
  std::uint32_t range_count{};
  std::uint32_t reserved{};
};

static_assert(sizeof(MetalWindowParams) == 64u);

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

struct MetalRange final {
  std::uint32_t location{};
  std::uint32_t length{};
};

static_assert(sizeof(MetalRange) == 8u);

struct MetalRangePlan final {
  MetalRange range{};
  std::uint32_t owner{std::numeric_limits<std::uint32_t>::max()};
  bool barrier{};
};

struct MetalSequence final {
  MetalAdapter *adapter{};
  std::vector<id<MTLResource>> declared;
  std::vector<id<MTLComputePipelineState>> pipelines;
  std::vector<MetalCommand> direct;
  std::vector<MetalRangePlan> ranges;
  std::vector<MetalRange> original_ranges;
  std::vector<MetalCommandBinding> indirect_bindings;
  std::vector<MetalThreadgroupBinding> indirect_threadgroups;
  std::vector<MetalPipelineTelemetryRecord> telemetry;
  std::vector<PreparedPipelineStepEvidence> step_evidence;
  std::shared_ptr<void> recurrence;
  id<MTLIndirectCommandBuffer> commands = nil;
  id<MTLBuffer> parameters = nil;
  id<MTLBuffer> raw_status = nil;
  id<MTLBuffer> control = nil;
  id<MTLBuffer> states = nil;
  id<MTLBuffer> gate_buffer = nil;
  id<MTLBuffer> range_buffer = nil;
  id<MTLBuffer> range_owners = nil;
  id<MTLBuffer> step_control = nil;
  NSUInteger command_count = 0u;
  std::uint32_t control_command_count{};
  std::uint32_t gate_count{};
  std::uint32_t state_count{};
  std::uint32_t range_capacity{};
  std::uint64_t retained_bytes{};
  std::uint64_t dispatch_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
  std::uint64_t instrumentation_byte_count{};
  bool uses_status_arena{};
  bool profile_steps{};
  id<MTLComputePipelineState> gate = nil;
  submission::State<MetalSequence> submission{};
};

[[nodiscard]] bool ValidMetalSequence(const MetalSequence *sequence) noexcept;
[[nodiscard]] bool EmptyMetalSequence(const MetalSequence &sequence) noexcept;

#endif

} // namespace rund::node::accel::detail

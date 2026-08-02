#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../kernel/memory.hpp"
#include "../../kernel.hpp"
#include "../../state.hpp"

#include <array>
#include <cstdint>
#include <memory>

#include <kernel/program/compute/graph/schema.hpp>

namespace rund::node::accel::detail {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalKernelImmutablePipelines;
using MetalPrepareStepFn =
    rund::AccelCheck (*)(const rund::AccelDevice &pick, const BoundStep &step,
                         const MetalKernelImmutablePipelines *pipelines,
                         std::shared_ptr<void> &resources);
using MetalEncodeStepFn =
    rund::AccelCheck (*)(MetalAdapter &adapter,
                         const std::shared_ptr<void> &resources, void *encoder);
using MetalFinishStepFn = rund::AccelCheck (*)(
    MetalAdapter &adapter, const std::shared_ptr<void> &resources);
using MetalMemoryStepFn = PreparedMemory (*)(
    const std::shared_ptr<void> &resources, std::uint64_t budget);

enum class MetalPipelineStatusEncoding : std::uint32_t {
  Nonzero,
  BitFlags,
  SegmentedScan,
  SegmentedReduce,
  Sentinel,
  Scatter,
  Limit,
  Mapping,
};

enum class MetalPipelineStatusTelemetry : std::uint32_t {
  None,
  Gather,
  IndexedControl,
};

struct MetalPipelineStatusBinding final {
  void *buffer = nullptr;
  std::uint64_t offset = 0u;
  std::uint64_t bytes = 0u;
  std::uint64_t limit = 0u;
  std::uint64_t work_item_count = 0u;
  std::array<std::uint32_t, 4u> reasons{};
  std::uint32_t reset = 0u;
  std::uint32_t observed = 0u;
  std::uint32_t observed_count = 1u;
  std::uint32_t indirect_dispatch_count = 0u;
  MetalPipelineStatusEncoding encoding = MetalPipelineStatusEncoding::Nonzero;
  MetalPipelineStatusTelemetry telemetry = MetalPipelineStatusTelemetry::None;
  bool replace = true;
};

inline constexpr std::size_t kMetalPipelineStatusBindingCapacity = 2u;

struct MetalPipelineStatusBindings final {
  std::array<MetalPipelineStatusBinding, kMetalPipelineStatusBindingCapacity>
      values{};
  std::size_t size = 0u;
};

using MetalPipelineStatusStepFn =
    bool (*)(const std::shared_ptr<void> &resources,
             MetalPipelineStatusBindings &bindings) noexcept;

enum class MetalPipelineTelemetryKind : std::uint8_t {
  None,
  ControlledMap,
  IndexedControl,
  ControlledCollective,
  GatherControl,
};

// Pipeline-private primitives cannot run their standalone Finish function:
// doing so would introduce one host wait per Program.  This descriptor keeps
// the small device-authored telemetry sources alive and lets the one terminal
// completion observation account for actual indirect work and conflicts.
struct MetalPipelineTelemetrySource final {
  MetalPipelineTelemetryKind kind{MetalPipelineTelemetryKind::None};
  void *primary_buffer{};
  void *count_buffer{};
  void *predicate_buffer{};
  rund::kernel::GraphControl control{};
  std::uint64_t count_offset{};
  std::uint64_t predicate_offset{};
  std::uint64_t capacity{};
  std::uint64_t work_item_count{};
  std::uint32_t primary_word_count{};
  std::uint32_t indirect_dispatch_count{};
};

using MetalPipelineTelemetryStepFn =
    bool (*)(const std::shared_ptr<void> &resources,
             MetalPipelineTelemetrySource &source) noexcept;
using MetalFailureFn = bool (*)(const std::shared_ptr<void> &resources,
                                std::uint64_t &ordinal) noexcept;

struct MetalKernelOps {
  MetalPrepareStepFn prepare = nullptr;
  MetalEncodeStepFn encode = nullptr;
  MetalFinishStepFn finish = nullptr;
  MetalMemoryStepFn memory = nullptr;
  MetalPipelineStatusStepFn pipeline_status = nullptr;
  MetalPipelineTelemetryStepFn pipeline_telemetry = nullptr;
  MetalFailureFn failure = nullptr;
};

#endif
} // namespace rund::node::accel::detail

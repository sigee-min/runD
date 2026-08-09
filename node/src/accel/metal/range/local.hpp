#pragma once

#include "../../range_aggregate/execution.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"
#include "api.hpp"
#include <accel/check.hpp>
#include <kernel/program/compute/graph/schema.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace rund::node::accel::detail {

struct MetalKernelImmutablePipelines;
struct BoundControl;

struct MetalRangeLimits final {
  rund::kernel::u32 maximum_workgroup_width{};
  rund::kernel::u64 static_shared_bytes{};
  rund::kernel::u64 shared_memory_limit{};
};

enum class MetalRangeSupport : std::uint8_t {
  Supported,
  Unsupported,
  Invalid,
};

[[nodiscard]] constexpr MetalRangeSupport
MetalRangeSupports(const RangeExec &execution,
                   const MetalRangeLimits limits) noexcept {
  if (execution.element_bytes() != 4u && execution.element_bytes() != 8u) {
    return MetalRangeSupport::Invalid;
  }
  if (limits.maximum_workgroup_width < execution.shape().width()) {
    return MetalRangeSupport::Unsupported;
  }
  const rund::kernel::u64 required_bytes = execution.static_shared_bytes();
  if (required_bytes == 0u) {
    return limits.static_shared_bytes == 0u ? MetalRangeSupport::Supported
                                            : MetalRangeSupport::Invalid;
  }
  if (limits.static_shared_bytes < required_bytes) {
    return MetalRangeSupport::Invalid;
  }
  return limits.static_shared_bytes <=
                 limits.shared_memory_limit / kRangeSharedReserve
             ? MetalRangeSupport::Supported
             : MetalRangeSupport::Unsupported;
}

enum class MetalRangeAttemptStatus : std::uint8_t {
  Ready,
  Unsupported,
  Failed,
};

struct MetalRangeAttempt final {
  MetalRangeAttemptStatus status{MetalRangeAttemptStatus::Failed};
  const char *reason{"accel_metal_pipeline_unavailable"};
  std::uint64_t create_ns{};
};

// Primitive adapters own public binding semantics.  The generic range path
// accepts exactly one authenticated read binding and one write binding.
class MetalRangeBinds final {
public:
  MetalRangeBinds() = delete;

  [[nodiscard]] static std::optional<MetalRangeBinds>
  make(MetalResidentBufferResult input, MetalResidentBufferResult output) {
    if (!input.check.ok || !output.check.ok || input.device_buffer == nullptr ||
        output.device_buffer == nullptr) {
      return std::nullopt;
    }
    return MetalRangeBinds{std::move(input), std::move(output)};
  }

  [[nodiscard]] const MetalResidentBufferResult &input() const noexcept {
    return input_;
  }

  [[nodiscard]] const MetalResidentBufferResult &output() const noexcept {
    return output_;
  }

private:
  MetalRangeBinds(MetalResidentBufferResult input,
                  MetalResidentBufferResult output)
      : input_(std::move(input)), output_(std::move(output)) {}

  MetalResidentBufferResult input_;
  MetalResidentBufferResult output_;
};

struct MetalRangeResources {
  MetalAdapter *adapter = nullptr;
  RangePlan range{RangePlan::rejected("compute_range_aggregate_unavailable")};
  MetalResidentBufferResult input{};
  MetalResidentBufferResult output{};
  std::array<std::shared_ptr<void>, kRangeStageCap> pipelines{};
  std::array<MetalRuntimeBuffer, kRangeTempCap> temporaries{};
  MetalResidentBufferResult control_count{};
  MetalRuntimeBuffer control_params{};
  MetalRuntimeBuffer control_indirect{};
  MetalRuntimeBuffer control_status{};
  std::shared_ptr<void> control_pipeline{};
  rund::kernel::GraphControl control{};
  std::uint32_t stage_count{};
  bool controlled{};
  bool indirect{};
};

void DestroyMetalRangeResources(void *raw);
[[nodiscard]] std::string MetalRangeSource(const RangeExec &execution);
[[nodiscard]] bool MetalRangeSourceUpperBytes(const RangeExec &execution,
                                              std::uint64_t &upper) noexcept;
[[nodiscard]] std::string MetalRangeControlSource(const RangePlan &plan);
[[nodiscard]] bool
MetalRangeControlSourceUpperBytes(const RangePlan &plan,
                                  std::uint64_t &upper) noexcept;
[[nodiscard]] MetalRangeAttempt CompileMetalRange(MetalAdapter &adapter,
                                                  const RangeExec &execution,
                                                  std::shared_ptr<void> &out);
[[nodiscard]] rund::AccelCheck
PrepareMetalRange(const rund::AccelDevice &pick, const RangePlan &range,
                  const MetalRangeBinds &bindings, const BoundControl *control,
                  std::shared_ptr<void> &resources,
                  const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeMetalRange(MetalAdapter &adapter, const std::shared_ptr<void> &resources,
                 void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalRange(MetalAdapter &adapter, const std::shared_ptr<void> &resources);
[[nodiscard]] bool
MetalRangeScratch(const MetalRangeResources &resources,
                  std::uint32_t stage_index,
                  const MetalRuntimeBuffer *&scratch0,
                  const MetalRuntimeBuffer *&scratch1) noexcept;

} // namespace rund::node::accel::detail

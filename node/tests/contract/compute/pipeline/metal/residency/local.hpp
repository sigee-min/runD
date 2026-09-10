#pragma once

#include "../../../allocation.hpp"
#include "../../local.hpp"

#include <rund/compute/virtual.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

#include "src/accel/kernel/fault.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/metal/kernel.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/device/residency/execution/owner.hpp"
#include "src/compute/device/residency/execution/plan.hpp"
#include "src/compute/device/residency/pool.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/pipeline/execution/schedule.hpp"
#include "src/compute/pipeline/execution/submit.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <node/runtime/compute/access.hpp>

#endif

namespace rund_node_test_pipeline {

class AdmissionBacking final : public rund::compute::VirtualBacking {
public:
  explicit AdmissionBacking(std::size_t bytes);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;

  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;

  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;

  [[nodiscard]] std::uint64_t callbacks() const noexcept;

  void fail_next_read() noexcept;

  [[nodiscard]] bool all_i32(std::int32_t expected) const noexcept;

private:
  std::vector<std::byte> bytes_;
  std::atomic_bool fail_read_;
  std::uint64_t reads_;
  std::uint64_t writes_;
};

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace execution = rund::compute::detail::residency::execution;
namespace residency = rund::compute::detail::residency;

class ScheduleRouteScope final {
public:
  explicit ScheduleRouteScope(
      rund::compute::detail::DeviceState &device) noexcept;
  ScheduleRouteScope(const ScheduleRouteScope &) = delete;
  ScheduleRouteScope &operator=(const ScheduleRouteScope &) = delete;
  ~ScheduleRouteScope();

  [[nodiscard]] explicit operator bool() const noexcept;

private:
  rund::compute::detail::DeviceState &device_;
  const rund::compute::detail::DeviceOps *original_;
  rund::compute::detail::DeviceOps routed_;
  bool active_;
};

struct NativeWait final {
  NativeWait() noexcept;
  std::atomic_bool done;
  execution::NativeEvidence evidence;
};

struct WindowWait final {
  WindowWait() noexcept;
  std::atomic_bool done;
  std::atomic_bool valid;
  std::atomic_bool wrong_rejected;
  std::atomic<std::size_t> release_count;
  rund::AccelContext context;
  std::array<rund::node::accel::detail::PreparedKernelPipeline, 2u> pipelines;
  std::uint64_t plan_identity;
  std::uint64_t token;
  std::uint64_t generation;
  std::uint32_t control_base;
  std::size_t suppressed;
  bool terminal_loss;
  bool aborted;
  rund::node::accel::detail::BackendResidencyWindowFinal final;
};

void CompleteNative(void *, execution::NativeEvidence &&) noexcept;
void CompleteWindowRelease(
    void *,
    rund::node::accel::detail::PreparedResidencyWindowRelease &&) noexcept;
void CompleteWindowFinal(
    void *,
    rund::node::accel::detail::PreparedResidencyWindowFinal &&) noexcept;

[[nodiscard]] rund::AccelCheck RejectPipelineExecution(
    const rund::compute::detail::DeviceState &,
    const rund::node::accel::detail::PreparedKernelPipeline &,
    std::span<const std::uint32_t>, std::shared_ptr<void>,
    rund::node::accel::detail::PreparedPipelineCompletion, void *) noexcept;

[[nodiscard]] rund::AccelCheck TamperPipelineExecution(
    const rund::compute::detail::DeviceState &,
    const rund::node::accel::detail::PreparedKernelPipeline &,
    std::span<const std::uint32_t>, std::shared_ptr<void>,
    rund::node::accel::detail::PreparedPipelineCompletion, void *) noexcept;

[[nodiscard]] execution::SealResult
ExecutionPlan(const residency::Pool &, std::uint64_t page_count = 1u) noexcept;

[[nodiscard]] bool SameCurrent(const rund::compute::MemoryStats &,
                               const rund::compute::MemoryStats &);

[[nodiscard]] bool AddedSubmissionBytes(const rund::compute::PipelinePlan &,
                                        const rund::compute::PipelinePlan &,
                                        std::uint64_t bytes);

[[nodiscard]] int CheckTransactionalStageRollback();
[[nodiscard]] int CheckVirtualAdmissionRollback();
[[nodiscard]] int CheckExecutionAdapter();
[[nodiscard]] int CheckSingleExecutionAdapter(
    const std::shared_ptr<rund::compute::detail::PipelineState> &,
    execution::Owner &, const execution::SealResult &);
[[nodiscard]] int CheckStagedLoopExecution(std::size_t epochs);
[[nodiscard]] int CheckSelectedNativeTerminalAndDeviceLoss();
[[nodiscard]] int CheckQueueTerminalLossQuarantine();
[[nodiscard]] int CheckScheduleTerminalLossQuarantine();
[[nodiscard]] int CheckScheduleBudgetFallback();
[[nodiscard]] int CheckScheduleKnownFailureRetry();

#endif

} // namespace rund_node_test_pipeline

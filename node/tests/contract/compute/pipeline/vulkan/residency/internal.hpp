#pragma once

#include "../../local.hpp"
#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "src/accel/backend/resource.hpp"
#include "src/accel/backend/token.hpp"
#include "src/accel/backend/usage.hpp"
#include "src/accel/kernel/fault.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/vulkan/adapter/state.hpp"
#include "src/accel/vulkan/buffer/resident/find.hpp"
#include "src/accel/vulkan/kernel.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/local.hpp"
#include "src/accel/vulkan/kernel/pipeline/state.hpp"
#include "src/accel/vulkan/resident/access.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace rund_node_test_pipeline_vulkan_residency {

namespace accel = rund::node::accel::detail;

using U32Program = rund::compute::Program<std::uint32_t(std::uint32_t)>;

struct WindowWait final {
  std::atomic_bool done{false};
  std::atomic_bool valid{true};
  std::atomic_size_t releases{};
  rund::AccelContext context{};
  std::array<accel::PreparedKernelPipeline, 2u> pipelines{};
  std::uint64_t plan{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t first_epoch{};
  std::uint32_t control{};
  std::size_t count{};
  std::size_t suppressed{std::numeric_limits<std::size_t>::max()};
  bool aborted{};
  accel::BackendResidencyWindowFinal final{};
};

struct ScheduleWait final {
  std::atomic_bool done{false};
  std::atomic_bool valid{true};
  std::atomic_uint64_t releases{};
  rund::AccelContext context{};
  std::array<accel::PreparedKernelPipeline, 2u> pipelines{};
  std::array<std::uint32_t, 4u> first_generation{};
  std::uint32_t generation_stride{};
  std::uint64_t plan{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t epoch_count{};
  accel::BackendResidencyScheduleFinal final{};
};

struct ResidencyFixture final {
  std::unique_ptr<rund::compute::Device> device;
  std::unique_ptr<U32Program> program;
  std::shared_ptr<rund_node_test_pipeline::WindowBacking> input_backing;
  std::shared_ptr<rund_node_test_pipeline::WindowBacking> output_backing;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
  std::shared_ptr<rund::compute::detail::PipelineState> primary;
  std::shared_ptr<rund::compute::detail::PipelineState> alternate;
  rund::compute::detail::AccelDeviceState *native{};
  WindowWait wait{};
  accel::PreparedResidencyWindowControl control{};
  bool skipped{};
};

[[nodiscard]] std::uint32_t
ScheduleControlGeneration(const ScheduleWait &wait,
                          std::uint64_t epoch) noexcept;

void CompleteScheduleRelease(
    void *raw, accel::PreparedResidencyScheduleRelease &&release) noexcept;

void CompleteScheduleFinal(
    void *raw, accel::PreparedResidencyScheduleFinal &&final) noexcept;

void CompleteRelease(void *raw,
                     accel::PreparedResidencyWindowRelease &&release) noexcept;

void CompleteFinal(void *raw,
                   accel::PreparedResidencyWindowFinal &&final) noexcept;

[[nodiscard]] int InitializeResidencyFixture(ResidencyFixture &fixture);
[[nodiscard]] int CheckWindow(ResidencyFixture &fixture);
[[nodiscard]] int CheckSchedule(ResidencyFixture &fixture);
[[nodiscard]] int CheckWarm(ResidencyFixture &fixture);
[[nodiscard]] int CheckAbort(ResidencyFixture &fixture);

} // namespace rund_node_test_pipeline_vulkan_residency

#endif

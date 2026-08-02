#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "src/accel/metal/kernel/template_memory.hpp"
#include "src/accel/metal/buffer/owner.hpp"
#include "src/accel/metal/kernel/local.hpp"
#include "src/accel/metal/kernel/pipeline/build.hpp"
#include "src/accel/metal/kernel/pipeline/icb.hpp"

#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>

namespace rund::node::accel::detail {
[[nodiscard]] rund::AccelDevice PickMetal();
} // namespace rund::node::accel::detail
#endif

namespace node_accel_contract {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool ExactMetalIcbDescriptorContract() {
  using namespace rund::node::accel::detail;
  @autoreleasepool {
    MTLIndirectCommandBufferDescriptor *const descriptor =
        MakeMetalPipelineIcbDescriptor();
    if (descriptor == nil ||
        descriptor.commandTypes !=
            (MTLIndirectCommandTypeConcurrentDispatch |
             MTLIndirectCommandTypeConcurrentDispatchThreads) ||
        descriptor.inheritPipelineState != NO ||
        descriptor.inheritBuffers != NO ||
        descriptor.maxKernelBufferBindCount !=
            MetalPipelineIcbBufferBindLimit) {
      return false;
    }
    if (@available(macOS 14.0, iOS 17.0, *)) {
      if (descriptor.maxKernelThreadgroupMemoryBindCount != 0u) {
        return false;
      }
    }
    return true;
  }
}

} // namespace

#endif

[[nodiscard]] bool MetalTemplateMemoryContract() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;

  const auto make_map = [] {
    auto prepared = std::make_shared<MetalMapTemplateResources>();
    prepared->input_plans.reserve(5u);
    prepared->input_plans.resize(2u);
    prepared->input_strides.reserve(4u);
    prepared->input_strides.resize(2u);
    prepared->output_strides.reserve(3u);
    prepared->output_strides.resize(1u);
    prepared->checks.reserve(6u);
    prepared->checks.resize(1u);
    // These model opaque MTLLibrary/MTLComputePipelineState targets. Registry
    // host telemetry must not invent bytes below their shared_ptr handles.
    prepared->pipeline = std::make_shared<std::uint64_t>(11u);
    prepared->control_pipeline = std::make_shared<std::uint64_t>(13u);
    prepared->check_pipeline = std::make_shared<std::uint64_t>(17u);
    return prepared;
  };
  const auto map_bytes = [](const MetalMapTemplateResources &prepared) {
    return static_cast<std::uint64_t>(sizeof(MetalMapTemplateResources)) +
           prepared.input_plans.capacity() * sizeof(InputWindowPlan) +
           prepared.input_strides.capacity() * sizeof(std::uint64_t) +
           prepared.output_strides.capacity() * sizeof(std::uint64_t) +
           prepared.checks.capacity() * sizeof(MetalMapCheck);
  };

  std::array<KernelExecutionStep, 2u> execution_steps{};
  execution_steps[1u].operation.set<operation::Scan>();
  std::array<BoundStep, 2u> bound_steps{};
  bound_steps[0u].step = &execution_steps[0u];
  bound_steps[1u].step = &execution_steps[1u];
  BackendRun signature{
      .steps = bound_steps.data(),
      .step_count = bound_steps.size(),
  };

  MetalKernelProgramTemplate program{};
  program.signature = &signature;
  program.steps.reserve(4u);
  program.steps.resize(bound_steps.size());
  const auto program_map = make_map();
  program.steps[0u].immutable = program_map;
  auto primitive = std::make_shared<MetalKernelImmutablePipelines>();
  primitive->stages[0u] = std::make_shared<std::uint64_t>(19u);
  primitive->count = 1u;
  program.steps[1u].immutable = primitive;
  const std::uint64_t expected_program =
      sizeof(MetalKernelProgramTemplate) +
      program.steps.capacity() * sizeof(MetalKernelProgramStepTemplate) +
      map_bytes(*program_map) + sizeof(MetalKernelImmutablePipelines);
  const PreparedMemory observed_program =
      ObserveMetalPipelineTemplate(&program);
  if (observed_program.current != expected_program ||
      observed_program.peak != expected_program ||
      observed_program.cumulative != expected_program ||
      observed_program.reused != 0u ||
      observed_program.budget != expected_program) {
    return false;
  }

  MetalMapRecurrenceTemplate recurrence{};
  BackendRun recurrence_signature{
      .steps = bound_steps.data(),
      .step_count = 1u,
  };
  recurrence.signature = &recurrence_signature;
  recurrence.history = true;
  recurrence.prepared = make_map();
  const std::uint64_t expected_recurrence =
      sizeof(MetalMapRecurrenceTemplate) + map_bytes(*recurrence.prepared);
  const PreparedMemory observed_recurrence =
      ObserveMetalPipelineTemplate(&recurrence);
  if (observed_recurrence.current != expected_recurrence ||
      observed_recurrence.peak != expected_recurrence ||
      observed_recurrence.cumulative != expected_recurrence ||
      observed_recurrence.reused != 0u ||
      observed_recurrence.budget != expected_recurrence) {
    return false;
  }
  const PreparedMemory invalid = ObserveMetalPipelineTemplate(nullptr);
  return invalid.current == std::numeric_limits<std::uint64_t>::max() &&
         invalid.peak == std::numeric_limits<std::uint64_t>::max() &&
         invalid.cumulative == std::numeric_limits<std::uint64_t>::max() &&
         invalid.budget == std::numeric_limits<std::uint64_t>::max();
#else
  return true;
#endif
}

[[nodiscard]] bool MetalIcbCalibrationContract() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  if (!ExactMetalIcbDescriptorContract()) {
    return false;
  }
  const rund::AccelDevice pick = PickMetal();
  if (!pick.check.ok) {
    // A compiled Metal surface may execute on a host with no Metal device.
    // Native parity is mandatory whenever an adapter actually opens.
    return pick.check.reason != nullptr &&
           std::string_view{pick.check.reason} ==
               "accel_metal_device_unavailable";
  }
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  if (!ValidMetalIcbCalibration(adapter->pipeline_icb_calibration)) {
    PreparedKernelPipelineReservation reservation{};
    const rund::AccelCheck check = PlanMetalPipelineStructureForCalibration(
        adapter->pipeline_icb_calibration, reservation);
    return !check.ok && check.reason != nullptr &&
           std::string_view{check.reason} ==
               "accel_metal_icb_calibration_failed";
  }
  id<MTLDevice> const device = (__bridge id<MTLDevice>)adapter->device.get();
  if (device == nil) {
    return false;
  }
  for (std::uint32_t index = 0u; index < MetalPipelineIcbClassCount; ++index) {
    @autoreleasepool {
      const NSUInteger capacity = NSUInteger{1u} << index;
      id<MTLIndirectCommandBuffer> const commands =
          AllocateMetalPipelineIcb(device, capacity);
      if (commands == nil || commands.size != capacity ||
          static_cast<std::uint64_t>(commands.allocatedSize) !=
              adapter->pipeline_icb_calibration.allocated_bytes[index]) {
        return false;
      }
    }
  }
  return true;
#else
  return true;
#endif
}

} // namespace node_accel_contract

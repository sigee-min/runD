#include "../../../buffer/access.hpp"

#include "local.hpp"

#include "../../../runtime/counter.hpp"
#include "../../control.hpp"
#include "../prepare/record.hpp"

#include <rund/compute/reason.hpp>

#include <algorithm>
#include <array>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool SelectVulkanResidencyArguments(VulkanPipeline &pipeline,
                                    const std::span<const std::uint32_t> locals,
                                    const bool execute) noexcept {
  VulkanResidencySelection *const selection = pipeline.residency.get();
  if (selection == nullptr || selection->arguments.mapped == nullptr ||
      selection->original_arguments.empty() ||
      selection->original_arguments.size() !=
          selection->argument_owners.size() ||
      locals.empty() || locals.size() > ResidencyWindowLocalCapacity) {
    return false;
  }
  auto *const arguments =
      static_cast<VkDispatchIndirectCommand *>(selection->arguments.mapped);
  std::array<bool, ResidencyWindowLocalCapacity> found{};
  for (std::size_t argument = 0u; argument < selection->argument_owners.size();
       ++argument) {
    for (std::size_t local = 0u; local < locals.size(); ++local) {
      if (selection->argument_owners[argument] == locals[local]) {
        arguments[argument] = execute ? selection->original_arguments[argument]
                                      : VkDispatchIndirectCommand{};
        found[local] = true;
      }
    }
  }
  return std::all_of(found.begin(), found.begin() + locals.size(),
                     [](const bool value) { return value; });
}

bool SeedVulkanResidencyControl(VulkanPipeline &pipeline,
                                const BackendResidencyWindowSignal &signal,
                                const bool) noexcept {
  if (signal.control_generation == 0u) {
    return false;
  }
  const PreparedPipelineControl initial{
      .generation =
          signal.control_generation - pipeline.control.generation_stride,
      .reserved = 0u,
  };
  return UploadVulkanBuffer(pipeline.control.summary, &initial,
                            sizeof(initial));
}

KernelResult ObserveVulkanResidency(
    VulkanPipeline &pipeline, const rund::AccelCheck admission,
    const std::uint64_t dispatch_count, const std::uint64_t control_count,
    const std::uint64_t reset_count, const std::uint64_t reset_bytes,
    const std::uint32_t control_generation) noexcept {
  KernelResult result{
      .check = admission,
      .stats =
          rund::RuntimeStats{
              .run =
                  {
                      .work =
                          {
                              .dispatch_count =
                                  admission.ok ? dispatch_count : 0u,
                              .command_submit_count = 1u,
                              .command_capacity = 1u,
                              .command_inflight_peak = 1u,
                              .reset_command_count =
                                  admission.ok ? reset_count : 0u,
                              .reset_bytes = admission.ok ? reset_bytes : 0u,
                          },
                  },
              .outcome = {.ok = admission.ok, .reason = admission.reason},
          },
      .pipeline = {.control_command_count = control_count, .submitted = true},
      .terminal = NativeTerminal::Known,
  };
  {
    std::lock_guard lock{pipeline.adapter->mutex};
    result.pipeline.control_observed =
        ReadVulkanPipelineControl(pipeline.control, result.pipeline.control);
    const bool canonical =
        result.pipeline.control_observed && pipeline.record != nullptr &&
        PreparedPipelineGenerationMatches(result.pipeline.control,
                                          control_generation) &&
        ValidPreparedPipelineControl(result.pipeline.control,
                                     pipeline.record->status);
    if (!canonical) {
      result.check = {false, "accel_kernel_pipeline_invalid"};
      result.terminal = NativeTerminal::UnknownMayWrite;
    } else if (result.pipeline.control.reason !=
               static_cast<std::uint32_t>(rund::compute::Reason::Ok)) {
      result.check = {false,
                      CanonicalReasonText(result.pipeline.control.reason)};
    }
    result.stats.outcome = {.ok = result.check.ok,
                            .reason = result.check.reason};
    if (admission.ok && canonical) {
      ::rund::detail::counter::Accumulate(pipeline.adapter->dispatch_count,
                                          dispatch_count);
    }
    if (!canonical) {
      result.stats.run.work.dispatch_count = 0u;
      result.stats.run.work.reset_command_count = 0u;
      result.stats.run.work.reset_bytes = 0u;
    }
  }
  return result;
}

#endif

} // namespace rund::node::accel::detail

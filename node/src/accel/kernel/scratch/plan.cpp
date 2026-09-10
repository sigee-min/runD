#include "../scratch.hpp"

#include "../../context/internal.hpp"
#include "../../diagnostic.hpp"
#include "internal.hpp"

#include <algorithm>
#include <array>

namespace rund::node::accel::detail {

KernelScratchPlan PlanKernelScratch(const rund::AccelContext &context,
                                    const rund::AccelKernel &kernel,
                                    const std::uint64_t alignment,
                                    const std::uint64_t page_bytes) {
  const auto reject = [&](const char *const reason) noexcept {
    diagnostic::RecordScratchExecutionAdmission(context, kernel, false, reason);
    return KernelScratchPlan{.reason = reason};
  };
  const auto accept = [&](const KernelScratchPlan &result) noexcept {
    diagnostic::RecordScratchExecutionAdmission(context, kernel, result.ok,
                                                result.reason);
    return result;
  };
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u ||
      page_bytes == 0u || page_bytes % alignment != 0u) {
    return reject("accel_kernel_scratch_invalid");
  }
  const KernelExecution execution = AdmitKernelForExecution(context, kernel);
  if (!execution.admission.check.ok || execution.steps.empty()) {
    return reject(execution.admission.check.reason);
  }
  std::size_t page_count = 0u;
  std::uint64_t last_bytes = 0u;
  std::uint64_t backing_bytes = 0u;
  std::uint64_t payload_bytes = 0u;
  for (const KernelExecutionStep &step : execution.steps) {
    if (const RangePlan *const range = RangePlanFor(step.operation);
        range != nullptr) {
      const KernelScratchBatchPlan batch =
          PlanRangeScratch(*range, alignment, page_bytes);
      if (!batch.ok()) {
        return reject(batch.reason());
      }
      payload_bytes = std::max(payload_bytes, batch.payload_bytes());
      if (batch.page_count() > page_count ||
          (batch.page_count() == page_count &&
           batch.last_bytes() > last_bytes)) {
        page_count = batch.page_count();
        last_bytes = batch.last_bytes();
        backing_bytes = batch.backing_bytes();
      }
      continue;
    }
    const kernel_scratch_internal::ScratchRequests requests =
        kernel_scratch_internal::BuildScratchRequests(step.operation,
                                                      context.pick.api);
    if (!requests.ok) {
      return reject("compute_pipeline_capacity");
    }
    std::array<KernelScratchRequirement, 8u> requirements{};
    for (std::size_t request = 0u; request < requests.count; ++request) {
      const std::uint64_t bytes = requests.bytes[request];
      requirements[request] = KernelScratchRequirement{
          .role = KernelScratchRole::from(static_cast<std::uint32_t>(request)),
          .bytes = bytes,
          .alignment = alignment,
      };
    }
    const KernelScratchBatchPlan batch = PlanKernelScratchBatch(
        std::span<const KernelScratchRequirement>{requirements.data(),
                                                  requests.count},
        alignment, page_bytes);
    if (!batch.ok()) {
      return reject(batch.reason());
    }
    payload_bytes = std::max(payload_bytes, batch.payload_bytes());
    if (batch.page_count() == 0u) {
      continue;
    }
    if (batch.page_count() > page_count ||
        (batch.page_count() == page_count && batch.last_bytes() > last_bytes)) {
      page_count = batch.page_count();
      last_bytes = batch.last_bytes();
      backing_bytes = batch.backing_bytes();
    }
  }
  if (page_count == 0u) {
    return accept(KernelScratchPlan{
        .payload_bytes = payload_bytes,
        .ok = true,
        .reason = "ok",
    });
  }
  return accept(KernelScratchPlan{.payload_bytes = payload_bytes,
                                  .backing_bytes = backing_bytes,
                                  .last_bytes = last_bytes,
                                  .page_count = page_count,
                                  .ok = true,
                                  .reason = "ok"});
}

} // namespace rund::node::accel::detail

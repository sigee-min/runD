#include "../../../accel/kernel/scratch.hpp"
#include "../../device/state.hpp"
#include "program.hpp"

#include "diagnostic.hpp"

#include "../../../accel/context/local.hpp"
#include "../../../accel/graph/token.hpp"
#include "../../../accel/graph/token/local.hpp"
#include "../../../accel/range_aggregate/model/plan.hpp"
#include "../../program/range.hpp"
#include "../../program/state.hpp"
#include "../../status.hpp"

#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <optional>

namespace rund::compute::detail::accel_backend {

RangeSnapshot program_ranges(const AccelProgram &program,
                             const std::span<RangeInfo> rows) noexcept {
  RangeSnapshot snapshot{};
  const auto token = node::accel::detail::LookupKernelToken(
      program.kernel.owner, program.kernel.kernel_id);
  if (token == nullptr) {
    return snapshot;
  }
  for (std::size_t index = 0u; index < token->steps.size(); ++index) {
    const auto *const range =
        node::accel::detail::RangePlanFor(token->steps[index].operation);
    if (range != nullptr) {
      append_program_range(*range, static_cast<std::uint32_t>(index), rows,
                           snapshot);
    }
  }
  return snapshot;
}

Status compile(DeviceState &device, AccelProgram &program,
               const rund::AccelGraph &graph) {
  AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::AccelProgramInvalid);
  }
  program.kernel = node::accel::CompileAccelKernel(accel->context, graph);
  if (!program.kernel.check.ok) {
    const Reason projected =
        project_reason(program.kernel.check.reason, Reason::LoweringInvalid);
    accel_diagnostic::RecordKernelCheckProjection(
        program.kernel, program.kernel.check,
        ::rund::compute::detail::reason_message(projected).data());
    return Status::fail(projected);
  }
  const std::optional<std::uint64_t> token_memory =
      node::accel::detail::MeasureKernelTokenRetainedMemory(program.kernel);
  if (!token_memory.has_value()) {
    return Status::fail(Reason::AccelProgramInvalid);
  }
  program.kernel_token_host_bytes = *token_memory;
  return Status::success();
}

node::accel::detail::KernelScratchPlan
plan_scratch(const DeviceState &device, const rund::AccelKernel &kernel,
             const std::uint64_t alignment, const std::uint64_t page_bytes) {
  const AccelDeviceState *const accel = accel_device(device);
  return accel == nullptr ? node::accel::detail::KernelScratchPlan{}
                          : node::accel::detail::PlanKernelScratch(
                                accel->context, kernel, alignment, page_bytes);
}

} // namespace rund::compute::detail::accel_backend

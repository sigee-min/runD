#include "internal.hpp"

#include "../../../device/residency/execution/model.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <atomic>

namespace rund::compute::detail {
namespace {

std::atomic<bool> retention_failure_once{};

[[nodiscard]] Status retain_residency_output_bindings(
    PipelineState &pipeline, const VirtualRunProjection &run,
    const std::span<const residency::CacheBinding> bindings,
    const std::size_t stride) noexcept {
  if (retention_failure_once.exchange(false, std::memory_order_acq_rel)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  // Canonical frame banks are the execution storage. This check is the whole
  // publication boundary: successful compute already wrote the authoritative
  // output frame, so retaining via a D2D mirror would create a second owner.
  if (pipeline.residency_pool == nullptr ||
      pipeline.residency_bank >= residency::Pool::BankCount ||
      pipeline.residency_output >= pipeline.resources.size() ||
      pipeline.resources[pipeline.residency_output].buffer == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (stride == 0u || bindings.size() % stride != 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < bindings.size(); index += stride) {
    const residency::CacheBinding &binding = bindings[index];
    std::size_t local = 0u;
    if (!virtual_cache_detail::output_bank_frame(pipeline, run, binding.frame,
                                                 local)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  return Status::success();
}

} // namespace

Status retain_residency_output(PipelineState &pipeline,
                               const VirtualRunProjection &run,
                               const residency::EpochLease lease,
                               Stats &) noexcept {
  return retain_residency_output_bindings(pipeline, run, lease.bindings, 1u);
}

Status retain_residency_output(PipelineState &pipeline,
                               const VirtualRunProjection &run,
                               const residency::ExecutionTicket &ticket,
                               Stats &) noexcept {
  if (ticket.phase !=
      static_cast<std::uint8_t>(residency::execution::Phase::Output)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  // Whole-run Output tickets bind [Device source, Host target] pairs. The
  // Device sources remain the sole retention authority.
  return retain_residency_output_bindings(pipeline, run, ticket.bindings, 2u);
}

void inject_residency_retention_failure_once() noexcept {
  retention_failure_once.store(true, std::memory_order_release);
}

bool cancel_residency_retention_failure_injection() noexcept {
  return retention_failure_once.exchange(false, std::memory_order_acq_rel);
}

} // namespace rund::compute::detail

#include <accel/context/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/runtime.hpp>

#include "evidence.hpp"

namespace rund::node::accel::detail {

rund::AccelEvidence RejectKernelEvidence(const rund::AccelContext &context,
                                         const KernelExecution &execution,
                                         const char *const reason) {
  if (!execution.admission.check.ok) {
    return rund::AccelEvidence{.outcome = {.ok = false, .reason = reason}};
  }
  return rund::AccelEvidence{
      .identity =
          {
              .graph_id_hi = execution.admission.graph_id_hi,
              .graph_id_lo = execution.admission.graph_id_lo,
              .kernel_id = execution.admission.kernel_id,
              .backend = context.api,
          },
      .run =
          {
              .work =
                  {
                      .original_operation_count =
                          execution.original_operation_count,
                      .fused_operation_count = execution.fused_operation_count,
                      .fusion_rejection_count =
                          execution.fusion_rejection_count,
                      .fusion_reason = execution.fusion_reason,
                  },
          },
      .outcome = {.ok = false, .reason = reason},
  };
}

rund::AccelEvidence BuildKernelEvidence(
    const rund::AccelContext &context, const KernelExecution &execution,
    const rund::RuntimeStats &stats,
    const std::uint64_t original_dispatch_count,
    const std::uint64_t final_dispatch_count, const bool ok,
    const char *const reason,
    const std::uint64_t internal_producer_consumer_roundtrip_bytes,
    const std::uint64_t external_producer_consumer_roundtrip_bytes,
    const std::uint64_t failed_batches, const std::uint64_t first_failed_batch,
    const std::uint32_t first_status) {
  rund::AccelRunFacts run = stats.run;
  run.work.original_operation_count = execution.original_operation_count;
  run.work.fused_operation_count = execution.fused_operation_count;
  run.work.original_dispatch_count = original_dispatch_count;
  run.work.final_dispatch_count = final_dispatch_count;
  run.work.fusion_rejection_count = execution.fusion_rejection_count;
  run.work.fusion_reason = execution.fusion_reason;
  run.transfer.internal_producer_consumer_roundtrip_bytes =
      internal_producer_consumer_roundtrip_bytes;
  run.transfer.external_producer_consumer_roundtrip_bytes =
      external_producer_consumer_roundtrip_bytes;
  return rund::AccelEvidence{
      .identity =
          {
              .graph_id_hi = execution.admission.graph_id_hi,
              .graph_id_lo = execution.admission.graph_id_lo,
              .kernel_id = execution.admission.kernel_id,
              .backend = context.api,
          },
      .run = run,
      .outcome =
          {
              .failed_batches = failed_batches,
              .first_failed_batch = first_failed_batch,
              .first_status = first_status,
              .ok = ok,
              .reason = reason,
          },
  };
}

} // namespace rund::node::accel::detail

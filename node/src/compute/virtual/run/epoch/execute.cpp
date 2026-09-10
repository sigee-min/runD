#include "internal.hpp"

namespace rund::compute::detail {

Status execute_virtual_epoch_pipeline(VirtualEpochContext &context) noexcept {
  const bool defer = context.transaction != nullptr &&
                     context.transaction->cursor.active &&
                     context.scan != nullptr && context.run.scan() &&
                     !context.run.graph_execution();
  const std::uint64_t control_generation =
      defer ? context.transaction->cursor.control_generation[context.bank]
            : PipelineNoGeneration;
  const std::uint8_t control_parity =
      defer ? context.transaction->cursor.base_parity[context.bank] : 0u;
  const std::uint64_t publication_generation =
      defer ? context.transaction->cursor.base_generation[context.bank]
            : PipelineNoGeneration;
  const std::uint8_t publication_parity =
      defer ? context.transaction->cursor.base_parity[context.bank] : 0u;
  if (context.transaction != nullptr && context.transaction->started &&
      context.transaction->scan) {
    for (std::size_t index = 0u; index < context.epoch_pages; ++index) {
      const residency::CacheBinding &binding =
          context.acquired.lease.bindings[context.epoch_pages + index];
      const Status journal = record_transaction_output(
          *context.transaction,
          context.pool->authority().virtual_transactions(),
          context.acquired.lease.token, binding.frame, binding.key,
          residency::FrameTier::Device);
      if (!journal) {
        return journal;
      }
    }
  }
  Status executed = Status::success();
  if (context.pool->device->backend == Backend::Cpu) {
    executed = run_residency_pipeline_lease(
        *context.selected, context.acquired.lease, defer, control_generation,
        control_parity, publication_generation, publication_parity);
  } else {
    if (!context.pool->submit_execution(
            context.bank, *context.selected, context.acquired.lease, defer,
            control_generation, control_parity, publication_generation,
            publication_parity)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    executed = context.pool->wait_execution(context.bank).status;
  }
  if (executed && defer) {
    executed = record_virtual_run_publication_terminal(*context.transaction,
                                                       context.bank);
  }
  return executed;
}

} // namespace rund::compute::detail

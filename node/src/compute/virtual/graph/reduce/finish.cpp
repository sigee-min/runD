#include "internal.hpp"

#include "../../../device/residency/registry/graph_persist_owner.hpp"
#include "../../backing.hpp"

namespace rund::compute::detail::graph_reduce {

VirtualGraphResult finish_tiled(GraphExecutionContext &context) noexcept {
  Status status = Status::success();
  if (context.persisted_output) {
    status = context.persists.finish(context.tickets, context.child_poison);
    if (!status) {
      return finish_failure(context, status, 0u, context.child_poison,
                            Phase::Persist, Check::Close, Fail::NoStage,
                            context.batches - 1u);
    }
  }
  const auto persist_owner = context.authority.graph_persists();
  if (context.cpu && persist_owner.cpu_graph_retry_active()) {
    return finish_failure(context, Status::fail(Reason::PipelineBusy), 0u, true,
                          Phase::Recovery, Check::Recover, Fail::NoStage,
                          Fail::NoStage);
  }
  if (!context.prefetch.quiescent()) {
    const Status failure = Status::fail(Reason::PipelineBusy);
    return finish_failure(context, failure, 0u, true, Phase::Recovery,
                          Check::Recover, Fail::NoStage, Fail::NoStage);
  }
  if (context.persisted_output) {
    VirtualBackingAccess::publish_write(*context.output);
    VirtualBackingAccess::clear_recovery(*context.output);
  }
  return VirtualGraphResult{.output_hash = context.persisted_output
                                               ? context.output_hash.Finish()
                                               : 0u};
}

} // namespace rund::compute::detail::graph_reduce

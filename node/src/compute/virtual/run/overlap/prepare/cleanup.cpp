#include "local.hpp"

namespace rund::compute::detail::virtual_run_overlap::prepare_detail {

Status abort(Context &context, const Status status) noexcept {
  bool completed = true;
  if (context.host_token != 0u) {
    completed = cancel_prefetch_receipt(context, context.prefetched, true);
    if (completed) {
      context.host_token = 0u;
    }
  } else {
    completed = release_prefetch_aliases(context, true);
  }
  completed =
      context.pool->authority().complete(context.prepared.token, false, true) &&
      completed;
  context.prepared.token = 0u;
  context.poison = !completed || context.poison;
  return completed ? status : Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail::virtual_run_overlap::prepare_detail

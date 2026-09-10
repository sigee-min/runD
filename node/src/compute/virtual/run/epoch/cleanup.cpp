#include "internal.hpp"

namespace rund::compute::detail {

bool cancel_virtual_prefetch(residency::Pool *const pool,
                             std::array<bool, 2u> &pending) noexcept {
  bool clean = pool != nullptr;
  for (std::size_t lane = 0u; lane < pending.size(); ++lane) {
    if (!pending[lane]) {
      continue;
    }
    if (pool == nullptr) {
      clean = false;
      continue;
    }
    const bool cancelled =
        pool->prefetch[lane].cancel(pool->authority(), true, true);
    clean = cancelled && clean;
    if (cancelled) {
      pending[lane] = false;
    }
  }
  return clean;
}

VirtualEpochResult abort_virtual_epoch(VirtualEpochContext &context,
                                       const Status status, const bool poison,
                                       const bool invalidate_all) noexcept {
  residency::Pool *const fallback_pool =
      context.pool != nullptr
          ? context.pool
          : (context.state.pipeline == nullptr
                 ? nullptr
                 : context.state.pipeline->residency_pool.get());
  bool rolled_back =
      cancel_virtual_prefetch(fallback_pool, context.prefetch_pending);
  if (context.host_token != 0u && context.pool != nullptr) {
    rolled_back =
        context.pool->prefetch[context.consume_lane].cancel(
            context.pool->authority(), context.prefetched, true, true) &&
        rolled_back;
    if (rolled_back) {
      context.host_token = 0u;
    }
  }
  if (context.pipeline != nullptr) {
    rolled_back = cancel_residency_output(*context.pipeline,
                                          context.output_reservation) &&
                  rolled_back;
  }
  if (context.execution_token != 0u && context.pool != nullptr) {
    rolled_back = context.pool->authority().complete(context.execution_token,
                                                     false, invalidate_all) &&
                  rolled_back;
    context.execution_token = 0u;
  }
  return VirtualEpochResult{
      .status = rolled_back ? status : Status::fail(Reason::PipelineInvalid),
      .failed_page = context.projected.failed_page,
      .poison_pipeline = poison || !rolled_back};
}

} // namespace rund::compute::detail

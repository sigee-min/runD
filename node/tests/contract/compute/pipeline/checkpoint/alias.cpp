#include "local.hpp"

namespace rund_node_test_pipeline::checkpoint {

int CheckAliasRejection(Context &context) {
  using namespace rund::compute;

  // An exposed duplicate selector and every partial/reversed alias are
  // rejected before mutation. The existing destination stays usable.
  auto stranded =
      pipeline(context.device)
          .state(context.first, context.second)
          .then(context.advance, read(context.first), write(context.second))
          .commit()
          .prepare();
  const std::shared_ptr<detail::PipelineState> stranded_state =
      stranded ? detail::PipelineStateAccess::state(*stranded)
               : std::shared_ptr<detail::PipelineState>{};
  const std::shared_ptr<detail::PipelinePublicationState> stranded_publication =
      stranded_state == nullptr ? nullptr : stranded_state->publication;
  const Status stranded_restore = stranded
                                      ? stranded->restore(context.latest)
                                      : Status::fail(Reason::PipelineInvalid);
  if (!stranded || stranded_restore ||
      stranded_restore.reason() != Reason::PipelineInvalid ||
      stranded->poisoned() || stranded->generation() != 0u ||
      stranded_state->publication != stranded_publication) {
    return 21;
  }

  auto partial_pending = context.device.buffer<std::int32_t>(Initial.size());
  auto partial = partial_pending
                     ? pipeline(context.device)
                           .state(context.first, *partial_pending)
                           .then(context.advance, read(context.first),
                                 write(*partial_pending))
                           .commit()
                           .prepare()
                     : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const std::shared_ptr<detail::PipelineState> partial_state =
      partial ? detail::PipelineStateAccess::state(*partial)
              : std::shared_ptr<detail::PipelineState>{};
  const std::shared_ptr<detail::PipelinePublicationState> partial_publication =
      partial_state == nullptr ? nullptr : partial_state->publication;
  const Status partial_restore = partial
                                     ? partial->restore(context.latest)
                                     : Status::fail(Reason::PipelineInvalid);
  if (!partial || partial_restore ||
      partial_restore.reason() != Reason::BindingDuplicate ||
      partial->poisoned() || partial->generation() != 0u ||
      partial_state->publication != partial_publication) {
    return 22;
  }

  auto reversed =
      pipeline(context.device)
          .state(context.second, context.first)
          .then(context.advance, read(context.second), write(context.first))
          .commit()
          .prepare();
  const Status reversed_restore = reversed
                                      ? reversed->restore(context.latest)
                                      : Status::fail(Reason::PipelineInvalid);
  if (!reversed || reversed_restore ||
      reversed_restore.reason() != Reason::BindingDuplicate ||
      reversed->poisoned() || reversed->generation() != 0u) {
    return 23;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::checkpoint

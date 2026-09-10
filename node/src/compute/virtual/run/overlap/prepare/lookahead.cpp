#include "local.hpp"

#include "../../../../pipeline/local.hpp"
#include "../../cache.hpp"

#include <limits>
#include <span>

namespace rund::compute::detail::virtual_run_overlap::prepare_detail {

Status schedule_lookahead(Context &context) noexcept {
  const std::uint64_t lookahead = context.run.active.stream.prefetch_distance();
  for (std::uint64_t distance = 1u; distance <= lookahead; ++distance) {
    if (context.epoch > std::numeric_limits<std::uint64_t>::max() - distance ||
        context.epoch + distance >= context.run.active.stream.epoch_count()) {
      break;
    }
    const std::size_t next_lane =
        static_cast<std::size_t>((context.epoch + distance) % 2u);
    if (context.prefetch_pending[next_lane]) {
      continue;
    }
    VirtualEpochProjection next{};
    if (!project_virtual_epoch(context.run, context.epoch + distance, next)) {
      return abort(context, Status::fail(Reason::PipelineInvalid));
    }
    const std::uint32_t next_bank = static_cast<std::uint32_t>(
        (context.epoch + distance) % residency::Pool::BankCount);
    const std::shared_ptr<PipelineState> &next_pipeline =
        next_bank == 0u ? context.state.pipeline
                        : context.state.alternate_pipeline;
    BufferWriteView available{};
    if (is_accelerator(context) && context.coherent_input_capable &&
        distance == 1u && next_pipeline != nullptr) {
      available = residency_input_view(*next_pipeline, context.run);
    }
    const BufferWriteView view =
        context.coherent_lookahead ? available : BufferWriteView{};
    const Status scheduled = schedule_virtual_prefetch(
        context.input, next, context.run, *context.pool,
        context.pool->prefetch[next_lane], true,
        context.prefetch_pending[next_lane], context.poison,
        view ? std::span<std::byte>{view.data, view.bytes}
             : std::span<std::byte>{},
        !context.coherent_lookahead && static_cast<bool>(available));
    if (!scheduled) {
      return abort(context, scheduled);
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail::virtual_run_overlap::prepare_detail

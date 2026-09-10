#include "../internal.hpp"

namespace rund::compute::detail::residency::execution {

bool SlidingInvocation::project(const std::uint64_t ordinal,
                                const std::span<residency::PageUse> scratch,
                                SlidingProjection &projection) const noexcept {
  projection = {};
  if (!*this || ordinal >= count()) {
    return false;
  }
  if (topology_ == SlidingTopology::Direct) {
    Node input{};
    Node output{};
    Forecast forecast{};
    WindowFootprintProjection footprint{};
    const bool window = direct_->has_window_footprint();
    if ((!window &&
         !direct_->project(NodeId{.epoch = ordinal, .phase = Phase::Input},
                           input)) ||
        (window && !direct_->window_footprint(ordinal, footprint)) ||
        !direct_->project(NodeId{.epoch = ordinal, .phase = Phase::Output},
                          output) ||
        !direct_->forecast(ordinal, forecast) ||
        (window ? footprint.source_count : input.input_count) >
            scratch.size() ||
        output.output_count > scratch.size() - (window ? footprint.source_count
                                                       : input.input_count)) {
      return false;
    }
    const std::size_t input_count =
        window ? footprint.source_count : input.input_count;
    for (std::size_t index = 0u; index < input_count; ++index) {
      const CacheUse &use =
          window ? footprint.sources[index] : input.input[index];
      scratch[index] = PageUse{
          .key = PageKey{.resource = direct_->input_resource(),
                         .page = use.key.page},
          .access = Access::Read,
          .next_use = use.next_use,
          .pin = PinInterval{.first_epoch = ordinal, .last_epoch = ordinal},
          .prefetch_epoch = forecast.prefetch_epoch,
          .ready_epoch = forecast.ready_epoch,
      };
    }
    for (std::size_t index = 0u; index < output.output_count; ++index) {
      const CacheUse &use = output.output[index];
      scratch[input_count + index] = PageUse{
          .key = PageKey{.resource = direct_->output_resource(),
                         .page = use.key.page},
          .access = Access::Write,
          .dirty =
              DirtyRange{.offset = use.dirty.offset, .bytes = use.dirty.bytes},
          .next_use = use.next_use,
          .pin = PinInterval{.first_epoch = ordinal, .last_epoch = ordinal},
          .prefetch_epoch = ordinal,
          .ready_epoch = ordinal,
      };
    }
    projection = SlidingProjection{
        .coordinate = SlidingCoordinate{.ordinal = ordinal,
                                        .batch = ordinal,
                                        .stage = 0u},
        .prefetch_epoch = forecast.prefetch_epoch,
        .ready_epoch = forecast.ready_epoch,
        .use_count = input_count + output.output_count,
        .fetch_count = input_count,
        .persist_count = output.output_count,
    };
  } else {
    const TiledGraphPlan &plan = graph_owner_->tiled_graph();
    const std::size_t stage_count = plan.stage_count();
    if (stage_count == 0u) {
      return false;
    }
    const std::size_t stage = static_cast<std::size_t>(ordinal % stage_count);
    const std::uint64_t batch = ordinal / stage_count;
    PageRun run{};
    if (!graph_.batch(batch, run) || stage >= plan.stages().size() ||
        run.page_count > std::numeric_limits<std::size_t>::max() /
                             plan.stages()[stage].ports.size()) {
      return false;
    }
    const std::size_t count = static_cast<std::size_t>(run.page_count) *
                              plan.stages()[stage].ports.size();
    if (count == 0u || count > scratch.size()) {
      return false;
    }
    Epoch epoch{};
    const auto uses = scratch.first(count);
    if (!graph_.project(batch, stage, uses, epoch) ||
        epoch.ordinal != ordinal) {
      return false;
    }
    projection = SlidingProjection{
        .coordinate =
            SlidingCoordinate{.ordinal = ordinal,
                              .batch = batch,
                              .stage = static_cast<std::uint32_t>(stage)},
        .prefetch_epoch = ordinal,
        .ready_epoch = ordinal,
        .use_count = count,
    };
    for (std::size_t index = 0u; index < count; ++index) {
      if (fetch_use(projection, uses, index)) {
        projection.prefetch_epoch =
            std::min(projection.prefetch_epoch, uses[index].prefetch_epoch);
        projection.ready_epoch =
            std::max(projection.ready_epoch, uses[index].ready_epoch);
        ++projection.fetch_count;
      }
      if (persist_use(projection, uses, index)) {
        ++projection.persist_count;
      }
    }
  }

  for (std::size_t index = 0u; index < projection.use_count; ++index) {
    std::uint64_t bytes = 0u;
    if (!use_bytes(projection, scratch, index, bytes)) {
      return false;
    }
    if (fetch_use(projection, scratch, index) &&
        !add(projection.fetch_bytes, bytes)) {
      return false;
    }
    if (persist_use(projection, scratch, index) &&
        !add(projection.persist_bytes, bytes)) {
      return false;
    }
  }
  return projection.use_count != 0u;
}

} // namespace rund::compute::detail::residency::execution

#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

void initialize_run(SlidingProductRun &wait, SlidingProductOwner &owner,
                    std::shared_ptr<void> cold_owner,
                    VirtualPipelineState &state, VirtualBacking &input,
                    VirtualBacking &output, const VirtualRunProjection &run,
                    Stats &stats, residency::Pool &pool,
                    residency::execution::Sliding sliding) noexcept {
  wait.cold_owner = std::move(cold_owner);
  wait.cold = &owner;
  wait.state = &state;
  wait.input = &input;
  wait.output = &output;
  wait.run = &run;
  wait.stats = &stats;
  wait.pool = &pool;
  wait.sliding = std::move(sliding);
  wait.result = {};
  wait.result.status = Status::fail(Reason::PipelineInvalid);
  wait.result.failed_page = ResidencyStats::no_failed_page;
  wait.failure = Status::success();
  wait.failure_coordinate = std::numeric_limits<std::uint64_t>::max();
  wait.backing_read_bytes = 0u;
  wait.backing_write_bytes = 0u;
  wait.backing_io_ns = 0u;
  wait.epoch_count = 0u;
  wait.recovery = false;
  wait.completed = false;
  wait.poison.store(false, std::memory_order_relaxed);
  wait.pipeline_started = {};
  wait.pipeline_submitted = {};
  wait.service_fault = {};
  wait.hash = ::rund::node::hash_detail::Fnv{};
}

void release_run_links(SlidingProductRun &wait) noexcept {
  wait.pipeline_started = {};
  wait.pipeline_submitted = {};
  if (wait.cold != nullptr && wait.cold->pending_valid) {
    discard_staged_roles(*wait.cold);
  }
  wait.cold_owner.reset();
  wait.cold = nullptr;
  wait.state = nullptr;
  wait.input = nullptr;
  wait.output = nullptr;
  wait.run = nullptr;
  wait.stats = nullptr;
  wait.pool = nullptr;
}

} // namespace rund::compute::detail::sliding_product_detail

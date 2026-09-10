#pragma once

#include "model.hpp"
#include "prefetch.hpp"
#include "wavefront.hpp"

namespace rund::compute {
class VirtualBacking;
struct Stats;
} // namespace rund::compute

namespace rund::compute::detail {
struct VirtualPipelineState;
}

namespace rund::compute::detail::graph_reduce {

class FailLog;

// Owns the input-side transition Projected -> SupplyReady -> PrefixReady.
// Backing Forecast and physical Promote remain separate capability owners;
// this controller only composes their authenticated results for one ticket.
class SupplyController final {
public:
  SupplyController(const VirtualRunProjection &, Stats &, residency::Pool &,
                   const residency::TiledGraphPlan &, Wavefront &,
                   PrefetchController &, FailLog &) noexcept;

  [[nodiscard]] Status prepare(Ticket &, Timeline *hidden_by,
                               bool prepared_before_ready) noexcept;
  [[nodiscard]] Status make_device_ready(Ticket &,
                                         Timeline *hidden_by) noexcept;
  void configure_retry(residency::Identity prepared, bool retry) noexcept {
    prepared_ = prepared;
    retry_ = retry;
  }

private:
  const VirtualRunProjection &run_;
  Stats &stats_;
  residency::Pool &pool_;
  residency::Authority &authority_;
  const residency::TiledGraphPlan &graph_;
  Wavefront &wavefront_;
  PrefetchController &prefetch_;
  FailLog &failure_;
  residency::Identity prepared_{};
  bool retry_{};
};

} // namespace rund::compute::detail::graph_reduce

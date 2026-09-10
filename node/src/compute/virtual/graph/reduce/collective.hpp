#pragma once

#include "model.hpp"
#include "stage.hpp"
#include "wavefront.hpp"

namespace rund::compute::detail::graph_reduce {

class FailLog;

// Owns terminal-stage Authority preparation and its success/failure terminal
// disposition. Native submission remains the StageController's authority.
class CollectiveController final {
public:
  CollectiveController(const VirtualRunProjection &, Stats &, residency::Pool &,
                       const residency::TiledGraphPlan &, Wavefront &,
                       StageController &, std::size_t terminal_stage,
                       FailLog &) noexcept;

  [[nodiscard]] Status prepare(Ticket &) noexcept;
  [[nodiscard]] Status finish(Ticket &, bool &child_poison) noexcept;

private:
  const VirtualRunProjection &run_;
  Stats &stats_;
  residency::Pool &pool_;
  residency::Authority &authority_;
  const residency::TiledGraphPlan &graph_;
  Wavefront &wavefront_;
  StageController &stages_;
  std::size_t terminal_stage_{};
  FailLog &failure_;
};

} // namespace rund::compute::detail::graph_reduce

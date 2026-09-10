#pragma once

#include "failure.hpp"
#include "model.hpp"
#include "prefetch.hpp"
#include "stage.hpp"
#include "wavefront.hpp"

namespace rund::compute::detail {
struct VirtualPipelineState;
}

namespace rund::compute::detail::graph_reduce {

class MiddleController final {
public:
  MiddleController(VirtualPipelineState &, const VirtualRunProjection &,
                   Stats &, residency::Pool &,
                   const residency::TiledGraphPlan &, Wavefront &,
                   PrefetchController &, StageController &,
                   std::uint64_t identity, std::uint32_t capacity,
                   std::size_t terminal_stage) noexcept;

  [[nodiscard]] Status execute(Ticket &, bool &child_poison) noexcept;

private:
  struct SelectResult final {
    Status status{Status::success()};
    Check check{Check::None};
    std::uint32_t stage{Fail::NoStage};
  };

  [[nodiscard]] bool ready_external_inputs(Ticket &,
                                           std::size_t stage) noexcept;
  [[nodiscard]] SelectResult select(Ticket &, StageScratch &,
                                    WavefrontCoordinate &,
                                    std::shared_ptr<PipelineState> &,
                                    bool &child_poison) noexcept;
  [[nodiscard]] std::uint32_t
  stage_id(const Ticket &, const WavefrontCoordinate &) const noexcept;
  [[nodiscard]] Status run_stage(Ticket &, const StageScratch &,
                                 const WavefrontCoordinate &,
                                 const std::shared_ptr<PipelineState> &,
                                 bool &child_poison) noexcept;
  [[nodiscard]] Status bind_terminal_input(Ticket &) noexcept;

  VirtualPipelineState &state_;
  const VirtualRunProjection &run_;
  Stats &stats_;
  residency::Pool &pool_;
  residency::Authority &authority_;
  const residency::TiledGraphPlan &graph_;
  Wavefront &wavefront_;
  PrefetchController &prefetch_;
  StageController &stages_;
  std::uint64_t identity_{};
  std::uint32_t capacity_{};
  std::size_t terminal_stage_{};
};

} // namespace rund::compute::detail::graph_reduce

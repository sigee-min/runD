#pragma once

#include "model.hpp"
#include "wavefront.hpp"

#include <span>

namespace rund::compute::detail::graph_reduce {

// Sole owner of the one-active-native-stage invariant and the submit/wait/fold
// transition shared by Prefix and Collective work.
class StageController final {
public:
  StageController(Stats &, residency::Pool &, Wavefront &,
                  std::uint64_t identity, std::size_t terminal_stage) noexcept;

  [[nodiscard]] Status submit(Ticket &, ExecutionStage) noexcept;
  [[nodiscard]] Status wait(Ticket &, ExecutionStage, bool fold_now,
                            bool &child_poison) noexcept;
  [[nodiscard]] Status fold(Ticket &, ExecutionStage) noexcept;
  void abort_active(bool &child_poison) noexcept;
  void fold_pending(std::span<Ticket>, bool &child_poison) noexcept;
  [[nodiscard]] bool idle() const noexcept { return active_ == nullptr; }

private:
  Stats &stats_;
  residency::Pool &pool_;
  Wavefront &wavefront_;
  std::uint64_t identity_{};
  std::size_t terminal_stage_{};
  Ticket *active_{};
};

} // namespace rund::compute::detail::graph_reduce

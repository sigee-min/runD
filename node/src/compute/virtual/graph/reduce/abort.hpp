#pragma once

#include "failure.hpp"
#include "output/persist.hpp"
#include "prefetch.hpp"
#include "stage.hpp"

#include <memory>
#include <span>

namespace rund::compute::detail {
struct VirtualGraphResult;
struct PipelineState;
} // namespace rund::compute::detail

namespace rund::compute::detail::graph_reduce {

// Whole-run failure closer. It first joins active native execution, then folds
// pending evidence, cancels shared Forecast work, and finally disposes every
// ticket before resetting Wavefront readiness.
class AbortController final {
public:
  AbortController(residency::Authority &, std::span<Ticket>,
                  PrefetchController &, PersistController &, StageController &,
                  Wavefront &, std::shared_ptr<CpuReceiptBook>,
                  std::shared_ptr<CpuGraphQuarantine>, FailLog &) noexcept;

  [[nodiscard]] VirtualGraphResult finish(Status, std::uint64_t page,
                                          bool child_poison) noexcept;

private:
  residency::Authority &authority_;
  std::span<Ticket> tickets_;
  std::shared_ptr<CpuReceiptBook> receipts_;
  PrefetchController &prefetch_;
  PersistController &persist_;
  StageController &stages_;
  Wavefront &wavefront_;
  FailLog &failure_;
  std::shared_ptr<CpuGraphQuarantine> quarantine_;
  bool receipts_bound_{true};
};

} // namespace rund::compute::detail::graph_reduce

#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

bool PrefetchController::cancel() noexcept {
  bool clean = true;
  for (std::size_t index = 0u; index < lanes_.size(); ++index) {
    PrefetchLane &selected = lanes_[index];
    if (!selected.pending && !selected.forecast) {
      continue;
    }
    clean = retire(index, selected) && clean;
  }
  return quiescent() && clean;
}

bool PrefetchController::quarantine() noexcept {
  if (!workers_quiescent()) {
    return false;
  }
  bool clean = true;
  for (PrefetchLane &selected : lanes_) {
    if (selected.pending) {
      clean = false;
      continue;
    }
    if (!selected.forecast) {
      continue;
    }
    if (!authority_.graph_forecasts().quarantine_graph_forecast(
            selected.forecast)) {
      clean = false;
      continue;
    }
    selected = {};
  }
  return clean && quiescent();
}

} // namespace rund::compute::detail::graph_reduce

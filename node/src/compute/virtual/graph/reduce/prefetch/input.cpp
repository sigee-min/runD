#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

bool PrefetchController::find_input(const std::uint32_t resource,
                                    std::size_t &index) const noexcept {
  const auto end = state_.graph_input_resources.begin() +
                   static_cast<std::ptrdiff_t>(input_count_);
  const auto found =
      std::find(state_.graph_input_resources.begin(), end, resource);
  if (found == end) {
    index = 0u;
    return false;
  }
  index = static_cast<std::size_t>(
      std::distance(state_.graph_input_resources.begin(), found));
  return index < input_count_ && inputs_[index] != nullptr;
}

} // namespace rund::compute::detail::graph_reduce

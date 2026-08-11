#include "model.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::compute::detail::residency {

bool StreamPlan::epoch(const std::uint64_t index, PageRun &run) const noexcept {
  std::uint64_t first_page = 0u;
  if (index >= epoch_count() || frame_capacity_ == 0u ||
      !kernel::checked::mul(index, frame_capacity_, first_page) ||
      first_page >= page_count_) {
    return false;
  }
  run = PageRun{
      .first_page = first_page,
      .page_count = std::min(frame_capacity_, page_count_ - first_page),
  };
  return true;
}

ResidencyPlan::ResidencyPlan(const std::uint64_t page_bytes,
                             const StreamPlan stream,
                             const Identity identity) noexcept
    : page_bytes_(page_bytes),
      frame_capacity_(static_cast<std::uint32_t>(stream.frame_capacity())),
      stream_(stream), identity_(identity) {}

ResidencyPlan::ResidencyPlan(const std::uint64_t page_bytes,
                             const std::uint32_t frame_capacity,
                             std::vector<PageUse> uses,
                             std::vector<Transition> transitions,
                             std::vector<Epoch> epochs,
                             const Identity identity) noexcept
    : page_bytes_(page_bytes), frame_capacity_(frame_capacity),
      uses_(std::move(uses)), transitions_(std::move(transitions)),
      epochs_(std::move(epochs)), identity_(identity) {}

} // namespace rund::compute::detail::residency

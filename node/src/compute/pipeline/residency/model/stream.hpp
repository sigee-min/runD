#pragma once

#include "base.hpp"

namespace rund::compute::detail::residency {

struct StreamPlanInput final {
  std::uint64_t page_bytes{};
  std::uint64_t page_count{};
  std::uint64_t requested_frames{};
  std::uint64_t max_frames{};
  DirtyRange dirty{};
  std::uint64_t dirty_bytes{};
  std::uint64_t prefetch_distance{};
};

class StreamPlan final {
public:
  constexpr StreamPlan() noexcept = default;
  constexpr StreamPlan(const std::uint64_t page_count,
                       const std::uint64_t frame_capacity,
                       const DirtyRange dirty, const std::uint64_t dirty_bytes,
                       const std::uint64_t prefetch_distance) noexcept
      : page_count_(page_count), frame_capacity_(frame_capacity), dirty_(dirty),
        dirty_bytes_(dirty_bytes), prefetch_distance_(prefetch_distance) {}

  [[nodiscard]] constexpr std::uint64_t page_count() const noexcept {
    return page_count_;
  }
  [[nodiscard]] constexpr std::uint64_t frame_capacity() const noexcept {
    return frame_capacity_;
  }
  [[nodiscard]] constexpr DirtyRange dirty_extent() const noexcept {
    return dirty_;
  }
  [[nodiscard]] constexpr std::uint64_t dirty_bytes() const noexcept {
    return dirty_bytes_;
  }
  [[nodiscard]] constexpr std::uint64_t prefetch_distance() const noexcept {
    return prefetch_distance_;
  }
  [[nodiscard]] constexpr std::uint64_t epoch_count() const noexcept {
    return frame_capacity_ == 0u ? 0u
                                 : page_count_ / frame_capacity_ +
                                       static_cast<std::uint64_t>(
                                           page_count_ % frame_capacity_ != 0u);
  }
  [[nodiscard]] bool epoch(std::uint64_t index, PageRun &run) const noexcept;
  [[nodiscard]] bool dirty_extent(std::uint64_t page,
                                  DirtyRange &dirty) const noexcept;
  [[nodiscard]] bool first_use(std::uint64_t page,
                               std::uint64_t &use) const noexcept;
  [[nodiscard]] bool next_use(std::uint64_t page,
                              std::uint64_t &next) const noexcept;

private:
  std::uint64_t page_count_{};
  std::uint64_t frame_capacity_{};
  DirtyRange dirty_{};
  std::uint64_t dirty_bytes_{};
  std::uint64_t prefetch_distance_{};
};

} // namespace rund::compute::detail::residency

#pragma once

#include <rund/compute/job.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>

namespace rund::compute {
namespace detail {

inline constexpr std::size_t BatchCapacity = 64u;

[[nodiscard]] Status
add_batch(std::span<std::shared_ptr<JobState>> jobs, std::size_t &size,
          const std::shared_ptr<JobState> &job, Stats &stats,
          std::shared_ptr<void> &workspace) noexcept;

[[nodiscard]] Status
run_batch(std::span<const std::shared_ptr<JobState>> jobs,
          Stats &stats, std::shared_ptr<void> &workspace);

} // namespace detail

class Batch final {
public:
  Batch() noexcept = default;
  Batch(const Batch &) = delete;
  Batch &operator=(const Batch &) = delete;
  Batch(Batch &&other) noexcept
      : jobs_(std::move(other.jobs_)), size_(std::exchange(other.size_, 0u)),
        stats_(std::exchange(other.stats_, Stats{})),
        workspace_(std::move(other.workspace_)) {}
  Batch &operator=(Batch &&other) noexcept {
    if (this != &other) {
      jobs_ = std::move(other.jobs_);
      size_ = std::exchange(other.size_, 0u);
      stats_ = std::exchange(other.stats_, Stats{});
      workspace_ = std::move(other.workspace_);
    }
    return *this;
  }

  template <class Signature>
  [[nodiscard]] Status add(Job<Signature> &job) noexcept {
    return detail::add_batch(jobs_, size_, job.state_, stats_, workspace_);
  }

  [[nodiscard]] Status run() {
    return detail::run_batch(
        std::span<const std::shared_ptr<detail::JobState>>{jobs_.data(), size_},
        stats_, workspace_);
  }

  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] static constexpr std::size_t capacity() noexcept {
    return detail::BatchCapacity;
  }
  [[nodiscard]] Stats stats() const noexcept { return stats_; }

private:
  std::array<std::shared_ptr<detail::JobState>, detail::BatchCapacity> jobs_{};
  std::size_t size_{};
  Stats stats_{};
  std::shared_ptr<void> workspace_{};
};

} // namespace rund::compute

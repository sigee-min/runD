#pragma once

#include <rund/compute/abi/virtual.hpp>

#include <memory>
#include <utility>

namespace rund::compute {

template <class Signature> class VirtualPipeline;

namespace detail {

struct VirtualPipelineState;
struct VirtualPipelineAccess;

class VirtualPipelineHandle {
public:
  [[nodiscard]] bool valid() const noexcept {
    return valid_virtual_pipeline(state_);
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] Status begin_samples() noexcept {
    return begin_virtual_pipeline_samples(state_);
  }
  [[nodiscard]] Status end_samples() noexcept {
    return end_virtual_pipeline_samples(state_);
  }
  [[nodiscard]] Status run() noexcept { return run_virtual_pipeline(state_); }
  [[nodiscard]] Status run(const std::uint64_t active_count) noexcept {
    return run_virtual_pipeline(state_, active_count);
  }
  [[nodiscard]] Stats stats() const noexcept {
    return virtual_pipeline_stats(state_);
  }
  [[nodiscard]] MemoryStats memory() const noexcept {
    return virtual_pipeline_memory(state_);
  }
  [[nodiscard]] PipelinePlan plan() const noexcept {
    return virtual_pipeline_plan(state_);
  }
  [[nodiscard]] Result<telemetry::Profile> profile() const noexcept {
    return virtual_pipeline_profile(state_);
  }

protected:
  explicit VirtualPipelineHandle(
      std::shared_ptr<VirtualPipelineState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<VirtualPipelineState> state_;
};

} // namespace detail

template <detail::ComputeValue R, detail::ComputeValue A>
class VirtualPipeline<R(A)> final : private detail::VirtualPipelineHandle {
public:
  VirtualPipeline(const VirtualPipeline &) = delete;
  VirtualPipeline &operator=(const VirtualPipeline &) = delete;
  VirtualPipeline(VirtualPipeline &&) noexcept = default;
  VirtualPipeline &operator=(VirtualPipeline &&) noexcept = default;

  using detail::VirtualPipelineHandle::begin_samples;
  using detail::VirtualPipelineHandle::end_samples;
  using detail::VirtualPipelineHandle::memory;
  using detail::VirtualPipelineHandle::operator bool;
  using detail::VirtualPipelineHandle::plan;
  using detail::VirtualPipelineHandle::profile;
  using detail::VirtualPipelineHandle::run;
  using detail::VirtualPipelineHandle::stats;
  using detail::VirtualPipelineHandle::valid;

private:
  friend struct detail::VirtualPipelineAccess;

  explicit VirtualPipeline(
      std::shared_ptr<detail::VirtualPipelineState> state) noexcept
      : detail::VirtualPipelineHandle(std::move(state)) {}
};

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue... Rest>
class VirtualPipeline<R(A, B, Rest...)> final
    : private detail::VirtualPipelineHandle {
public:
  VirtualPipeline(const VirtualPipeline &) = delete;
  VirtualPipeline &operator=(const VirtualPipeline &) = delete;
  VirtualPipeline(VirtualPipeline &&) noexcept = default;
  VirtualPipeline &operator=(VirtualPipeline &&) noexcept = default;

  using detail::VirtualPipelineHandle::begin_samples;
  using detail::VirtualPipelineHandle::end_samples;
  using detail::VirtualPipelineHandle::memory;
  using detail::VirtualPipelineHandle::operator bool;
  using detail::VirtualPipelineHandle::plan;
  using detail::VirtualPipelineHandle::profile;
  using detail::VirtualPipelineHandle::run;
  using detail::VirtualPipelineHandle::stats;
  using detail::VirtualPipelineHandle::valid;

private:
  friend struct detail::VirtualPipelineAccess;

  explicit VirtualPipeline(
      std::shared_ptr<detail::VirtualPipelineState> state) noexcept
      : detail::VirtualPipelineHandle(std::move(state)) {}
};

namespace detail {

struct VirtualPipelineAccess final {
  template <class Signature>
  [[nodiscard]] static const std::shared_ptr<VirtualPipelineState> &
  state(const VirtualPipeline<Signature> &pipeline) noexcept {
    return pipeline.state_;
  }

  template <class Signature>
  [[nodiscard]] static VirtualPipeline<Signature>
  make(std::shared_ptr<VirtualPipelineState> state) noexcept {
    return VirtualPipeline<Signature>{std::move(state)};
  }
};

} // namespace detail

} // namespace rund::compute

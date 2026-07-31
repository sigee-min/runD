#pragma once

#include <rund/compute/job.hpp>
#include <rund/compute/pipeline.hpp>

namespace rund::compute::detail {

struct JobAccess final {
  template <class Signature>
  [[nodiscard]] static std::shared_ptr<JobState> state(
      Job<Signature> &job) noexcept {
    return job.state_;
  }
};

struct PipelineStateAccess final {
  [[nodiscard]] static const std::shared_ptr<PipelineState> &
  state(const Pipeline &pipeline) noexcept {
    return pipeline.state_;
  }
};

} // namespace rund::compute::detail

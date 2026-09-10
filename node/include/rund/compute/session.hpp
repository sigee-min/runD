#pragma once

#include <rund/compute/job.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session/await.hpp>
#include <rund/compute/virtual.hpp>
#include <rund/session.hpp>

namespace rund {

template <class Signature>
compute::Request Session::compute(compute::Job<Signature> &job) noexcept {
  return compute_job(job.state_);
}

template <class Signature>
compute::Request
Session::compute(compute::VirtualPipeline<Signature> &pipeline) noexcept {
  return compute_virtual(
      compute::detail::VirtualPipelineAccess::state(pipeline));
}

inline compute::Request Session::compute(compute::Pipeline &pipeline) noexcept {
  return compute_pipeline(pipeline.state_);
}

} // namespace rund

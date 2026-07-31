#pragma once

#include <rund/compute/job.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session/await.hpp>
#include <rund/session.hpp>

namespace rund {

template <class Signature>
compute::Request Session::compute(compute::Job<Signature> &job) noexcept {
  return compute_job(job.state_);
}

inline compute::Request Session::compute(compute::Pipeline &pipeline) noexcept {
  return compute_pipeline(pipeline.state_);
}

} // namespace rund

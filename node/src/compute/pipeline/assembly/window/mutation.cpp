#include "internal.hpp"

namespace rund::compute::detail {

PipelineBuildMutation::PipelineBuildMutation(PipelineBuildState &build) noexcept
    : build_{build}, steps_{build.steps.size()}, bindings_{build.binding_count},
      internals_{build.internals.size()},
      publications_{build.publications.size()},
      window_controls_{build.window_controls.size()},
      nested_windows_{build.nested_windows.size()} {}

PipelineBuildMutation::~PipelineBuildMutation() noexcept {
  if (!committed_) {
    build_.steps.resize(steps_);
    build_.internals.resize(internals_);
    build_.publications.resize(publications_);
    build_.window_controls.resize(window_controls_);
    build_.nested_windows.resize(nested_windows_);
    build_.binding_count = bindings_;
  }
}

void PipelineBuildMutation::fail(const Reason reason) noexcept {
  build_.failure = reason;
}

void PipelineBuildMutation::commit() noexcept { committed_ = true; }

} // namespace rund::compute::detail

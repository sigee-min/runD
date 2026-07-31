#pragma once

#include "../../state.hpp"
#include "../model/context.hpp"

namespace rund::node {

class Scheduler::ControlCommitScope final {
public:
  explicit ControlCommitScope(Scheduler &scheduler) noexcept;
  ControlCommitScope(const ControlCommitScope &) = delete;
  ControlCommitScope &operator=(const ControlCommitScope &) = delete;
  ~ControlCommitScope();

private:
  Scheduler *scheduler_ = nullptr;
  SchedulerThreadContext context_{};
  bool installed_ = false;
};

} // namespace rund::node

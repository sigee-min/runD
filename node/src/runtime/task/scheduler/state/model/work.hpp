#pragma once

namespace rund::node {

struct SchedulerWork {
  SchedulerWork *next = nullptr;
  void *context = nullptr;
  void (*invoke)(void *context) noexcept = nullptr;
  void (*released)(void *context) noexcept = nullptr;
};

} // namespace rund::node

#pragma once

#include <memory>

namespace rund::node {

struct ReactorPlatformState;

struct ReactorPlatformStateDelete {
  void operator()(ReactorPlatformState *state) const noexcept;
};

struct ReactorPlatform {
  std::unique_ptr<ReactorPlatformState, ReactorPlatformStateDelete> state{};
};

} // namespace rund::node

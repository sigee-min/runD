#pragma once

#include "../frame.hpp"

namespace rund::node {

class BindFrameArena final {
public:
  explicit BindFrameArena(FrameArena &arena) noexcept;
  ~BindFrameArena();
  BindFrameArena(const BindFrameArena &) = delete;
  BindFrameArena &operator=(const BindFrameArena &) = delete;

private:
  FrameArena *prior_{};
};

} // namespace rund::node

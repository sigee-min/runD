#include "bind.hpp"

namespace rund::detail::task::frame {
namespace {

thread_local ::rund::node::FrameArena *active = nullptr;
thread_local ReasonCode failure = ReasonCode::TaskFrameRuntimeMissing;
thread_local bool blocked = false;

} // namespace

void Bind(::rund::node::FrameArena *const arena) noexcept {
  active = arena;
  blocked = false;
  failure =
      arena == nullptr ? ReasonCode::TaskFrameRuntimeMissing : ReasonCode::Ok;
}

void Block() noexcept {
  active = nullptr;
  blocked = true;
  failure = ReasonCode::TaskFrameRuntimeMismatch;
}

::rund::node::FrameArena *Active() noexcept { return active; }

void *Acquire(const std::size_t bytes, const std::size_t alignment) noexcept {
  if (blocked) {
    failure = ReasonCode::TaskFrameRuntimeMismatch;
    return nullptr;
  }
  if (active == nullptr) {
    failure = ReasonCode::TaskFrameRuntimeMissing;
    return nullptr;
  }
  const ::rund::node::FrameLease lease = active->acquire(bytes, alignment);
  failure = lease ? ReasonCode::Ok : active->code();
  return lease.data;
}

void Release(void *const frame) noexcept {
  if (frame == nullptr) {
    return;
  }
  ::rund::node::FrameArena::release_frame(frame);
}

ReasonCode TakeFailure() noexcept {
  const ReasonCode result = failure;
  failure = ReasonCode::TaskFrameRuntimeMissing;
  return result;
}

std::uint32_t Bytes(void *const frame) noexcept {
  return ::rund::node::FrameArena::frame_bytes(frame);
}

bool Reused(void *const frame) noexcept {
  return ::rund::node::FrameArena::frame_reused(frame);
}

} // namespace rund::detail::task::frame

namespace rund::node {

BindFrameArena::BindFrameArena(FrameArena &arena) noexcept
    : prior_(::rund::detail::task::frame::Active()) {
  if (prior_ != nullptr && prior_ != &arena) {
    ::rund::detail::task::frame::Block();
  } else {
    ::rund::detail::task::frame::Bind(&arena);
  }
}

BindFrameArena::~BindFrameArena() { ::rund::detail::task::frame::Bind(prior_); }

} // namespace rund::node

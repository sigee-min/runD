#pragma once

#include "../readiness/state.hpp"
#include "result.hpp"
#include "state.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::node {

struct ReactorPlatformRegistration {
  ReactorHandle handle = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
};

class ReactorRegistrationChange final {
public:
  enum class Kind : std::uint8_t {
    Add,
    Modify,
    CleanupRemove,
  };

  [[nodiscard]] static constexpr ReactorRegistrationChange
  add(const ReactorHandle handle, const ReactorInterest interest,
      const std::uint64_t fd_generation) noexcept {
    return ReactorRegistrationChange{Kind::Add, handle, interest,
                                     fd_generation};
  }

  [[nodiscard]] static constexpr ReactorRegistrationChange
  modify(const ReactorHandle handle, const ReactorInterest interest,
         const std::uint64_t fd_generation) noexcept {
    return ReactorRegistrationChange{Kind::Modify, handle, interest,
                                     fd_generation};
  }

  [[nodiscard]] static constexpr ReactorRegistrationChange
  cleanup_remove(const ReactorHandle handle,
                 const std::uint64_t fd_generation) noexcept {
    return ReactorRegistrationChange{Kind::CleanupRemove, handle,
                                     ReactorInterest::None, fd_generation};
  }

  [[nodiscard]] constexpr Kind kind() const noexcept { return kind_; }
  [[nodiscard]] constexpr ReactorHandle handle() const noexcept {
    return handle_;
  }
  [[nodiscard]] constexpr ReactorInterest interest() const noexcept {
    return interest_;
  }
  [[nodiscard]] constexpr std::uint64_t fd_generation() const noexcept {
    return fd_generation_;
  }
  [[nodiscard]] constexpr bool is_cleanup_remove() const noexcept {
    return kind_ == Kind::CleanupRemove;
  }

private:
  constexpr ReactorRegistrationChange(
      const Kind kind, const ReactorHandle handle,
      const ReactorInterest interest,
      const std::uint64_t fd_generation) noexcept
      : handle_(handle), fd_generation_(fd_generation), interest_(interest),
        kind_(kind) {}

  ReactorHandle handle_;
  std::uint64_t fd_generation_;
  ReactorInterest interest_;
  Kind kind_;
};

[[nodiscard]] ReactorPlatformOpResult
AddReactorPlatformInterest(ReactorPlatform &, ReactorHandle,
                           ReactorInterest) noexcept;
[[nodiscard]] ReactorPlatformOpResult
ModifyReactorPlatformInterest(ReactorPlatform &, ReactorHandle,
                              ReactorInterest) noexcept;
[[nodiscard]] ReactorPlatformOpResult
RemoveReactorPlatformInterest(ReactorPlatform &, ReactorHandle) noexcept;
[[nodiscard]] ReactorPlatformBatchResult ApplyReactorPlatformChanges(
    ReactorPlatform &, const ReactorRegistrationChange *, std::size_t) noexcept;

} // namespace rund::node

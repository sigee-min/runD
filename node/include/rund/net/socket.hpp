#pragma once

#include <rund/net/result.hpp>

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rund::net {

namespace detail {
struct SocketAccess;
}
struct CloseResult;

class Socket;

class SocketView final {
public:
  constexpr SocketView() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return slot_ != nullptr && generation_ != 0u;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
  [[nodiscard]] std::uint64_t id() const noexcept;

  friend constexpr bool operator==(const SocketView &,
                                   const SocketView &) noexcept = default;

private:
  friend class Socket;
  friend struct detail::SocketAccess;

  constexpr SocketView(void *const slot,
                       const std::uint64_t generation) noexcept
      : slot_(slot), generation_(generation) {}

  void *slot_ = nullptr;
  std::uint64_t generation_ = 0u;
};

static_assert(sizeof(SocketView) == 16u);
static_assert(std::is_trivially_copyable_v<SocketView>);

class Socket final {
public:
  constexpr Socket() noexcept = default;
  ~Socket() noexcept;

  Socket(const Socket &) = delete;
  Socket &operator=(const Socket &) = delete;
  Socket(Socket &&other) noexcept
      : slot_(std::exchange(other.slot_, nullptr)),
        generation_(std::exchange(other.generation_, 0u)) {}
  Socket &operator=(Socket &&other) noexcept;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return slot_ != nullptr && generation_ != 0u;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
  [[nodiscard]] std::uint64_t id() const noexcept;
  [[nodiscard]] constexpr SocketView view() const & noexcept {
    return SocketView{slot_, generation_};
  }
  SocketView view() const && = delete;
  [[nodiscard]] CloseResult close() noexcept;

private:
  friend struct detail::SocketAccess;

  constexpr Socket(void *const slot, const std::uint64_t generation) noexcept
      : slot_(slot), generation_(generation) {}

  void *slot_ = nullptr;
  std::uint64_t generation_ = 0u;
};

static_assert(sizeof(Socket) == 16u);
static_assert(!std::is_copy_constructible_v<Socket>);
static_assert(!std::is_copy_assignable_v<Socket>);
static_assert(std::is_nothrow_move_constructible_v<Socket>);
static_assert(std::is_nothrow_move_assignable_v<Socket>);

struct CloseResult : Status {
  using Status::Status;

  int native_error = 0;
};

struct NonblockingResult : Status {
  using Status::Status;

  int native_error = 0;
  bool enabled = false;
};

[[nodiscard]] NonblockingResult nonblocking(SocketView socket,
                                            bool enabled) noexcept;

} // namespace rund::net

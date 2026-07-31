#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

namespace rund::host::io {

namespace detail {
struct Access;
}

struct CloseResult;
class Fd;

class FdView final {
public:
  constexpr FdView() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return native_ >= 0 ? host_id_ == static_cast<std::uint64_t>(native_) + 1u
                        : native_ == -1 && host_id_ != 0u;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
  [[nodiscard]] constexpr std::uint64_t id() const noexcept {
    return valid() ? host_id_ : 0u;
  }

  friend constexpr bool operator==(const FdView &,
                                   const FdView &) noexcept = default;

private:
  friend class Fd;
  friend struct detail::Access;

  constexpr FdView(const int native, const std::uint64_t host_id) noexcept
      : native_(native), host_id_(host_id) {}

  int native_ = -1;
  std::uint64_t host_id_ = 0u;
};

static_assert(sizeof(FdView) == 16u);
static_assert(std::is_trivially_copyable_v<FdView>);

class Fd final {
public:
  constexpr Fd() noexcept = default;
  ~Fd() noexcept;

  Fd(const Fd &) = delete;
  Fd &operator=(const Fd &) = delete;
  Fd(Fd &&other) noexcept
      : native_(std::exchange(other.native_, -1)),
        host_id_(std::exchange(other.host_id_, 0u)) {}
  Fd &operator=(Fd &&other) noexcept;

  [[nodiscard]] constexpr bool valid() const noexcept { return view().valid(); }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
  [[nodiscard]] constexpr std::uint64_t id() const noexcept {
    return view().id();
  }
  [[nodiscard]] constexpr FdView view() const & noexcept {
    return FdView{native_, host_id_};
  }
  FdView view() const && = delete;
  [[nodiscard]] CloseResult close() noexcept;

private:
  friend Fd take_native_fd(int &) noexcept;
  friend Fd take_native_fd(int &&) noexcept;
  friend Fd replay_fd(std::uint64_t) noexcept;
  friend struct detail::Access;

  constexpr Fd(const int native, const std::uint64_t host_id) noexcept
      : native_(native), host_id_(host_id) {}

  int native_ = -1;
  std::uint64_t host_id_ = 0u;
};

static_assert(sizeof(Fd) == 16u);
static_assert(!std::is_copy_constructible_v<Fd>);
static_assert(!std::is_copy_assignable_v<Fd>);
static_assert(std::is_nothrow_move_constructible_v<Fd>);
static_assert(std::is_nothrow_move_assignable_v<Fd>);

} // namespace rund::host::io

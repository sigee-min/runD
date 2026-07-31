#pragma once

#include <rund/telemetry/event.hpp>

#include <concepts>
#include <memory>

namespace rund::telemetry {

class Sink;

namespace detail {

template <class Callable>
concept SinkObserver = requires(Callable &observer, const Event &event) {
  { observer(event) } -> std::same_as<void>;
};

void emit(const Sink &sink, const Event &event);

} // namespace detail

template <detail::SinkObserver Callable>
[[nodiscard]] Sink bind(Callable &observer,
                        Level level = Level::Basic) noexcept;

class Sink final {
public:
  constexpr Sink() noexcept = default;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return callback_ != nullptr;
  }
  [[nodiscard]] constexpr Level level() const noexcept { return level_; }

private:
  using Callback = void (*)(void *, const Event &);

  constexpr Sink(void *const context, const Callback callback,
                 const Level level) noexcept
      : context_(context), callback_(callback), level_(level) {}

  void *context_ = nullptr;
  Callback callback_ = nullptr;
  Level level_ = Level::Basic;

  template <detail::SinkObserver Callable>
  friend Sink bind(Callable &, Level) noexcept;
  friend void detail::emit(const Sink &, const Event &);
};

template <detail::SinkObserver Callable>
[[nodiscard]] Sink bind(Callable &observer, const Level level) noexcept {
  return Sink{
      const_cast<void *>(static_cast<const void *>(std::addressof(observer))),
      +[](void *const raw, const Event &event) {
        (*static_cast<Callable *>(raw))(event);
      },
      level};
}

} // namespace rund::telemetry

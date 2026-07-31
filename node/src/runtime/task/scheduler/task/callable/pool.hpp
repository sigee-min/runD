#pragma once

#include <rund/task/callable.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node {

class CallablePool final {
public:
  CallablePool() noexcept = default;
  ~CallablePool();
  CallablePool(const CallablePool &) = delete;
  CallablePool &operator=(const CallablePool &) = delete;

  [[nodiscard]] bool configure(std::uint32_t capacity) noexcept;
  void reset() noexcept;
  [[nodiscard]] ::rund::detail::task::Callable *
  claim(::rund::detail::task::Callable &&value) noexcept;
  // Lane destruction touches one owned slot; deterministic commit recycles it.
  void destroy(::rund::detail::task::Callable *value) noexcept;
  void release(::rund::detail::task::Callable *value) noexcept;

private:
  enum class SlotState : std::uint8_t { Free, Live, Destroyed };

  struct Slot final {
    Slot *next{};
    SlotState state{SlotState::Free};
    alignas(::rund::detail::task::Callable)
        std::byte storage[sizeof(::rund::detail::task::Callable)]{};
  };
  struct Page final {
    Slot *slots{};
    std::uint32_t count{};
  };

  [[nodiscard]] bool grow() noexcept;

  std::vector<Page> pages_{};
  Slot *free_{};
  std::uint32_t capacity_{};
  std::uint32_t created_{};
  std::uint32_t live_{};
};

} // namespace rund::node

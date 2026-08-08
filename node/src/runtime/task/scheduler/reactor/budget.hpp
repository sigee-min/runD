#pragma once

#include <cstddef>
#include <vector>

#include "model.hpp"

namespace rund::node {

class ReactorBudgetSelection final {
public:
  [[nodiscard]] constexpr bool ok() const noexcept { return ready_ != nullptr; }

  [[nodiscard]] constexpr const std::vector<ReactorReady> &
  ready() const noexcept {
    return *ready_;
  }

  [[nodiscard]] constexpr std::size_t consumed() const noexcept {
    return ready_ == nullptr ? 0u : ready_->size();
  }

private:
  friend ReactorBudgetSelection
  ReactorBudgetSelect(ReactorRuntime &reactor,
                      const std::vector<ReactorReady> &ordered,
                      std::size_t budget) noexcept;

  [[nodiscard]] static constexpr ReactorBudgetSelection failed() noexcept {
    return ReactorBudgetSelection{nullptr};
  }

  [[nodiscard]] static constexpr ReactorBudgetSelection
  selected(const std::vector<ReactorReady> &ready) noexcept {
    return ReactorBudgetSelection{&ready};
  }

  constexpr explicit ReactorBudgetSelection(
      const std::vector<ReactorReady> *const ready) noexcept
      : ready_(ready) {}

  const std::vector<ReactorReady> *ready_ = nullptr;
};

[[nodiscard]] ReactorBudgetSelection
ReactorBudgetSelect(ReactorRuntime &reactor,
                    const std::vector<ReactorReady> &ordered,
                    std::size_t budget) noexcept;

[[nodiscard]] std::size_t
ReactorBudgetExtendInvalidFdPrefix(const std::vector<ReactorReady> &ordered,
                                   std::size_t consumed) noexcept;

} // namespace rund::node

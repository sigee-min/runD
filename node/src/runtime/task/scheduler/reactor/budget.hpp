#pragma once

#include <cstddef>
#include <vector>

#include "model.hpp"

namespace rund::node {

struct ReactorBudgetSelection {
  const std::vector<ReactorReady> *ready = nullptr;
  std::size_t consumed = 0u;
  bool ok = true;
};

[[nodiscard]] ReactorBudgetSelection
ReactorBudgetSelect(ReactorRuntime &reactor,
                    const std::vector<ReactorReady> &ordered,
                    std::size_t budget) noexcept;

[[nodiscard]] std::size_t
ReactorBudgetExtendInvalidFdPrefix(const std::vector<ReactorReady> &ordered,
                                   std::size_t consumed) noexcept;

} // namespace rund::node

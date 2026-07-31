#pragma once

#include <span>

namespace rund::node::test_contract {

struct Case final {
  const char *name;
  int (*run)();
};

[[nodiscard]] std::span<const Case> case_table() noexcept;

} // namespace rund::node::test_contract

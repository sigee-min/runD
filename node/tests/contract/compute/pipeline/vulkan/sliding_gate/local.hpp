#pragma once

#include "../residency/local.hpp"

#include "src/accel/kernel/prepared/interface/api.hpp"

#include <atomic>
#include <cstdint>

namespace rund_node_test_pipeline::sliding_gate_detail {

struct Wait final {
  std::atomic_bool done{false};
  std::atomic<std::uint32_t> callback_count{};
  rund::node::accel::detail::KernelResult result{};
};

void Complete(void *, rund::node::accel::detail::KernelResult result) noexcept;

[[nodiscard]] bool RunTerminalFrontier();

} // namespace rund_node_test_pipeline::sliding_gate_detail

#pragma once

// Included by backend/run.hpp inside rund::node::accel::detail.

// Exact shared-template route demand frozen by Pipeline preparation before
// the first private backend route is materialized. `owner_count` counts unique
// prepared route owners in one stream and `route_copies` is the public
// generation stride (one ordinary stream or two transactional streams).
// `capacity` is their checked product and therefore sizes the complete shared
// template pool once, independent of which route observes the first miss.
struct BackendTemplateRouteDemand final {
  std::uint32_t owner_count{};
  std::uint32_t route_copies{};
  std::uint32_t capacity{};

  [[nodiscard]] constexpr bool empty() const noexcept {
    return owner_count == 0u && route_copies == 0u && capacity == 0u;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return owner_count != 0u && (route_copies == 1u || route_copies == 2u) &&
           capacity != 0u &&
           static_cast<std::uint64_t>(owner_count) * route_copies == capacity;
  }
};

struct BackendRun final {
  const rund::AccelDevice *pick = nullptr;
  const BackendOps *ops = nullptr;
  const KernelExecution *execution = nullptr;
  const BoundResets *resets = nullptr;
  const BoundStep *steps = nullptr;
  std::size_t step_count = 0u;
  std::uint64_t original_dispatch_count = 0u;
  std::uint64_t final_dispatch_count = 0u;
  std::uint64_t *traffic = nullptr;
  const KernelViewLayout *views = nullptr;
  const RunBinds *view_binds = nullptr;
  const KernelScratchLayout *scratch = nullptr;
  // Non-owning cold-preparation cursor. Prepared route resources retain any
  // immutable template owners they acquire from this registry.
  PreparedKernelTemplateRegistry *templates = nullptr;
  // Non-owning scalar cursor valid only during one private route
  // materialization. The cursor clears it before returning to the caller.
  BackendTemplateRouteDemand template_route_demand{};
  std::uint32_t *failed_node = nullptr;
};

#pragma once

#include <rund/compute/backend.hpp>
#include <rund/compute/flow/builder.hpp>
#include <rund/compute/target.hpp>

#include <cstdlib>
#include <cstdint>
#include <span>

namespace rund::node::test_contract {

[[nodiscard]] std::span<const rund::compute::Backend>
selected_compute_backends() noexcept;

[[nodiscard]] inline std::span<const rund::compute::Backend>
selected_accelerators() noexcept {
  auto backends = selected_compute_backends();
  if (!backends.empty() && backends.front() == rund::compute::Backend::Cpu) {
    backends = backends.subspan(1u);
  }
  return backends;
}

[[nodiscard]] inline bool
backend_selected(const rund::compute::Backend backend) noexcept {
  for (const auto selected : selected_compute_backends()) {
    if (selected == backend) {
      return true;
    }
  }
  return false;
}

inline void require_selected_backend(
    const rund::compute::Backend backend) noexcept {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  if (backend != rund::compute::Backend::Cpu) {
    std::abort();
  }
#else
  (void)backend;
#endif
}

[[nodiscard]] inline auto
target_for(const rund::compute::Backend backend,
           const std::uint32_t cpu_workers = 0u) noexcept {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  require_selected_backend(backend);
  return rund::compute::Target::cpu(cpu_workers);
#else
  switch (backend) {
  case rund::compute::Backend::Cpu:
    return rund::compute::Target::cpu(cpu_workers);
  case rund::compute::Backend::Metal:
    return rund::compute::Target::metal();
  case rund::compute::Backend::Vulkan:
    return rund::compute::Target::vulkan();
  case rund::compute::Backend::Unavailable:
    std::abort();
  }
  std::abort();
#endif
}

[[nodiscard]] inline auto flow_on(const rund::compute::Backend backend,
                                  const rund::compute::Target cpu) noexcept {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  require_selected_backend(backend);
  return rund::compute::on(rund::compute::Target::cpu(cpu.workers()));
#else
  return rund::compute::on(target_for(backend, cpu.workers()));
#endif
}

[[nodiscard]] inline auto
flow_on(const rund::compute::Backend backend) noexcept {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  require_selected_backend(backend);
  return rund::compute::on(rund::compute::Target::cpu());
#else
  return rund::compute::on(target_for(backend));
#endif
}

template <class Range>
[[nodiscard]] inline auto flow_on(const rund::compute::Backend backend,
                                  Range &input) {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  require_selected_backend(backend);
  return rund::compute::on(rund::compute::Target::cpu(), input);
#else
  return rund::compute::on(target_for(backend), input);
#endif
}

template <class Range>
[[nodiscard]] inline auto flow_on(const rund::compute::Backend backend,
                                  const rund::compute::Target cpu,
                                  Range &input) {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  require_selected_backend(backend);
  return rund::compute::on(rund::compute::Target::cpu(cpu.workers()), input);
#else
  return rund::compute::on(target_for(backend, cpu.workers()), input);
#endif
}

} // namespace rund::node::test_contract

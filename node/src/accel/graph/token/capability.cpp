#include <accel/kernel/value.hpp>

#include "../../backend/match.hpp"
#include "local.hpp"

#include <atomic>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace rund::node::accel::detail {
namespace {

struct KernelTokenDelete final {
  KernelToken *token = nullptr;

  void operator()(KernelToken *const value) const noexcept { delete value; }
};

[[nodiscard]] std::uint64_t NextKernelId() noexcept {
  static std::atomic<std::uint64_t> next{1u};
  std::uint64_t current = next.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<std::uint64_t>::max()) {
    if (next.compare_exchange_weak(current, current + 1u,
                                   std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      return current;
    }
  }
  return 0u;
}

[[nodiscard]] std::uint64_t
StepDynamicMemory(const KernelExecutionStep &step) noexcept {
  using rund::kernel::compute_retained_detail::Add;
  using rund::kernel::compute_retained_detail::VectorCapacityBytes;
  std::uint64_t bytes = step.artifact.retained_dynamic_memory_bytes();
  bytes = Add(bytes, step.cpu_input.retained_dynamic_memory_bytes());
  return Add(bytes,
             VectorCapacityBytes(step.graph_binding_indices.overflow_indices));
}

[[nodiscard]] std::uint64_t
StepListMemory(const std::vector<KernelExecutionStep> &steps) noexcept {
  using rund::kernel::compute_retained_detail::Add;
  using rund::kernel::compute_retained_detail::VectorCapacityBytes;
  std::uint64_t bytes = VectorCapacityBytes(steps);
  for (const KernelExecutionStep &step : steps) {
    bytes = Add(bytes, StepDynamicMemory(step));
  }
  return bytes;
}

} // namespace

std::shared_ptr<KernelToken> MakeKernelToken(KernelToken token) noexcept {
  token.kernel_id = NextKernelId();
  if (token.kernel_id == 0u) {
    return {};
  }
  KernelToken *const raw = new (std::nothrow) KernelToken(std::move(token));
  if (raw == nullptr) {
    return {};
  }
  try {
    std::unique_ptr<KernelToken, KernelTokenDelete> owner{
        raw, KernelTokenDelete{.token = raw}};
    return std::shared_ptr<KernelToken>{std::move(owner)};
  } catch (...) {
    return {};
  }
}

std::shared_ptr<KernelToken>
LookupKernelToken(const std::shared_ptr<void> &owner,
                  const std::uint64_t kernel_id) noexcept {
  if (owner == nullptr || kernel_id == 0u) {
    return {};
  }
  const auto *const capability = std::get_deleter<KernelTokenDelete>(owner);
  if (capability == nullptr || capability->token == nullptr ||
      capability->token->kernel_id != kernel_id) {
    return {};
  }
  return std::shared_ptr<KernelToken>{owner, capability->token};
}

KernelTokenRetainedMemory
MeasureKernelTokenRetainedMemory(const rund::AccelKernel &kernel) noexcept {
  try {
    const std::shared_ptr<KernelToken> token =
        LookupKernelToken(kernel.owner, kernel.kernel_id);
    if (token == nullptr || !SameObject(token, kernel.owner) ||
        token->kernel_id != kernel.kernel_id ||
        token->context_id != kernel.context_id ||
        token->graph_id_hi != kernel.graph_id_hi ||
        token->graph_id_lo != kernel.graph_id_lo ||
        token->node_count != kernel.node_count || token->api != kernel.api ||
        token->scalar != kernel.scalar || token->domain != kernel.domain) {
      return {};
    }

    using rund::kernel::compute_retained_detail::Add;
    using rund::kernel::compute_retained_detail::VectorCapacityBytes;
    std::uint64_t bytes = sizeof(KernelToken);
    bytes = Add(bytes, VectorCapacityBytes(token->graph_roles));
    bytes = Add(bytes, VectorCapacityBytes(token->graph_shapes));
    bytes = Add(bytes, VectorCapacityBytes(token->graph_visibilities));
    bytes = Add(bytes, VectorCapacityBytes(token->graph_alias_representatives));
    bytes = Add(bytes, VectorCapacityBytes(token->resets));
    bytes = Add(bytes, VectorCapacityBytes(token->required_barriers));
    bytes = Add(bytes, StepListMemory(token->steps));
    return KernelTokenRetainedMemory{.host_bytes = bytes, .exact = true};
  } catch (...) {
    return {};
  }
}

} // namespace rund::node::accel::detail

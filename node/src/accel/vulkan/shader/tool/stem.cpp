#include "local.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <atomic>
#if defined(_WIN32)
#include <process.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] std::uint64_t ProcessId() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(_getpid());
#elif defined(__unix__) || defined(__APPLE__)
  return static_cast<std::uint64_t>(getpid());
#else
  return 0u;
#endif
}

}  // namespace

std::string BuildVulkanShaderStem(const rund::kernel::ComputePlan& plan) {
  static std::atomic_uint64_t counter{0u};
  const std::uint64_t id = counter.fetch_add(1u, std::memory_order_relaxed);
  std::string stem = "shader-";
  stem += std::to_string(static_cast<unsigned long long>(ProcessId()));
  stem += "-";
  stem += std::to_string(id);
  stem += "-";
  AppendHex64Digits(stem, plan.op_hash_hi);
  stem += "-";
  AppendHex64Digits(stem, plan.op_hash_lo);
  return stem;
}

}  // namespace rund::node::accel::detail

#endif  // defined(RUND_NODE_HAVE_VULKAN_SDK)

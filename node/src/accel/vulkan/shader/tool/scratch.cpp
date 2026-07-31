#include "local.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <cstdlib>
#include <system_error>

namespace rund::node::accel::detail {

std::filesystem::path VulkanShaderScratchDir() {
  const char* const configured = std::getenv("RUND_NODE_VULKAN_SCRATCH_DIR");
  if (configured != nullptr && configured[0] != '\0') {
    return std::filesystem::path{configured};
  }
  std::error_code ec{};
  std::filesystem::path root = std::filesystem::temp_directory_path(ec);
  if (ec || root.empty()) {
    root = std::filesystem::current_path();
  }
  return root / "rund-node-vulkan";
}

} // namespace rund::node::accel::detail

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

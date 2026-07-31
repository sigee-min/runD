#pragma once

#include "../../adapter/api.hpp"
#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <filesystem>
#include <string>
#include <vector>

namespace rund::node::accel::detail {

void AppendHex64Digits(std::string &out, rund::kernel::u64 value);

[[nodiscard]] std::filesystem::path VulkanShaderScratchDir();

[[nodiscard]] bool RunProcess(const std::vector<std::string> &args);

[[nodiscard]] bool ReadSpirv(const std::filesystem::path &path,
                             VulkanShader &shader);

[[nodiscard]] std::string
BuildVulkanShaderStem(const rund::kernel::ComputePlan &plan);

} // namespace rund::node::accel::detail

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../clock.hpp"
#include "../../runtime/counter.hpp"
#include "../cache.hpp"
#include "local.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <fstream>
#include <system_error>

namespace rund::node::accel::detail {

bool CompileVulkanSourceWithTools(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact, VulkanShader &shader) {
  if (FindValidatedVulkanSpirv(adapter.glslang_validator_path,
                               adapter.spirv_val_path, artifact.source_text,
                               shader)) {
    return true;
  }
  std::error_code ec{};
  const std::filesystem::path scratch_dir = VulkanShaderScratchDir();
  std::filesystem::create_directories(scratch_dir, ec);
  if (ec) {
    SetVulkanLastError(adapter, "accel_vulkan_shader_compile_failed");
    return false;
  }

  const std::string stem = BuildVulkanShaderStem(plan);
  const std::filesystem::path source_path = scratch_dir / (stem + ".comp");
  const std::filesystem::path spirv_path = scratch_dir / (stem + ".spv");

  {
    std::ofstream out{source_path, std::ios::binary};
    out.write(artifact.source_text.data(),
              static_cast<std::streamsize>(artifact.source_text.size()));
    if (!out) {
      std::filesystem::remove(source_path, ec);
      SetVulkanLastError(adapter, "accel_vulkan_shader_compile_failed");
      return false;
    }
  }

  const std::uint64_t compile_begin = MonotonicNanoseconds();
  const bool compiled = RunProcess({
      adapter.glslang_validator_path,
      "-V",
      "--target-env",
      "vulkan1.1",
      "-S",
      "comp",
      "-e",
      "main",
      "-o",
      spirv_path.string(),
      source_path.string(),
  });
  RecordVulkanSpirvCompileNs(adapter, MonotonicNanoseconds() - compile_begin);
  if (!compiled) {
    std::filesystem::remove(source_path, ec);
    std::filesystem::remove(spirv_path, ec);
    SetVulkanLastError(adapter, "accel_vulkan_shader_process_failed");
    return false;
  }

  if (!ReadSpirv(spirv_path, shader)) {
    std::filesystem::remove(source_path, ec);
    std::filesystem::remove(spirv_path, ec);
    SetVulkanLastError(adapter, "accel_vulkan_spirv_read_failed");
    return false;
  }

  if (!adapter.spirv_val_path.empty()) {
    const std::uint64_t validate_begin = MonotonicNanoseconds();
    const bool validated = RunProcess({
        adapter.spirv_val_path,
        "--target-env",
        "vulkan1.1",
        spirv_path.string(),
    });
    RecordVulkanSpirvCompileNs(adapter,
                               MonotonicNanoseconds() - validate_begin);
    if (!validated) {
      std::filesystem::remove(source_path, ec);
      std::filesystem::remove(spirv_path, ec);
      SetVulkanLastError(adapter, "accel_vulkan_spirv_invalid");
      return false;
    }
  }

  std::filesystem::remove(source_path, ec);
  std::filesystem::remove(spirv_path, ec);
  CacheValidatedVulkanSpirv(adapter.glslang_validator_path,
                            adapter.spirv_val_path, artifact.source_text,
                            shader);
  return true;
}

} // namespace rund::node::accel::detail

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

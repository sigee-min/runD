#include "support.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "../../../../../src/accel/vulkan/shader/api.hpp"
#include "../../../../../src/accel/vulkan/adapter/state.hpp"
#include "../../../../../src/accel/vulkan/kernel/pipeline/source.hpp"
#include "../../../../../src/accel/vulkan/shader/cache.hpp"
#include "../../../../../src/accel/vulkan/shader/module.hpp"

#include <rund/counter.hpp>
#include <kernel/program/compute/artifact.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#endif

namespace rund::node::vulkan_cache_contract {

[[nodiscard]] int VulkanShaderCache() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  ClearValidatedVulkanSpirvCache();
  auto words = std::make_shared<const std::vector<std::uint32_t>>(
      std::vector<std::uint32_t>{0x07230203u, 1u, 2u, 3u});
  const VulkanShader original{.words = words, .hash = 17u};
  CacheValidatedVulkanSpirv("compiler-a", "validator-a", "source-a", original);
  if (ValidatedVulkanSpirvCacheSize() != 1u ||
      ValidatedVulkanSpirvCacheBytes() == 0u ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity) {
    return 2;
  }

  VulkanShader found{};
  if (!FindValidatedVulkanSpirv("compiler-a", "validator-a", "source-a",
                                found) ||
      found.words != words || found.hash != original.hash) {
    return 3;
  }
  if (FindValidatedVulkanSpirv("compiler-b", "validator-a", "source-a",
                               found) ||
      FindValidatedVulkanSpirv("compiler-a", "validator-b", "source-a",
                               found) ||
      FindValidatedVulkanSpirv("compiler-a", "validator-a", "source-b",
                               found)) {
    return 4;
  }

  for (std::size_t index = 0u; index < kVulkanSpirvCacheCapacity; ++index) {
    const std::string source = "bounded-source-" + std::to_string(index);
    CacheValidatedVulkanSpirv("compiler-a", "validator-a", source, original);
  }
  if (ValidatedVulkanSpirvCacheSize() != kVulkanSpirvCacheCapacity ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity ||
      FindValidatedVulkanSpirv("compiler-a", "validator-a", "source-a",
                               found) ||
      !FindValidatedVulkanSpirv(
          "compiler-a", "validator-a",
          "bounded-source-" + std::to_string(kVulkanSpirvCacheCapacity - 1u),
          found) ||
      found.words != words) {
    return 5;
  }
  ClearValidatedVulkanSpirvCache();
  if (ValidatedVulkanSpirvCacheSize() != 0u ||
      ValidatedVulkanSpirvCacheBytes() != 0u || found.words != words ||
      found.words->size() != 4u || found.words->front() != 0x07230203u) {
    return 6;
  }
  const std::string oversized_source(kVulkanSpirvCacheByteCapacity, 'x');
  CacheValidatedVulkanSpirv("compiler-a", "validator-a", oversized_source,
                            original);
  if (ValidatedVulkanSpirvCacheSize() != 0u ||
      ValidatedVulkanSpirvCacheBytes() != 0u) {
    return 7;
  }

  constexpr std::size_t kByteEntryCount = 20u;
  constexpr std::size_t kByteSourceSize = 1u * 1024u * 1024u;
  constexpr std::string_view kByteCompiler = "byte-compiler";
  constexpr std::string_view kByteValidator = "byte-validator";
  const std::size_t byte_entry_size = kByteSourceSize + kByteCompiler.size() +
                                      kByteValidator.size() +
                                      words->size() * sizeof(std::uint32_t);
  const std::size_t expected_byte_entries =
      std::min(kVulkanSpirvCacheCapacity,
               kVulkanSpirvCacheByteCapacity / byte_entry_size);
  std::string byte_source(kByteSourceSize, 'x');
  for (std::size_t index = 0u; index < kByteEntryCount; ++index) {
    byte_source.back() = static_cast<char>('a' + index);
    CacheValidatedVulkanSpirv(kByteCompiler, kByteValidator, byte_source,
                              original);
  }
  std::string first_byte_source(kByteSourceSize, 'x');
  first_byte_source.back() = 'a';
  std::string last_byte_source(kByteSourceSize, 'x');
  last_byte_source.back() = static_cast<char>('a' + kByteEntryCount - 1u);
  if (ValidatedVulkanSpirvCacheSize() != expected_byte_entries ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity ||
      FindValidatedVulkanSpirv(kByteCompiler, kByteValidator, first_byte_source,
                               found) ||
      !FindValidatedVulkanSpirv(kByteCompiler, kByteValidator, last_byte_source,
                                found) ||
      found.words != words) {
    return 8;
  }
  ClearValidatedVulkanSpirvCache();

  constexpr std::size_t kThreadCount = 8u;
  constexpr std::size_t kExactEntriesPerThread = 16u;
  std::atomic_bool concurrent_exact{true};
  std::array<std::thread, kThreadCount> exact_threads{};
  for (std::size_t thread_index = 0u; thread_index < kThreadCount;
       ++thread_index) {
    exact_threads[thread_index] =
        std::thread{[thread_index, &original, &words, &concurrent_exact]() {
          const std::string compiler =
              "concurrent-compiler-" + std::to_string(thread_index);
          const std::string validator =
              "concurrent-validator-" + std::to_string(thread_index);
          for (std::size_t entry_index = 0u;
               entry_index < kExactEntriesPerThread; ++entry_index) {
            const std::string source = "concurrent-source-" +
                                       std::to_string(thread_index) + "-" +
                                       std::to_string(entry_index);
            CacheValidatedVulkanSpirv(compiler, validator, source, original);
            VulkanShader concurrent_found{};
            if (!FindValidatedVulkanSpirv(compiler, validator, source,
                                          concurrent_found) ||
                concurrent_found.words != words ||
                concurrent_found.hash != original.hash ||
                FindValidatedVulkanSpirv(compiler + "-other", validator, source,
                                         concurrent_found) ||
                FindValidatedVulkanSpirv(compiler, validator + "-other", source,
                                         concurrent_found) ||
                FindValidatedVulkanSpirv(compiler, validator, source + "-other",
                                         concurrent_found)) {
              concurrent_exact.store(false, std::memory_order_relaxed);
            }
          }
        }};
  }
  for (std::thread &thread : exact_threads) {
    thread.join();
  }
  if (!concurrent_exact.load(std::memory_order_relaxed) ||
      ValidatedVulkanSpirvCacheSize() !=
          kThreadCount * kExactEntriesPerThread ||
      ValidatedVulkanSpirvCacheBytes() == 0u ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity) {
    return 9;
  }

  ClearValidatedVulkanSpirvCache();
  constexpr std::size_t kBoundedEntriesPerThread = 64u;
  std::array<std::thread, kThreadCount> bounded_threads{};
  for (std::size_t thread_index = 0u; thread_index < kThreadCount;
       ++thread_index) {
    bounded_threads[thread_index] = std::thread{[thread_index, &original]() {
      for (std::size_t entry_index = 0u; entry_index < kBoundedEntriesPerThread;
           ++entry_index) {
        const std::string source = "concurrent-bounded-source-" +
                                   std::to_string(thread_index) + "-" +
                                   std::to_string(entry_index);
        CacheValidatedVulkanSpirv("concurrent-compiler", "concurrent-validator",
                                  source, original);
      }
    }};
  }
  for (std::thread &thread : bounded_threads) {
    thread.join();
  }
  if (ValidatedVulkanSpirvCacheSize() != kVulkanSpirvCacheCapacity ||
      ValidatedVulkanSpirvCacheBytes() == 0u ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity) {
    return 10;
  }
  ClearValidatedVulkanSpirvCache();

  constexpr std::string_view kToolSource = R"(
#version 450
layout(local_size_x = 1) in;
void main() {}
)";
  rund::kernel::ComputePlan tool_plan{};
  rund::kernel::LoweringArtifact tool_artifact{};
  tool_artifact.source_text = kToolSource;
  VulkanAdapter tool_adapter{};
  tool_adapter.glslang_validator_path =
      "/rund-node-intentionally-missing-compiler";
  VulkanShader tool_shader{};
  if (CompileVulkanSourceWithTools(tool_adapter, tool_plan, tool_artifact,
                                   tool_shader) ||
      ValidatedVulkanSpirvCacheSize() != 0u) {
    return 11;
  }
#if defined(RUND_NODE_TEST_GLSLANG_VALIDATOR_PATH)
  tool_adapter.glslang_validator_path = RUND_NODE_TEST_GLSLANG_VALIDATOR_PATH;
  tool_adapter.spirv_val_path.clear();
  if (!CompileVulkanSourceWithTools(tool_adapter, tool_plan, tool_artifact,
                                    tool_shader) ||
      ValidatedVulkanSpirvCacheSize() != 1u) {
    return 12;
  }
  VulkanShader published = tool_shader;
  tool_shader = {};
  if (!CompileVulkanSourceWithTools(tool_adapter, tool_plan, tool_artifact,
                                    tool_shader) ||
      tool_shader.words != published.words ||
      ValidatedVulkanSpirvCacheSize() != 1u) {
    return 13;
  }
  ClearValidatedVulkanSpirvCache();
  tool_adapter.spirv_val_path = "/rund-node-intentionally-missing-validator";
  tool_shader = {};
  if (CompileVulkanSourceWithTools(tool_adapter, tool_plan, tool_artifact,
                                   tool_shader) ||
      ValidatedVulkanSpirvCacheSize() != 0u ||
      ValidatedVulkanSpirvCacheBytes() != 0u) {
    return 14;
  }
#endif
  return 0;
#else
  return 0;
#endif
}

} // namespace rund::node::vulkan_cache_contract

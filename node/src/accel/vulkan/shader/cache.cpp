#include "cache.hpp"

#include "../../../hash/fnv.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <string>

namespace rund::node::accel::detail {
namespace {

struct VulkanSpirvCacheEntry final {
  std::uint64_t key_hash = 0u;
  std::size_t byte_size = 0u;
  std::string compiler{};
  std::string validator{};
  std::string source{};
  VulkanShader shader{};
};

struct VulkanSpirvCache final {
  std::mutex mutex{};
  std::deque<VulkanSpirvCacheEntry> entries{};
  std::size_t bytes = 0u;
};

[[nodiscard]] VulkanSpirvCache& SpirvCache() {
  static VulkanSpirvCache cache{};
  return cache;
}

void Mix(::rund::node::hash_detail::Fnv &hash,
         const std::string_view text) noexcept {
  for (const char value : text) {
    hash.Byte(static_cast<unsigned char>(value));
  }
  hash.Byte(0xffu);
}

[[nodiscard]] std::uint64_t CacheKeyHash(const std::string_view compiler,
                                         const std::string_view validator,
                                         const std::string_view source) {
  ::rund::node::hash_detail::Fnv hash{};
  Mix(hash, compiler);
  Mix(hash, validator);
  Mix(hash, source);
  return hash.Finish();
}

[[nodiscard]] bool ValidShader(const VulkanShader& shader) noexcept {
  return shader.words != nullptr && !shader.words->empty() &&
         shader.words->front() == 0x07230203u && shader.hash != 0u;
}

[[nodiscard]] std::optional<std::size_t>
CacheEntryBytes(const std::string_view compiler,
                const std::string_view validator,
                const std::string_view source,
                const VulkanShader& shader) noexcept {
  if (!ValidShader(shader) ||
      shader.words->size() >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
    return std::nullopt;
  }
  std::size_t bytes = shader.words->size() * sizeof(std::uint32_t);
  for (const std::size_t size :
       {compiler.size(), validator.size(), source.size()}) {
    if (size > std::numeric_limits<std::size_t>::max() - bytes) {
      return std::nullopt;
    }
    bytes += size;
  }
  return bytes;
}

} // namespace

bool FindValidatedVulkanSpirv(const std::string_view compiler,
                              const std::string_view validator,
                              const std::string_view source,
                              VulkanShader& shader) {
  const std::uint64_t key_hash = CacheKeyHash(compiler, validator, source);
  VulkanSpirvCache& cache = SpirvCache();
  std::lock_guard lock{cache.mutex};
  for (const VulkanSpirvCacheEntry& entry : cache.entries) {
    if (entry.key_hash == key_hash && entry.compiler == compiler &&
        entry.validator == validator && entry.source == source) {
      shader = entry.shader;
      return true;
    }
  }
  return false;
}

void CacheValidatedVulkanSpirv(const std::string_view compiler,
                               const std::string_view validator,
                               const std::string_view source,
                               const VulkanShader& shader) {
  const std::optional<std::size_t> byte_size =
      CacheEntryBytes(compiler, validator, source, shader);
  if (!byte_size || *byte_size > kVulkanSpirvCacheByteCapacity) {
    return;
  }
  const std::uint64_t key_hash = CacheKeyHash(compiler, validator, source);
  VulkanSpirvCache& cache = SpirvCache();
  std::lock_guard lock{cache.mutex};
  for (const VulkanSpirvCacheEntry& entry : cache.entries) {
    if (entry.key_hash == key_hash && entry.compiler == compiler &&
        entry.validator == validator && entry.source == source) {
      return;
    }
  }
  while (!cache.entries.empty() &&
         (cache.entries.size() == kVulkanSpirvCacheCapacity ||
          cache.bytes > kVulkanSpirvCacheByteCapacity - *byte_size)) {
    cache.bytes -= cache.entries.front().byte_size;
    cache.entries.pop_front();
  }
  cache.entries.push_back(VulkanSpirvCacheEntry{
      .key_hash = key_hash,
      .byte_size = *byte_size,
      .compiler = std::string{compiler},
      .validator = std::string{validator},
      .source = std::string{source},
      .shader = shader,
  });
  cache.bytes += *byte_size;
}

std::size_t ValidatedVulkanSpirvCacheSize() {
  VulkanSpirvCache& cache = SpirvCache();
  std::lock_guard lock{cache.mutex};
  return cache.entries.size();
}

std::size_t ValidatedVulkanSpirvCacheBytes() {
  VulkanSpirvCache& cache = SpirvCache();
  std::lock_guard lock{cache.mutex};
  return cache.bytes;
}

void ClearValidatedVulkanSpirvCache() {
  VulkanSpirvCache& cache = SpirvCache();
  std::lock_guard lock{cache.mutex};
  cache.entries.clear();
  cache.bytes = 0u;
}

} // namespace rund::node::accel::detail

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

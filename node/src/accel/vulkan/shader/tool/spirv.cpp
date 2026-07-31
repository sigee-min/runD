#include "local.hpp"

#include "../../../../hash/fnv.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <fstream>
#include <limits>

namespace rund::node::accel::detail {

bool ReadSpirv(const std::filesystem::path &path, VulkanShader &shader) {
  std::ifstream input{path, std::ios::binary | std::ios::ate};
  if (!input.is_open()) {
    return false;
  }
  const std::streamoff end = input.tellg();
  if (end <= 0) {
    return false;
  }
  const std::uint64_t size = static_cast<std::uint64_t>(end);
  if (size >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      size > static_cast<std::uint64_t>(
                 std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  input.seekg(0, std::ios::beg);
  if (!input.read(reinterpret_cast<char *>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()))) {
    return false;
  }
  if (bytes.empty() || bytes.size() % 4u != 0u) {
    return false;
  }

  const std::size_t word_count = bytes.size() / 4u;
  ::rund::node::hash_detail::Fnv hash{};
  std::vector<std::uint32_t> words{};
  words.reserve(word_count);
  for (std::size_t index = 0u; index < word_count; ++index) {
    const std::size_t offset = index * 4u;
    hash.Byte(static_cast<std::uint8_t>(bytes[offset]));
    hash.Byte(static_cast<std::uint8_t>(bytes[offset + 1u]));
    hash.Byte(static_cast<std::uint8_t>(bytes[offset + 2u]));
    hash.Byte(static_cast<std::uint8_t>(bytes[offset + 3u]));
    words.push_back(static_cast<std::uint32_t>(bytes[offset]) |
                    (static_cast<std::uint32_t>(bytes[offset + 1u]) << 8u) |
                    (static_cast<std::uint32_t>(bytes[offset + 2u]) << 16u) |
                    (static_cast<std::uint32_t>(bytes[offset + 3u]) << 24u));
  }
  if (words.empty() || words[0] != 0x07230203u || hash.Finish() == 0u) {
    return false;
  }
  shader.words =
      std::make_shared<const std::vector<std::uint32_t>>(std::move(words));
  shader.hash = hash.Finish();
  return true;
}

} // namespace rund::node::accel::detail

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

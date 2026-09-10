#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace rund::node::accel::detail::device_vsm_source {

inline constexpr std::string_view CanonicalVariant =
    "// artifact_variant=canonical";
inline constexpr std::string_view DeviceVariant =
    "// artifact_variant=device_vsm";

[[nodiscard]] inline bool replace_one(std::string &source,
                                      const std::string_view before,
                                      const std::string_view after) {
  const std::size_t at = source.find(before);
  if (at == std::string::npos ||
      source.find(before, at + before.size()) != std::string::npos) {
    return false;
  }
  source.replace(at, before.size(), after);
  return true;
}

[[nodiscard]] bool transform_metal(std::string &, std::uint64_t, std::uint32_t);
[[nodiscard]] bool transform_vulkan(std::string &, std::uint64_t,
                                    std::uint32_t);

} // namespace rund::node::accel::detail::device_vsm_source

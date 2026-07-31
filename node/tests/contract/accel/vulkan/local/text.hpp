#pragma once

#include "tile.hpp"

#include <string>
#include <string_view>

namespace node_accel_contract::vulkan {

[[nodiscard]] inline std::string Hex64WithPrefix(
    const rund::kernel::u64 value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string out = "0x";
  for (int shift = 60; shift >= 0; shift -= 4) {
    out.push_back(kHex[(value >> static_cast<unsigned>(shift)) & 0xfu]);
  }
  return out;
}

[[nodiscard]] inline std::string KeyLine(const std::string_view name,
                                         const rund::kernel::u64 value) {
  std::string line = "// ";
  line += name;
  line += "=";
  line += Hex64WithPrefix(value);
  line += "\n";
  return line;
}

[[nodiscard]] inline bool ReplaceAll(std::string& text,
                                     const std::string_view from,
                                     const std::string_view to) {
  bool changed = false;
  std::size_t offset = 0u;
  while ((offset = text.find(from, offset)) != std::string::npos) {
    text.replace(offset, from.size(), to);
    offset += to.size();
    changed = true;
  }
  return changed;
}

}  // namespace node_accel_contract::vulkan

#pragma once

#include <kernel/core/model.hpp>

#include <string>
#include <string_view>

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] constexpr char HexDigit(const u8 value) noexcept {
  return value < 10u ? static_cast<char>('0' + value)
                     : static_cast<char>('a' + (value - 10u));
}

inline void AppendHexByte(std::string& out, const u8 value) {
  out += HexDigit(static_cast<u8>((value >> 4u) & 0x0fu));
  out += HexDigit(static_cast<u8>(value & 0x0fu));
}

inline void AppendHex64Digits(std::string& out, const u64 value) {
  constexpr char kHex[] = "0123456789abcdef";
  for (int shift = 60; shift >= 0; shift -= 4) {
    out += kHex[(value >> static_cast<unsigned>(shift)) & 0xfu];
  }
}

inline void AppendHex64(std::string& out, const u64 value) {
  out += "0x";
  AppendHex64Digits(out, value);
}

[[nodiscard]] inline std::string HexText(const std::string_view text) {
  std::string out;
  out.reserve(text.size() * 2u);
  for (const char c : text) {
    AppendHexByte(out, static_cast<u8>(static_cast<unsigned char>(c)));
  }
  return out;
}

[[nodiscard]] inline std::string SafeIdentifier(
    const std::string_view text) {
  if (text.empty()) {
    return "empty";
  }
  std::string out;
  out.reserve(text.size() * 2u);
  for (const char c : text) {
    AppendHexByte(out, static_cast<u8>(static_cast<unsigned char>(c)));
  }
  return out;
}

}  // namespace rund::kernel::compute_lowering_detail

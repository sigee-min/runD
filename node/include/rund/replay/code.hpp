#pragma once

#include <cstdint>
#include <string_view>

namespace rund::replay {

enum class Family : std::uint8_t {
  None = 0u,
  Replay = 1u,
  Runtime = 2u,
  Host = 3u,
  Accel = 4u,
  Kernel = 5u,
  Codec = 6u,
};

enum class Code : std::uint32_t {
  Ok = 0u,
#define RUND_REPLAY_DOMAIN_Replay 1u
#define RUND_REPLAY_DOMAIN_Host 3u
#define RUND_REPLAY_DOMAIN_Accel 4u
#define RUND_REPLAY_DOMAIN_Kernel 5u
#define RUND_REPLAY_DOMAIN_Codec 6u
#define RUND_REPLAY_CODE(domain, value, name, text)                            \
  name = (RUND_REPLAY_DOMAIN_##domain << 16u) | value,
#include <rund/replay/code.def>
#undef RUND_REPLAY_CODE
#undef RUND_REPLAY_DOMAIN_Codec
#undef RUND_REPLAY_DOMAIN_Kernel
#undef RUND_REPLAY_DOMAIN_Accel
#undef RUND_REPLAY_DOMAIN_Host
#undef RUND_REPLAY_DOMAIN_Replay
#define RUND_REPLAY_RUNTIME_Universal(value, name, text)
#define RUND_REPLAY_RUNTIME_General(value, name, text)                         \
  name = (2u << 16u) | value,
#define RUND_REPLAY_RUNTIME_PreparedMemory(value, name, text)                  \
  name = (2u << 16u) | value,
#define RUND_NODE_REASON(value, name, text, category)                          \
  RUND_REPLAY_RUNTIME_##category(value, name, text)
#include <rund/reason.def>
#undef RUND_NODE_REASON
#undef RUND_REPLAY_RUNTIME_PreparedMemory
#undef RUND_REPLAY_RUNTIME_General
#undef RUND_REPLAY_RUNTIME_Universal
};

[[nodiscard]] constexpr std::uint32_t raw(const Code code) noexcept {
  return static_cast<std::uint32_t>(code);
}

[[nodiscard]] constexpr bool ok(const Code code) noexcept {
  return code == Code::Ok;
}

[[nodiscard]] constexpr int exit_code(const Code code) noexcept {
  return ok(code) ? 0 : 1;
}

[[nodiscard]] constexpr Family family(const Code code) noexcept {
  const auto domain = static_cast<std::uint16_t>(raw(code) >> 16u);
  return domain <= static_cast<std::uint16_t>(Family::Codec)
             ? static_cast<Family>(domain)
             : Family::None;
}

[[nodiscard]] bool valid(Code code) noexcept;
[[nodiscard]] std::string_view error(Code code) noexcept;

} // namespace rund::replay

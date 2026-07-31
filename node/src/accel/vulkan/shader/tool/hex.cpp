#include "local.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund::node::accel::detail {

void AppendHex64Digits(std::string& out, const rund::kernel::u64 value) {
  constexpr char kHex[] = "0123456789abcdef";
  for (int shift = 60; shift >= 0; shift -= 4) {
    out.push_back(kHex[(value >> static_cast<unsigned>(shift)) & 0xfu]);
  }
}

}  // namespace rund::node::accel::detail

#endif  // defined(RUND_NODE_HAVE_VULKAN_SDK)

#pragma once

#include "../../state.hpp"
#include <string>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

void AppendMetalArtifactHex64(std::string& out,
                              const rund::kernel::u64 value) {
  constexpr char kHex[] = "0123456789abcdef";
  for (int shift = 60; shift >= 0; shift -= 4) {
    out.push_back(kHex[(value >> static_cast<unsigned>(shift)) & 0xfu]);
  }
}

[[nodiscard]] NSString* MetalMapArtifactFunctionName(
    const rund::kernel::ArtifactKey& key) {
  std::string name = "rund_compute_map_";
  AppendMetalArtifactHex64(name, key.op_hash_hi);
  name.push_back('_');
  AppendMetalArtifactHex64(name, key.op_hash_lo);
  if (key.variant == rund::kernel::LoweringArtifactVariant::Controlled) {
    name += "_controlled";
  } else if (key.variant ==
             rund::kernel::LoweringArtifactVariant::Recurrence) {
    name += "_recurrence";
  }
  return [NSString stringWithUTF8String:name.c_str()];
}

#endif

}  // namespace rund::node::accel::detail

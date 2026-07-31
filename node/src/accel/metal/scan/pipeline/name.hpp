#pragma once

#include "../pipeline.hpp"

#include <string>
#include <string_view>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] NSString* MetalScanFunctionName(
    const char* const base,
    const rund::kernel::ScanElement elem) {
  std::string name = base;
  name += elem == rund::kernel::ScanElement::U64 ? "_u64" : "_u32";
  return [NSString stringWithUTF8String:name.c_str()];
}

[[nodiscard]] std::string MetalScanPipelineKey(
    const std::string_view pass,
    const rund::kernel::ScanElement element) {
  std::string key = "scan.";
  key += pass;
  key += element == rund::kernel::ScanElement::U64 ? ".u64" : ".u32";
  return key;
}

}  // namespace
#endif

}  // namespace rund::node::accel::detail

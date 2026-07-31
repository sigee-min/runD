#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalSegmentedPrelude() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint kSegmentedWidth = 256u;

struct SegmentedScanParams {
  ulong element_count;
  ulong block_size;
  ulong block_count;
  uint inclusive;
  uint reserved;
};
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail

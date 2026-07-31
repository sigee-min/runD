#include <accel/graph/node.hpp>

#include "desc.hpp"

#include "local.hpp"

namespace rund::node::accel::detail {

rund::kernel::SortDesc
SortDescFor(const rund::AccelGraphNode &node) noexcept {
  const bool bounded =
      node.sort.count_source != rund::kernel::ComputeCountSource::Descriptor;
  const std::uint64_t identity = bounded ? 4u : 3u;
  const std::uint64_t values = bounded ? 5u : 4u;
  if ((node.buffer_count != identity && node.buffer_count != values) ||
      node.buffers == nullptr || !AccelGraphBufferShapeValid(node.buffers[0]) ||
      !SortKeyWidthOk(
          AccelGraphBufferShape(node.buffers[0]).scalar_width_bytes)) {
    return rund::kernel::SortDesc{};
  }
  return rund::kernel::SortDesc{
      .key = SortKeyForWidth(
          AccelGraphBufferShape(node.buffers[0]).scalar_width_bytes),
      .value = node.buffer_count == identity
                   ? rund::kernel::SortValue::IdentityU32
                   : rund::kernel::SortValue::U32,
      .element_count = AccelGraphBufferShape(node.buffers[0]).count,
      .radix_bits = 8u,
      .key_bits = node.sort.key_bits,
      .stable = true,
      .count_source = node.sort.count_source,
  };
}

rund::kernel::CompactDesc
CompactDescFor(const rund::AccelGraphNode &node) noexcept {
  if (node.buffer_count != 2u || node.buffers == nullptr ||
      !AccelGraphBufferShapeValid(node.buffers[0]) ||
      !AccelGraphBufferShapeValid(node.buffers[1])) {
    return rund::kernel::CompactDesc{};
  }
  return rund::kernel::CompactDesc{
      .element_count = AccelGraphBufferShape(node.buffers[0]).count,
      .output_capacity = AccelGraphBufferShape(node.buffers[1]).count,
      .flag_bytes = static_cast<rund::kernel::u32>(
          AccelGraphBufferShape(node.buffers[0]).scalar_width_bytes),
      .output_bytes = static_cast<rund::kernel::u32>(
          AccelGraphBufferShape(node.buffers[1]).scalar_width_bytes),
  };
}

rund::kernel::ScatterDesc
ScatterDescFor(const rund::AccelGraphNode &node) noexcept {
  if (node.buffer_count != 3u || node.buffers == nullptr ||
      !AccelGraphBufferShapeValid(node.buffers[0]) ||
      !AccelGraphBufferShapeValid(node.buffers[2])) {
    return rund::kernel::ScatterDesc{};
  }
  return rund::kernel::ScatterDesc{
      .element = AccelGraphBufferShape(node.buffers[0]).scalar_width_bytes ==
                         sizeof(rund::kernel::u64)
                     ? rund::kernel::ScatterElement::U64
                     : rund::kernel::ScatterElement::U32,
      .element_count = AccelGraphBufferShape(node.buffers[0]).count,
      .output_count = AccelGraphBufferShape(node.buffers[2]).count,
  };
}

} // namespace rund::node::accel::detail

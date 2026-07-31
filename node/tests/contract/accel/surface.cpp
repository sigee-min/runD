#include <accel/context/buffer.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/primitive/transform/node.hpp>
#include <accel/graph/factory/scan/basic.hpp>
#include <accel/graph/factory/scan/segmented.hpp>

#include <accel/graph.hpp>
#include <kernel/program/compute/dsl.hpp>

#include "test/assert.hpp"

#include <array>
#include <string_view>

namespace {

[[nodiscard]] bool GraphFactoriesBuildNodes() {
  rund::AccelBuffer input{};
  rund::AccelBuffer heads{};
  rund::AccelBuffer output{};
  input.scalar_width_bytes = sizeof(rund::kernel::u32);
  input.count = 16u;
  heads.scalar_width_bytes = sizeof(rund::kernel::u32);
  heads.count = 16u;
  output.scalar_width_bytes = sizeof(rund::kernel::u32);
  output.count = 16u;
  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelRead(input, "in"),
      rund::AccelWrite(output, "out"),
  };
  const std::array<rund::AccelGraphBufferRef, 3u> segmented_refs{
      rund::AccelRead(input, "in"),
      rund::AccelRead(heads, "heads"),
      rund::AccelWrite(output, "out"),
  };
  const rund::kernel::ScanDesc desc{
      .op = rund::kernel::ScanOp::InclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = input.count,
      .block_size = 8u,
  };
  const rund::kernel::ScanHash hash = rund::kernel::HashScan(desc);
  const rund::AccelGraphNode node =
      rund::AccelScan(refs.data(), refs.size(), desc);
  const rund::AccelGraphNode default_node =
      rund::AccelScan(refs.data(), refs.size(), input.count);
  const rund::AccelGraphNode default_segmented_node = rund::AccelSegmentedScan(
      segmented_refs.data(), segmented_refs.size(), input.count);
  const rund::AccelGraphNode transform_node = rund::AccelTransform(
      refs.data(), refs.size(),
      rund::kernel::TransformDesc{
          .op = rund::kernel::TransformOp::Fourier,
          .direction = rund::kernel::TransformDir::Forward,
          .layout = rund::kernel::TransformLayout::Split,
          .normalization = rund::kernel::TransformNorm::None,
          .element_count = input.count,
      });
  TEST_ASSERT(node.kind == rund::kernel::NodeKind::Scan);
  TEST_ASSERT(node.buffers == refs.data());
  TEST_ASSERT(node.buffer_count == refs.size());
  TEST_ASSERT(node.primitive_hash_hi == hash.hi);
  TEST_ASSERT(node.primitive_hash_lo == hash.lo);
  TEST_ASSERT(node.element_count == desc.element_count);
  TEST_ASSERT(node.scan.block_size == desc.block_size);
  TEST_ASSERT(default_node.kind == rund::kernel::NodeKind::Scan);
  TEST_ASSERT(default_node.scan.op == rund::kernel::ScanOp::ExclusiveSum);
  TEST_ASSERT(default_node.scan.element == rund::kernel::ScanElement::U32);
  TEST_ASSERT(default_node.scan.element_count == input.count);
  TEST_ASSERT(default_node.scan.block_size == input.count);
  TEST_ASSERT(default_segmented_node.kind == rund::kernel::NodeKind::SegmentedScan);
  TEST_ASSERT(default_segmented_node.buffers == segmented_refs.data());
  TEST_ASSERT(default_segmented_node.buffer_count == segmented_refs.size());
  TEST_ASSERT(default_segmented_node.segmented_scan.op ==
              rund::kernel::SegmentedScanOp::ExclusiveSum);
  TEST_ASSERT(default_segmented_node.segmented_scan.element ==
              rund::kernel::SegmentedScanElement::U32);
  TEST_ASSERT(default_segmented_node.segmented_scan.element_count ==
              input.count);
  TEST_ASSERT(default_segmented_node.segmented_scan.block_size == input.count);
  TEST_ASSERT(transform_node.kind == rund::kernel::NodeKind::Transform);
  TEST_ASSERT(transform_node.transform.layout ==
              rund::kernel::TransformLayout::Split);
  TEST_ASSERT(transform_node.transform.normalization ==
              rund::kernel::TransformNorm::None);
  TEST_ASSERT(refs[0].binding_name == std::string_view{"in"});
  TEST_ASSERT(refs[1].role == rund::kernel::BufferRole::Write);
  return true;
}

} // namespace

int RunAccelSurfaceContract() {
  TEST_ASSERT(GraphFactoriesBuildNodes());
  return 0;
}

#include "pipeline.hpp"

#include "pipeline/create/run.hpp"
#include "pipeline/name.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

bool CompileMetalScanPipelines(MetalAdapter& adapter,
                               const rund::kernel::ScanElement element,
                               std::shared_ptr<void>& block,
                               std::shared_ptr<void>& prefix,
                               std::shared_ptr<void>& offset) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  return CompileMetalScanPipelineSet(
      adapter, MetalScanFunctionName("rund_compute_scan_block", element),
      MetalScanFunctionName("rund_compute_scan_prefix", element),
      MetalScanFunctionName("rund_compute_scan_offset", element),
      MetalScanPipelineKey("block", element),
      MetalScanPipelineKey("prefix", element),
      MetalScanPipelineKey("offset", element), block, prefix, offset);
#else
  (void)adapter;
  (void)element;
  (void)block;
  (void)prefix;
  (void)offset;
  return false;
#endif
}

bool CompileMetalScanFlagPipelines(MetalAdapter& adapter,
                                   std::shared_ptr<void>& block,
                                   std::shared_ptr<void>& prefix,
                                   std::shared_ptr<void>& offset) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  return CompileMetalScanPipelineSet(
      adapter, @"rund_compute_scan_block_flag_u32",
      @"rund_compute_scan_prefix_u32", @"rund_compute_scan_offset_u32",
      "scan.block.flag.u32",
      MetalScanPipelineKey("prefix", rund::kernel::ScanElement::U32),
      MetalScanPipelineKey("offset", rund::kernel::ScanElement::U32),
      block, prefix, offset);
#else
  (void)adapter;
  (void)block;
  (void)prefix;
  (void)offset;
  return false;
#endif
}

}  // namespace rund::node::accel::detail

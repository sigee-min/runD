#include "../../domain.hpp"
#include "../pipeline/named.hpp"
#include "local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] NSString *
FunctionName(const char *const phase,
             const rund::kernel::SegmentedScanElement element,
             const rund::kernel::ComputeDomain domain) {
  std::string name = "rund_compute_segmented_scan_";
  name += phase;
  const bool signed_domain = IsSignedDomain(domain);
  name += signed_domain ? "_i" : "_u";
  name += element == rund::kernel::SegmentedScanElement::U64 ? "64" : "32";
  return [NSString stringWithUTF8String:name.c_str()];
}

[[nodiscard]] std::string
PipelineKey(const char *const phase,
            const rund::kernel::SegmentedScanElement element,
            const rund::kernel::ComputeDomain domain) {
  std::string key = "segmented_scan.";
  key += phase;
  const bool signed_domain = IsSignedDomain(domain);
  key += signed_domain ? ".i" : ".u";
  key += element == rund::kernel::SegmentedScanElement::U64 ? "64" : "32";
  return key;
}

} // namespace

bool CompileMetalSegmentedScanPipelines(
    MetalAdapter &adapter, const rund::kernel::SegmentedScanElement element,
    const rund::kernel::ComputeDomain domain, std::shared_ptr<void> &block,
    std::shared_ptr<void> &prefix, std::shared_ptr<void> &offset) {
  const std::string block_key = PipelineKey("block", element, domain);
  const std::string prefix_key = PipelineKey("prefix", element, domain);
  const std::string offset_key = PipelineKey("offset", element, domain);
  block = LookupMetalNamedPipeline(adapter, block_key);
  prefix = LookupMetalNamedPipeline(adapter, prefix_key);
  offset = LookupMetalNamedPipeline(adapter, offset_key);
  if (block != nullptr && prefix != nullptr && offset != nullptr) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return false;
  }
  const std::uint64_t begin_ns = MonotonicNanoseconds();
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalSegmentedScanSource());
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  if (library == nil ||
      !MakeNamedMetalPipeline(device, library,
                              FunctionName("block", element, domain), block) ||
      !MakeNamedMetalPipeline(
          device, library, FunctionName("prefix", element, domain), prefix) ||
      !MakeNamedMetalPipeline(
          device, library, FunctionName("offset", element, domain), offset)) {
    return false;
  }
  const std::uint64_t create_ns = MonotonicNanoseconds() - begin_ns;
  StoreMetalNamedPipeline(adapter, block_key, block, create_ns);
  StoreMetalNamedPipeline(adapter, prefix_key, prefix, 0u);
  StoreMetalNamedPipeline(adapter, offset_key, offset, 0u);
  return true;
}
#endif

} // namespace rund::node::accel::detail

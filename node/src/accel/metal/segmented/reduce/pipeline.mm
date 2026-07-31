#include "model.hpp"

#include "../../pipeline/cache.hpp"
#include "../../pipeline/named.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
CompileMetalSegmentedReduce(MetalAdapter &adapter,
                            const rund::kernel::SegmentedReducePlan &plan,
                            const rund::kernel::ComputeDomain domain,
                            MetalSegmentedReducePipelines &out) {
  constexpr const char *kClassify = "segmented-reduce.classify";
  constexpr const char *kPrefix = "segmented-reduce.prefix";
  constexpr const char *kScatter = "segmented-reduce.scatter";
  const std::string reduce_key = MetalSegmentedReduceKey(plan, domain);
  out.classify = LookupMetalNamedPipeline(adapter, kClassify);
  out.prefix = LookupMetalNamedPipeline(adapter, kPrefix);
  out.scatter = LookupMetalNamedPipeline(adapter, kScatter);
  out.reduce = LookupMetalNamedPipeline(adapter, reduce_key);
  if (out.classify != nullptr && out.prefix != nullptr &&
      out.scatter != nullptr && out.reduce != nullptr) {
    return {true, "ok"};
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalSegmentedReduceSource(plan.op, domain));
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  if (library == nil) {
    return {false, "accel_metal_pipeline_unavailable"};
  }
  const std::uint64_t begin = MonotonicNanoseconds();
  const std::string reduce_name = MetalSegmentedReduceName(plan, domain);
  std::shared_ptr<void> classify = out.classify;
  std::shared_ptr<void> prefix = out.prefix;
  std::shared_ptr<void> scatter = out.scatter;
  std::shared_ptr<void> reduce = out.reduce;
  if ((classify == nullptr &&
       !MakeNamedMetalPipeline(device, library,
                               "rund_compute_segmented_reduce_classify",
                               classify)) ||
      (prefix == nullptr &&
       !MakeNamedMetalPipeline(
           device, library, "rund_compute_segmented_reduce_prefix", prefix)) ||
      (scatter == nullptr &&
       !MakeNamedMetalPipeline(device, library,
                               "rund_compute_segmented_reduce_scatter",
                               scatter)) ||
      (reduce == nullptr &&
       !MakeNamedMetalPipeline(device, library, reduce_name.c_str(), reduce))) {
    return {false, "accel_metal_pipeline_unavailable"};
  }
  std::uint64_t unreported_ns = MonotonicNanoseconds() - begin;
  if (out.classify == nullptr) {
    StoreMetalNamedPipeline(adapter, kClassify, classify, unreported_ns);
    unreported_ns = 0u;
  }
  if (out.prefix == nullptr) {
    StoreMetalNamedPipeline(adapter, kPrefix, prefix, unreported_ns);
    unreported_ns = 0u;
  }
  if (out.scatter == nullptr) {
    StoreMetalNamedPipeline(adapter, kScatter, scatter, unreported_ns);
    unreported_ns = 0u;
  }
  if (out.reduce == nullptr) {
    StoreMetalNamedPipeline(adapter, reduce_key, reduce, unreported_ns);
  }
  out.classify = std::move(classify);
  out.prefix = std::move(prefix);
  out.scatter = std::move(scatter);
  out.reduce = std::move(reduce);
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail

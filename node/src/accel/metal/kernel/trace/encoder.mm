#include "encoder.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
@interface RUNDMetalDispatchTraceEncoder : NSObject {
@private
  id<MTLComputeCommandEncoder> _encoder;
  rund::node::accel::detail::MetalDispatchTrace *_trace;
}
- (void)bindEncoder:(id<MTLComputeCommandEncoder>)encoder
              trace:(rund::node::accel::detail::MetalDispatchTrace *)trace;
- (void)clearEncoder;
@end

@implementation RUNDMetalDispatchTraceEncoder
- (void)bindEncoder:(id<MTLComputeCommandEncoder>)encoder
              trace:(rund::node::accel::detail::MetalDispatchTrace *)trace {
  _encoder = encoder;
  _trace = trace;
}
- (void)clearEncoder {
  _encoder = nil;
  _trace = nullptr;
}
- (id)forwardingTargetForSelector:(SEL)selector {
  (void)selector;
  return _encoder;
}
- (void)dispatchThreadgroups:(MTLSize)groups
       threadsPerThreadgroup:(MTLSize)threads {
  rund::node::accel::detail::SampleMetalDispatchTrace(_encoder, *_trace);
  [_encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  rund::node::accel::detail::SampleMetalDispatchTrace(_encoder, *_trace);
}
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group {
  rund::node::accel::detail::SampleMetalDispatchTrace(_encoder, *_trace);
  [_encoder dispatchThreads:threads threadsPerThreadgroup:group];
  rund::node::accel::detail::SampleMetalDispatchTrace(_encoder, *_trace);
}
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buffer
                          indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)threads {
  rund::node::accel::detail::SampleMetalDispatchTrace(_encoder, *_trace);
  [_encoder dispatchThreadgroupsWithIndirectBuffer:buffer
                              indirectBufferOffset:offset
                             threadsPerThreadgroup:threads];
  rund::node::accel::detail::SampleMetalDispatchTrace(_encoder, *_trace);
}
@end

namespace rund::node::accel::detail {

id CreateMetalDispatchTraceEncoder() noexcept {
  return [[RUNDMetalDispatchTraceEncoder alloc] init];
}

void BindMetalDispatchTraceEncoder(id const proxy,
                                   id<MTLComputeCommandEncoder> const encoder,
                                   MetalDispatchTrace &trace) noexcept {
  [(RUNDMetalDispatchTraceEncoder *)proxy bindEncoder:encoder trace:&trace];
}

void ClearMetalDispatchTraceEncoder(id const proxy) noexcept {
  [(RUNDMetalDispatchTraceEncoder *)proxy clearEncoder];
}

} // namespace rund::node::accel::detail
#endif

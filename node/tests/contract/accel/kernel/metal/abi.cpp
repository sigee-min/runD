#include "abi.hpp"

#include "src/accel/metal/kernel/pipeline/abi.hpp"
#include "src/accel/metal/kernel/pipeline/source.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace node_accel_contract {
namespace {

using namespace rund::node::accel::detail;

constexpr std::uint32_t Mask32 = 0xa5c37e19u;
constexpr std::uint64_t Mask64 = 0xa5c37e19d40bf286ull;

constexpr std::uint32_t Seed32(const std::size_t offset) {
  return 0xe1390000u | static_cast<std::uint32_t>(offset + 1u);
}

constexpr std::uint64_t Seed64(const std::size_t offset) {
  return 0xfedcba9876540000ull | static_cast<std::uint64_t>(offset + 1u);
}

// Validate preserved defaults before filling every scalar with a distinct
// value. Upper 32 bits are nonzero so a truncated ulong transfer cannot pass.
#define RUND_METAL_ABI_BEGIN(Host, Gpu)                                        \
  bool Fill(Host &value, const bool transformed) {
#define RUND_METAL_ABI_U32(name, initial, offset)                              \
  if (value.name != (initial))                                                 \
    return false;                                                              \
  value.name = Seed32(offset) ^ (transformed ? Mask32 : 0u);
#define RUND_METAL_ABI_U64(name, initial, offset)                              \
  if (value.name != (initial))                                                 \
    return false;                                                              \
  value.name = Seed64(offset) ^ (transformed ? Mask64 : 0u);
#define RUND_METAL_ABI_ARRAY64(name, count, offset)                            \
  for (std::size_t i = 0; i < count; ++i) {                                    \
    if (value.name[i] != 0u)                                                   \
      return false;                                                            \
    value.name[i] = Seed64(offset + i * 8u) ^ (transformed ? Mask64 : 0u);     \
  }
#define RUND_METAL_ABI_POLICIES(name, g0, g1, g2, g3, offset)                  \
  for (std::size_t i = 0; i < 4u; ++i) {                                       \
    if (value.name[i] != 0u)                                                   \
      return false;                                                            \
    value.name[i] = Seed32(offset + i * 4u) ^ (transformed ? Mask32 : 0u);     \
  }
#define RUND_METAL_ABI_END(Host, Gpu, size, alignment)                         \
  return true;                                                                 \
  }
#include "src/accel/metal/kernel/pipeline/abi/schema/records.def"
#undef RUND_METAL_ABI_BEGIN
#undef RUND_METAL_ABI_U32
#undef RUND_METAL_ABI_U64
#undef RUND_METAL_ABI_ARRAY64
#undef RUND_METAL_ABI_POLICIES
#undef RUND_METAL_ABI_END

// Compare fields, not padding bytes. This also checks the array/scalar policy
// representation mapping rather than accepting an untyped memcpy roundtrip.
#define RUND_METAL_ABI_BEGIN(Host, Gpu)                                        \
  bool Matches(const Host &a, const Host &b) {
#define RUND_METAL_ABI_U32(name, initial, offset)                              \
  if (a.name != b.name)                                                        \
    return false;
#define RUND_METAL_ABI_U64 RUND_METAL_ABI_U32
#define RUND_METAL_ABI_ARRAY64(name, count, offset)                            \
  if (a.name != b.name)                                                        \
    return false;
#define RUND_METAL_ABI_POLICIES(name, g0, g1, g2, g3, offset)                  \
  if (a.name != b.name)                                                        \
    return false;
#define RUND_METAL_ABI_END(Host, Gpu, size, alignment)                         \
  return true;                                                                 \
  }
#include "src/accel/metal/kernel/pipeline/abi/schema/records.def"
#undef RUND_METAL_ABI_BEGIN
#undef RUND_METAL_ABI_U32
#undef RUND_METAL_ABI_U64
#undef RUND_METAL_ABI_ARRAY64
#undef RUND_METAL_ABI_POLICIES
#undef RUND_METAL_ABI_END

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

const char *ProbeSource() {
  return
#define RUND_METAL_ABI_BEGIN(Host, Gpu)                                        \
  "kernel void probe_" #Gpu "(device const " #Gpu                              \
  " *input [[buffer(0)]], device " #Gpu " *output [[buffer(1)]]) {\n"          \
  "typedef " #Gpu " Record;\n"
#define RUND_METAL_ABI_U32(name, initial, offset)                              \
  "static_assert(__builtin_offsetof(Record, " #name ") == " #offset            \
  ", \"field offset\");\n"                                                     \
  "output->" #name " = input->" #name " ^ 0xa5c37e19u;\n"
#define RUND_METAL_ABI_U64(name, initial, offset)                              \
  "static_assert(__builtin_offsetof(Record, " #name ") == " #offset            \
  ", \"field offset\");\n"                                                     \
  "output->" #name " = input->" #name " ^ 0xa5c37e19d40bf286ul;\n"
#define RUND_METAL_ABI_ARRAY64(name, count, offset)                            \
  "static_assert(__builtin_offsetof(Record, " #name ") == " #offset            \
  ", \"array offset\");\n"                                                     \
  "for (uint i = 0; i < " #count "; ++i) output->" #name "[i] = input->" #name \
  "[i] ^ 0xa5c37e19d40bf286ul;\n"
#define RUND_METAL_ABI_POLICY(gpu, offset, delta)                              \
  "static_assert(__builtin_offsetof(Record, " #gpu ") == " #offset             \
  " + " #delta ", \"policy offset\");\n"                                       \
  "output->" #gpu " = input->" #gpu " ^ 0xa5c37e19u;\n"
#define RUND_METAL_ABI_POLICIES(name, g0, g1, g2, g3, offset)                  \
  RUND_METAL_ABI_POLICY(g0, offset, 0)                                         \
  RUND_METAL_ABI_POLICY(g1, offset, 4)                                         \
  RUND_METAL_ABI_POLICY(g2, offset, 8)                                         \
  RUND_METAL_ABI_POLICY(g3, offset, 12)
#define RUND_METAL_ABI_END(Host, Gpu, size, alignment)                         \
  "static_assert(sizeof(Record) == " #size ", \"record size\");\n"             \
  "static_assert(alignof(Record) == " #alignment                               \
  ", \"record alignment\");\n}\n"
#include "src/accel/metal/kernel/pipeline/abi/schema/records.def"
      ;
#undef RUND_METAL_ABI_BEGIN
#undef RUND_METAL_ABI_U32
#undef RUND_METAL_ABI_U64
#undef RUND_METAL_ABI_ARRAY64
#undef RUND_METAL_ABI_POLICY
#undef RUND_METAL_ABI_POLICIES
#undef RUND_METAL_ABI_END
}

template <class Record>
bool Roundtrip(id<MTLDevice> device, id<MTLCommandQueue> queue,
               id<MTLLibrary> library, const char *name) {
  Record input{}, expected{}, observed{};
  if (!Fill(input, false) || !Fill(expected, true))
    return false;
  NSString *const function_name = [NSString stringWithUTF8String:name];
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  if (function == nil)
    return false;
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  id<MTLBuffer> source =
      [device newBufferWithBytes:&input
                          length:sizeof(Record)
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> target =
      [device newBufferWithLength:sizeof(Record)
                          options:MTLResourceStorageModeShared];
  if (pipeline == nil || source == nil || target == nil)
    return false;
  void *contents = [target contents];
  if (contents == nullptr)
    return false;
  std::memset(contents, 0x3c, sizeof(Record));
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (command == nil || encoder == nil)
    return false;
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:source offset:0 atIndex:0];
  [encoder setBuffer:target offset:0 atIndex:1];
  [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::fprintf(stderr, "Metal ABI dispatch failed (%s): %s\n", name,
                 command.error == nil
                     ? "command did not complete"
                     : command.error.localizedDescription.UTF8String);
    return false;
  }
  std::memcpy(&observed, contents, sizeof(Record));
  if (!Matches(observed, expected)) {
    std::fprintf(stderr, "Metal ABI field mismatch: %s\n", name);
    return false;
  }
  return true;
}
#endif

} // namespace

bool MetalPipelineAbiContract() {
  std::size_t records = 0;
#define RUND_METAL_ABI_BEGIN(Host, Gpu)                                        \
  {                                                                            \
    Host a{}, b{};                                                             \
    if (!Fill(a, false) || !Fill(b, true) || Matches(a, b))                    \
      return false;                                                            \
  }
#define RUND_METAL_ABI_U32(name, initial, offset)
#define RUND_METAL_ABI_U64(name, initial, offset)
#define RUND_METAL_ABI_ARRAY64(name, count, offset)
#define RUND_METAL_ABI_POLICIES(name, g0, g1, g2, g3, offset)
#define RUND_METAL_ABI_END(Host, Gpu, size, alignment) ++records;
#include "src/accel/metal/kernel/pipeline/abi/schema/records.def"
#undef RUND_METAL_ABI_BEGIN
#undef RUND_METAL_ABI_U32
#undef RUND_METAL_ABI_U64
#undef RUND_METAL_ABI_ARRAY64
#undef RUND_METAL_ABI_POLICIES
#undef RUND_METAL_ABI_END
  if (records != 7u)
    return false;
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "Metal ABI native unavailable: no device\n");
      return false;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      std::fprintf(stderr, "Metal ABI command queue creation failed\n");
      return false;
    }
    const std::string source = MetalPipelineSource() + ProbeSource();
    NSString *text = [[NSString alloc] initWithBytes:source.data()
                                              length:source.size()
                                            encoding:NSUTF8StringEncoding];
    NSError *error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:text options:nil error:&error];
    if (library == nil) {
      std::fprintf(stderr, "Metal ABI shader compile failed: %s\n",
                   error == nil ? "unavailable"
                                : error.localizedDescription.UTF8String);
      return false;
    }
#define RUND_METAL_ABI_BEGIN(Host, Gpu)                                        \
  if (!Roundtrip<Host>(device, queue, library, "probe_" #Gpu))                 \
    return false;
#define RUND_METAL_ABI_U32(name, initial, offset)
#define RUND_METAL_ABI_U64(name, initial, offset)
#define RUND_METAL_ABI_ARRAY64(name, count, offset)
#define RUND_METAL_ABI_POLICIES(name, g0, g1, g2, g3, offset)
#define RUND_METAL_ABI_END(Host, Gpu, size, alignment)
#include "src/accel/metal/kernel/pipeline/abi/schema/records.def"
#undef RUND_METAL_ABI_BEGIN
#undef RUND_METAL_ABI_U32
#undef RUND_METAL_ABI_U64
#undef RUND_METAL_ABI_ARRAY64
#undef RUND_METAL_ABI_POLICIES
#undef RUND_METAL_ABI_END
    std::fprintf(stderr, "Metal ABI: 7 typed GPU roundtrips passed\n");
  }
#else
  std::fprintf(stderr,
               "Metal ABI: 7 host records passed; native SDK unavailable\n");
#endif
  return true;
}

} // namespace node_accel_contract

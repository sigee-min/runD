#include "local.hpp"

#include "../../pipeline/named.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
rund::AccelCheck RunMetalResources(const rund::AccelDevice &pick,
                                   const std::shared_ptr<void> &prepared) {
  @autoreleasepool {
    auto *const resources = static_cast<MetalKernelResources *>(prepared.get());
    if (resources == nullptr || resources->size() == 0u) {
      return rund::AccelCheck{false, "accel_metal_unavailable"};
    }
    MetalKernelContext context{};
    const rund::AccelCheck valid = ValidateMetalKernelContext(pick, context);
    if (!valid.ok) {
      return valid;
    }
    CommandRun command{};
    const rund::AccelCheck ready =
        OpenCommand<ResourceRefs::Borrowed>(*context.adapter, command);
    if (!ready.ok) {
      return ready;
    }
    const rund::AccelCheck encoded =
        EncodeMetalSteps(*context.adapter, *resources, command);
    if (!encoded.ok) {
      return encoded;
    }
    const rund::AccelCheck submitted =
        WaitCommand(*context.adapter, (__bridge void *)command.buffer);
    return submitted.ok ? FinishMetalSteps(*context.adapter, *resources)
                        : submitted;
  }
}
#else
rund::AccelCheck RunMetalResources(const rund::AccelDevice &,
                                   const std::shared_ptr<void> &) {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}
#endif

} // namespace rund::node::accel::detail

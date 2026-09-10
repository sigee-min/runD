#include "../control.hpp"
#include "../source/upper.hpp"

#include "../../../../clock.hpp"
#include "../../../../kernel/backend/source/storage.hpp"
#include "../../../pipeline/named.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

bool MetalMapControlFinalSourceUpperBytes(std::uint64_t &upper) noexcept {
  return PipelinePrivateMetalSourceUpperBytes(
      MetalMapControlSourceText().size(), 1u,
      IsPipelinePrivatePreparation(CurrentKernelPreparationMode()), upper);
}

std::string MetalMapControlSource() {
  std::uint64_t upper = 0u;
  if (!MetalMapControlFinalSourceUpperBytes(upper)) {
    return {};
  }
  const auto recipe = []<typename Sink>(Sink &sink) noexcept(
                          noexcept(sink.append(std::string_view{}))) {
    return sink.append(MetalMapControlSourceText());
  };
  return backend_source_recipe::materialize(
      recipe, MetalMapControlSourceText().size(), upper);
}

std::shared_ptr<void> MetalMapControlPipeline(MetalAdapter &adapter) {
  constexpr const char *key = "map.control.dispatch.v1";
  std::shared_ptr<void> pipeline = LookupMetalNamedPipeline(adapter, key);
  if (pipeline != nullptr) {
    return pipeline;
  }
  id<MTLDevice> const device = (__bridge id<MTLDevice>)adapter.device.get();
  const std::uint64_t begin = MonotonicNanoseconds();
  std::uint64_t source_upper = 0u;
  if (!MetalMapControlFinalSourceUpperBytes(source_upper)) {
    return {};
  }
  const std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalMapControlSource(), source_upper);
  id<MTLLibrary> const library = (__bridge id<MTLLibrary>)library_owner.get();
  if (library == nil ||
      !MakeNamedMetalPipeline(device, library, "rund_map_control_dispatch",
                              pipeline)) {
    return {};
  }
  StoreMetalNamedPipeline(adapter, key, pipeline,
                          MonotonicNanoseconds() - begin);
  return pipeline;
}

#endif

} // namespace rund::node::accel::detail

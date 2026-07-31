#pragma once

#include "../../scatter.hpp"
#include "../../scatter/model.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"
#include <memory>
#include <string>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kScatterThreadgroupSize = 256u;

struct MetalScatterEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::ScatterPlan plan{};
  MetalResidentBufferResult values{};
  MetalResidentBufferResult indices{};
  MetalResidentBufferResult output{};
  MetalRuntimeBuffer status{};
  std::shared_ptr<void> pipeline{};
};

void DestroyMetalScatterEncodeResources(void *raw);
[[nodiscard]] std::string MetalScatterSource();
[[nodiscard]] bool
CompileMetalScatterPipeline(MetalAdapter &adapter,
                            rund::kernel::ScatterElement element,
                            std::shared_ptr<void> &out);

} // namespace rund::node::accel::detail

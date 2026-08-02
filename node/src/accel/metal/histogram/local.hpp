#pragma once

#include "../../histogram.hpp"
#include "../../histogram/model.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"

#include <memory>
#include <string>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kHistogramThreadgroupSize = 256u;

struct MetalHistogramPipelines {
  std::shared_ptr<void> clear{};
  std::shared_ptr<void> count{};
};

struct MetalHistogramEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::HistogramPlan plan{};
  MetalResidentBufferResult bins{};
  MetalResidentBufferResult counts{};
  MetalRuntimeBuffer status{};
  MetalHistogramPipelines pipelines{};
};

void DestroyMetalHistogramEncodeResources(void *raw);
[[nodiscard]] std::string MetalHistogramSource();
[[nodiscard]] std::uint64_t MetalHistogramSourceUpperBytes() noexcept;
[[nodiscard]] bool CompileMetalHistogramPipelines(MetalAdapter &adapter,
                                                  MetalHistogramPipelines &out);

} // namespace rund::node::accel::detail

#pragma once

#include "../../stencil.hpp"
#include "../../stencil/model.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"
#include <memory>
#include <string>

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kStencilThreadgroupSize = 256u;

struct MetalStencilEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::StencilPlan plan{};
  MetalResidentBufferResult input{};
  MetalResidentBufferResult output{};
  std::shared_ptr<void> pipeline{};
};

void DestroyMetalStencilEncodeResources(void *raw);
[[nodiscard]] std::string MetalStencilSource(rund::kernel::StencilOp op);
[[nodiscard]] bool MetalStencilSourceUpperBytes(rund::kernel::StencilOp op,
                                                std::uint64_t &upper) noexcept;
[[nodiscard]] bool
CompileMetalStencilPipeline(MetalAdapter &adapter, rund::kernel::StencilOp op,
                            rund::kernel::StencilElement element,
                            rund::kernel::ComputeDomain domain,
                            std::shared_ptr<void> &out);

} // namespace rund::node::accel::detail

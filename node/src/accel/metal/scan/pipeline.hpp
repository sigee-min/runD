#pragma once

#include "../../scan/metal.hpp"
#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] bool CompileMetalScanPipelines(
    MetalAdapter& adapter,
    rund::kernel::ScanElement element,
    std::shared_ptr<void>& block,
    std::shared_ptr<void>& prefix,
    std::shared_ptr<void>& offset);

[[nodiscard]] bool CompileMetalScanFlagPipelines(
    MetalAdapter& adapter,
    std::shared_ptr<void>& block,
    std::shared_ptr<void>& prefix,
    std::shared_ptr<void>& offset);

}  // namespace rund::node::accel::detail

#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/reduce/model.hpp>

#include <string>

namespace rund::node::accel::detail::device_vsm_reduce_source {

[[nodiscard]] std::string metal_source(const rund::kernel::ArtifactKey &,
                                       rund::kernel::ReduceElement,
                                       rund::kernel::ReduceOp);
[[nodiscard]] std::string vulkan_source(const rund::kernel::ArtifactKey &,
                                        rund::kernel::ReduceElement,
                                        rund::kernel::ReduceOp);

[[nodiscard]] std::string
metal_additive_source(const rund::kernel::ArtifactKey &,
                      rund::kernel::ReduceElement, rund::kernel::ReduceOp);
[[nodiscard]] std::string
metal_extreme_source(const rund::kernel::ArtifactKey &,
                     rund::kernel::ReduceElement, rund::kernel::ReduceOp);
[[nodiscard]] std::string
vulkan_additive_source(const rund::kernel::ArtifactKey &,
                       rund::kernel::ReduceElement, rund::kernel::ReduceOp);
[[nodiscard]] std::string
vulkan_extreme_source(const rund::kernel::ArtifactKey &,
                      rund::kernel::ReduceElement, rund::kernel::ReduceOp);

} // namespace rund::node::accel::detail::device_vsm_reduce_source

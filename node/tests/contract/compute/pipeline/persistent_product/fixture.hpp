#pragma once

#include "model.hpp"

#include <rund/compute/device.hpp>

#include <cstdint>
#include <memory>

namespace rund::compute::detail {
struct PipelineState;
}

namespace rund_node_test_persistent_product {

[[nodiscard]] PublicationSnapshot SnapshotPublication(
    const std::shared_ptr<rund::compute::detail::PipelineState> &);
[[nodiscard]] std::uint64_t BackingVersion(PersistentProductBacking &) noexcept;
[[nodiscard]] std::uint64_t
BackingRecovery(PersistentProductBacking &) noexcept;
[[nodiscard]] bool PrepareProduct(rund::compute::Backend, std::uint64_t,
                                  PreparedProduct &, bool &);
[[nodiscard]] bool ExactOutput(const PreparedProduct &) noexcept;

} // namespace rund_node_test_persistent_product

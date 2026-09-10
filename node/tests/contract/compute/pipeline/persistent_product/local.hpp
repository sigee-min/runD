#pragma once

#include <rund/compute/device.hpp>

#include <cstdint>
#include <memory>

namespace rund::compute::detail {
struct VirtualPipelineState;
}

namespace rund_node_test_persistent_product {

using NativeQueueCounter = bool (*)(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &,
    std::uint64_t &) noexcept;

[[nodiscard]] int CheckPersistentProduct(rund::compute::Backend,
                                         NativeQueueCounter) noexcept;
[[nodiscard]] bool RunPersistentProductCase(rund::compute::Backend,
                                            NativeQueueCounter, std::uint64_t,
                                            bool &) noexcept;
[[nodiscard]] bool CheckPersistentWindowProduct(rund::compute::Backend,
                                                NativeQueueCounter,
                                                bool &) noexcept;
[[nodiscard]] bool CheckPersistentPublicationFailure(rund::compute::Backend,
                                                     NativeQueueCounter,
                                                     bool &) noexcept;
[[nodiscard]] bool CheckPersistentPublicationObservers(rund::compute::Backend,
                                                       bool &) noexcept;
[[nodiscard]] bool CheckPersistentStartFailure(rund::compute::Backend,
                                               NativeQueueCounter,
                                               bool &) noexcept;
[[nodiscard]] bool CheckPersistentCapabilityTaxonomy() noexcept;
[[nodiscard]] bool CheckPersistentTerminalUnsupported(rund::compute::Backend,
                                                      NativeQueueCounter,
                                                      bool &) noexcept;
[[nodiscard]] bool CheckPersistentMemoryRetry(rund::compute::Backend,
                                              bool &) noexcept;
[[nodiscard]] bool CheckPersistentKnownAdmissionFailure(rund::compute::Backend,
                                                        NativeQueueCounter,
                                                        bool &) noexcept;
[[nodiscard]] bool CheckPersistentUnknownFailure(rund::compute::Backend,
                                                 NativeQueueCounter,
                                                 bool &) noexcept;

} // namespace rund_node_test_persistent_product

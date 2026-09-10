#pragma once

#include "../../persistent_product/backing.hpp"
#include "../../persistent_product/local.hpp"

#include "src/compute/virtual/state.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_device_vsm_product::binary_test {

inline constexpr std::uint64_t PageElements = 16u;

struct PreparedBinary final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      first{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      second{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      output{};
  std::vector<std::byte> expected{};
  std::uint64_t element_count{};
};

[[nodiscard]] bool PrepareBinaryProduct(rund::compute::Backend, std::uint64_t,
                                        PreparedBinary &, bool &);
[[nodiscard]] bool ExactBinaryOutput(const PreparedBinary &) noexcept;
[[nodiscard]] bool RunBinaryProductCases(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;
[[nodiscard]] bool RunPublicBinaryProduct(rund::compute::Backend,
                                          bool &) noexcept;
[[nodiscard]] bool RunPublicMaximumInputProduct(rund::compute::Backend,
                                                bool &) noexcept;
[[nodiscard]] bool RunResidentProductCases(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;
[[nodiscard]] bool
RunResidentHashCacheCase(rund::compute::Backend,
                         rund_node_test_persistent_product::NativeQueueCounter,
                         bool &) noexcept;

} // namespace rund_node_test_device_vsm_product::binary_test

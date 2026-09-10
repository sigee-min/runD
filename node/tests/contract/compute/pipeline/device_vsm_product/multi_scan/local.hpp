#pragma once

#include "../../persistent_product/backing.hpp"
#include "../../persistent_product/local.hpp"

#include "src/compute/virtual/state.hpp"

#include <kernel/program/compute/scan/model.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_device_vsm_product::multi_scan_test {

inline constexpr std::uint64_t PageElements = 16u;

struct PreparedMultiScan final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      first{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      second{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      output{};
  std::vector<std::uint64_t> expected{};
  std::uint64_t element_count{};
  rund::kernel::ScanOp operation{rund::kernel::ScanOp::InclusiveSum};
};

[[nodiscard]] bool PrepareMultiScanProduct(rund::compute::Backend,
                                           std::uint64_t, rund::kernel::ScanOp,
                                           PreparedMultiScan &, bool &);
[[nodiscard]] bool ExactMultiScanOutput(const PreparedMultiScan &) noexcept;
[[nodiscard]] bool RejectWideMultiScan(rund::compute::Backend, bool &) noexcept;
[[nodiscard]] bool
RunMaximumInputScan(rund::compute::Backend,
                    rund_node_test_persistent_product::NativeQueueCounter,
                    rund::kernel::ScanOp, bool &) noexcept;
[[nodiscard]] bool RunMultiScanProductCases(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;

} // namespace rund_node_test_device_vsm_product::multi_scan_test

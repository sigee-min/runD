#pragma once

#include "../../persistent_product/backing.hpp"
#include "../../persistent_product/local.hpp"

#include "src/compute/virtual/state.hpp"

#include <kernel/program/compute/scan/model.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_device_vsm_product::scan_u32_test {

inline constexpr std::uint64_t PageElements = 16u;

struct PreparedScan final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      input{};
  std::shared_ptr<rund_node_test_persistent_product::PersistentProductBacking>
      output{};
  std::vector<std::uint32_t> expected{};
  std::uint64_t element_count{};
  rund::kernel::ScanOp operation{rund::kernel::ScanOp::InclusiveSum};
};

[[nodiscard]] bool PrepareScanProduct(rund::compute::Backend, std::uint64_t,
                                      rund::kernel::ScanOp, bool,
                                      PreparedScan &, bool &);
[[nodiscard]] bool SeedSafe(PreparedScan &) noexcept;
[[nodiscard]] bool ExactScanOutput(const PreparedScan &) noexcept;
[[nodiscard]] bool RunScanProductCases(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;

} // namespace rund_node_test_device_vsm_product::scan_u32_test

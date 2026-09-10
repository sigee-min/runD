#pragma once

#include "../internal.hpp"

#include "src/accel/kernel/residency/device_vsm.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::measure::compute::virtual_graph_pointwise::run_detail {

using Clock = std::chrono::steady_clock;
using Owner =
    ::rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
using Access = ::rund::compute::detail::VirtualBackingAccess;

struct Case final {
  std::array<std::shared_ptr<::rund::compute::VirtualBacking>, Spec::InputCount>
      inputs{};
  std::shared_ptr<::rund::compute::VirtualBacking> output;
  std::unique_ptr<Pipeline> pipeline;
  std::uint64_t graph_hi{};
  std::uint64_t graph_lo{};
};

void record_status(Facts &facts, const ::rund::compute::Status &status);
[[nodiscard]] double micros(Clock::duration duration) noexcept;
[[nodiscard]] bool prepare(::rund::compute::Device &device, Case &result);
[[nodiscard]] bool input_digest(const Case &test_case,
                                std::uint64_t &digest) noexcept;
[[nodiscard]] bool
read_output(const std::shared_ptr<::rund::compute::VirtualBacking> &backing,
            std::vector<std::uint64_t> &values) noexcept;
[[nodiscard]] bool
expected_output(const std::vector<std::uint64_t> &values) noexcept;
[[nodiscard]] Facts facts(Case &test_case, Backend backend,
                          ::rund::compute::Status status,
                          const virtual_residency::ProductRouteEvidence &route,
                          std::uint64_t version_before,
                          std::uint64_t version_after, bool output_match,
                          const std::vector<std::uint64_t> &values,
                          const ::rund::compute::Stats &stats);

} // namespace rund::measure::compute::virtual_graph_pointwise::run_detail

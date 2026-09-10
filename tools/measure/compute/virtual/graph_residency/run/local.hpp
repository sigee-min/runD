#pragma once

#include "../internal.hpp"

#include "../../backing.hpp"

#include <array>
#include <chrono>
#include <memory>
#include <vector>

namespace rund::measure::compute::virtual_graph_residency::run_detail {

using Clock = std::chrono::steady_clock;
using Owner =
    ::rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
using Access = ::rund::compute::detail::VirtualBackingAccess;

struct Case final {
  std::array<std::shared_ptr<::rund::compute::VirtualBacking>, Spec::InputCount>
      inputs{};
  std::shared_ptr<::rund::compute::VirtualBacking> output;
  std::unique_ptr<Pipeline> pipeline;
};

[[nodiscard]] bool input_digest(const Case &, std::uint64_t &) noexcept;
[[nodiscard]] bool
read_output(const std::shared_ptr<::rund::compute::VirtualBacking> &,
            std::vector<std::uint64_t> &) noexcept;
[[nodiscard]] bool expected_output(const std::vector<std::uint64_t> &);
[[nodiscard]] Facts facts(const Case &, ::rund::compute::Status,
                          const virtual_residency::ProductRouteEvidence &,
                          std::uint64_t, std::uint64_t, bool,
                          const std::vector<std::uint64_t> &,
                          const ::rund::compute::Stats &);
[[nodiscard]] double micros(Clock::duration) noexcept;
[[nodiscard]] std::uint64_t nanos() noexcept;
[[nodiscard]] double phase_us(std::uint64_t, std::uint64_t) noexcept;
[[nodiscard]] bool
phase_ready(::rund::compute::Status, const std::shared_ptr<Owner> &,
            const ::rund::node::accel::detail::DeviceVsmEvidence &,
            std::uint64_t, std::uint64_t, std::uintptr_t,
            std::uintptr_t) noexcept;
[[nodiscard]] bool prepare(::rund::compute::Device &, Backend, Case &);

} // namespace rund::measure::compute::virtual_graph_residency::run_detail

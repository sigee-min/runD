#pragma once

#include "../residency.hpp"

#include <rund/compute/device.hpp>
#include <rund/compute/virtual.hpp>

#include <cstddef>
#include <memory>

namespace rund::measure::compute::virtual_residency {

struct ResidencyBackings final {
  std::shared_ptr<::rund::compute::VirtualBacking> input;
  std::shared_ptr<::rund::compute::VirtualBacking> output;
  bool resident{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return input != nullptr && output != nullptr;
  }
};

[[nodiscard]] ResidencyBackings
CreateResidencyBackings(::rund::compute::Device &device,
                        VirtualResidencyBacking mode,
                        std::size_t element_count) noexcept;

} // namespace rund::measure::compute::virtual_residency

#include "backing.hpp"

#include "../backing.hpp"

#include <cstdint>

namespace rund::measure::compute::virtual_residency {

ResidencyBackings
CreateResidencyBackings(::rund::compute::Device &device,
                        const VirtualResidencyBacking mode,
                        const std::size_t element_count) noexcept {
  if (mode == VirtualResidencyBacking::Callback) {
    try {
      return {
          .input = std::make_shared<MemoryBacking>(element_count *
                                                   sizeof(std::int32_t)),
          .output = std::make_shared<MemoryBacking>(element_count *
                                                    sizeof(std::int32_t)),
      };
    } catch (...) {
      return {};
    }
  }

  auto input = ::rund::compute::resident_virtual_backing<std::int32_t>(
      device, element_count);
  auto output = ::rund::compute::resident_virtual_backing<std::int32_t>(
      device, element_count);
  if (!input || !output) {
    return {};
  }
  return {
      .input = std::move(input).value(),
      .output = std::move(output).value(),
      .resident = true,
  };
}

} // namespace rund::measure::compute::virtual_residency

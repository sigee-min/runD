#include "../projection.hpp"

#include <kernel/core/checked.hpp>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool region_coordinate(
    const std::array<residency::FrameRegion, residency::Pool::BankCount>
        &regions,
    const std::uint32_t frame, std::size_t &bank, std::size_t &local) noexcept {
  for (std::size_t index = 0u; index < regions.size(); ++index) {
    const residency::FrameRegion region = regions[index];
    if (region.count != 0u && frame >= region.first &&
        frame - region.first < region.count) {
      bank = index;
      local = static_cast<std::size_t>(frame - region.first);
      return true;
    }
  }
  return false;
}

} // namespace

std::byte *virtual_host_input_frame(const VirtualRunProjection &run,
                                    const std::uint32_t frame) noexcept {
  std::uint64_t bank_capacity = 0u;
  if (run.host_frame_capacity == 0u || run.host_input_count == 0u ||
      !kernel::checked::mul(run.host_frame_capacity, run.host_input_count,
                            bank_capacity) ||
      frame < run.first_host_input_frame ||
      static_cast<std::uint64_t>(frame - run.first_host_input_frame) >=
          bank_capacity * residency::Pool::BankCount) {
    return nullptr;
  }
  const std::uint64_t relative = frame - run.first_host_input_frame;
  const std::size_t bank = static_cast<std::size_t>(relative / bank_capacity);
  const std::size_t local = static_cast<std::size_t>(relative % bank_capacity);
  std::byte *const base = run.input_host_banks[bank];
  return base == nullptr ? nullptr : base + local * run.input_page_bytes;
}

std::byte *virtual_host_output_frame(const VirtualRunProjection &run,
                                     const std::uint32_t frame) noexcept {
  const std::uint64_t capacity = run.host_output_frame_capacity;
  if (capacity == 0u || frame < run.first_host_output_frame ||
      static_cast<std::uint64_t>(frame - run.first_host_output_frame) >=
          capacity * residency::Pool::BankCount) {
    return nullptr;
  }
  const std::uint64_t relative = frame - run.first_host_output_frame;
  const std::size_t bank = static_cast<std::size_t>(relative / capacity);
  const std::size_t local = static_cast<std::size_t>(relative % capacity);
  std::byte *const base = run.output_host_banks[bank];
  return base == nullptr ? nullptr : base + local * run.output_page_bytes;
}

std::byte *virtual_input_frame(const VirtualRunProjection &run,
                               const std::uint32_t frame) noexcept {
  std::size_t bank = 0u;
  std::size_t local = 0u;
  if (!region_coordinate(run.input_regions, frame, bank, local)) {
    return nullptr;
  }
  std::byte *const base = run.input_host_banks[bank];
  return base == nullptr ? nullptr : base + local * run.input_page_bytes;
}

std::byte *virtual_output_frame(const VirtualRunProjection &run,
                                const std::uint32_t frame) noexcept {
  std::size_t bank = 0u;
  std::size_t local = 0u;
  if (!region_coordinate(run.output_regions, frame, bank, local)) {
    return nullptr;
  }
  std::byte *const base = run.output_host_banks[bank];
  return base == nullptr ? nullptr : base + local * run.output_page_bytes;
}

std::byte *virtual_resident_output_frame(const VirtualRunProjection &run,
                                         const std::uint32_t frame) noexcept {
  if (std::byte *const execution = virtual_output_frame(run, frame);
      execution != nullptr) {
    return execution;
  }
  return virtual_host_output_frame(run, frame);
}

} // namespace rund::compute::detail

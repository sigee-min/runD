#pragma once

#include "../../../pipeline/state.hpp"
#include "../backing.hpp"
#include "../execution.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace rund::compute::detail::execution_io {

[[nodiscard]] Status fill_input(VirtualBacking &, const VirtualRunProjection &,
                                const residency::ExecutionTicket &,
                                PipelineState *, ResidencyStats &,
                                std::uint64_t &,
                                VirtualInputReuseSeed * = nullptr) noexcept;

[[nodiscard]] Status supply_input(PipelineState &, const VirtualRunProjection &,
                                  const residency::ExecutionTicket &, Stats &,
                                  std::uint64_t &) noexcept;

[[nodiscard]] Status
collect_output(PipelineState &, const VirtualRunProjection &,
               const residency::ExecutionTicket &,
               std::span<const std::byte *const> &,
               std::array<const std::byte *, PipelineLeafCapacity> &) noexcept;

[[nodiscard]] std::uint64_t
failed_page(const residency::ExecutionTicket &) noexcept;

} // namespace rund::compute::detail::execution_io

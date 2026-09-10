#pragma once

#include "../../../device/residency/pool.hpp"

#include <cstdint>
#include <span>

namespace rund::compute::detail::graph_reduce {

struct CpuEpochReceipt;
class CpuEpochPermit;
enum class CpuBindResult : std::uint8_t;

[[nodiscard]] bool
close_cpu_epoch(residency::Authority &, CpuEpochReceipt &, bool success,
                bool invalidate_all = false,
                residency::CloseInfo *info = nullptr) noexcept;

[[nodiscard]] CpuBindResult
bind_cpu_epoch(residency::Authority &, CpuEpochPermit &, std::uint64_t token,
               std::uint64_t generation,
               residency::CloseInfo *info = nullptr) noexcept;

[[nodiscard]] Status
authority_status(const residency::AuthorityResult &) noexcept;

[[nodiscard]] bool discard_keys(residency::Authority &,
                                std::span<const residency::CacheKey>,
                                residency::FrameRegion) noexcept;

} // namespace rund::compute::detail::graph_reduce

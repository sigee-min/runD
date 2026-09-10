#pragma once

#include "credentials/cpu.hpp"
#include "credentials/epoch.hpp"
#include "model/frame.hpp"
#include "model/lease.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund::compute::detail::residency {

[[nodiscard]] std::size_t
find_frame(const std::vector<registry_model::Frame> &frames, CacheKey key,
           std::size_t first, std::size_t count) noexcept;
[[nodiscard]] bool in_region(FrameRegion region, std::size_t frame) noexcept;
[[nodiscard]] bool in_regions(std::span<const FrameRegion> regions,
                              std::size_t frame) noexcept;
[[nodiscard]] bool demanded(std::span<const CacheUse> uses,
                            CacheKey key) noexcept;
[[nodiscard]] bool valid_cpu_key(CpuReservationKey key) noexcept;
[[nodiscard]] bool supplied_cpu_key(CpuReservationKey key) noexcept;

void clear(registry_model::LeaseSlot &slot) noexcept;
void save(registry_model::LeaseSlot &slot, std::uint32_t frame,
          const registry_model::Frame &prior);
void rollback(std::vector<registry_model::Frame> &frames,
              registry_model::LeaseSlot &slot, bool invalidate_fetches,
              bool invalidate_all = false);
[[nodiscard]] bool complete_epoch(std::vector<registry_model::Frame> &frames,
                                  registry_model::LeaseSlot &epoch,
                                  bool success, bool invalidate_all,
                                  CloseInfo *info = nullptr) noexcept;

[[nodiscard]] std::uint64_t next_token(std::uint64_t &next) noexcept;
[[nodiscard]] std::uint64_t next_generation(std::uint64_t &next) noexcept;
[[nodiscard]] bool project_graph_use(PageUse use,
                                     GraphMaterialization materialization,
                                     Access expected, std::uint64_t epoch,
                                     CacheUse &projected) noexcept;

} // namespace rund::compute::detail::residency

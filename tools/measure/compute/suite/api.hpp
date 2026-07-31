#pragma once

#include "../model.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rund::measure::compute {

#if defined(RUND_COMPUTE_FOCUS)
[[nodiscard]] bool ParseBackend(std::string_view name, Backend &backend) noexcept;
#endif
bool ReportEnvironment(Backend backend);
void PrintStatsColumns();
void PrintWarmColumns();
void PrintWorkloadColumns();

#if defined(RUND_COMPUTE_FOCUS)
void PrintBulkColumns();
bool Bulk(Backend backend, std::size_t samples);
#endif

bool SparseWorkloads(Backend backend, std::size_t count, std::size_t samples);
bool CollectiveWorkloads(Backend backend, std::size_t count,
                         std::size_t samples);
bool ResidentSetup(Backend backend, std::size_t count, std::size_t samples);
bool BatchJobs(Backend backend, std::size_t count, std::size_t samples);

#if !defined(RUND_COMPUTE_FOCUS)
bool Map(const char *label, ::rund::compute::Target target, std::size_t count,
         std::size_t iterations);
bool NodeMap(std::uint32_t workers, std::size_t count, std::size_t iterations);
bool NodeOrchestration(Backend backend, std::uint32_t workers,
                       std::size_t count, std::size_t iterations);
bool InflightVulkan(std::uint32_t workers, std::size_t count,
                    std::size_t samples);
bool Mixed(std::uint32_t workers, std::size_t count, std::size_t iterations);
bool Families(Backend backend);
bool FixedWidening32(Backend backend);
bool FixedWidening64(Backend backend);
#endif

} // namespace rund::measure::compute

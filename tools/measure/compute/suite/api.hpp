#pragma once

#include "../model.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rund::measure::compute {

#if defined(RUND_COMPUTE_FOCUS)
[[nodiscard]] bool ParseBackend(std::string_view name,
                                Backend &backend) noexcept;
#endif
bool ReportEnvironment(Backend backend);
void PrintProductColumns();
bool ProductScenarios(Backend backend);

#if defined(RUND_COMPUTE_FOCUS)
void PrintStatsColumns();
void PrintWarmColumns();
void PrintWorkloadColumns();
void PrintBulkColumns();
bool Bulk(Backend backend, std::size_t samples);
bool SparseWorkloads(Backend backend, std::size_t count, std::size_t samples);
bool CollectiveWorkloads(Backend backend, std::size_t count,
                         std::size_t samples);
bool ResidentSetup(Backend backend, std::size_t count, std::size_t samples);
bool BatchJobs(Backend backend, std::size_t count, std::size_t samples);
#endif

} // namespace rund::measure::compute

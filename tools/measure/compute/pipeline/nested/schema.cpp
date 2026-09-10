#include "../../pipeline.hpp"

#include <cstdio>

namespace rund::measure::compute {

void PrintNestedRepeatColumns() {
  std::fputs("window_repeat_columns,backend,command_path,status,maximum,"
             "outer_windows,tile,inner_iterations,samples_per_pair,"
             "serial_pair_serial_first,serial_pair_nested_first,"
             "latency_pair_nested_first,latency_pair_sealed_first,"
             "throughput_pair_repeated_first,"
             "throughput_pair_sealed_first,templates,commands,"
             "sealed_repetitions,serial_wall_median_us,"
             "serial_pair_nested_wall_median_us,"
             "latency_pair_nested_wall_median_us,"
             "latency_pair_sealed_wall_median_us,"
             "repeated_nested_wall_median_us,"
             "throughput_pair_sealed_wall_median_us,"
             "sealed_equivalent_median_us,nested_speedup,"
             "sealed_measured_throughput_speedup,"
             "sealed_single_execution_ratio,serial_command_submits,"
             "nested_command_submits,repeated_command_submits,"
             "sealed_command_submits,serial_dispatches,nested_dispatches,"
             "repeated_dispatches,sealed_dispatches,"
             "nested_control_commands,"
             "serial_warm_buffer_allocations,nested_warm_buffer_allocations,"
             "repeated_warm_buffer_allocations,"
             "sealed_warm_buffer_allocations,"
             "serial_warm_uploaded_bytes,nested_warm_uploaded_bytes,"
             "repeated_warm_uploaded_bytes,sealed_warm_uploaded_bytes,"
             "serial_warm_download_events,nested_warm_download_events,"
             "repeated_warm_download_events,sealed_warm_download_events,"
             "serial_warm_downloaded_bytes,nested_warm_downloaded_bytes,"
             "repeated_warm_downloaded_bytes,sealed_warm_downloaded_bytes,"
             "serial_fallback,"
             "nested_fallback,repeated_fallback,sealed_fallback,serial_result,"
             "nested_result,repeated_result,sealed_result,result_parity,"
             "warm_zero\n",
             stdout);
}

} // namespace rund::measure::compute

#include "output.hpp"

#include <cstdio>

namespace rund::measure::compute {

void PrintCsv(const std::string_view text) {
  std::putchar('"');
  for (const char value : text) {
    if (value == '"') {
      std::fputs("\"\"", stdout);
    } else {
      std::putchar(static_cast<unsigned char>(value));
    }
  }
  std::putchar('"');
}

bool ReportEnvironment(const Backend backend,
                       const rund::compute::Device &device) {
  auto info = device.info();
  if (!info) {
    std::printf("environment,%s,info_failed,%u,", Name(backend),
                static_cast<unsigned>(info.code()));
    PrintCsv(info.error());
    std::fputs(",\"\",\"\",\"\"\n", stdout);
    return false;
  }
  if (info->backend != backend) {
    std::printf("environment,%s,backend_mismatch,%u,", Name(backend),
                static_cast<unsigned>(rund::compute::Code::Invalid));
    PrintCsv("compute_device_info_backend_mismatch");
    std::putchar(',');
    PrintCsv(info->name);
    std::putchar(',');
    PrintCsv(info->driver);
    std::putchar(',');
    PrintCsv(info->driver_details);
    std::putchar('\n');
    return false;
  }
  std::printf("environment,%s,ok,%u,\"\",", Name(backend),
              static_cast<unsigned>(rund::compute::Code::Ok));
  PrintCsv(info->name);
  std::putchar(',');
  PrintCsv(info->driver);
  std::putchar(',');
  PrintCsv(info->driver_details);
  std::putchar('\n');
  return true;
}

bool ReportEnvironment(const Backend backend) {
  auto device = rund::compute::open(TargetFor(backend));
  if (!device) {
    std::printf("environment,%s,open_failed,%u,", Name(backend),
                static_cast<unsigned>(device.code()));
    PrintCsv(device.error());
    std::fputs(",\"\",\"\",\"\"\n", stdout);
    return false;
  }
  return ReportEnvironment(backend, *device);
}

void PrintWarm(const WarmCounters &warm) {
  std::printf(",%llu,%llu,%llu,%llu,%u",
              static_cast<unsigned long long>(warm.pipeline_compiles),
              static_cast<unsigned long long>(warm.buffer_allocations),
              static_cast<unsigned long long>(warm.download_events),
              static_cast<unsigned long long>(warm.uploaded_bytes),
              warm.zero() ? 1u : 0u);
}

#if defined(RUND_COMPUTE_FOCUS)
void PrintStatsColumns() {
  std::fputs(
      ",pipeline_compiles,buffer_allocations,download_events,dispatches,"
      "command_submits,uploaded_bytes,downloaded_bytes,pipeline_cache_hits,"
      "pipeline_cache_evictions,buffer_reuses,descriptor_pool_creations,"
      "descriptor_set_allocations,descriptor_reuses,original_dispatches,"
      "final_dispatches,fusions,fusion_rejections,internal_roundtrip_bytes,"
      "external_roundtrip_bytes,kernel_ns,kernel_samples,shader_compile_ns,"
      "spirv_compile_ns,pipeline_create_ns,descriptor_setup_ns,submit_wait_ns,"
      "readback_ns,graph_hash,output_hash,workers,participating,tiles,tile_"
      "size,vector_chunks,tail_chunks",
      stdout);
}

void PrintWarmColumns() {
  std::fputs(",warm_pipeline_compiles,warm_buffer_allocations,"
             "warm_download_events,warm_uploaded_bytes,warm_zero",
             stdout);
}

void PrintWorkloadColumns() {
  std::fputs(
      "workload_columns,backend,family,variant,status,input_count,active_count,"
      "samples,prime_runs,median_us,input_items_per_s,active_items_per_s",
      stdout);
  PrintStatsColumns();
  PrintWarmColumns();
  std::fputs(",resident_bytes,staging_bytes\n", stdout);
}
#endif

} // namespace rund::measure::compute

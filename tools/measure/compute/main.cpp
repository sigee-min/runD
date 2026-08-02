#include "suite/core.hpp"
#if defined(RUND_COMPUTE_FOCUS)
#include "pipeline.hpp"
#endif

#include <cstdio>
#include <string_view>
#include <thread>

using namespace rund::measure::compute;

int main(const int argc, char **const argv) {
#if defined(RUND_COMPUTE_FOCUS)
  Backend focus = Backend::Unavailable;
  const bool collective = argc == 3 &&
                          std::string_view{argv[1]} == "--collective" &&
                          ParseBackend(argv[2], focus);
  const bool bulk = argc == 3 && std::string_view{argv[1]} == "--bulk" &&
                    ParseBackend(argv[2], focus);
  const bool sort = argc == 3 && std::string_view{argv[1]} == "--sort" &&
                    ParseBackend(argv[2], focus);
  const bool resident = argc == 3 &&
                        std::string_view{argv[1]} == "--resident" &&
                        ParseBackend(argv[2], focus);
  const bool batch = argc == 3 && std::string_view{argv[1]} == "--batch" &&
                     ParseBackend(argv[2], focus) && focus != Backend::Cpu;
  const bool pipeline = argc == 3 &&
                        std::string_view{argv[1]} == "--pipeline" &&
                        ParseBackend(argv[2], focus) && focus != Backend::Cpu;
  const bool checkpoint = argc == 3 &&
                          std::string_view{argv[1]} == "--checkpoint" &&
                          ParseBackend(argv[2], focus);
  const bool recurrence = argc == 3 &&
                          std::string_view{argv[1]} == "--recurrence" &&
                          ParseBackend(argv[2], focus) && focus != Backend::Cpu;
  const bool window_repeat =
      argc == 3 && std::string_view{argv[1]} == "--window-repeat" &&
      ParseBackend(argv[2], focus) && focus != Backend::Cpu;
  const bool pipeline_profile =
      argc == 3 && std::string_view{argv[1]} == "--pipeline-profile" &&
      ParseBackend(argv[2], focus) && focus != Backend::Cpu;
  const bool plan_memory = argc == 3 &&
                           std::string_view{argv[1]} == "--plan-memory" &&
                           ParseBackend(argv[2], focus);
  const bool prepare_memory = argc == 3 &&
                              std::string_view{argv[1]} == "--prepare-memory" &&
                              ParseBackend(argv[2], focus);
  const bool focused = collective || sort || bulk || resident || batch ||
                       pipeline || checkpoint || recurrence || window_repeat ||
                       pipeline_profile || plan_memory || prepare_memory;
  if (!focused) {
#else
  (void)argv;
  if (argc != 1) {
#endif
    std::fputs("usage: runD-compute-measure "
               "[--resident|--collective|--sort|--bulk|--batch|--pipeline|"
               "--checkpoint|--recurrence|--window-repeat|--pipeline-profile|"
               "--plan-memory|--prepare-memory "
               "cpu|metal|vulkan]\n",
               stderr);
    return 2;
  }
  const unsigned hint = std::thread::hardware_concurrency();
  const std::uint32_t workers = hint == 0u ? 1u : hint;
  std::printf("hardware_workers,%u\n", workers);
  bool ok = true;
  std::fputs("environment_columns,backend,status,code,error,name,driver,"
             "driver_details\n",
             stdout);
#if defined(RUND_COMPUTE_FOCUS)
  if (focused) {
    ok = ReportEnvironment(Backend::Cpu) && ok;
    if (focus != Backend::Cpu) {
      ok = ReportEnvironment(focus) && ok;
    }
    if (checkpoint) {
      rund::measure::compute::PrintCheckpointColumns();
      ok = rund::measure::compute::MeasureCheckpoints(focus, 1u << 20u, 12u) &&
           ok;
    } else if (plan_memory || prepare_memory) {
      rund::measure::compute::PrintPreparationMemoryColumns();
      ok = rund::measure::compute::MeasurePreparationMemory(focus,
                                                            prepare_memory) &&
           ok;
    } else if (pipeline_profile) {
      rund::measure::compute::PrintPipelineProfileColumns();
      ok = rund::measure::compute::MeasurePipelineProfile(focus, 4096u, 12u) &&
           ok;
    } else if (pipeline) {
      rund::measure::compute::PrintPipelineColumns();
      ok = rund::measure::compute::MeasurePipeline(focus, 4096u, 12u) && ok;
    } else if (recurrence) {
      rund::measure::compute::PrintRecurrenceColumns();
      ok = rund::measure::compute::MeasureRecurrence(focus, 4096u, 12u) && ok;
    } else if (window_repeat) {
      rund::measure::compute::PrintNestedRepeatColumns();
      ok = rund::measure::compute::MeasureNestedRepeat(focus, 12u) && ok;
    } else if (batch) {
      std::fputs("batch_columns,backend,status,jobs,elements_per_job,samples,"
                 "serial_first,batch_first,serial_wall_median_us,"
                 "batch_wall_median_us,serial_submit_wait_us,"
                 "batch_submit_wait_us,serial_kernel_us,batch_kernel_us,"
                 "serial_host_residual_us,batch_host_residual_us,"
                 "serial_jobs_per_s,batch_jobs_per_s,speedup,paired_speedup,"
                 "serial_command_submits,batch_command_submits,"
                 "job_command_submits,serial_dispatches,batch_dispatches,"
                 "graph_hash,output_hash,hash_parity,warm_zero\n",
                 stdout);
      for (const std::size_t count :
           {64u, 256u, 1024u, 4096u, 16384u, 65536u}) {
        batch_reference = {};
        ok = BatchJobs(focus, count, 12u) && ok;
      }
    } else if (resident) {
      std::fputs("resident_setup_columns,backend,status,count,samples,"
                 "median_us,transfer_bytes,resident_bytes,graph_hash,"
                 "output_hash\n",
                 stdout);
      ok = ResidentSetup(Backend::Cpu, 1024u, 21u) && ok;
      ok = ResidentSetup(Backend::Cpu, 1u << 20u, 7u) && ok;
      if (focus != Backend::Cpu) {
        ok = ResidentSetup(focus, 1024u, 21u) && ok;
        ok = ResidentSetup(focus, 1u << 20u, 7u) && ok;
      }
    } else if (collective) {
      PrintWorkloadColumns();
      ok = CollectiveWorkloads(Backend::Cpu, 4096u, 7u) && ok;
      ok = CollectiveWorkloads(Backend::Cpu, 1u << 18u, 5u) && ok;
      if (focus != Backend::Cpu) {
        ok = CollectiveWorkloads(focus, 4096u, 7u) && ok;
        ok = CollectiveWorkloads(focus, 1u << 18u, 5u) && ok;
      }
    } else if (bulk) {
      PrintBulkColumns();
      constexpr std::size_t measured_samples = 15u;
      const std::size_t cpu_samples =
          focus == Backend::Cpu ? measured_samples : 1u;
      ok = Bulk(Backend::Cpu, cpu_samples) && ok;
      if (focus != Backend::Cpu) {
        ok = Bulk(focus, measured_samples) && ok;
      }
    } else {
      PrintWorkloadColumns();
      constexpr std::size_t measured_samples = 15u;
      const std::size_t cpu_samples =
          focus == Backend::Cpu ? measured_samples : 1u;
      ok = SparseWorkloads(Backend::Cpu, 1u << 18u, cpu_samples) && ok;
      if (focus != Backend::Cpu) {
        ok = SparseWorkloads(focus, 1u << 18u, measured_samples) && ok;
      }
    }
    return ok ? 0 : 1;
  }
#else
  for (const Backend backend : kBackends) {
    ok = ReportEnvironment(backend) && ok;
  }
  std::fputs("warm_columns,backend,family,status,median_us", stdout);
  PrintStatsColumns();
  PrintWarmColumns();
  std::fputs(",resident_bytes,staging_bytes\n", stdout);
  PrintWorkloadColumns();
  std::printf("resident_setup_columns,backend,status,count,samples,median_us,"
              "transfer_bytes,resident_bytes,graph_hash,output_hash\n");
  for (const Backend backend : kBackends) {
    ok = ResidentSetup(backend, 1024u, 21u) && ok;
    ok = ResidentSetup(backend, 1u << 20u, 7u) && ok;
  }
  ok = Map("map_1_small", rund::compute::Target::cpu(1u), 1024u, 101u) && ok;
  ok =
      Map("map_host_small", rund::compute::Target::cpu(workers), 1024u, 101u) &&
      ok;
  ok = Map("map_1_large", rund::compute::Target::cpu(1u), 1u << 20u, 21u) && ok;
  ok = Map("map_host_large", rund::compute::Target::cpu(workers), 1u << 20u,
           21u) &&
       ok;
  std::fputs("host_map_columns,host,backend,family,count,median_us", stdout);
  PrintStatsColumns();
  PrintWarmColumns();
  std::fputs(",resident_bytes,staging_bytes\n", stdout);
  ok = NodeMap(workers, 1u << 20u, 21u) && ok;
  std::fputs("orchestration_columns,backend,status,submit_median_us,"
             "peer_median_us,completion_median_us,total_median_us,"
             "peer_completed,external_parks,external_wakes,parked,resumed,"
             "task_workers,configured_compute_workers",
             stdout);
  PrintStatsColumns();
  PrintWarmColumns();
  std::putchar('\n');
  for (const Backend backend : kBackends) {
    ok = NodeOrchestration(backend, workers, 4096u, 21u) && ok;
  }
  std::fputs("inflight_columns,backend,status,k,count,samples,serial_median_us,"
             "concurrent_median_us,serial_jobs_per_s,concurrent_jobs_per_s,"
             "concurrent_items_per_s,speedup,command_capacity,"
             "command_inflight_peak,command_capacity_rejections,graph_hash,"
             "output_hash,hash_parity,warm_zero,command_submits,dispatches\n",
             stdout);
  ok = InflightVulkan(workers, 1u << 18u, 7u) && ok;
  std::fputs(
      "batch_columns,backend,status,jobs,elements_per_job,samples,"
      "serial_first,batch_first,serial_wall_median_us,batch_wall_median_us,"
      "serial_submit_wait_us,"
      "batch_submit_wait_us,serial_kernel_us,batch_kernel_us,"
      "serial_host_residual_us,batch_host_residual_us,serial_jobs_per_s,"
      "batch_jobs_per_s,speedup,paired_speedup,serial_command_submits,"
      "batch_command_submits,job_command_submits,serial_dispatches,"
      "batch_dispatches,graph_hash,"
      "output_hash,hash_parity,warm_zero\n",
      stdout);
  ok = BatchJobs(Backend::Metal, 64u, 8u) && ok;
  ok = BatchJobs(Backend::Vulkan, 64u, 8u) && ok;
  std::printf(
      "mixed_columns,backends,status,serial_median_us,mixed_median_us,"
      "serial_jobs_per_s,mixed_jobs_per_s,external_parks,external_wakes,"
      "cpu_graph_hash,cpu_output_hash,metal_graph_hash,metal_output_hash,"
      "vulkan_graph_hash,vulkan_output_hash,"
      "hash_parity,warm_pipeline_compiles,warm_buffer_allocations,"
      "warm_download_events,warm_uploaded_bytes,warm_zero\n");
  ok = Mixed(workers, 4096u, 7u) && ok;
  for (const Backend backend : kBackends) {
    ok = FixedWidening32(backend) && ok;
    ok = FixedWidening64(backend) && ok;
    ok = SparseWorkloads(backend, 1u << 18u, 5u) && ok;
    ok = CollectiveWorkloads(backend, 4096u, 7u) && ok;
    ok = CollectiveWorkloads(backend, 1u << 18u, 5u) && ok;
    ok = Families(backend) && ok;
  }
  return ok ? 0 : 1;
#endif
}

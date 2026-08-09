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
  [[maybe_unused]] const std::uint32_t workers = hint == 0u ? 1u : hint;
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
  PrintProductColumns();
  for (const Backend backend : kBackends) {
    ok = ProductScenarios(backend) && ok;
  }
  return ok ? 0 : 1;
#endif
}

#include "suite/core.hpp"
#include "suite/reference.hpp"
#if defined(RUND_COMPUTE_FOCUS)
#include "pipeline.hpp"
#include "virtual/crossover.hpp"
#include "virtual/graph_pointwise.hpp"
#include "virtual/graph_residency.hpp"
#include "virtual/residency.hpp"
#include "virtual/window.hpp"
#endif

#include <cstdio>
#include <string>
#include <string_view>
#include <thread>

using namespace rund::measure::compute;

int main(const int argc, char **const argv) {
#if defined(RUND_COMPUTE_FOCUS)
  if (argc == 2 &&
      std::string_view{argv[1]} == "--virtual-crossover-aggregate") {
    return rund::measure::compute::AggregateVirtualCrossover() ? 0 : 1;
  }
  if (argc == 2 && std::string_view{argv[1]} == "--virtual-window-aggregate") {
    return rund::measure::compute::AggregateVirtualWindow() ? 0 : 1;
  }
  Backend focus = Backend::Unavailable;
  const bool virtual_crossover =
      argc == 2 && std::string_view{argv[1]} == "--virtual-crossover";
  if (virtual_crossover) {
    focus = Backend::Metal;
  }
  const bool virtual_window =
      argc == 2 && std::string_view{argv[1]} == "--virtual-window";
  if (virtual_window) {
    focus = Backend::Metal;
  }
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
  const bool virtual_residency =
      argc == 3 && std::string_view{argv[1]} == "--virtual-residency" &&
      ParseBackend(argv[2], focus);
  const bool virtual_residency_resident =
      argc == 3 &&
      std::string_view{argv[1]} == "--virtual-residency-resident" &&
      ParseBackend(argv[2], focus);
  const bool graph_residency_mode =
      argc == 3 && std::string_view{argv[1]} == "--virtual-graph-residency" &&
      ParseBackend(argv[2], focus);
  const bool graph_pointwise_mode =
      argc == 3 && std::string_view{argv[1]} == "--virtual-graph-pointwise" &&
      ParseBackend(argv[2], focus);
  const bool focused = virtual_crossover || virtual_window || collective ||
                       sort || bulk || resident || batch || pipeline ||
                       checkpoint || recurrence || window_repeat ||
                       pipeline_profile || plan_memory || prepare_memory ||
                       virtual_residency || virtual_residency_resident;
  const bool graph_focused = graph_residency_mode || graph_pointwise_mode;
  const bool any_focused = focused || graph_focused;
  if (!any_focused) {
    std::fputs("usage: runD-compute-focus "
               "[--resident|--collective|--sort|--bulk|--batch|--pipeline|"
               "--checkpoint|--recurrence|--window-repeat|--pipeline-profile|"
               "--plan-memory|--prepare-memory|--virtual-residency|"
               "--virtual-residency-resident|--virtual-graph-residency|"
               "--virtual-graph-pointwise cpu|metal|vulkan]|"
               "--virtual-crossover|--virtual-window\n",
               stderr);
#else
  (void)argv;
  if (argc != 1) {
    std::fputs("usage: runD-compute-measure\n", stderr);
#endif
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
  if (any_focused) {
    if (!virtual_residency && !virtual_residency_resident &&
        !graph_residency_mode && !graph_pointwise_mode) {
      ok = ReportEnvironment(Backend::Cpu) && ok;
      if (focus != Backend::Cpu) {
        ok = ReportEnvironment(focus) && ok;
      }
    }
    if (graph_residency_mode) {
      rund::measure::compute::virtual_graph_residency::Result result{};
      result.backend = focus;
      const auto report_failure = [focus](const char *status,
                                          const std::uint32_t code,
                                          const std::string_view error) {
        std::printf("environment,%s,%s,%u,", Name(focus), status,
                    static_cast<unsigned>(code));
        PrintCsv(error);
        std::fputs(",\"\",\"\",\"\"\n", stdout);
      };
      auto device = ::rund::compute::open(TargetFor(focus));
      if (!device) {
        result.device_code = static_cast<std::uint32_t>(device.code());
        result.device_error = std::string(device.error());
        report_failure("open_failed", result.device_code, device.error());
        ok = false;
      } else {
        const auto info = device->info();
        if (!info) {
          result.device_code = static_cast<std::uint32_t>(info.code());
          result.device_error = std::string(info.error());
          report_failure("info_failed", result.device_code, info.error());
          ok = false;
        } else {
          ok = rund::measure::compute::virtual_graph_residency::
                   ReportEnvironment(focus, *device, *info) &&
               ok;
          ok = info->backend == focus && ok;
          result.device_valid = info->backend == focus;
          result.device = info->name;
          result.driver = info->driver;
          result.driver_details = info->driver_details;
          result.device_code = static_cast<std::uint32_t>(
              info->backend == focus ? ::rund::compute::Code::Ok
                                     : ::rund::compute::Code::Invalid);
          if (info->backend != focus) {
            result.device_error = "compute_device_info_backend_mismatch";
          }
          rund::measure::compute::PrintVirtualGraphResidencyColumns();
          ok = rund::measure::compute::virtual_graph_residency::Run(
                   *device, *info, result) &&
               ok;
          rund::measure::compute::virtual_graph_residency::Report(result);
          return ok ? 0 : 1;
        }
      }
      rund::measure::compute::PrintVirtualGraphResidencyColumns();
      rund::measure::compute::virtual_graph_residency::Report(result);
    } else if (graph_pointwise_mode) {
      ok =
          rund::measure::compute::virtual_graph_pointwise::Measure(focus) && ok;
    } else if (virtual_crossover) {
      rund::measure::compute::PrintVirtualCrossoverColumns();
      ok = rund::measure::compute::MeasureVirtualCrossover() && ok;
    } else if (virtual_window) {
      rund::measure::compute::PrintVirtualWindowColumns();
      ok = rund::measure::compute::MeasureVirtualWindow() && ok;
    } else if (virtual_residency) {
      rund::measure::compute::PrintVirtualResidencyColumns();
      ok = rund::measure::compute::MeasureVirtualResidency(
               focus,
               rund::measure::compute::VirtualResidencyBacking::Callback) &&
           ok;
    } else if (virtual_residency_resident) {
      rund::measure::compute::PrintVirtualResidencyColumns();
      ok = rund::measure::compute::MeasureVirtualResidency(
               focus,
               rund::measure::compute::VirtualResidencyBacking::Resident) &&
           ok;
    } else if (checkpoint) {
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

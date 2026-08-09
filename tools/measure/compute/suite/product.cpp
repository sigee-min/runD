#include "core.hpp"

#include <rund/compute/cache.hpp>
#include <rund/compute/pipeline.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rund::measure::compute {
namespace {

// clang-format off
#include "product/model.hpp"
#include "product/oracle.hpp"
#include "product/report.hpp"
#include "product/exact.hpp"
#include "product/bounded.hpp"
// clang-format on

} // namespace

void PrintProductColumns() {
  std::fputs(
      "product_cold_columns,backend,chain,shape,status,capacity,active_count,"
      "radius,window_size,stride,candidate,width,stages,shared_capacity,"
      "candidate_scratch_bytes,source_hi,source_lo,execution_hi,execution_lo,"
      "first_result_us,author_us,compile_us,upload_us,prepare_us,run_us,"
      "read_us,dispatches,run_submissions,terminal_submissions,"
      "terminal_readbacks,logical_upload_bytes,logical_readback_bytes,"
      "downloaded_bytes,peak_retained_bytes,"
      "resident_peak_bytes,scratch_payload_bytes,scratch_backing_bytes,"
      "pipeline_compiles,pipeline_cache_hits,prepared_templates,graph_hash,"
      "output_hash,result\n",
      stdout);
  std::fputs(
      "product_warm_columns,backend,chain,shape,status,capacity,active_count,"
      "radius,window_size,stride,candidate,width,stages,shared_capacity,"
      "candidate_scratch_bytes,source_hi,source_lo,execution_hi,execution_lo,"
      "samples,warm_p50_us,warm_p95_us,active_elements_per_s,"
      "dispatches_per_run,submissions_per_run,terminal_submissions,"
      "terminal_readbacks,downloaded_bytes,"
      "peak_retained_bytes,resident_peak_bytes,scratch_payload_bytes,"
      "scratch_backing_bytes,sampled_runs,clean_runs,warm_cache_hits,"
      "graph_hash,"
      "output_hash,result\n",
      stdout);
}

bool ProductScenarios(const Backend backend) {
  using namespace rund::compute;
  auto opened = open(TargetFor(backend));
  if (!opened) {
    std::fprintf(stderr, "product %s device open failed: %.*s\n", Name(backend),
                 static_cast<int>(opened.error().size()),
                 opened.error().data());
    return false;
  }
  Device device = std::move(opened).value();
  auto cache_result = program_cache(device, 32u);
  if (!cache_result) {
    std::fprintf(stderr, "product %s cache failed: %.*s\n", Name(backend),
                 static_cast<int>(cache_result.error().size()),
                 cache_result.error().data());
    return false;
  }
  ProgramCache cache = std::move(cache_result).value();
  bool ok = true;
  const auto exact_window = [&](const std::size_t count,
                                const std::size_t radius,
                                const std::string_view name) {
    std::vector<std::uint32_t> input(count);
    fill_input(input, count);
    const auto mapped = mapped_prefix(input, count);
    const std::uint32_t expected = window_sum_oracle(mapped, radius);
    const ProductShape shape{.chain = "window",
                             .name = name,
                             .capacity = count,
                             .active = count,
                             .radius = radius,
                             .window = radius * 2u + 1u,
                             .stride = 1u};
    return measure_exact(device, backend, shape, input, expected, [&] {
      return on(device, cache)
          .map<std::uint32_t>("product-window-map", count,
                              [](auto value) { return (value & 3u) + 1u; })
          .window({.op = Window::Sum, .radius = radius})
          .filter([](auto value) { return (value & 7u) != 0u; })
          .reduce(Reduce::Sum);
    });
  };
  ok = exact_window(4096u, 4u, "n4096_r4") && ok;
  ok = exact_window(4096u, 1024u, "n4096_r1024") && ok;
  ok = exact_window(BoundedCapacity, 4u, "n262144_r4") && ok;
  ok = exact_window(BoundedCapacity, 1024u, "n262144_r1024") && ok;

  const auto exact_pool = [&](const std::size_t count, const std::size_t width,
                              const std::size_t stride,
                              const std::string_view name) {
    std::vector<std::uint32_t> input(count);
    fill_input(input, count);
    const auto mapped = mapped_prefix(input, count);
    const std::uint32_t expected = pool_sum_oracle(mapped, width, stride);
    const ProductShape shape{.chain = "pool",
                             .name = name,
                             .capacity = count,
                             .active = count,
                             .radius = 0u,
                             .window = width,
                             .stride = stride};
    return measure_exact(device, backend, shape, input, expected, [&] {
      return on(device, cache)
          .map<std::uint32_t>("product-pool-map", count,
                              [](auto value) { return (value & 3u) + 1u; })
          .pool({.op = Window::Sum,
                 .width = width,
                 .stride = stride,
                 .edge = WindowEdge::Clip,
                 .tail = PoolTail::Keep})
          .filter([](auto value) { return (value & 7u) != 0u; })
          .reduce(Reduce::Sum);
    });
  };
  ok = exact_pool(4096u, 129u, 2u, "n4096_k129_s2") && ok;
  ok = exact_pool(BoundedCapacity, 2049u, 2u, "n262144_k2049_s2") && ok;
  ok = measure_bounded(device, cache, backend) && ok;
  return ok;
}

} // namespace rund::measure::compute

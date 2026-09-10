#include "local.hpp"

#include <array>
#include <limits>

namespace rund_node_test_pipeline::view {
namespace {

[[nodiscard]] bool CheckNestedCpuViewPhase(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 4u> initial_values{1u, 2u, 3u, 4u};
  constexpr std::array<std::uint32_t, 8u> backing_values{99u, 1u, 99u, 2u,
                                                         99u, 3u, 99u, 4u};
  constexpr std::array<std::uint32_t, 1u> count_value{2u};
  auto seed = on(device)
                  .input<std::uint32_t>(4u)
                  .zip_input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto external, auto total, auto ordinal) {
                    (void)total;
                    auto sum = external.reduce(Reduce::Sum);
                    return sum.combine(
                        "pipeline-view-nested-seed", ordinal.scalar(),
                        [](auto value, auto outer) { return value + outer; });
                  })
                  .compile();
  auto fold =
      on(device)
          .input<std::uint32_t>(4u)
          .zip_input<std::uint32_t>(1u)
          .branch([](auto outer, auto tile) {
            return outer.combine(
                "pipeline-view-nested-fold", tile.scalar(),
                [](auto value, auto increment) { return value + increment; });
          })
          .compile();
  auto initial = device.upload<std::uint32_t>(initial_values);
  auto backing = device.upload<std::uint32_t>(backing_values);
  auto count = device.upload<std::uint32_t>(count_value);
  auto target = device.buffer<std::uint32_t>(4u);
  if (!seed || !fold || !initial || !backing || !count || !target) {
    return false;
  }
  auto external = backing->view(1u, 4u, 2u);
  if (!external) {
    return false;
  }
  // Fold consumes the sealed dense recurrent banks, so the reachable CPU View
  // transfer is the Seed-only external input. This proves record_cpu_view uses
  // the common route projector while retaining the compact Seed outer ordinal.
  const auto body = tile_repeat<0u>(*seed, *fold);
  auto builder = pipeline(device);
  builder.windows<2u, 1u>(body, window(*count), read(*initial, *external),
                          write_final(*target));
  const auto plan = builder.plan();
  if (!plan || plan->view_nested_phase != PipelineNestedPhase::Seed ||
      plan->view_outer_window != 0u ||
      plan->view_inner_iteration != std::numeric_limits<std::size_t>::max() ||
      plan->view_count != 4u ||
      plan->view_stride_bytes != 2u * sizeof(std::uint32_t) ||
      plan->view_span_bytes != 7u * sizeof(std::uint32_t)) {
    std::fprintf(
        stderr,
        "nested CPU view plan status=%u reason=%u phase=%u step=%llu "
        "iteration=%llu coordinates=%llu/%llu count=%llu stride=%llu "
        "span=%llu\n",
        static_cast<unsigned>(plan.ok()), static_cast<unsigned>(plan.reason()),
        static_cast<unsigned>(plan ? plan->view_nested_phase
                                   : PipelineNestedPhase::None),
        static_cast<unsigned long long>(plan ? plan->view_step : 0u),
        static_cast<unsigned long long>(plan ? plan->view_iteration : 0u),
        static_cast<unsigned long long>(plan ? plan->view_outer_window : 0u),
        static_cast<unsigned long long>(plan ? plan->view_inner_iteration : 0u),
        static_cast<unsigned long long>(plan ? plan->view_count : 0u),
        static_cast<unsigned long long>(plan ? plan->view_stride_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->view_span_bytes : 0u));
    return false;
  }
  auto prepared = std::move(builder).prepare();
  std::array<std::uint32_t, 4u> observed{};
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const bool read = prepared && ReadExact(*prepared, *target, observed);
  if (!prepared || !ran || !read ||
      observed != std::array<std::uint32_t, 4u>{22u, 23u, 24u, 25u}) {
    std::fprintf(stderr,
                 "nested CPU view run prepared=%u/%u run=%u/%u read=%u "
                 "values=%u,%u,%u,%u\n",
                 static_cast<unsigned>(prepared.ok()),
                 static_cast<unsigned>(prepared.reason()),
                 static_cast<unsigned>(ran.ok()),
                 static_cast<unsigned>(ran.reason()),
                 static_cast<unsigned>(read), observed[0u], observed[1u],
                 observed[2u], observed[3u]);
    return false;
  }
  return true;
}

} // namespace

[[nodiscard]] int CheckBasic(rund::compute::Device &device,
                             const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (backend == Backend::Cpu && !CheckNestedCpuViewPhase(device)) {
    return 9;
  }
  constexpr std::array<std::int32_t, 8u> source_values{0, 1, 2, 3, 4, 5, 6, 7};
  auto source = Upload(device, source_values);
  auto target = device.buffer<std::int32_t>(source_values.size());
  auto program = on(device)
                     .map<std::int32_t>("pipeline-strided-view", 3u,
                                        [](auto value) { return value * 10; })
                     .compile();
  if (!source || !target || !program) {
    return 1;
  }
  auto input_view = source->view(1u, 3u, 2u);
  auto output_view = target->view(0u, 3u, 2u);
  if (!input_view || !output_view || input_view->offset() != 1u ||
      input_view->size() != 3u || input_view->stride() != 2u ||
      input_view->span_bytes() != 5u * sizeof(std::int32_t)) {
    return 2;
  }
  auto prepared = pipeline(device)
                      .then(*program, read(*input_view), write(*output_view))
                      .prepare();
  if (!prepared) {
    return 3;
  }
  const Status view_run = prepared->run();
  if (!view_run || prepared->stats().pipeline.barrier_count != 0u) {
    return 3;
  }
  std::array<std::int32_t, source_values.size()> observed{};
  if (!ReadExact(*prepared, *target, observed) ||
      observed != std::array<std::int32_t, 8u>{10, 0, 30, 0, 50, 0, 0, 0}) {
    return 4;
  }

  auto even = on(device)
                  .map<std::int32_t>("pipeline-even-view", 4u,
                                     [](auto value) { return value + 1; })
                  .compile();
  auto odd = on(device)
                 .map<std::int32_t>("pipeline-odd-view", 4u,
                                    [](auto value) { return value + 2; })
                 .compile();
  auto disjoint_target = device.buffer<std::int32_t>(source_values.size());
  auto even_input = source->view(0u, 4u, 2u);
  auto odd_input = source->view(1u, 4u, 2u);
  auto even_output = disjoint_target->view(0u, 4u, 2u);
  auto odd_output = disjoint_target->view(1u, 4u, 2u);
  if (!even || !odd || !disjoint_target || !even_input || !odd_input ||
      !even_output || !odd_output) {
    return 5;
  }
  auto disjoint = pipeline(device)
                      .then(*even, read(*even_input), write(*even_output))
                      .then(*odd, read(*odd_input), write(*odd_output))
                      .prepare();
  if (!disjoint || !disjoint->run() ||
      disjoint->stats().pipeline.barrier_count != 0u ||
      !ReadExact(*disjoint, *disjoint_target, observed) ||
      observed != std::array<std::int32_t, 8u>{1, 3, 3, 5, 5, 7, 7, 9}) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::view

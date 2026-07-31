#include "model.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <utility>
#include <vector>

namespace rund_node_collective_modes {

template <class T>
[[nodiscard]] bool CheckEmpty(const rund::compute::Backend backend,
                              DomainEvidence &evidence) {
  using namespace rund::compute;
  std::vector<T> empty;
  std::vector<std::uint32_t> empty_heads;
  auto exact_target = flow_on(backend, Target::cpu(2u));
  auto exact = std::move(exact_target)
                   .template input<T>(0u)
                   .branch([](auto values) {
                     return outputs(values, values.scan(Scan::InclusiveSum),
                                    values.scan(Scan::ExclusiveSum),
                                    values.reduce(Reduce::Sum));
                   })
                   .compile();
  if (!exact) {
    std::fprintf(stderr,
                 "compute modes exact empty compile backend=%u width=%zu "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(exact.error().size()), exact.error().data());
    return false;
  }
  auto exact_job = exact->resident(empty);
  if (!exact_job) {
    std::fprintf(stderr,
                 "compute modes exact empty resident backend=%u width=%zu "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(exact_job.error().size()),
                 exact_job.error().data());
    return false;
  }
  if (!SameSuccess(*exact_job, backend, "exact-empty", evidence.exact_empty)) {
    return false;
  }
  auto exact_output = exact_job->read_all();
  if (!exact_output || !std::get<0>(*exact_output).empty() ||
      !std::get<1>(*exact_output).empty() ||
      !std::get<2>(*exact_output).empty() ||
      std::get<3>(*exact_output) != Zero<T>()) {
    std::fprintf(stderr,
                 "compute modes exact empty identity mismatch backend=%u "
                 "width=%zu\n",
                 static_cast<unsigned>(backend), sizeof(T));
    return false;
  }

  auto minimum_target = flow_on(backend, Target::cpu(2u));
  auto exact_minimum =
      std::move(minimum_target)
          .template input<T>(0u)
          .branch([](auto values) { return values.reduce(Reduce::Min); })
          .compile();
  auto maximum_target = flow_on(backend, Target::cpu(2u));
  auto exact_maximum =
      std::move(maximum_target)
          .template input<T>(0u)
          .branch([](auto values) { return values.reduce(Reduce::Max); })
          .compile();
  if (exact_minimum || exact_maximum ||
      exact_minimum.error() != "compute_reduce_count_zero" ||
      exact_maximum.error() != "compute_reduce_count_zero") {
    std::fprintf(stderr,
                 "compute modes exact empty reduce rejection mismatch "
                 "backend=%u width=%zu min=%.*s max=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(exact_minimum.error().size()),
                 exact_minimum.error().data(),
                 static_cast<int>(exact_maximum.error().size()),
                 exact_maximum.error().data());
    return false;
  }

  auto segmented_target = flow_on(backend, Target::cpu(2u));
  auto segmented =
      std::move(segmented_target)
          .template input<T>(0u)
          .template zip_input<std::uint32_t>(0u)
          .branch([](auto values, auto segments) {
            return outputs(values.segmented_scan(segments, Scan::InclusiveSum),
                           values.segmented_scan(segments, Scan::ExclusiveSum),
                           values.segmented_reduce(segments, Reduce::Sum),
                           values.segmented_reduce(segments, Reduce::Min),
                           values.segmented_reduce(segments, Reduce::Max));
          })
          .compile();
  if (!segmented) {
    std::fprintf(
        stderr,
        "compute modes segmented empty compile backend=%u reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(segmented.error().size()), segmented.error().data());
    return false;
  }
  auto segmented_job = segmented->resident(empty, empty_heads);
  if (!segmented_job) {
    std::fprintf(stderr,
                 "compute modes segmented empty resident backend=%u "
                 "width=%zu reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(segmented_job.error().size()),
                 segmented_job.error().data());
    return false;
  }
  if (!SameSuccess(*segmented_job, backend, "segmented-empty",
                   evidence.segmented_empty)) {
    return false;
  }
  auto segmented_output = segmented_job->read_all();
  if (!segmented_output || !std::get<0>(*segmented_output).empty() ||
      !std::get<1>(*segmented_output).empty() ||
      !std::get<2>(*segmented_output).empty() ||
      !std::get<3>(*segmented_output).empty() ||
      !std::get<4>(*segmented_output).empty()) {
    std::fprintf(stderr,
                 "compute modes segmented empty identity mismatch backend=%u "
                 "width=%zu\n",
                 static_cast<unsigned>(backend), sizeof(T));
    return false;
  }

  std::size_t edge_index = 0u;
  for (const WindowEdge edge : {WindowEdge::Clamp, WindowEdge::Clip}) {
    for (const Window operation : {Window::Sum, Window::Min, Window::Max}) {
      auto window_target = flow_on(backend, Target::cpu(2u));
      auto window = std::move(window_target)
                        .template input<T>(0u)
                        .branch([=](auto values) {
                          return values.window(
                              {.op = operation, .radius = 1u, .edge = edge});
                        })
                        .compile();
      if (operation == Window::Sum) {
        if (!window) {
          std::fprintf(
              stderr,
              "compute modes exact empty window compile backend=%u edge=%u "
              "reason=%.*s\n",
              static_cast<unsigned>(backend), static_cast<unsigned>(edge),
              static_cast<int>(window.error().size()), window.error().data());
          return false;
        }
        auto window_job = window->resident(empty);
        const char *const family = edge == WindowEdge::Clamp
                                       ? "exact-empty-window-clamp"
                                       : "exact-empty-window-clip";
        if (!window_job) {
          std::fprintf(stderr,
                       "compute modes exact empty window resident backend=%u "
                       "width=%zu edge=%u reason=%.*s\n",
                       static_cast<unsigned>(backend), sizeof(T),
                       static_cast<unsigned>(edge),
                       static_cast<int>(window_job.error().size()),
                       window_job.error().data());
          return false;
        }
        if (!SameSuccess(*window_job, backend, family,
                         evidence.exact_empty_window[edge_index])) {
          return false;
        }
        auto output = window_job->read();
        if (!output || !output->empty()) {
          std::fprintf(stderr,
                       "compute modes exact empty window identity mismatch "
                       "backend=%u edge=%u width=%zu\n",
                       static_cast<unsigned>(backend),
                       static_cast<unsigned>(edge), sizeof(T));
          return false;
        }
        continue;
      }
      if (window || window.error() != "compute_stencil_count_zero") {
        std::fprintf(
            stderr,
            "compute modes exact empty window rejection mismatch "
            "backend=%u width=%zu edge=%u operation=%u reason=%.*s\n",
            static_cast<unsigned>(backend), sizeof(T),
            static_cast<unsigned>(edge), static_cast<unsigned>(operation),
            static_cast<int>(window.error().size()), window.error().data());
        return false;
      }
    }
    ++edge_index;
  }

  std::array<T, 1u> input{Zero<T>()};
  const auto make_bounded = [&] {
    auto target = flow_on(backend, Target::cpu(2u));
    return std::move(target)
        .template map<T>("mode-bounded-empty", input.size(),
                         [](auto value) { return Store<T>(value); })
        .filter([](auto value) { return value != value; });
  };
  auto bounded = make_bounded()
                     .branch([](auto values) {
                       return outputs(values, values.scan(Scan::InclusiveSum),
                                      values.scan(Scan::ExclusiveSum),
                                      values.reduce(Reduce::Sum), values.sort(),
                                      values.argsort(), values.count());
                     })
                     .compile();
  auto minimum = make_bounded().reduce(Reduce::Min).compile();
  auto maximum = make_bounded().reduce(Reduce::Max).compile();
  auto bounded_window =
      make_bounded()
          .branch([](auto values) {
            return outputs(values.window({.op = Window::Sum, .radius = 1u}),
                           values.window({.op = Window::Min, .radius = 1u}),
                           values.window({.op = Window::Max, .radius = 1u}),
                           values.window({.op = Window::Sum,
                                          .radius = 1u,
                                          .edge = WindowEdge::Clip}),
                           values.window({.op = Window::Min,
                                          .radius = 1u,
                                          .edge = WindowEdge::Clip}),
                           values.window({.op = Window::Max,
                                          .radius = 1u,
                                          .edge = WindowEdge::Clip}));
          })
          .compile();
  if (!bounded || !minimum || !maximum || !bounded_window) {
    const auto reason = [&]() -> std::string_view {
      if (!bounded) {
        return bounded.error();
      }
      if (!minimum) {
        return minimum.error();
      }
      if (!maximum) {
        return maximum.error();
      }
      return bounded_window.error();
    }();
    const char *const family = !bounded   ? "bounded"
                               : !minimum ? "minimum"
                               : !maximum ? "maximum"
                                          : "bounded-window";
    std::fprintf(stderr,
                 "compute modes bounded empty compile backend=%u width=%zu "
                 "family=%s reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T), family,
                 static_cast<int>(reason.size()), reason.data());
    return false;
  }
  auto bounded_job = bounded->resident(input);
  if (!bounded_job) {
    std::fprintf(stderr,
                 "compute modes bounded empty resident backend=%u width=%zu "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(bounded_job.error().size()),
                 bounded_job.error().data());
    return false;
  }
  if (!SameSuccess(*bounded_job, backend, "bounded-empty",
                   evidence.bounded_empty)) {
    return false;
  }
  auto bounded_output = bounded_job->read_all();
  if (!bounded_output || !std::get<0>(*bounded_output).empty() ||
      !std::get<1>(*bounded_output).empty() ||
      !std::get<2>(*bounded_output).empty() ||
      std::get<3>(*bounded_output) != Zero<T>() ||
      !std::get<4>(*bounded_output).empty() ||
      !std::get<5>(*bounded_output).empty() ||
      std::get<6>(*bounded_output) != 0u) {
    std::fprintf(stderr,
                 "compute modes bounded empty identity mismatch backend=%u "
                 "width=%zu\n",
                 static_cast<unsigned>(backend), sizeof(T));
    return false;
  }
  auto bounded_window_job = bounded_window->resident(input);
  if (!bounded_window_job) {
    std::fprintf(stderr,
                 "compute modes bounded empty window resident backend=%u "
                 "width=%zu reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(bounded_window_job.error().size()),
                 bounded_window_job.error().data());
    return false;
  }
  if (!SameSuccess(*bounded_window_job, backend, "bounded-empty-window",
                   evidence.bounded_empty_window)) {
    return false;
  }
  auto bounded_window_output = bounded_window_job->read_all();
  if (!bounded_window_output || !std::get<0>(*bounded_window_output).empty() ||
      !std::get<1>(*bounded_window_output).empty() ||
      !std::get<2>(*bounded_window_output).empty() ||
      !std::get<3>(*bounded_window_output).empty() ||
      !std::get<4>(*bounded_window_output).empty() ||
      !std::get<5>(*bounded_window_output).empty()) {
    std::fprintf(stderr,
                 "compute modes bounded empty window mismatch backend=%u "
                 "width=%zu\n",
                 static_cast<unsigned>(backend), sizeof(T));
    return false;
  }
  return SameFailure(*minimum, backend, "bounded-empty-min",
                     "compute_reduce_count_zero", evidence.bounded_empty_min,
                     input) &&
         SameFailure(*maximum, backend, "bounded-empty-max",
                     "compute_reduce_count_zero", evidence.bounded_empty_max,
                     input);
}

[[nodiscard]] bool CheckEmpty(const rund::compute::Backend backend,
                              DomainEvidence &evidence, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckEmpty<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckEmpty<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckEmpty<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckEmpty<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckEmpty<rund::compute::Fixed<16, 16>>(backend, evidence);
  case Domain::Fixed20x44:
    return CheckEmpty<rund::compute::Fixed<20, 44>>(backend, evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

} // namespace rund_node_collective_modes

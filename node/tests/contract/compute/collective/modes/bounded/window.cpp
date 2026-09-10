#include "../model.hpp"

#include <array>
#include <cstdio>
#include <ranges>
#include <tuple>
#include <vector>

namespace rund_node_collective_modes::bounded {
namespace {

template <class T>
[[nodiscard]] bool CheckWindowFor(const rund::compute::Backend backend,
                                  DomainEvidence &evidence) {
  using namespace rund::compute;
  auto input = TailValues<T>();
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template map<T>("mode-bounded-window-input", input.size(),
                           [](auto value) { return Store<T>(value); })
          .filter([](auto value) { return value != Zero<T>(); })
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
  if (!program) {
    std::fprintf(
        stderr,
        "compute modes compile backend=%u family=bounded-window reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return false;
  }
  auto job = program->resident(input);
  if (!job) {
    return false;
  }
  const bool host_same =
      SameSuccess(*job, backend, "bounded-window", evidence.bounded_window);
  std::vector<std::int64_t> logical;
  logical.reserve(input.size());
  for (std::size_t index = 0u; index < input.size(); ++index) {
    const std::int64_t value = TailInteger<T>(index);
    if (value != 0) {
      logical.push_back(value);
    }
  }
  const auto clamped = ExpectedWindows<T>(logical, false);
  const auto clipped = ExpectedWindows<T>(logical, true);
  auto output = job->read_all();
  const std::array<bool, 6u> fields{
      output && std::get<0>(*output) == clamped[0u],
      output && std::get<1>(*output) == clamped[1u],
      output && std::get<2>(*output) == clamped[2u],
      output && std::get<3>(*output) == clipped[0u],
      output && std::get<4>(*output) == clipped[1u],
      output && std::get<5>(*output) == clipped[2u]};
  const bool same =
      std::ranges::all_of(fields, [](const bool value) { return value; });
  if (!same) {
    std::fprintf(stderr,
                 "compute modes bounded window golden mismatch backend=%u "
                 "width=%zu fields=%u%u%u%u%u%u\n",
                 static_cast<unsigned>(backend), sizeof(T), fields[0u],
                 fields[1u], fields[2u], fields[3u], fields[4u], fields[5u]);
  }
  return host_same && same;
}

} // namespace

bool CheckWindow(const rund::compute::Backend backend,
                 DomainEvidence &evidence, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckWindowFor<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckWindowFor<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckWindowFor<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckWindowFor<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckWindowFor<rund::compute::Fixed<16, 16>>(backend, evidence);
  case Domain::Fixed20x44:
    return CheckWindowFor<rund::compute::Fixed<20, 44>>(backend, evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

} // namespace rund_node_collective_modes::bounded

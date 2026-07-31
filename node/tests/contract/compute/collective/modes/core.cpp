#include "model.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <utility>
#include <vector>

namespace rund_node_collective_modes {

template <class T>
[[nodiscard]] bool CheckModes(const rund::compute::Backend backend,
                              DomainEvidence &evidence) {
  using namespace rund::compute;
  auto input = TailValues<T>();
  auto heads = SegmentHeads(input.size());
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template input<T>(input.size())
          .template zip_input<std::uint32_t>(heads.size())
          .branch([](auto values, auto segments) {
            return outputs(
                values.scan(Scan::InclusiveSum),
                values.scan(Scan::ExclusiveSum), values.reduce(Reduce::Sum),
                values.reduce(Reduce::Min), values.reduce(Reduce::Max),
                values.window({.op = Window::Sum, .radius = 1u}),
                values.window({.op = Window::Min, .radius = 1u}),
                values.window({.op = Window::Max, .radius = 1u}),
                values.segmented_scan(segments, Scan::InclusiveSum),
                values.segmented_scan(segments, Scan::ExclusiveSum),
                values.segmented_reduce(segments, Reduce::Sum),
                values.segmented_reduce(segments, Reduce::Min),
                values.segmented_reduce(segments, Reduce::Max));
          })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "compute modes compile backend=%u family=all reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return false;
  }
  auto job = program->resident(input, heads);
  if (!job) {
    std::fprintf(stderr,
                 "compute modes resident backend=%u family=all reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  return SameSuccess(*job, backend, "all", evidence.modes);
}

template <class T>
[[nodiscard]] bool CheckClip(const rund::compute::Backend backend,
                             DomainEvidence &evidence) {
  using namespace rund::compute;
  auto input = TailValues<T>();
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template input<T>(input.size())
          .branch([](auto values) {
            return outputs(values.window({.op = Window::Sum,
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
        stderr, "compute modes compile backend=%u family=clip reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return false;
  }
  auto job = program->resident(input);
  if (!job || !SameSuccess(*job, backend, "clip", evidence.clip)) {
    return false;
  }
  std::vector<std::int64_t> logical(input.size());
  for (std::size_t index = 0u; index < logical.size(); ++index) {
    logical[index] = TailInteger<T>(index);
  }
  const auto expected = ExpectedWindows<T>(logical, true);
  auto output = job->read_all();
  const bool same = output && std::get<0>(*output) == expected[0u] &&
                    std::get<1>(*output) == expected[1u] &&
                    std::get<2>(*output) == expected[2u];
  if (!same) {
    std::fprintf(stderr,
                 "compute modes clip golden mismatch backend=%u width=%zu\n",
                 static_cast<unsigned>(backend), sizeof(T));
  }
  return same;
}

template <class T>
[[nodiscard]] bool CheckExtrema(const rund::compute::Backend backend,
                                DomainEvidence &evidence) {
  using namespace rund::compute;
  std::array<T, 2u> input{Minimum<T>(), Maximum<T>()};
  std::array<std::uint32_t, 2u> heads{1u, 0u};
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template input<T>(input.size())
          .template zip_input<std::uint32_t>(heads.size())
          .branch([](auto values, auto segments) {
            return outputs(values.reduce(Reduce::Min),
                           values.reduce(Reduce::Max),
                           values.window({.op = Window::Sum, .radius = 1u}),
                           values.window({.op = Window::Min, .radius = 1u}),
                           values.window({.op = Window::Max, .radius = 1u}),
                           values.segmented_reduce(segments, Reduce::Min),
                           values.segmented_reduce(segments, Reduce::Max),
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
        stderr, "compute modes compile backend=%u family=extrema reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return false;
  }
  auto job = program->resident(input, heads);
  return job && SameSuccess(*job, backend, "extrema", evidence.extrema);
}

template <class T>
[[nodiscard]] bool CheckCoreDomain(const rund::compute::Backend backend,
                                   DomainEvidence &evidence) {
  return CheckModes<T>(backend, evidence) && CheckClip<T>(backend, evidence) &&
         CheckExtrema<T>(backend, evidence);
}

[[nodiscard]] bool CheckCore(const rund::compute::Backend backend,
                             DomainEvidence &evidence, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckCoreDomain<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckCoreDomain<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckCoreDomain<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckCoreDomain<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckCoreDomain<rund::compute::Fixed<16, 16>>(backend, evidence);
  case Domain::Fixed20x44:
    return CheckCoreDomain<rund::compute::Fixed<20, 44>>(backend, evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

} // namespace rund_node_collective_modes

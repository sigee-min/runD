#include "model.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <type_traits>
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
[[nodiscard]] bool CheckPool(const rund::compute::Backend backend,
                             DomainEvidence &evidence) {
  using namespace rund::compute;
  auto input = TailValues<T>();
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template input<T>(input.size())
          .branch([](auto values) {
            constexpr PoolSpec drop{.op = Window::Sum,
                                    .width = 129u,
                                    .stride = 2u,
                                    .edge = WindowEdge::Clip,
                                    .tail = PoolTail::Drop};
            constexpr PoolSpec keep{.op = Window::Sum,
                                    .width = 129u,
                                    .stride = 2u,
                                    .edge = WindowEdge::Clamp,
                                    .tail = PoolTail::Keep};
            return outputs(values.pool(drop),
                           values.pool(PoolSpec{.op = Window::Min,
                                                .width = drop.width,
                                                .stride = drop.stride,
                                                .edge = drop.edge,
                                                .tail = drop.tail}),
                           values.pool(PoolSpec{.op = Window::Max,
                                                .width = drop.width,
                                                .stride = drop.stride,
                                                .edge = drop.edge,
                                                .tail = drop.tail}),
                           values.pool(keep),
                           values.pool(PoolSpec{.op = Window::Min,
                                                .width = keep.width,
                                                .stride = keep.stride,
                                                .edge = keep.edge,
                                                .tail = keep.tail}),
                           values.pool(PoolSpec{.op = Window::Max,
                                                .width = keep.width,
                                                .stride = keep.stride,
                                                .edge = keep.edge,
                                                .tail = keep.tail}));
          })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "compute modes compile backend=%u family=pool reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return false;
  }
  auto job = program->resident(input);
  if (!job || !SameSuccess(*job, backend, "pool", evidence.pool)) {
    return false;
  }
  constexpr std::size_t width = 129u;
  constexpr std::size_t stride = 2u;
  const auto expected = [&](const bool keep, const bool clamp,
                            const Window op) {
    const std::size_t count = keep ? 1u + (input.size() - 1u) / stride
                                   : 1u + (input.size() - width) / stride;
    std::vector<T> values;
    values.reserve(count);
    for (std::size_t output = 0u; output < count; ++output) {
      const std::size_t begin = output * stride;
      std::int64_t value = TailInteger<T>(begin);
      for (std::size_t offset = 1u; offset < width; ++offset) {
        const std::size_t source = begin + offset;
        if (!clamp && source >= input.size()) {
          continue;
        }
        const std::int64_t next = TailInteger<T>(
            std::min(source, static_cast<std::size_t>(input.size() - 1u)));
        value = op == Window::Sum   ? value + next
                : op == Window::Min ? std::min(value, next)
                                    : std::max(value, next);
      }
      values.push_back(DomainValue<T>(value));
    }
    return values;
  };
  auto output = job->read_all();
  const bool same =
      output && std::get<0>(*output) == expected(false, false, Window::Sum) &&
      std::get<1>(*output) == expected(false, false, Window::Min) &&
      std::get<2>(*output) == expected(false, false, Window::Max) &&
      std::get<3>(*output) == expected(true, true, Window::Sum) &&
      std::get<4>(*output) == expected(true, true, Window::Min) &&
      std::get<5>(*output) == expected(true, true, Window::Max);
  if (!same) {
    std::fprintf(stderr,
                 "compute modes pool golden mismatch backend=%u width=%zu\n",
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
[[nodiscard]] bool CheckFixedWrapPool(const rund::compute::Backend backend,
                                      DomainEvidence &evidence) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  using Unsigned = std::make_unsigned_t<Raw>;
  constexpr std::size_t count = 257u;
  constexpr std::size_t width = 129u;
  std::array<T, count> input{};
  constexpr std::array<Raw, 4u> pattern{std::numeric_limits<Raw>::max(), Raw{2},
                                        std::numeric_limits<Raw>::min(),
                                        Raw{-2}};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = T::from_raw(pattern[index % pattern.size()]);
  }
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template input<T>(count)
          .branch([](auto values) {
            return values
                .map(
                    "pool-wrap-source",
                    [](auto value) {
                      return quantize<T, Rounding::NearestEven, Overflow::Wrap>(
                          value);
                    })
                .pool(PoolSpec{.op = Window::Sum,
                               .width = width,
                               .stride = 1u,
                               .edge = WindowEdge::Clip,
                               .tail = PoolTail::Drop});
          })
          .compile();
  if (!program) {
    std::fprintf(stderr,
                 "compute modes pool wrap compile backend=%u width=%zu "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  auto job = program->resident(input);
  if (!job || !SameSuccess(*job, backend, "pool-wrap", evidence.pool_wrap)) {
    if (!job) {
      std::fprintf(stderr,
                   "compute modes pool wrap resident backend=%u width=%zu "
                   "reason=%.*s\n",
                   static_cast<unsigned>(backend), sizeof(T),
                   static_cast<int>(job.error().size()), job.error().data());
    }
    return false;
  }
  auto output = job->read();
  if (!output || output->size() != count - width + 1u) {
    std::fprintf(stderr,
                 "compute modes pool wrap output backend=%u width=%zu\n",
                 static_cast<unsigned>(backend), sizeof(T));
    return false;
  }
  for (std::size_t begin = 0u; begin < output->size(); ++begin) {
    Unsigned accumulator = 0u;
    for (std::size_t offset = 0u; offset < width; ++offset) {
      accumulator += static_cast<Unsigned>(input[begin + offset].raw());
    }
    if ((*output)[begin].raw() != std::bit_cast<Raw>(accumulator)) {
      std::fprintf(stderr,
                   "compute modes pool wrap golden backend=%u width=%zu "
                   "index=%zu expected=%lld actual=%lld\n",
                   static_cast<unsigned>(backend), sizeof(T), begin,
                   static_cast<long long>(std::bit_cast<Raw>(accumulator)),
                   static_cast<long long>((*output)[begin].raw()));
      return false;
    }
  }
  return true;
}

template <class T>
[[nodiscard]] bool CheckLargeWindow(const rund::compute::Backend backend,
                                    DomainEvidence &evidence) {
  using namespace rund::compute;
  constexpr std::size_t count = 257u;
  constexpr std::size_t radius = 128u;
  const auto source_value = [](const std::size_t index) {
    if constexpr (std::is_unsigned_v<T>) {
      return static_cast<std::int64_t>(index % 11u);
    } else {
      return static_cast<std::int64_t>(index % 11u) - 5;
    }
  };
  std::array<T, count> input{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = DomainValue<T>(source_value(index));
  }
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template input<T>(count)
          .branch([](auto values) {
            return outputs(values.window({.op = Window::Sum,
                                          .radius = radius,
                                          .edge = WindowEdge::Clamp}),
                           values.window({.op = Window::Min,
                                          .radius = radius,
                                          .edge = WindowEdge::Clamp}),
                           values.window({.op = Window::Max,
                                          .radius = radius,
                                          .edge = WindowEdge::Clamp}),
                           values.window({.op = Window::Sum,
                                          .radius = radius,
                                          .edge = WindowEdge::Clip}),
                           values.window({.op = Window::Min,
                                          .radius = radius,
                                          .edge = WindowEdge::Clip}),
                           values.window({.op = Window::Max,
                                          .radius = radius,
                                          .edge = WindowEdge::Clip}));
          })
          .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input);
  if (!job ||
      !SameSuccess(*job, backend, "large-window", evidence.large_window)) {
    return false;
  }
  auto output = job->read_all();
  if (!output) {
    return false;
  }
  const auto expected = [&](const Window operation, const bool clamp) {
    std::vector<T> result;
    result.reserve(count);
    for (std::size_t center = 0u; center < count; ++center) {
      std::int64_t value =
          operation == Window::Min   ? std::numeric_limits<std::int64_t>::max()
          : operation == Window::Max ? std::numeric_limits<std::int64_t>::min()
                                     : 0;
      for (std::int64_t delta = -static_cast<std::int64_t>(radius);
           delta <= static_cast<std::int64_t>(radius); ++delta) {
        const std::int64_t logical = static_cast<std::int64_t>(center) + delta;
        if (!clamp &&
            (logical < 0 || logical >= static_cast<std::int64_t>(count))) {
          continue;
        }
        const std::size_t source =
            static_cast<std::size_t>(std::clamp<std::int64_t>(
                logical, 0, static_cast<std::int64_t>(count - 1u)));
        const std::int64_t next = source_value(source);
        value = operation == Window::Sum   ? value + next
                : operation == Window::Min ? std::min(value, next)
                                           : std::max(value, next);
      }
      result.push_back(DomainValue<T>(value));
    }
    return result;
  };
  return std::get<0>(*output) == expected(Window::Sum, true) &&
         std::get<1>(*output) == expected(Window::Min, true) &&
         std::get<2>(*output) == expected(Window::Max, true) &&
         std::get<3>(*output) == expected(Window::Sum, false) &&
         std::get<4>(*output) == expected(Window::Min, false) &&
         std::get<5>(*output) == expected(Window::Max, false);
}

template <class T>
[[nodiscard]] bool CheckCoreDomain(const rund::compute::Backend backend,
                                   DomainEvidence &evidence) {
  return CheckModes<T>(backend, evidence) && CheckClip<T>(backend, evidence) &&
         CheckPool<T>(backend, evidence) && CheckExtrema<T>(backend, evidence);
}

[[nodiscard]] bool CheckCore(const rund::compute::Backend backend,
                             DomainEvidence &evidence, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckCoreDomain<std::int32_t>(backend, evidence) &&
           CheckLargeWindow<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckCoreDomain<std::uint32_t>(backend, evidence) &&
           CheckLargeWindow<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckCoreDomain<std::int64_t>(backend, evidence) &&
           CheckLargeWindow<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckCoreDomain<std::uint64_t>(backend, evidence) &&
           CheckLargeWindow<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckCoreDomain<rund::compute::Fixed<16, 16>>(backend, evidence) &&
           CheckFixedWrapPool<rund::compute::Fixed<16, 16>>(backend, evidence);
  case Domain::Fixed20x44:
    return CheckCoreDomain<rund::compute::Fixed<20, 44>>(backend, evidence) &&
           CheckFixedWrapPool<rund::compute::Fixed<20, 44>>(backend, evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

} // namespace rund_node_collective_modes

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace {

using rund::compute::Buffer;
using rund::compute::LatestDeviceState;
using rund::compute::Pipeline;
using rund::compute::PipelineBuilder;
using rund::compute::SnapshotStorage;
using rund::compute::StateSnapshot;

static_assert(!std::copy_constructible<Pipeline>);
static_assert(std::is_nothrow_move_constructible_v<Pipeline>);
static_assert(!std::copy_constructible<PipelineBuilder>);
static_assert(std::is_nothrow_move_constructible_v<PipelineBuilder>);
static_assert(std::copy_constructible<StateSnapshot>);
static_assert(std::copy_constructible<LatestDeviceState>);
static_assert(!std::copy_constructible<SnapshotStorage>);
static_assert(std::is_nothrow_move_constructible_v<SnapshotStorage>);
static_assert(std::same_as<decltype(std::declval<Pipeline &>().begin_samples()),
                           rund::compute::Status>);
static_assert(std::same_as<decltype(std::declval<Pipeline &>().end_samples()),
                           rund::compute::Status>);
static_assert(std::same_as<decltype(std::declval<rund::compute::PipelineStats>()
                                        .samples_clean(1u)),
                           bool>);
static_assert(std::same_as<decltype(std::declval<rund::Session &>().compute(
                               std::declval<Pipeline &>())),
                           rund::compute::Request>);

template <class T>
concept ReadsTemporary =
    requires(T value) { rund::compute::read(std::move(value)); };

template <class T>
concept WritesConst = requires(const T value) { rund::compute::write(value); };

template <class T>
concept ViewsTemporary = requires(T value) { std::move(value).view(); };

template <class T>
concept WritesConstView =
    requires(const T &value) { rund::compute::write(value.view()); };

template <class T>
concept WritesMutableView =
    requires(T &value) { rund::compute::write(value.view()); };

static_assert(!ReadsTemporary<Buffer<std::int32_t>>);
static_assert(!WritesConst<Buffer<std::int32_t>>);
static_assert(!ViewsTemporary<Buffer<std::int32_t>>);
static_assert(!WritesConstView<Buffer<std::int32_t>>);
static_assert(WritesMutableView<Buffer<std::int32_t>>);

} // namespace

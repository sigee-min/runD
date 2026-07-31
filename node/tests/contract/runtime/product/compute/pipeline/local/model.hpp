#pragma once

#include <rund/compute/fixed.hpp>

#include <cstdint>
#include <span>

namespace rund {
class Session;

namespace compute {
class Device;
class Pipeline;
template <class T> class Buffer;
} // namespace compute
} // namespace rund

namespace runtime_compute_pipeline {

template <class T>
[[nodiscard]] bool ReadExact(const rund::compute::Pipeline &pipeline,
                             const rund::compute::Buffer<T> &buffer,
                             std::span<T> output);

extern template bool
ReadExact<std::int32_t>(const rund::compute::Pipeline &,
                        const rund::compute::Buffer<std::int32_t> &,
                        std::span<std::int32_t>);
extern template bool
ReadExact<std::uint32_t>(const rund::compute::Pipeline &,
                         const rund::compute::Buffer<std::uint32_t> &,
                         std::span<std::uint32_t>);
extern template bool ReadExact<rund::compute::Fixed<20u, 44u>>(
    const rund::compute::Pipeline &,
    const rund::compute::Buffer<rund::compute::Fixed<20u, 44u>> &,
    std::span<rund::compute::Fixed<20u, 44u>>);

int Lifecycle(rund::Session &session, rund::compute::Device &device);
int Claims(rund::Session &session, rund::compute::Device &device);
int Cancellation(rund::Session &session, rund::compute::Device &device);
int Semantic(rund::Session &session, rund::compute::Device &device);
int View(rund::Session &session, rund::compute::Device &device);
int Publish(rund::Session &session, rund::compute::Device &device);
int Failure(rund::Session &session, rund::compute::Device &device);
int StateCancel(rund::Session &session, rund::compute::Device &device);
int NestedWorkTotals(rund::Session &session, rund::compute::Device &device);
int Close(rund::compute::Device &device);
int Run();

} // namespace runtime_compute_pipeline

#include "common.hpp"

#include <memory>
#include <type_traits>

static_assert(!std::is_default_constructible_v<rund::compute::Device>);
static_assert(!std::is_copy_constructible_v<rund::compute::Device>);
static_assert(std::is_move_constructible_v<rund::compute::Device>);
static_assert(HasWorkers<rund::compute::Target>);
static_assert(!HasWorkers<rund::compute::Backend>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Resource{}.rounding),
                   rund::compute::Rounding>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Resource{}.overflow),
                   rund::compute::Overflow>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Resource{}.approximation),
                   rund::compute::Approximation>);
static_assert(std::is_same_v<decltype(rund::compute::graph::Resource{}.active),
                             std::uint32_t>);
static_assert(std::is_same_v<decltype(rund::compute::graph::Resource{}.parent),
                             std::uint32_t>);
static_assert(std::is_same_v<decltype(rund::compute::graph::Resource{}.source),
                             std::uint32_t>);
static_assert(std::is_same_v<decltype(rund::compute::graph::Info{}.memory),
                             rund::compute::graph::MemoryPlan>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Info{}.authored_nodes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::Info{}.lowered_nodes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.logical_bytes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.live_bytes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.physical_bytes),
                   std::uint64_t>);
static_assert(std::is_same_v<
              decltype(rund::compute::graph::MemoryPlan{}.allocation_count),
              std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.reset_bytes),
                   std::uint64_t>);
static_assert(
    std::is_same_v<decltype(rund::compute::graph::MemoryPlan{}.reset_count),
                   std::uint64_t>);
static_assert(std::is_same_v<decltype(rund::compute::Stats{}.reset_bytes),
                             std::uint64_t>);
static_assert(std::is_same_v<decltype(rund::compute::Stats{}.reset_commands),
                             std::uint64_t>);
static_assert(
    !std::is_default_constructible_v<rund::compute::Buffer<std::int32_t>>);

using ComputeDeviceResult = rund::compute::Result<rund::compute::Device>;
static_assert(
    std::is_same_v<decltype(rund::compute::open(rund::compute::Target::cpu())),
                   ComputeDeviceResult>);
static_assert(
    std::is_same_v<decltype(std::declval<ComputeDeviceResult &>().operator->()),
                   rund::compute::Device *>);
static_assert(std::is_same_v<decltype(*std::declval<ComputeDeviceResult &>()),
                             rund::compute::Device &>);
static_assert(std::same_as<
              decltype(std::declval<const ComputeDeviceResult &>().exit_code()),
              int>);

#pragma once

#include <accel/context/buffer.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include "../roundtrip.hpp"
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] ProducerConsumerRoundtrip RejectRoundtrip() noexcept;

[[nodiscard]] bool BindingSpanBytes(const rund::AccelRunBinding &binding,
                                    std::uint64_t &bytes) noexcept;

[[nodiscard]] bool SameBinding(const rund::AccelRunBinding &left,
                               const rund::AccelRunBinding &right) noexcept;

[[nodiscard]] bool BindingRoleIs(const KernelExecution &execution,
                                 std::uint64_t binding,
                                 std::uint64_t binding_count,
                                 rund::kernel::BufferRole role) noexcept;

[[nodiscard]] bool BindingVisibilityIsInternal(const KernelExecution &execution,
                                               std::uint64_t binding,
                                               std::uint64_t binding_count,
                                               bool &internal) noexcept;

[[nodiscard]] bool AccumulateLatestProducerRoundtrip(
    const KernelExecution &execution, const ScheduledStepOrder &step_order,
    const rund::AccelRun &run, std::size_t consumer_order,
    std::uint64_t read_index, const rund::AccelRunBinding &read_binding,
    std::uint64_t read_bytes, ProducerConsumerRoundtrip &result) noexcept;

} // namespace rund::node::accel::detail

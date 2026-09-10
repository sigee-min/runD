#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] bool MapSpecializedSourceUpperBytes(
    std::uint64_t source_bytes, std::uint64_t source_upper_bytes,
    const rund::kernel::ComputePlan &plan, std::uint64_t &upper) noexcept;
[[nodiscard]] bool
MapSpecializedSourceUpperBytes(const rund::kernel::LoweringArtifact &source,
                               const rund::kernel::ComputePlan &plan,
                               std::uint64_t &upper) noexcept;

enum class MetalMapWordClass : std::uint8_t { Bytewise, Word32 };

inline constexpr std::uint64_t MetalMapWordBytes = sizeof(std::uint32_t);

[[nodiscard]] constexpr MetalMapWordClass
MetalMapBindingWordClass(const std::uint64_t offset_bytes,
                         const std::uint64_t stride_bytes) noexcept {
  return offset_bytes % MetalMapWordBytes == 0u &&
                 stride_bytes % MetalMapWordBytes == 0u
             ? MetalMapWordClass::Word32
             : MetalMapWordClass::Bytewise;
}

[[nodiscard]] constexpr bool MetalMapBindingWordAligned(
    const rund::kernel::ResidentBufferRef &ref) noexcept {
  return MetalMapBindingWordClass(ref.offset_bytes, ref.stride_bytes) ==
         MetalMapWordClass::Word32;
}

[[nodiscard]] rund::kernel::LoweringArtifact
SpecializeMap(const rund::kernel::LoweringArtifact &source,
              const rund::kernel::ComputePlan &plan,
              const rund::kernel::BindingSet &bindings,
              std::uint64_t alignment = 1u, std::uint64_t reserve_upper = 0u);

[[nodiscard]] rund::kernel::LoweringArtifact
SpecializeMapInPlace(rund::kernel::LoweringArtifact &&source,
                     const rund::kernel::ComputePlan &plan,
                     const rund::kernel::BindingSet &bindings,
                     std::uint64_t alignment = 1u,
                     std::uint64_t reserve_upper = 0u);

} // namespace rund::node::accel::detail

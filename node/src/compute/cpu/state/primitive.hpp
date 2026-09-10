#pragma once

#include <kernel/program/compute/model.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <variant>

namespace rund::compute::detail {

template <class Key> struct CpuSortPrimitiveScratch final {
  std::span<Key> keys;
  std::span<kernel::u32> values;
  std::array<kernel::u64, 256u> counts{};
  std::array<kernel::u64, 256u> offsets{};
};

struct CpuScatterPrimitiveScratch final {
  std::span<kernel::u32> keys;
  std::span<kernel::u32> marks;
  // All Scatter descriptors in one serial execution envelope share these
  // physical marks and their epoch. mark_capacity covers the complete max
  // slab so epoch wrap clears stale marks outside a smaller active prefix.
  std::span<kernel::u32> mark_capacity;
  kernel::u32 *epoch{};

  [[nodiscard]] bool scatter_ready() const noexcept { return epoch != nullptr; }
  [[nodiscard]] kernel::u32 &scatter_epoch() noexcept { return *epoch; }
  [[nodiscard]] std::span<kernel::u32> scatter_marks() noexcept {
    return mark_capacity;
  }
};

struct CpuScatterReducePrimitiveScratch final {
  std::span<kernel::u32> sorted_indices;
};

template <class Lane> struct CpuTransformScratch final {
  std::span<Lane> twiddle;
};

template <class Lane> struct CpuFactorQrScratch final {
  std::array<std::span<Lane>, 3u> values;
};

template <class Lane> struct CpuSolveLuScratch final {
  std::span<Lane> factor;
  std::span<kernel::u32> pivots;
};

template <class Lane> struct CpuSolveCholeskyScratch final {
  std::span<Lane> factor;
};

template <class Lane> struct CpuSolveQrMatrixScratch final {
  std::span<Lane> y;
  std::array<std::span<Lane>, 3u> qr;
};

template <class Lane> struct CpuSolveQrFactorScratch final {
  std::span<Lane> y;
};

template <class Lane> struct CpuSpectrumEigenScratch final {
  std::array<std::span<Lane>, 3u> values;
};

template <class Lane> struct CpuSpectrumSvdValuesScratch final {
  std::array<std::span<Lane>, 3u> values;
  std::span<kernel::u64> order;
};

template <class Lane> struct CpuSpectrumSvdVectorsScratch final {
  std::array<std::span<Lane>, 4u> values;
  std::span<kernel::u64> order;
};

template <class Lane> struct CpuRangeScratch final {
  std::span<Lane> first;
  std::span<Lane> second;
};

using CpuPrimitiveScratch = std::variant<
    std::monostate, CpuSortPrimitiveScratch<kernel::u32> *,
    CpuSortPrimitiveScratch<kernel::u64> *, CpuScatterPrimitiveScratch *,
    CpuScatterReducePrimitiveScratch *, CpuTransformScratch<kernel::i32> *,
    CpuTransformScratch<kernel::i64> *, CpuFactorQrScratch<kernel::i32> *,
    CpuFactorQrScratch<kernel::i64> *, CpuSolveLuScratch<kernel::i32> *,
    CpuSolveLuScratch<kernel::i64> *, CpuSolveCholeskyScratch<kernel::i32> *,
    CpuSolveCholeskyScratch<kernel::i64> *,
    CpuSolveQrMatrixScratch<kernel::i32> *,
    CpuSolveQrMatrixScratch<kernel::i64> *,
    CpuSolveQrFactorScratch<kernel::i32> *,
    CpuSolveQrFactorScratch<kernel::i64> *, CpuSpectrumEigenScratch<kernel::i32> *,
    CpuSpectrumEigenScratch<kernel::i64> *,
    CpuSpectrumSvdValuesScratch<kernel::i32> *,
    CpuSpectrumSvdValuesScratch<kernel::i64> *,
    CpuSpectrumSvdVectorsScratch<kernel::i32> *,
    CpuSpectrumSvdVectorsScratch<kernel::i64> *, CpuRangeScratch<kernel::i32> *,
    CpuRangeScratch<kernel::u32> *, CpuRangeScratch<kernel::i64> *,
    CpuRangeScratch<kernel::u64> *>;

} // namespace rund::compute::detail

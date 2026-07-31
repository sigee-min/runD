#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

inline constexpr std::array<ExecuteFn, kCpuSimdExecutorCount> kExecutorTable{{
    ExecuteParam,
    ExecuteRead,
    ExecuteWrite,
    ExecuteAdd,
    ExecuteSub,
    ExecuteMul,
    ExecuteMulWrap,
    ExecuteMin,
    ExecuteMax,
    ExecuteClamp,
    ExecuteSelect,
    ExecuteEq,
    ExecuteLt,
    ExecuteLe,
    ExecuteConstant,
    ExecuteNeg,
    ExecuteAbs,
    ExecuteAbsMagnitude,
    ExecuteSign,
    ExecuteNe,
    ExecuteGt,
    ExecuteGe,
    ExecutePredicateNot,
    ExecutePredicateAnd,
    ExecutePredicateOr,
    ExecuteBitAnd,
    ExecuteBitOr,
    ExecuteBitXor,
    ExecuteBitNot,
    ExecuteShlConst,
    ExecuteShrLogicalConst,
    ExecuteShrArithmeticConst,
    ExecuteAddSat,
    ExecuteAddSatUnsigned,
    ExecuteSubSat,
    ExecuteNegPositiveFixed,
    ExecuteMulFixed,
    ExecuteMulFixedScaled,
    ExecuteMulUnsignedFixed,
    ExecuteMulAddFixed,
    ExecuteDivFixed,
    ExecuteRecip,
    ExecuteSqrt,
    ExecuteRsqrt,
    ExecuteSin,
    ExecuteCos,
    ExecuteTan,
    ExecuteExp,
    ExecuteLog,
    ExecuteAtan2,
    ExecuteDivSigned,
    ExecuteDivUnsigned,
    ExecuteMinUnsigned,
    ExecuteMaxUnsigned,
    ExecuteClampUnsigned,
    ExecuteLtUnsigned,
    ExecuteLeUnsigned,
    ExecuteGtUnsigned,
    ExecuteGeUnsigned,
    ExecuteIndex,
    ExecuteQuantize,
    ExecuteReadAt,
    ExecuteReadFull,
    ExecuteReadStridedFull,
    ExecuteWriteFull,
}};
static_assert(kCpuSimdBaseExecutorCount == 62u);
static_assert(kCpuSimdExecutorCount == 65u);

[[nodiscard]] ExecuteFn
ExecutorFor(const CpuSimdExecutorSlot executor_slot) noexcept {
  return kExecutorTable[static_cast<std::size_t>(executor_slot)];
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail

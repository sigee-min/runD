#pragma once

#include "../context.hpp"

#include <math64/fixed/lane.hpp>
#include <math64/fixed/sqrt/lane.hpp>
#include <math64/nonlinear/exp.hpp>
#include <math64/nonlinear/log.hpp>
#include <math64/simd/compare.hpp>
#include <math64/simd/memory.hpp>
#include <math64/simd/select.hpp>
#include <math64/turn/atan.hpp>
#include <math64/turn/trig.hpp>

#define RUND_CPU_SIMD_RUN RunFixedLane64
#define RUND_CPU_SIMD_SCRATCH_BYTES ScratchBytesFixedLane64
#define RUND_CPU_SIMD_SCALAR rund::kernel::i64
#define RUND_CPU_SIMD_BITS_SCALAR rund::kernel::u64
#define RUND_CPU_SIMD_VEC rund::math64::simd::I64x
#define RUND_CPU_SIMD_BITS_VEC rund::math64::simd::U64x
#define RUND_CPU_SIMD_MASK rund::math64::simd::Mask64x
#define RUND_CPU_SIMD_BITS 64u
#define RUND_CPU_SIMD_LANES rund::math64::simd::LaneCount
#define RUND_CPU_SIMD_SPLAT(value) rund::math64::simd::SplatI64(value)
#define RUND_CPU_SIMD_SPLAT_BITS(value) rund::math64::simd::SplatU64(value)
#define RUND_CPU_SIMD_LOAD(ptr) rund::math64::simd::LoadI64(ptr)
#define RUND_CPU_SIMD_STORE(ptr, value) rund::math64::simd::Store(ptr, value)
#define RUND_CPU_SIMD_SELECT(mask, lhs, rhs)                                   \
  rund::math64::simd::Select(mask, lhs, rhs)
#define RUND_CPU_SIMD_EQ(lhs, rhs) rund::math64::simd::Eq(lhs, rhs)
#define RUND_CPU_SIMD_NE(lhs, rhs) rund::math64::simd::Ne(lhs, rhs)
#define RUND_CPU_SIMD_LT(lhs, rhs) rund::math64::simd::Lt(lhs, rhs)
#define RUND_CPU_SIMD_LE(lhs, rhs) rund::math64::simd::Le(lhs, rhs)
#define RUND_CPU_SIMD_GT(lhs, rhs) rund::math64::simd::Gt(lhs, rhs)
#define RUND_CPU_SIMD_GE(lhs, rhs) rund::math64::simd::Ge(lhs, rhs)
#define RUND_CPU_SIMD_LT_UNSIGNED(lhs, rhs)                                    \
  rund::math64::simd::Lt(std::bit_cast<rund::math64::simd::U64x>(lhs),         \
                         std::bit_cast<rund::math64::simd::U64x>(rhs))
#define RUND_CPU_SIMD_LE_UNSIGNED(lhs, rhs)                                    \
  rund::math64::simd::Le(std::bit_cast<rund::math64::simd::U64x>(lhs),         \
                         std::bit_cast<rund::math64::simd::U64x>(rhs))
#define RUND_CPU_SIMD_GT_UNSIGNED(lhs, rhs)                                    \
  rund::math64::simd::Gt(std::bit_cast<rund::math64::simd::U64x>(lhs),         \
                         std::bit_cast<rund::math64::simd::U64x>(rhs))
#define RUND_CPU_SIMD_GE_UNSIGNED(lhs, rhs)                                    \
  rund::math64::simd::Ge(std::bit_cast<rund::math64::simd::U64x>(lhs),         \
                         std::bit_cast<rund::math64::simd::U64x>(rhs))
#define RUND_CPU_SIMD_ADD_WRAP(lhs, rhs) rund::math64::AddWrap(lhs, rhs)
#define RUND_CPU_SIMD_SUB_WRAP(lhs, rhs) rund::math64::SubWrap(lhs, rhs)
#define RUND_CPU_SIMD_MUL_LOW(lhs, rhs) rund::math64::MulLow(lhs, rhs)
#define RUND_CPU_SIMD_VALUE_SELECT(mask, lhs, rhs)                             \
  rund::math64::Select(mask, lhs, rhs)
#define RUND_CPU_SIMD_ABS_MAGNITUDE(value) rund::math64::AbsMagnitude(value)
#define RUND_CPU_SIMD_ABS(value) rund::math64::Abs(value)
#define RUND_CPU_SIMD_SIGN(value) rund::math64::Sign(value)
#define RUND_CPU_SIMD_ADD_SAT_UNSIGNED(lhs, rhs)                               \
  rund::math64::AddSatUnsigned(lhs, rhs)
#define RUND_CPU_SIMD_ADD_SAT(lhs, rhs) rund::math64::AddSat(lhs, rhs)
#define RUND_CPU_SIMD_SUB_SAT(lhs, rhs) rund::math64::SubSat(lhs, rhs)
#define RUND_CPU_SIMD_NEG_POSITIVE_FIXED(value)                                \
  rund::math64::NegPositiveFixed(value)
#define RUND_CPU_SIMD_MUL_UNSIGNED_FIXED(lhs, rhs)                             \
  rund::math64::MulUnsignedFixed(lhs, rhs)
#define RUND_CPU_SIMD_MUL_FIXED_SCALED(lhs, rhs)                               \
  rund::math64::MulFixedScaled(lhs, rhs)
#define RUND_CPU_SIMD_MUL_ADD_FIXED(lhs, rhs, add)                             \
  rund::math64::MulAddFixed(lhs, rhs, add)
#define RUND_CPU_SIMD_MUL_FIXED(lhs, rhs) rund::math64::MulFixed(lhs, rhs)
#define RUND_CPU_SIMD_DIV_FIXED(lhs, rhs) rund::math64::DivFixed(lhs, rhs)
#define RUND_CPU_SIMD_RECIP(value) rund::math64::Recip(value)
#define RUND_CPU_SIMD_RSQRT(value) rund::math64::Rsqrt(value)
#define RUND_CPU_SIMD_SQRT(value) rund::math64::Sqrt(value)
#define RUND_CPU_SIMD_SIN(value)                                               \
  rund::math64::TurnSin(std::bit_cast<rund::math64::simd::U64x>(value))
#define RUND_CPU_SIMD_COS(value)                                               \
  rund::math64::TurnCos(std::bit_cast<rund::math64::simd::U64x>(value))
#define RUND_CPU_SIMD_TAN(value)                                               \
  rund::math64::DivFixed(RUND_CPU_SIMD_SIN(value), RUND_CPU_SIMD_COS(value))
#define RUND_CPU_SIMD_EXP(value) rund::math64::Exp2(value)
#define RUND_CPU_SIMD_LOG(value) rund::math64::Log2(value)
#define RUND_CPU_SIMD_ATAN2(lhs, rhs)                                          \
  std::bit_cast<rund::math64::simd::I64x>(rund::math64::TurnAtan2(lhs, rhs))
#define RUND_CPU_SIMD_CONSTANT(node)                                           \
  RUND_CPU_SIMD_SPLAT(std::bit_cast<rund::kernel::i64>(                        \
      static_cast<rund::kernel::u64>((node).lhs) |                             \
      (static_cast<rund::kernel::u64>((node).rhs) << 32u)))

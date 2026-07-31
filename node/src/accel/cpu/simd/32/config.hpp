#pragma once

#include "../context.hpp"

#include <math32/fixed/lane.hpp>
#include <math32/fixed/sqrt/lane.hpp>
#include <math32/nonlinear/exp.hpp>
#include <math32/nonlinear/log.hpp>
#include <math32/simd/compare.hpp>
#include <math32/simd/memory.hpp>
#include <math32/simd/select.hpp>
#include <math32/turn/atan.hpp>
#include <math32/turn/trig.hpp>
#define RUND_CPU_SIMD_RUN RunFixedLane32
#define RUND_CPU_SIMD_SCRATCH_BYTES ScratchBytesFixedLane32
#define RUND_CPU_SIMD_SCALAR rund::kernel::i32
#define RUND_CPU_SIMD_BITS_SCALAR rund::kernel::u32
#define RUND_CPU_SIMD_VEC rund::math32::simd::I32x
#define RUND_CPU_SIMD_BITS_VEC rund::math32::simd::U32x
#define RUND_CPU_SIMD_MASK rund::math32::simd::Mask32x
#define RUND_CPU_SIMD_BITS 32u
#define RUND_CPU_SIMD_LANES rund::math32::simd::LaneCount
#define RUND_CPU_SIMD_SPLAT(value) rund::math32::simd::SplatI32(value)
#define RUND_CPU_SIMD_SPLAT_BITS(value) rund::math32::simd::SplatU32(value)
#define RUND_CPU_SIMD_LOAD(ptr) rund::math32::simd::LoadI32(ptr)
#define RUND_CPU_SIMD_STORE(ptr, value) rund::math32::simd::Store(ptr, value)
#define RUND_CPU_SIMD_SELECT(mask, lhs, rhs)                                   \
  rund::math32::simd::Select(mask, lhs, rhs)
#define RUND_CPU_SIMD_EQ(lhs, rhs) rund::math32::simd::Eq(lhs, rhs)
#define RUND_CPU_SIMD_NE(lhs, rhs) rund::math32::simd::Ne(lhs, rhs)
#define RUND_CPU_SIMD_LT(lhs, rhs) rund::math32::simd::Lt(lhs, rhs)
#define RUND_CPU_SIMD_LE(lhs, rhs) rund::math32::simd::Le(lhs, rhs)
#define RUND_CPU_SIMD_GT(lhs, rhs) rund::math32::simd::Gt(lhs, rhs)
#define RUND_CPU_SIMD_GE(lhs, rhs) rund::math32::simd::Ge(lhs, rhs)
#define RUND_CPU_SIMD_LT_UNSIGNED(lhs, rhs)                                    \
  rund::math32::simd::Lt(std::bit_cast<rund::math32::simd::U32x>(lhs),         \
                         std::bit_cast<rund::math32::simd::U32x>(rhs))
#define RUND_CPU_SIMD_LE_UNSIGNED(lhs, rhs)                                    \
  rund::math32::simd::Le(std::bit_cast<rund::math32::simd::U32x>(lhs),         \
                         std::bit_cast<rund::math32::simd::U32x>(rhs))
#define RUND_CPU_SIMD_GT_UNSIGNED(lhs, rhs)                                    \
  rund::math32::simd::Gt(std::bit_cast<rund::math32::simd::U32x>(lhs),         \
                         std::bit_cast<rund::math32::simd::U32x>(rhs))
#define RUND_CPU_SIMD_GE_UNSIGNED(lhs, rhs)                                    \
  rund::math32::simd::Ge(std::bit_cast<rund::math32::simd::U32x>(lhs),         \
                         std::bit_cast<rund::math32::simd::U32x>(rhs))
#define RUND_CPU_SIMD_ADD_WRAP(lhs, rhs) rund::math32::AddWrap(lhs, rhs)
#define RUND_CPU_SIMD_SUB_WRAP(lhs, rhs) rund::math32::SubWrap(lhs, rhs)
#define RUND_CPU_SIMD_MUL_LOW(lhs, rhs) rund::math32::MulLow(lhs, rhs)
#define RUND_CPU_SIMD_VALUE_SELECT(mask, lhs, rhs)                             \
  rund::math32::Select(mask, lhs, rhs)
#define RUND_CPU_SIMD_ABS_MAGNITUDE(value) rund::math32::AbsMagnitude(value)
#define RUND_CPU_SIMD_ABS(value) rund::math32::Abs(value)
#define RUND_CPU_SIMD_SIGN(value) rund::math32::Sign(value)
#define RUND_CPU_SIMD_ADD_SAT_UNSIGNED(lhs, rhs)                               \
  rund::math32::AddSatUnsigned(lhs, rhs)
#define RUND_CPU_SIMD_ADD_SAT(lhs, rhs) rund::math32::AddSat(lhs, rhs)
#define RUND_CPU_SIMD_SUB_SAT(lhs, rhs) rund::math32::SubSat(lhs, rhs)
#define RUND_CPU_SIMD_NEG_POSITIVE_FIXED(value)                                \
  rund::math32::NegPositiveFixed(value)
#define RUND_CPU_SIMD_MUL_UNSIGNED_FIXED(lhs, rhs)                             \
  rund::math32::MulUnsignedFixed(lhs, rhs)
#define RUND_CPU_SIMD_MUL_FIXED_SCALED(lhs, rhs)                               \
  rund::math32::MulFixedScaled(lhs, rhs)
#define RUND_CPU_SIMD_MUL_ADD_FIXED(lhs, rhs, add)                             \
  rund::math32::MulAddFixed(lhs, rhs, add)
#define RUND_CPU_SIMD_MUL_FIXED(lhs, rhs) rund::math32::MulFixed(lhs, rhs)
#define RUND_CPU_SIMD_DIV_FIXED(lhs, rhs) rund::math32::DivFixed(lhs, rhs)
#define RUND_CPU_SIMD_RECIP(value) rund::math32::Recip(value)
#define RUND_CPU_SIMD_RSQRT(value) rund::math32::Rsqrt(value)
#define RUND_CPU_SIMD_SQRT(value) rund::math32::Sqrt(value)
#define RUND_CPU_SIMD_SIN(value)                                               \
  rund::math32::TurnSin(std::bit_cast<rund::math32::simd::U32x>(value))
#define RUND_CPU_SIMD_COS(value)                                               \
  rund::math32::TurnCos(std::bit_cast<rund::math32::simd::U32x>(value))
#define RUND_CPU_SIMD_TAN(value)                                               \
  rund::math32::DivFixed(RUND_CPU_SIMD_SIN(value), RUND_CPU_SIMD_COS(value))
#define RUND_CPU_SIMD_EXP(value) rund::math32::Exp2(value)
#define RUND_CPU_SIMD_LOG(value) rund::math32::Log2(value)
#define RUND_CPU_SIMD_ATAN2(lhs, rhs)                                          \
  std::bit_cast<rund::math32::simd::I32x>(rund::math32::TurnAtan2(lhs, rhs))
#define RUND_CPU_SIMD_CONSTANT(node)                                           \
  RUND_CPU_SIMD_SPLAT(std::bit_cast<rund::kernel::i32>(                        \
      static_cast<rund::kernel::u32>((node).lhs)))

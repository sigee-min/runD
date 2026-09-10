#pragma once

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/virtual.hpp>

#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

template <class T>
concept HasRemainder = requires(T value) { value % value; };

template <class T>
concept HasWorkers = requires(T value) {
  { value.workers() } -> std::same_as<std::uint32_t>;
};

template <class T>
concept AcceptsRvalueFlowInput = requires(T value) {
  rund::compute::on(rund::compute::Target::cpu(), std::move(value));
};

template <class Range>
concept AcceptsFlowRange = requires(Range &value) {
  rund::compute::on(rund::compute::Target::cpu(), value);
};

template <class... Args>
concept MakesComputeFailure = requires(Args &&...args) {
  rund::compute::Status::fail(std::forward<Args>(args)...);
};

template <class Range>
concept AcceptsConstFlowRange = requires(const Range &value) {
  rund::compute::on(rund::compute::Target::cpu(), value);
};

template <class T>
concept AcceptsBufferValue = requires(const rund::compute::Device &device) {
  device.template buffer<T>(1u);
};

template <class T>
concept CastsToFloat = requires(T value) { static_cast<float>(value); };

template <class T>
concept CastsToDouble = requires(T value) { static_cast<double>(value); };

template <unsigned IntegerBits, unsigned FractionBits>
concept DefinesFixed =
    requires { typename rund::compute::Fixed<IntegerBits, FractionBits>; };

template <class T>
concept AddsFloatExpression =
    requires(rund::compute::Expr<T> value) { value + 1.0f; };

template <class Left, class Right>
concept AddsExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      left + right;
    };

template <class Left, class Right>
concept SubtractsExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      left - right;
    };

template <class Left, class Right>
concept MultipliesExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      left * right;
    };

template <class Left, class Right>
concept DividesExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      left / right;
    };

template <class Left, class Right>
concept MinimizesExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      rund::compute::min(left, right);
    };

template <class Left, class Right>
concept MaximizesExpressions =
    requires(rund::compute::Expr<Left> left, rund::compute::Expr<Right> right) {
      rund::compute::max(left, right);
    };

template <class T>
concept HasFixedDivide =
    requires(const T &value) { rund::compute::div_fixed(value, value); };

template <class T>
concept HasUnsignedStorageSaturate =
    requires(const T &value) { rund::compute::add_sat_unsigned(value, value); };

template <class T>
concept HasArithmeticShift =
    requires(const T &value) { rund::compute::shr_arithmetic<1u>(value); };

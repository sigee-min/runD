#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/definition.hpp>
#include <kernel/program/compute/dsl/geometry.hpp>
#include <kernel/program/compute/dsl/matrix.hpp>
#include <kernel/program/compute/dsl/metric.hpp>

// Composite DSL headers consume the primitive expression surface above them.
// clang-format off
#include <kernel/program/compute/dsl/functions/core.hpp>
#include <kernel/program/compute/dsl/functions/linear.hpp>
#include <kernel/program/compute/dsl/functions/metric.hpp>
#include <kernel/program/compute/dsl/functions/tolerance.hpp>
#include <kernel/program/compute/dsl/functions/piece.hpp>
#include <kernel/program/compute/dsl/functions/predicate.hpp>
#include <kernel/program/compute/dsl/functions/mask.hpp>
#include <kernel/program/compute/dsl/functions/geometry.hpp>
#include <kernel/program/compute/dsl/functions/range.hpp>
#include <kernel/program/compute/dsl/functions/stats.hpp>
#include <kernel/program/compute/dsl/functions/moment.hpp>
#include <kernel/program/compute/dsl/functions/aggregate.hpp>
#include <kernel/program/compute/dsl/functions/corr.hpp>
#include <kernel/program/compute/dsl/functions/ratio.hpp>
#include <kernel/program/compute/dsl/functions/standardize.hpp>
#include <kernel/program/compute/dsl/functions/approx.hpp>
#include <kernel/program/compute/dsl/functions/transcendental.hpp>
#include <kernel/program/compute/dsl/functions/complex.hpp>
#include <kernel/program/compute/dsl/functions/matrix.hpp>
#include <kernel/program/compute/dsl/functions/affine/axis.hpp>
#include <kernel/program/compute/dsl/functions/mix.hpp>
#include <kernel/program/compute/dsl/functions/poly.hpp>
#include <kernel/program/compute/dsl/functions/bit.hpp>
#include <kernel/program/compute/dsl/functions/hash.hpp>
#include <kernel/program/compute/dsl/functions/interp/base.hpp>
#include <kernel/program/compute/dsl/functions/noise.hpp>
#include <kernel/program/compute/dsl/functions/interp.hpp>
#include <kernel/program/compute/dsl/functions/reflect/axis.hpp>
// clang-format on

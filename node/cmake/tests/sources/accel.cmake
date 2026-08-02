set(NODE_TEST_ACCEL_SURFACE_TEST_SOURCES
  tests/contract/accel/surface.cpp
)

set(NODE_TEST_ACCEL_CPU_SIMD_VECTOR_TEST_SOURCES
  tests/contract/accel/cpu/vector/contract.cpp
  tests/contract/accel/cpu/vector.cpp
)

set(NODE_TEST_ACCEL_CPU_SIMD_DSL_BASIC_TEST_SOURCES
  tests/contract/accel/cpu/dsl/basic.cpp
  tests/contract/accel/cpu/hash.cpp
  tests/contract/accel/cpu/noise.cpp
  tests/contract/accel/cpu/noise/grid.cpp
  tests/contract/accel/cpu/norm.cpp
  tests/contract/accel/cpu/range.cpp
  tests/contract/accel/cpu/mask.cpp
  tests/contract/accel/cpu/tol.cpp
  tests/contract/accel/cpu/piece.cpp
  tests/contract/accel/cpu/poly.cpp
)

set(NODE_TEST_ACCEL_CPU_SIMD_DSL_GEOMETRY_TEST_SOURCES
  tests/contract/accel/cpu/dsl/geometry.cpp
  tests/contract/accel/cpu/vec.cpp
  tests/contract/accel/cpu/sq.cpp
  tests/contract/accel/cpu/metric.cpp
  tests/contract/accel/cpu/proj.cpp
  tests/contract/accel/cpu/cross.cpp
)

set(NODE_TEST_ACCEL_CPU_SIMD_DSL_STATS_TEST_SOURCES
  tests/contract/accel/cpu/dsl/stats.cpp
  tests/contract/accel/cpu/stats.cpp
  tests/contract/accel/cpu/moment.cpp
  tests/contract/accel/cpu/corr.cpp
)

set(NODE_TEST_ACCEL_CPU_SIMD_DSL_MATRIX_TEST_SOURCES
  tests/contract/accel/cpu/dsl/matrix.cpp
  tests/contract/accel/cpu/linear.cpp
  tests/contract/accel/cpu/matrix.cpp
  tests/contract/accel/cpu/affine.cpp
  tests/contract/accel/cpu/mix.cpp
  tests/contract/accel/cpu/interp.cpp
)

set(NODE_TEST_ACCEL_CPU_SIMD_OPS_TEST_SOURCES
  tests/contract/accel/cpu/ops.cpp
  tests/contract/accel/cpu/ops/32/composite.cpp
  tests/contract/accel/cpu/ops/32/scalar.cpp
  tests/contract/accel/cpu/ops/32/bit.cpp
  tests/contract/accel/cpu/ops/32/arithmetic.cpp
  tests/contract/accel/cpu/ops/32/nonlinear.cpp
  tests/contract/accel/cpu/ops/64/contract.cpp
  tests/contract/accel/cpu/order.cpp
)

set(NODE_TEST_ACCEL_CPU_SIMD_BACKEND_TEST_SOURCES
  tests/contract/accel/cpu/backend.cpp
  tests/contract/accel/cpu/forged.cpp
)

set(NODE_TEST_ACCEL_KERNEL_CORE_TEST_SOURCES
  tests/contract/accel/kernel/core.cpp
  tests/contract/accel/kernel/authority.cpp
  tests/contract/accel/kernel/binding/indices.cpp
  tests/contract/accel/kernel/collective.cpp
  tests/contract/accel/kernel/collective/backend.cpp
  tests/contract/accel/kernel/collective/graph.cpp
  tests/contract/accel/kernel/collective/required.cpp
  tests/contract/accel/kernel/collective/surface.cpp
  tests/contract/accel/kernel/segmented/reduce/model.cpp
  tests/contract/accel/kernel/compile.cpp
  tests/contract/accel/kernel/compile/binding.cpp
  tests/contract/accel/kernel/compile/fixture.cpp
  tests/contract/accel/kernel/compile/graph.cpp
  tests/contract/accel/kernel/compile/identity.cpp
  tests/contract/accel/kernel/compile/reject.cpp
  tests/contract/accel/kernel/compile/support.cpp
  tests/contract/accel/kernel/compact/run.cpp
  tests/contract/accel/kernel/compact/match.cpp
  tests/contract/accel/kernel/compact/reject.cpp
  tests/contract/accel/kernel/cpu/map/parity.cpp
  tests/contract/accel/kernel/gather.cpp
  tests/contract/accel/kernel/histogram.cpp
  tests/contract/accel/kernel/histogram/match.cpp
  tests/contract/accel/kernel/model.cpp
  tests/contract/accel/kernel/metal/template_memory.cpp
  tests/contract/accel/kernel/partition.cpp
  tests/contract/accel/kernel/partition/compile.cpp
  tests/contract/accel/kernel/partition/fixture.cpp
  tests/contract/accel/kernel/partition/match.cpp
  tests/contract/accel/kernel/reduce.cpp
  tests/contract/accel/kernel/recurrence.cpp
  tests/contract/accel/kernel/reset.cpp
  tests/contract/accel/kernel/run.cpp
  tests/contract/accel/kernel/run/evidence.cpp
  tests/contract/accel/kernel/run/fixture.cpp
  tests/contract/accel/kernel/run/reject.cpp
  tests/contract/accel/kernel/run/window.cpp
  tests/contract/accel/kernel/scatter.cpp
  tests/contract/accel/kernel/stencil.cpp
)

if(RUND_NODE_HAVE_METAL_SDK)
  set_source_files_properties(
    tests/contract/accel/kernel/metal/template_memory.cpp
    PROPERTIES
      LANGUAGE OBJCXX
      COMPILE_DEFINITIONS RUND_NODE_HAVE_METAL_SDK=1
      COMPILE_OPTIONS "-fobjc-arc")
endif()

set(NODE_TEST_ACCEL_KERNEL_NUMERIC_TEST_SOURCES
  tests/contract/accel/kernel/numeric.cpp
  tests/contract/accel/kernel/numeric/metal.cpp
  tests/contract/accel/kernel/numeric/topology.cpp
  tests/contract/accel/kernel/factor.cpp
  tests/contract/accel/kernel/matrix.cpp
  tests/contract/accel/kernel/solve.cpp
  tests/contract/accel/kernel/solve/raw.cpp
  tests/contract/accel/kernel/solve/reuse.cpp
  tests/contract/accel/kernel/spectrum.cpp
  tests/contract/accel/kernel/transform.cpp
)

set(NODE_TEST_ACCEL_CPU_KERNEL_TEST_SOURCES
  tests/contract/accel/kernel/cpu.cpp
  tests/contract/accel/kernel/cpu/map/run.cpp
  tests/contract/accel/kernel/cpu/segmented.cpp
)

set(NODE_TEST_ACCEL_BACKEND_RUNTIME_TEST_SOURCES
  tests/contract/accel/backend/runtime.cpp
  tests/contract/accel/buffer.cpp
  tests/contract/accel/buffer/backend.cpp
  tests/contract/accel/buffer/desc.cpp
  tests/contract/accel/buffer/metal.cpp
  tests/contract/accel/metal/resident.cpp
  tests/contract/accel/resident.cpp
)

set(NODE_TEST_ACCEL_BACKEND_WINDOW_TEST_SOURCES
  tests/contract/accel/backend/window.cpp
  tests/contract/accel/backend/window/caps.cpp
  tests/contract/accel/backend/window/obligation.cpp
  tests/contract/accel/backend/window/simple.cpp
  tests/contract/accel/backend/window/staging.cpp
  tests/contract/accel/runtime/window/direct.cpp
  tests/contract/accel/runtime/window/dispatch.cpp
  tests/contract/accel/runtime/window/pick.cpp
)

set(NODE_TEST_ACCEL_BACKEND_PICK_TEST_SOURCES
  tests/contract/accel/backend/pick.cpp
  tests/contract/accel/vulkan/pick.cpp
)

set(NODE_TEST_ACCEL_BACKEND_CORE_TEST_SOURCES
  tests/contract/accel/backend/core.cpp
  tests/contract/accel/backend/run/core.cpp
  tests/contract/accel/context.cpp
  tests/contract/accel/context/invalid.cpp
  tests/contract/accel/context/open.cpp
  tests/contract/accel/context/reject.cpp
  tests/contract/accel/context/state.cpp
  tests/contract/accel/context/transfer.cpp
  tests/contract/accel/runtime/policy.cpp
)

set(NODE_TEST_ACCEL_KERNEL_FUSION_TEST_SOURCES
  tests/contract/accel/kernel/fusion.cpp
  tests/contract/accel/kernel/fusion/success/extra.cpp
  tests/contract/accel/kernel/fusion/read/two.cpp
  tests/contract/accel/kernel/fusion/visibility.cpp
)

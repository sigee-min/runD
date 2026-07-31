#pragma once

#include <rund/compute/abi/model.hpp>
#include <rund/compute/target.hpp>
#include <span>
#include <string_view>
namespace rund::compute::detail {
[[nodiscard]] std::shared_ptr<FlowState>
make_flow(Target target, Type type, std::size_t count,
          FixedFormat fixed_format = {});
[[nodiscard]] std::shared_ptr<FlowState>
make_flow_on(std::shared_ptr<DeviceState> device, Type type, std::size_t count,
             std::shared_ptr<ProgramCacheState> cache = {},
             FixedFormat fixed_format = {});
void flow_bind(const std::shared_ptr<FlowState> &flow, HostView input);
void flow_map(const std::shared_ptr<FlowState> &flow, std::string_view name,
              ExprRef expression);
[[nodiscard]] std::uint32_t
flow_map_value(const std::shared_ptr<FlowState> &flow,
               std::span<const std::uint32_t> inputs, std::string_view name,
               ExprRef expression);
[[nodiscard]] ValueIds flow_map_multi(const std::shared_ptr<FlowState> &flow,
                                      std::span<const std::uint32_t> inputs,
                                      std::string_view name,
                                      std::span<const ExprRef> expressions);
[[nodiscard]] std::uint32_t
flow_map_value_controlled(const std::shared_ptr<FlowState> &flow,
                          std::span<const std::uint32_t> inputs,
                          std::string_view name, ExprRef expression,
                          FlowControl control);
[[nodiscard]] ValueIds flow_map_multi_controlled(
    const std::shared_ptr<FlowState> &flow,
    std::span<const std::uint32_t> inputs, std::string_view name,
    std::span<const ExprRef> expressions, FlowControl control);
[[nodiscard]] std::size_t
flow_step_count(const std::shared_ptr<FlowState> &flow) noexcept;
void flow_tag_iteration(const std::shared_ptr<FlowState> &flow,
                        std::size_t first_step,
                        std::uint32_t iteration) noexcept;
[[nodiscard]] std::uint32_t flow_zero(const std::shared_ptr<FlowState> &flow,
                                      std::size_t count);
[[nodiscard]] std::uint32_t flow_index(const std::shared_ptr<FlowState> &flow,
                                       Type type, std::size_t count);
[[nodiscard]] std::uint32_t flow_retype(const std::shared_ptr<FlowState> &flow,
                                        std::uint32_t input, Type output);
[[nodiscard]] std::uint32_t
flow_retype_like(const std::shared_ptr<FlowState> &flow, std::uint32_t input,
                 std::uint32_t format_source);
[[nodiscard]] std::uint32_t
flow_fixed_select_value(const std::shared_ptr<FlowState> &flow,
                        std::uint32_t value, std::uint32_t selected,
                        std::uint64_t otherwise_bits, bool invert,
                        std::string_view name);
[[nodiscard]] std::uint32_t
flow_fixed_merge_values(const std::shared_ptr<FlowState> &flow,
                        std::uint32_t left, std::uint32_t right,
                        Window operation, std::string_view name);
void flow_reject(const std::shared_ptr<FlowState> &flow, Reason reason);
[[nodiscard]] std::uint32_t
flow_binary_values(const std::shared_ptr<FlowState> &flow, Primitive operation,
                   std::span<const std::uint32_t> inputs, Type output,
                   std::size_t count, PrimitiveOptions options);
void flow_scan(const std::shared_ptr<FlowState> &flow, Scan scan);
[[nodiscard]] std::uint32_t
flow_scan_value(const std::shared_ptr<FlowState> &flow, std::uint32_t input,
                Scan scan);
void flow_bounded_scan(const std::shared_ptr<FlowState> &flow,
                       std::uint32_t count, Scan scan);
[[nodiscard]] std::uint32_t
flow_bounded_scan_value(const std::shared_ptr<FlowState> &flow,
                        std::uint32_t input, std::uint32_t count, Scan scan);
void flow_bounded_reduce(const std::shared_ptr<FlowState> &flow,
                         std::uint32_t count, Reduce operation);
[[nodiscard]] std::uint32_t
flow_bounded_reduce_value(const std::shared_ptr<FlowState> &flow,
                          std::uint32_t input, std::uint32_t count,
                          Reduce operation);
void flow_bounded_sort(const std::shared_ptr<FlowState> &flow,
                       std::uint32_t count, bool indices);
[[nodiscard]] std::uint32_t
flow_bounded_sort_value(const std::shared_ptr<FlowState> &flow,
                        std::uint32_t input, std::uint32_t count, bool indices);
[[nodiscard]] BoundedIds flow_filter(const std::shared_ptr<FlowState> &flow,
                                     ExprRef selected, ExprRef rejected);
[[nodiscard]] BoundedIds
flow_filter_value(const std::shared_ptr<FlowState> &flow, std::uint32_t input,
                  ExprRef selected, ExprRef rejected);
[[nodiscard]] BoundedIds
flow_filter_masks(const std::shared_ptr<FlowState> &flow, std::uint32_t input,
                  std::uint32_t selected, std::uint32_t rejected);
[[nodiscard]] BoundedIds
flow_compact_value(const std::shared_ptr<FlowState> &flow, std::uint32_t input,
                   std::size_t capacity);
void flow_unary(const std::shared_ptr<FlowState> &flow, Primitive operation,
                Type output, std::size_t count, PrimitiveOptions options);
[[nodiscard]] std::uint32_t
flow_unary_value(const std::shared_ptr<FlowState> &flow, std::uint32_t input,
                 Primitive operation, Type output, std::size_t count,
                 PrimitiveOptions options);
[[nodiscard]] std::uint32_t
flow_independent_input(const std::shared_ptr<FlowState> &flow, HostView input,
                       FixedFormat fixed_format);
void flow_mark_bounded_input(const std::shared_ptr<FlowState> &flow,
                             std::uint32_t count,
                             std::size_t capacity) noexcept;
[[nodiscard]] std::uint32_t flow_side(const std::shared_ptr<FlowState> &flow,
                                      HostView input, bool bind,
                                      FixedFormat fixed_format);
void flow_binary(const std::shared_ptr<FlowState> &flow, Primitive operation,
                 std::uint32_t side, bool side_first, Type output,
                 std::size_t count, PrimitiveOptions options);
[[nodiscard]] std::uint32_t
flow_complex_side(const std::shared_ptr<FlowState> &flow, HostView input,
                  bool bind, FixedFormat fixed_format);
[[nodiscard]] ComplexIds flow_transform(const std::shared_ptr<FlowState> &flow,
                                        std::uint32_t real, std::uint32_t imag,
                                        PrimitiveOptions options);
void flow_pick(const std::shared_ptr<FlowState> &flow, std::uint32_t value);
void flow_outputs(const std::shared_ptr<FlowState> &flow,
                  std::span<const std::uint32_t> values);
void flow_matrix_view(const std::shared_ptr<FlowState> &flow, std::size_t rows,
                      std::size_t cols, std::size_t batches);
[[nodiscard]] std::size_t
flow_matrix_extent(const std::shared_ptr<FlowState> &flow, std::size_t rows,
                   std::size_t cols, std::size_t batches);
[[nodiscard]] std::size_t flow_matrix_product(
    const std::shared_ptr<FlowState> &flow, std::size_t left_rows,
    std::size_t left_cols, std::size_t left_batches, std::size_t right_rows,
    std::size_t right_cols, std::size_t right_batches, std::size_t right_count);
[[nodiscard]] FactorIds flow_factor(const std::shared_ptr<FlowState> &flow,
                                    FactorOp operation, std::size_t rows,
                                    std::size_t cols, std::size_t batches);
[[nodiscard]] SolveIds
flow_factor_solve(const std::shared_ptr<FlowState> &flow, FactorOp operation,
                  std::uint32_t packed, std::uint32_t pivots, std::uint32_t rhs,
                  std::size_t rows, std::size_t rhs_cols, std::size_t batches);
[[nodiscard]] SolveIds
flow_matrix_solve(const std::shared_ptr<FlowState> &flow, FactorOp operation,
                  std::uint32_t matrix, std::uint32_t rhs, std::size_t rows,
                  std::size_t cols, std::size_t rhs_cols, std::size_t batches);
[[nodiscard]] SpectrumIds
flow_spectrum(const std::shared_ptr<FlowState> &flow, SpectrumOp operation,
              SpectrumVectors vectors, std::size_t rows, std::size_t cols,
              std::size_t batches, std::uint32_t iterations);
[[nodiscard]] std::size_t
flow_count(const std::shared_ptr<FlowState> &flow) noexcept;
[[nodiscard]] std::size_t
flow_output_count(const std::shared_ptr<FlowState> &flow) noexcept;
[[nodiscard]] std::size_t
flow_value_count(const std::shared_ptr<FlowState> &flow,
                 std::uint32_t value) noexcept;
[[nodiscard]] FixedFormat
flow_value_format(const std::shared_ptr<FlowState> &flow,
                  std::uint32_t value) noexcept;
[[nodiscard]] std::uint32_t
flow_value(const std::shared_ptr<FlowState> &flow) noexcept;
[[nodiscard]] std::span<const HostView>
flow_bindings(const std::shared_ptr<FlowState> &flow) noexcept;
[[nodiscard]] Result<std::shared_ptr<ProgramState>>
compile_flow(const std::shared_ptr<FlowState> &flow);
} // namespace rund::compute::detail

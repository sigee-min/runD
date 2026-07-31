#include <kernel/program/compute/dsl/operation.hpp>

#include <utility>

namespace rund::compute_dsl {
namespace {

void AppendParamStorage(std::vector<rund::kernel::u8> &storage,
                        const std::vector<detail::BindingRuntime> &bindings) {
  storage.clear();
  for (const detail::BindingRuntime &binding : bindings) {
    if (binding.kind == detail::BindingKind::Param) {
      storage.insert(storage.end(), binding.value_bytes.begin(),
                     binding.value_bytes.end());
    }
  }
}

void AppendInputSpans(std::vector<rund::kernel::BufferSpan> &spans,
                      const std::vector<detail::BindingRuntime> &bindings) {
  spans.clear();
  for (const detail::BindingRuntime &binding : bindings) {
    if (binding.kind == detail::BindingKind::Read) {
      spans.push_back(rund::kernel::BufferSpan{
          .data = binding.runtime_data,
          .element_bytes = binding.element_bytes,
          .stride_bytes = binding.runtime_stride_bytes,
          .count = binding.runtime_count,
      });
    }
  }
}

void AppendInputShapeBytes(
    std::vector<rund::kernel::u64> &widths,
    const std::vector<detail::BindingRuntime> &bindings) {
  widths.clear();
  for (const detail::BindingRuntime &binding : bindings) {
    if (binding.kind == detail::BindingKind::Read) {
      widths.push_back(binding.element_bytes);
    }
  }
}

void AppendOutputSpans(std::vector<rund::kernel::OutputSpan> &spans,
                       const std::vector<detail::BindingRuntime> &bindings) {
  spans.clear();
  for (const detail::BindingRuntime &binding : bindings) {
    if (binding.kind == detail::BindingKind::Write) {
      spans.push_back(rund::kernel::OutputSpan{
          .data = binding.runtime_write_data,
          .element_bytes = binding.element_bytes,
          .stride_bytes = binding.runtime_stride_bytes,
          .count = binding.runtime_count,
      });
    }
  }
}

void AppendOutputShapeBytes(
    std::vector<rund::kernel::u64> &widths,
    const std::vector<detail::BindingRuntime> &bindings) {
  widths.clear();
  for (const detail::BindingRuntime &binding : bindings) {
    if (binding.kind == detail::BindingKind::Write) {
      widths.push_back(binding.element_bytes);
    }
  }
}

} // namespace

ComputeOp::ComputeOp(rund::kernel::ComputeIR ir,
                     const rund::kernel::ComputeMap map,
                     std::vector<detail::BindingRuntime> bindings,
                     const rund::kernel::u64 tile_count)
    : ir_(std::move(ir)), map_(map), bindings_(std::move(bindings)),
      tile_count_(tile_count) {
  AppendParamStorage(param_storage_, bindings_);
  AppendInputSpans(input_spans_, bindings_);
  AppendInputShapeBytes(input_element_bytes_, bindings_);
  AppendOutputSpans(output_spans_, bindings_);
  AppendOutputShapeBytes(output_element_bytes_, bindings_);
}

rund::kernel::ComputeMap ComputeOp::map() const noexcept { return map_; }

const rund::kernel::ComputeIR &ComputeOp::ir() const noexcept { return ir_; }

bool ComputeOp::ok() const noexcept { return ir_.ok; }

const char *ComputeOp::reason() const noexcept { return ir_.reason; }

rund::kernel::InputShapeBytes ComputeOp::input_element_bytes() const noexcept {
  return rund::kernel::InputShapeBytes{
      .data =
          input_element_bytes_.empty() ? nullptr : input_element_bytes_.data(),
      .count = static_cast<rund::kernel::u64>(input_element_bytes_.size()),
  };
}

detail::ComputeOpMetadataView ComputeOp::metadata_view() const noexcept {
  return detail::ComputeOpMetadataView{
      .param_data = param_storage_.empty() ? nullptr : param_storage_.data(),
      .param_bytes = static_cast<rund::kernel::u64>(param_storage_.size()),
      .input_element_bytes =
          input_element_bytes_.empty() ? nullptr : input_element_bytes_.data(),
      .input_count =
          static_cast<rund::kernel::u64>(input_element_bytes_.size()),
      .output_element_bytes = output_element_bytes_.empty()
                                  ? nullptr
                                  : output_element_bytes_.data(),
      .output_count =
          static_cast<rund::kernel::u64>(output_element_bytes_.size()),
  };
}

rund::kernel::BindingSet
ComputeOp::staged(const rund::kernel::u64 phase_id,
                  const rund::kernel::u64 lane_count,
                  const rund::kernel::ComputeApi api) const {
  if (!ok()) {
    return rund::kernel::BindingSet{.reason = reason()};
  }
  if (tile_count_ == 0u || map_.output_bytes_per_tile == 0u) {
    return rund::kernel::BindingSet{.reason = "compute_binding_invalid"};
  }

  rund::kernel::u32 write_count = 0u;
  for (const detail::BindingRuntime &binding : bindings_) {
    if ((binding.kind == detail::BindingKind::Read ||
         binding.kind == detail::BindingKind::Write) &&
        !binding.has_runtime_storage) {
      return rund::kernel::BindingSet{
          .tile_count = tile_count_,
          .reason = "compute_binding_runtime_missing",
      };
    }
    if (binding.kind == detail::BindingKind::Write) {
      ++write_count;
    }
  }
  if (write_count == 0u || write_count != output_spans_.size()) {
    return rund::kernel::BindingSet{
        .tile_count = tile_count_,
        .reason = "compute_write_missing",
    };
  }

  return rund::kernel::BindingSet{
      .phase_id = phase_id,
      .tile_count = tile_count_,
      .lane_count = lane_count,
      .op_hash_hi = map_.op_hash_hi,
      .op_hash_lo = map_.op_hash_lo,
      .api = api,
      .scalar = map_.scalar,
      .domain = map_.domain,
      .input_bytes_per_tile = map_.input_bytes_per_tile,
      .output_bytes_per_tile = map_.output_bytes_per_tile,
      .param_bytes = map_.param_bytes,
      .metadata_bytes_per_tile = map_.metadata_bytes_per_tile,
      .input_buffers = input_spans_.empty() ? nullptr : input_spans_.data(),
      .input_buffer_count = static_cast<rund::kernel::u64>(input_spans_.size()),
      .output_buffers = output_spans_.data(),
      .output_buffer_count =
          static_cast<rund::kernel::u64>(output_spans_.size()),
      .input_element_bytes =
          input_element_bytes_.empty() ? nullptr : input_element_bytes_.data(),
      .input_element_byte_count =
          static_cast<rund::kernel::u64>(input_element_bytes_.size()),
      .output_element_bytes = output_element_bytes_.data(),
      .output_element_byte_count =
          static_cast<rund::kernel::u64>(output_element_bytes_.size()),
      .param_data = param_storage_.empty() ? nullptr : param_storage_.data(),
      .param_data_bytes = static_cast<rund::kernel::u64>(param_storage_.size()),
      .staged_output =
          output_spans_.size() == 1u ? output_spans_.front().data : nullptr,
      .staged_output_stride =
          output_spans_.size() == 1u ? output_spans_.front().stride_bytes : 0u,
      .staged_output_count =
          output_spans_.size() == 1u ? output_spans_.front().count : 0u,
      .ok = true,
      .reason = "ok",
  };
}

rund::kernel::BindingSet
ComputeOp::resident(const rund::kernel::u64 phase_id,
                    const rund::kernel::u64 lane_count,
                    const rund::kernel::ComputeApi api,
                    const rund::kernel::ResidentBufferRef *const inputs,
                    const rund::kernel::u64 input_count,
                    const rund::kernel::ResidentBufferRef &output,
                    const std::shared_ptr<void> *const input_handles,
                    const rund::kernel::u64 input_handle_count,
                    const std::shared_ptr<void> *const output_handle) const {
  if (!ok()) {
    return rund::kernel::BindingSet{.reason = reason()};
  }
  if (tile_count_ == 0u || map_.output_bytes_per_tile == 0u) {
    return rund::kernel::BindingSet{.reason = "compute_binding_invalid"};
  }

  rund::kernel::u64 read_count = 0u;
  rund::kernel::u64 write_count = 0u;
  for (const detail::BindingRuntime &binding : bindings_) {
    if (binding.kind == detail::BindingKind::Read) {
      ++read_count;
    } else if (binding.kind == detail::BindingKind::Write) {
      ++write_count;
    }
  }
  if (write_count != 1u) {
    return rund::kernel::BindingSet{
        .tile_count = tile_count_,
        .reason = write_count == 0u ? "compute_write_missing"
                                    : "compute_multi_write_unsupported",
    };
  }
  if (input_count != read_count || (input_count != 0u && inputs == nullptr)) {
    return rund::kernel::BindingSet{
        .tile_count = tile_count_,
        .reason = "compute_binding_input_count_mismatch",
    };
  }
  const bool handle_shape_invalid =
      (input_handles == nullptr) != (input_handle_count == 0u) ||
      (input_handles != nullptr && input_handle_count != input_count);
  if (handle_shape_invalid) {
    return rund::kernel::BindingSet{
        .tile_count = tile_count_,
        .reason = "compute_binding_input_count_mismatch",
    };
  }

  return rund::kernel::BindingSet{
      .phase_id = phase_id,
      .tile_count = tile_count_,
      .lane_count = lane_count,
      .op_hash_hi = map_.op_hash_hi,
      .op_hash_lo = map_.op_hash_lo,
      .api = api,
      .scalar = map_.scalar,
      .domain = map_.domain,
      .input_bytes_per_tile = map_.input_bytes_per_tile,
      .output_bytes_per_tile = map_.output_bytes_per_tile,
      .param_bytes = map_.param_bytes,
      .metadata_bytes_per_tile = map_.metadata_bytes_per_tile,
      .input_buffers = nullptr,
      .input_buffer_count = 0u,
      .input_element_bytes =
          input_element_bytes_.empty() ? nullptr : input_element_bytes_.data(),
      .input_element_byte_count =
          static_cast<rund::kernel::u64>(input_element_bytes_.size()),
      .output_element_bytes = output_element_bytes_.data(),
      .output_element_byte_count =
          static_cast<rund::kernel::u64>(output_element_bytes_.size()),
      .param_data = param_storage_.empty() ? nullptr : param_storage_.data(),
      .param_data_bytes = static_cast<rund::kernel::u64>(param_storage_.size()),
      .staged_output = nullptr,
      .staged_output_stride = 0u,
      .staged_output_count = 0u,
      .ok = true,
      .reason = "ok",
      .resident_inputs =
          rund::kernel::ResidentBindingRange{
              .refs = inputs,
              .handles = input_handles,
              .storage_count = input_count,
              .count = input_count,
          },
      .resident_outputs =
          rund::kernel::ResidentBindingRange{
              .refs = &output,
              .handles = output_handle,
              .storage_count = 1u,
              .count = 1u,
          },
  };
}

} // namespace rund::compute_dsl

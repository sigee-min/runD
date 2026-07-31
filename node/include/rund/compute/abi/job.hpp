#pragma once

#include <limits>
#include <new>
#include <rund/compute/abi/model.hpp>
#include <span>
namespace rund::compute::detail {
[[nodiscard]] Result<std::shared_ptr<JobState>>
make_job_raw(const std::shared_ptr<ProgramState> &program,
             std::span<const HostView> inputs);
template <class... T>
[[nodiscard]] Result<std::shared_ptr<JobState>>
make_job(const std::shared_ptr<ProgramState> &program,
         const std::span<const T>... inputs) {
  const std::array<HostView, sizeof...(T)> views{
      HostView{inputs.data(), inputs.size(), type<T>()}...};
  return make_job_raw(program, views);
}
[[nodiscard]] Status run_job(const std::shared_ptr<JobState> &state);
[[nodiscard]] Status write_job_raw(const std::shared_ptr<JobState> &state,
                                   std::span<const HostView> inputs);
template <class... T>
[[nodiscard]] Status write_job(const std::shared_ptr<JobState> &state,
                               const std::span<const T>... inputs) {
  const std::array<HostView, sizeof...(T)> views{
      HostView{inputs.data(), inputs.size(), type<T>()}...};
  return write_job_raw(state, views);
}
[[nodiscard]] Result<std::size_t>
job_read_size(const std::shared_ptr<JobState> &state, std::size_t output,
              Type type);
[[nodiscard]] Status job_read_data(const std::shared_ptr<JobState> &state,
                                   std::size_t output, Type type, void *data,
                                   std::size_t bytes, std::size_t logical_count,
                                   bool destination_zeroed);
template <class T>
[[nodiscard]] Result<std::vector<T>>
read_job_prefix(const std::shared_ptr<JobState> &state,
                const std::size_t output, const std::size_t count) {
  auto size = job_read_size(state, output, type<T>());
  if (!size) {
    return Result<std::vector<T>>::fail(size.reason());
  }
  const std::size_t logical =
      count == std::numeric_limits<std::size_t>::max() ? *size : count;
  if (logical > *size) {
    return Result<std::vector<T>>::fail(Reason::BoundedCountInvalid);
  }
  try {
    std::vector<T> values(logical);
    const Status read = job_read_data(state, output, type<T>(), values.data(),
                                      values.size() * sizeof(T), logical, true);
    return read ? Result<std::vector<T>>::success(std::move(values))
                : Result<std::vector<T>>::fail(read.reason());
  } catch (const std::bad_alloc &) {
    return Result<std::vector<T>>::fail(Reason::BufferCapacity);
  }
}
template <class T>
[[nodiscard]] Result<std::vector<T>>
read_job(const std::shared_ptr<JobState> &state,
         const std::size_t output = 0u) {
  return read_job_prefix<T>(state, output,
                            std::numeric_limits<std::size_t>::max());
}
template <class Schema> struct SchemaReader final {
  [[nodiscard]] static Result<HostValueT<Schema>>
  read(const std::shared_ptr<JobState> &state, std::size_t &output) {
    auto value = read_job<Schema>(state, output);
    ++output;
    return value;
  }
};
template <class T> struct SchemaReader<Scalar<T>> final {
  [[nodiscard]] static Result<T> read(const std::shared_ptr<JobState> &state,
                                      std::size_t &output) {
    auto value = read_job<T>(state, output);
    ++output;
    if (!value) {
      return Result<T>::fail(value.reason());
    }
    if (value->size() != 1u) {
      return Result<T>::fail(Reason::ScalarCountInvalid);
    }
    return Result<T>::success(std::move(value->front()));
  }
};
template <class T, class Count> struct SchemaReader<Bounded<T, Count>> final {
  [[nodiscard]] static Result<std::vector<T>>
  read(const std::shared_ptr<JobState> &state, std::size_t &output) {
    const std::size_t value_output = output++;
    auto count = read_job<Count>(state, output++);
    if (!count) {
      return Result<std::vector<T>>::fail(count.reason());
    }
    if (count->size() != 1u) {
      return Result<std::vector<T>>::fail(Reason::BoundedCountInvalid);
    }
    const std::size_t logical = static_cast<std::size_t>(count->front());
    auto values = read_job_prefix<T>(state, value_output, logical);
    if (!values) {
      return Result<std::vector<T>>::fail(values.reason());
    }
    if (logical > values->size()) {
      return Result<std::vector<T>>::fail(Reason::BoundedCountInvalid);
    }
    values->resize(logical);
    return values;
  }
};
template <class Tag, class T> struct SchemaReader<Field<Tag, T>> final {
  [[nodiscard]] static Result<HostValueT<T>>
  read(const std::shared_ptr<JobState> &state, std::size_t &output) {
    return SchemaReader<T>::read(state, output);
  }
};
template <class... Schema, std::size_t... I>
[[nodiscard]] Result<std::tuple<HostValueT<Schema>...>>
read_schema_tuple(const std::shared_ptr<JobState> &state, std::size_t &output,
                  std::index_sequence<I...>) {
  std::tuple<HostValueT<Schema>...> values{};
  Status status = Status::success();
  auto read = [&]<std::size_t Index, class Value>() {
    if (!status) {
      return;
    }
    auto value = SchemaReader<Value>::read(state, output);
    if (!value) {
      status = Status::fail(value.reason());
      return;
    }
    std::get<Index>(values) = std::move(value).value();
  };
  (read.template operator()<I, Schema>(), ...);
  if (!status) {
    return Result<std::tuple<HostValueT<Schema>...>>::fail(status.reason());
  }
  return Result<std::tuple<HostValueT<Schema>...>>::success(std::move(values));
}
template <class... Schema> struct SchemaReader<Record<Schema...>> final {
  [[nodiscard]] static Result<std::tuple<HostValueT<Schema>...>>
  read(const std::shared_ptr<JobState> &state, std::size_t &output) {
    return read_schema_tuple<Schema...>(state, output,
                                        std::index_sequence_for<Schema...>{});
  }
};
} // namespace rund::compute::detail

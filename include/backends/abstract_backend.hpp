#pragma once

#include <extension/format.hpp>

namespace einops {
namespace backends {

template <class Tensor>
class AbstractBackend
{
public:
	using Shape = std::vector<int64_t>;

	virtual ~AbstractBackend() = default;

	virtual inline bool is_float_type(Tensor const& x) const = 0;

	virtual inline Shape shape(Tensor const& x) = 0;
	virtual inline Tensor reshape(Tensor const& x, Shape const& shape) = 0;

	virtual inline Tensor add_axis(Tensor const& x, int64_t new_position) = 0;
	virtual inline Tensor add_axes(Tensor const& x, int64_t n_axes, std::map<int64_t, int64_t> const& pos2len) = 0;
	virtual inline Tensor stack_on_zeroth_dimension(std::vector<Tensor> const& list) = 0;

	virtual inline Tensor arange(int64_t start, int64_t stop) = 0;
	virtual inline Tensor reduce(Tensor const& x, std::string const& operation, std::vector<int64_t> const& reduced_axes) = 0;
	virtual inline Tensor transpose(Tensor const& x, std::vector<int64_t> const& axes) = 0;
	virtual inline Tensor tile(Tensor const& x, std::vector<int64_t> const& repeats) = 0;
	virtual inline Tensor concat(std::vector<Tensor> const& tensors, int64_t axis) = 0;
	virtual inline Tensor einsum(std::string const& pattern, std::vector<Tensor> const& tensors) = 0;
};

template <typename T>
inline auto sort_and_reverse(std::vector<T> const& in) -> std::vector<T>
{
	auto out = std::vector<T>(std::begin(in), std::end(in));
	std::sort(std::begin(out), std::end(out));
	std::reverse(std::begin(out), std::end(out));
	return out;
}

} // namespace backends
} // namespace einops
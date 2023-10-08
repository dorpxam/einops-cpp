#pragma once

#ifdef EINOPS_TENSORFLOW_BACKEND

#include <backends/abstract_backend.hpp>
#include <tensorflow/core/framework/tensor.h>

namespace einops {
namespace backends {

class TensorflowBackend : public AbstractBackend<tensorflow::Tensor>
{
public:
	using Tensor = tensorflow::Tensor;

	inline bool is_float_type(Tensor const& x) const final
	{
		return (x.dtype() == torch::kFloat16 ||
				x.dtype() == torch::kFloat32 || 
				x.dtype() == torch::kFloat64 || 
				x.dtype() == torch::kBFloat16) ? true : false;
	}

	inline Shape shape(Tensor const& x) final
	{
		return x.sizes().vec();
	}

	inline Tensor reshape(Tensor const& x, Shape const& shape) final
	{
		return x.reshape(shape);
	}

	inline Tensor add_axis(Tensor const& x, int64_t new_position) final
	{
		return torch::unsqueeze(x, new_position);
	}

	inline Tensor add_axes(Tensor const& x, int64_t n_axes, std::map<int64_t, int64_t> const& pos2len) final
	{
		auto y = x;
		std::vector<int64_t> repeats (n_axes, -1);
		for (auto&& [axis_position, axis_length] : pos2len)
		{
			y = add_axis(y, axis_position);
			repeats[axis_position] = axis_length;
		}
		return y.expand(repeats);
	}

	inline Tensor stack_on_zeroth_dimension(at::TensorList const& tensors)
	{
		return torch::stack(tensors);
	}

	inline Tensor stack_on_zeroth_dimension(std::vector<Tensor> const& tensors) final
	{
		return torch::stack(tensors);
	}

	inline Tensor arange(int64_t start, int64_t stop) final
	{
		return torch::arange(start, stop, c10::TensorOptions().dtype(torch::kInt64));
	}

	inline Tensor reduce(Tensor const& x, std::string const& operation, std::vector<int64_t> const& reduced_axes) final
	{
		if (operation == "min")  
			return x.amin(reduced_axes); 
		else
		if (operation == "max")  
			return x.amax(reduced_axes); 
		else
		if (operation == "sum")
			return x.sum(reduced_axes);
		else
		if (operation == "mean")
			return x.mean(reduced_axes);
		else
		if (operation == "prod" ||
			operation == "any" ||
			operation == "all")
		{
			auto y = x;
			for (auto dim : sort_and_reverse(reduced_axes))
			{
				if (operation == "prod")
					y = y.prod(dim);
				else
				if (operation == "any")
					y = y.any(dim);
				else
				if (operation == "all")
					y = y.all(dim);
			}
			return y;
		}
		else
			throw std::runtime_error(std::format("TorchBackend::reduce : Unknown reduction {}", operation).c_str());
	}

	inline Tensor transpose(Tensor const& x, std::vector<int64_t> const& axes) final
	{
		return x.permute(axes);
	}

	inline Tensor tile(Tensor const& x, std::vector<int64_t> const& repeats) final
	{
		return x.repeat(repeats);
	}

	inline Tensor concat(std::vector<Tensor> const& tensors, int64_t axis) final
	{
		return torch::cat(tensors, axis);
	}

	inline Tensor einsum(std::string const& pattern, std::vector<Tensor> const& tensors) final
	{
		return torch::einsum(pattern, tensors);
	}
};

template <typename Tensor>
auto get_backend(Tensor const& tensor) -> std::tuple<TensorflowBackend, TensorflowBackend::Tensor>
{
	auto backend = TensorflowBackend();
	if constexpr (std::is_same_v<Tensor, at::TensorList> 
			   || std::is_same_v<Tensor, std::vector<tensorflow::Tensor>>)
		return { backend, backend.stack_on_zeroth_dimension(tensor) };
	else
	if constexpr (std::is_same_v<Tensor, tensorflow::Tensor>)
		return { backend, tensor };
	else
		throw std::runtime_error("Tensorflow backend only support tensorflow::Tensor, tensorflow::TensorList or a std::vector<tensorflow::Tensor>.");
}

/*
inline std::string dump(torch::IntArrayRef const& vector)
{
	std::string result = "(";
	for (auto value : vector)
		result += std::to_string(value) + ", ";
	result = result.substr(0, result.size() - 2);
	return result + ")";
}

inline bool array_equal(torch::Tensor const& lhs, torch::Tensor const& rhs)
{
	return dump(lhs.sizes()) == dump(rhs.sizes());
}
*/

} // namespace backends
} // namespace einops

#endif // EINOPS_TENSORFLOW_BACKEND
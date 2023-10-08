#pragma once

#ifdef EINOPS_TORCH_BACKEND

#include <einops.hpp>
#include <torch/torch.h>

namespace einops {
namespace backends {

template <typename T>
inline auto sort_reverse(std::vector<T> const& in) -> std::vector<T>
{
	auto out = std::vector<T>(std::begin(in), std::end(in));
					std::sort(std::begin(out), std::end(out));
				 std::reverse(std::begin(out), std::end(out));
	return out;
}
	
class TorchJitBackend
{
public:
	static torch::Tensor reduce(torch::Tensor const& x, std::string const& operation, Axes const& reduced_axes)
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
		if (operation == "prod")
		{
			auto y = x;
			for (auto dim : sort_reverse(reduced_axes))
				y = y.prod(dim);
			return y;
		}
		else
			throw std::runtime_error(std::format("Unknown reduction {}", operation).c_str());
	}

	static torch::Tensor transpose(torch::Tensor const& x, Axes const& axes)
	{
		return x.permute(axes);
	}

	static torch::Tensor stack_on_zeroth_dimension(torch::TensorList const& tensors)
	{
		return torch::stack(tensors);
	}

	static torch::Tensor tile(torch::Tensor const& x, Axes const& repeats)
	{
		return x.repeat(repeats);
	}

	static torch::Tensor add_axes(torch::Tensor const& x, const int64_t n_axes, AxesMap const& pos2len)
	{
		auto y = x;
		std::vector<int64_t> repeats (n_axes, -1);
		for (auto&& [axis_position, axis_length] : pos2len)
		{
			y = torch::unsqueeze(y, axis_position);
			repeats[axis_position] = axis_length;
		}
		return y.expand(repeats);
	}
	
	static bool is_float_type(torch::Tensor const& x)
	{
		return (x.dtype() == torch::kFloat16 ||
				x.dtype() == torch::kFloat32 || 
				x.dtype() == torch::kFloat64 || 
				x.dtype() == torch::kBFloat16) ? true : false;
	}

	static Shape shape(torch::Tensor const& x)
	{
		return x.sizes().vec();;
	}

	static torch::Tensor reshape(torch::Tensor const& x, Axes const& shape)
	{
		return x.reshape(shape);
	}
};

inline void apply_for_scriptable_torch(TransformRecipe const& recipe, 
									   torch::Tensor& tensor,
									   std::string const& reduction_type,
									   AxesLengths const& axes_dims)
{
	auto&& backend = TorchJitBackend();
	auto&& [init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added] 
		= _reconstruct_from_shape_uncached(recipe, backend.shape(tensor), axes_dims);
	if (init_shapes.has_value())
		tensor = backend.reshape(tensor, init_shapes.value());
	if (axes_reordering.has_value())
		tensor = backend.transpose(tensor, axes_reordering.value());
	if (reduced_axes.size() > 0)
		tensor = backend.reduce(tensor, reduction_type, reduced_axes);
	if (added_axes.size() > 0)
		tensor = backend.add_axes(tensor, n_axes_w_added, added_axes);
	if (final_shapes.has_value())
		tensor = backend.reshape(tensor, final_shapes.value());
}

} // namespace backends
} // namespace einops

#endif // EINOPS_TENSORFLOW_BACKEND
#pragma once

#include <einops.hpp>

namespace einops {
namespace implementation {

inline auto analyze_pattern(std::string const& pattern, std::string const& opname) -> std::tuple<int, int, int>
{
	auto axes = splits(pattern, " ");
	auto axes_set = std::set(axes.begin(), axes.end());
	if (axes.size() != axes_set.size())
		throw Exception(::format("Duplicates in axes names in {}(..., \"{}\")", opname, pattern));
	if (axes_set.count(_asterisk) == 0)
		throw Exception(::format("No *-axis in {}(..., \"{}\")", opname, pattern));
	for (auto&& axis : axes)
	{
		if (axis != _asterisk)
		{
			auto&& [is_valid, reason] = ParsedExpression::check_axis_name_return_reason(axis);
			if (!is_valid)
				throw Exception(format("Invalid axis name {} in {}(..., \"{}\")", axis, opname, pattern));
		}
	}
	auto n_axes_before = index(axes, _asterisk);
	auto n_axes_after = axes.size() - n_axes_before - 1;
	auto min_axes = n_axes_before + n_axes_after;
	return { n_axes_before, n_axes_after, min_axes };
}

} // namespace implementation

using namespace backends::implementation;

/// @brief Packs several tensors into one.
///		See einops tutorial for introduction into packing (and how it replaces stack and concatenation).
/// @param tensors tensors to be packed, can be of different dimensionality
/// @param pattern pattern that is shared for all inputs and output, e.g. "i j * k" or "batch seq *"
///		where * designates an axis to be unpacked
/// @return tuple with { packed_tensor, packed_shapes }.
template <typename Tensor>
auto pack(std::vector<Tensor> const& tensors, std::string const& pattern) -> std::tuple<Tensor, std::vector<std::vector<int64_t>>>
{
	auto&& [n_axes_before, n_axes_after, min_axes] = implementation::analyze_pattern(pattern, "pack");

	auto backend = get_packing_backend(tensors.front());

	std::vector<Tensor> reshaped_tensors;
	std::vector<std::vector<int64_t>> packed_shapes;

	for (auto&& [i, tensor] : iters::enumerate(tensors))
	{
		auto shape = backend.shape(tensor);
		if (shape.size() < min_axes)
			throw Exception(format("packed tensor #{} (enumeration starts with 0) has shape {}, " \
								   "while pattern {} assumes at least {} axes", i, implementation::print(shape), pattern, implementation::print(min_axes)));

		auto axis_after_packed_axes = shape.size() - n_axes_after;
		auto packed_shape = subvec(shape, n_axes_before, axis_after_packed_axes);
		packed_shapes.push_back(packed_shape);
		std::vector<int64_t> reshape_axes;
		{
			for (auto&& axis : iters::range(n_axes_before))
				reshape_axes.push_back(shape[axis]);
			reshape_axes.push_back(-1);
			for (auto&& axis : iters::range(axis_after_packed_axes, shape.size()))
				reshape_axes.push_back(shape[axis]);
		}
		reshaped_tensors.push_back(backend.reshape(tensor, reshape_axes));
	}
	return std::make_tuple(backend.concat(reshaped_tensors, n_axes_before), packed_shapes);
}

/// @brief Unpacks a single tensor into several by splitting over a selected axes.
///		See einops tutorial for introduction into packing (and how it replaces stack and concatenation).
/// @param tensor tensor to be unpacked.
/// @param packed_shapes packed_shapes (aka PS) is a list of shapes that take place of '*' in each output.
///		output will contain a single tensor for every provided shape.
/// @param pattern pattern that is shared for input and all outputs, e.g. "i j * k" or "batch seq *",
///		where * designates an axis to be unpacked
/// @return list of tensors.
template <typename Tensor>
auto unpack(Tensor const& tensor, std::vector<std::vector<int64_t>> const& packed_shapes, std::string const& pattern) -> std::vector<Tensor>
{
	auto&& [n_axes_before, n_axes_after, min_axes] = implementation::analyze_pattern(pattern, "unpack");

	auto backend = get_packing_backend(tensor);
	auto input_shape = backend.shape(tensor);
	if (input_shape.size() != n_axes_before + 1 + n_axes_after)
		throw Exception(format("unpack(..., {}) received input of wrong dim with shape {}", pattern, implementation::print(input_shape)));

	auto unpacked_axis = n_axes_before;

	auto lengths_of_composed_axes = processor<int>(packed_shapes, [=](auto&& x) { return contains<int64_t>(x, -1) ? -1 : prod(x); });

	auto n_unknown_composed_axes = sum<int>(processor<bool>(lengths_of_composed_axes, [=](auto&& x) { return bool(x == -1); }));

	if (n_unknown_composed_axes > 1)
		throw Exception(::format("unpack(..., {}) received more than one -1 in {packed_shapes} and can't infer dimensions", pattern));

	auto split_positions = std::vector<int64_t>(packed_shapes.size(), 0);
		 split_positions.push_back(input_shape[unpacked_axis]);

	if (n_unknown_composed_axes == 0)
	{
		for (auto&& [i, x] : iters::enumerate(lengths_of_composed_axes))
			if (i < lengths_of_composed_axes.size()-1)
				split_positions[i + 1] = split_positions[i] + x;
	}
	else
	{
		auto unknown_composed_axis = index(lengths_of_composed_axes, -1);

		for (auto&& i : iters::range<int64_t>(unknown_composed_axis))
			split_positions[i + 1] = split_positions[i] + lengths_of_composed_axes[i];

		for (auto&& j : reverse(iters::range<int64_t>(unknown_composed_axis + 1, lengths_of_composed_axes.size()).vec()))
			split_positions[j] = split_positions[j + 1] - lengths_of_composed_axes[j];
	}

	auto shape_start = subvec(input_shape, 0, unpacked_axis);
	auto shape_end   = subvec(input_shape, unpacked_axis + 1, input_shape.size());

	std::vector<TensorIndex> slice_filler;
	for (auto&& _ : iters::range(unpacked_axis))
		slice_filler.push_back(Slice(None, None));

	try
	{
		std::vector<Tensor> output;
		for (auto&& [i, element_shape] : iters::enumerate(packed_shapes))
		{
			auto slices = concat(slice_filler, Slice(split_positions[i], split_positions[i + 1]));
			auto shapes = concat(shape_start, element_shape, shape_end);
			output.push_back(backend.reshape(tensor.index(refarray(slices)), shapes));
		}
		return output;
	}
	catch (...)
	{
		throw Exception(format("Error during unpack(..., \"{}\"): could not split axis of size {}" \
							   " into requested {}", pattern, split_positions.back(), implementation::print(packed_shapes)));
	}
}

} // namespace einops
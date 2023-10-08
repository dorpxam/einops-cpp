#pragma once

#include <einops.hpp>

namespace einops::implementation {

template <typename Tensor>
class RearrangeMixin
{
public:
	template <typename... Args>
	RearrangeMixin(Pattern const& pattern, Args... axes_lengths)
		: _pattern(pattern)
	{
		_multirecipe = multirecipe();
		_axes_lengths = to_vector(std::tuple<Args...>(axes_lengths...));
	}

	MultiRecipe multirecipe() const
	{
		try
		{
			return ::_prepare_recipes_for_all_dims(_pattern, "rearrange", _axes_lengths);
		}
		catch (Exception const& e)
		{
			throw Exception(format(" Error while preparing {}\n {}", to_string(), e.what()));
		}
	}

	virtual Tensor _apply_recipe(Tensor const& x)
	{
		return ::_apply_recipe(get_backend(x), _multirecipe[x.sizes()], x, "rearrange", _axes_lengths);
	}

	std::string to_string() const
	{
		std::string params = _pattern;
		for (auto&& [axis, length] : _axes_lengths)
			params += format(", {}={}", print(axis), length);
		return format("{}({})", "RearrangeMixin", params);
	}

protected:
	Pattern _pattern;
	MultiRecipe _multirecipe;
	AxesLengths _axes_lengths;
};

template <typename Tensor>
class ReduceMixin
{
public:
	template <typename... Args>
	ReduceMixin(Pattern const& pattern, Reduction const& reduction, Args... axes_lengths)
		: _pattern(pattern)
		, _reduction(reduction)
	{
		_multirecipe = multirecipe();
		_axes_lengths = to_vector(std::tuple<Args...>(axes_lengths...));
	}

	MultiRecipe multirecipe() const
	{
		try
		{
			return ::_prepare_recipes_for_all_dims(_pattern, _reduction, _axes_lengths);
		}
		catch (Exception const& e)
		{
			throw Exception(format(" Error while preparing {}\n {}", to_string(), e.what()));
		}
	}

	virtual Tensor _apply_recipe(Tensor const& x)
	{
		return ::_apply_recipe(get_backend(x), _multirecipe[x.ndimension()], x, _reduction, _axes_lengths);
	}

	std::string to_string() const
	{
		std::string params = format("{}, {}", _pattern, _reduction);
		for (auto&& [axis, length] : _axes_lengths)
			params += format(", {}={}", print(axis), length);
		return format("{}({})", "ReduceMixin", params);
	}

protected:
	Pattern _pattern;
	Reduction _reduction;
	MultiRecipe _multirecipe;
	AxesLengths _axes_lengths;
};

} // namespace einops::implementation
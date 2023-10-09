#pragma once

#ifdef EINOPS_TORCH_BACKEND

#include <layers/common.hpp>
#include <layers/einmix.hpp>
using namespace einops;
using namespace einops::backends;
using namespace einops::implementation;

#include <torchjit.hpp> // _torch_specific in python

class RearrangeImpl : public RearrangeMixin<torch::Tensor>
					, public torch::nn::Module
{
	using Base = RearrangeMixin<torch::Tensor>;

public:
	template <typename... Args>
	RearrangeImpl(Pattern const& pattern, Args... axes_lengths)
		: Base(pattern, axes_lengths...)
	{}

	virtual ~RearrangeImpl() throw() {}

	torch::Tensor forward(torch::Tensor input)
	{
		auto&& recipe = Base::_multirecipe[input.ndimension()];
		apply_for_scriptable_torch(recipe, input, "rearrange", Base::_axes_lengths);
		return input;
	}

	torch::Tensor _apply_recipe(torch::Tensor const& x) final {}
};

TORCH_MODULE(Rearrange);

class ReduceImpl : public ReduceMixin<torch::Tensor>
				 , public torch::nn::Module
{
	using Base = ReduceMixin<torch::Tensor>;

public:
	template <typename... Args>
	ReduceImpl(Pattern const& pattern, Reduction const& reduction, Args... axes_lengths)
		: Base(pattern, reduction, axes_lengths...)
	{}

	virtual ~ReduceImpl() throw() {}

	torch::Tensor forward(torch::Tensor input)
	{
		auto&& recipe = Base::_multirecipe[input.ndimension()];
		apply_for_scriptable_torch(recipe, input, Base::_reduction, Base::_axes_lengths);
		return input;
	}

	torch::Tensor _apply_recipe(torch::Tensor const& x) final {}
};

TORCH_MODULE(Reduce);

class EinMixImpl : public _EinmixMixin
				 , public torch::nn::Module
{
public:
	template <typename... Args>
	EinMixImpl(Pattern const& pattern, std::string const& weight_shape, std::string const& bias_shape, Args... axes_lengths)
		: _EinmixMixin(pattern, weight_shape, bias_shape, axes_lengths...)
	{}

	virtual ~EinMixImpl() throw() {}

	torch::Tensor forward(torch::Tensor x)
	{}
};

TORCH_MODULE(EinMix);

#endif // EINOPS_TENSORFLOW_BACKEND
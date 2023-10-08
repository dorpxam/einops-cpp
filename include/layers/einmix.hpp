#pragma once

#include <einops.hpp>

namespace einops::implementation {

class _EinmixMixin
{
public:
	template <typename... Args>
	_EinmixMixin(Pattern const& pattern, std::string const& weight_shape, std::string const& bias_shape, Args... axes_lengths)
	{}

	virtual void _create_rearrange_layers(std::optional<Pattern> const& pre_reshape_pattern,
										  std::optional<AxesMap> const& pre_reshape_lengths,
										  std::optional<Pattern> const& post_reshape_pattern,
										  std::optional<AxesMap> const& post_reshape_lengths) = 0;

	virtual void _create_parameters() = 0;

	std::string to_string() const
	{
		return "";
	}
};

} // namespace einops::implementation
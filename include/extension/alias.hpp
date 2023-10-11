#pragma once

#include <extension/anonymous.hpp>
#include <extension/hash.hpp>
#include <extension/tools.hpp>

namespace einops::implementation {

using Length = int64_t;
using Lengths = std::vector<int64_t>;
using Position = int64_t;
using Shape = std::vector<int64_t>;
using Shapes = std::vector<Shape>;

using Pattern = std::string;
using Reduction = std::string;
using Reductions = std::vector<std::string>;

using Identifier = std::variant<std::string, AnonymousAxis>;
using Identifiers = std::set<Identifier>;
using IdentifiersMap = std::map<Identifier, int64_t>;

using Composition = std::vector<std::variant<std::vector<std::string>, std::string>>;
using FlatComposition = std::vector<std::string>;
using BracketGroup = std::vector<std::string>;

using Axis = int64_t;
using Axes = std::vector<int64_t>;
using AxesMap = std::map<int64_t, int64_t>;
using AxesNames = std::vector<std::string>;
using AxesLengths = std::vector<std::tuple<std::string, int64_t>>;
using AxesLengthsMap = std::map<std::string, int64_t>;

using OptionalAxes = std::optional<Axes>;
using CompositeAxes = std::vector<Axes>;
using InputCompositeAxes = std::vector<std::tuple<Axes, Axes>>;
using OutputCompositeAxes = std::vector<Axes>;

struct TransformRecipe
{
	Axes elementary_axes_lengths;
	IdentifiersMap axis_name2elementary_axis;
	InputCompositeAxes input_composition_known_unknown;
	Axes axes_permutation;
	Axis first_reduced_axis{ -1 };
	AxesMap added_axes;
	OutputCompositeAxes output_composite_axes;
	Hash hash{ 0 }; // trick for LRU cache
};

using MultiRecipe = std::map<int64_t, TransformRecipe>;

using CookedRecipe = std::tuple<OptionalAxes, 
							    OptionalAxes, 
										Axes, 
										AxesMap, 
								OptionalAxes, 
										Axis>;

// string printing helpers

inline auto print(int64_t value) -> std::string
{
	return std::to_string(value);
}

inline auto print(std::vector<int64_t> const& values) -> std::string
{
	std::string result;
	for (auto&& value : values)
		result += print(value);
	return result;
}

inline auto print(std::vector<std::string> const& values) -> std::string
{
	return "[" + join(values, ", ") + "]";
}

inline auto print(Identifier const& id) -> std::string
{
	return id.index() == 0 ? std::get<0>(id) : std::get<1>(id).to_string();
}

inline auto print(Identifiers const& ids) -> std::string
{
	std::vector<std::string> values;
	for (auto&& x : ids)
		values.push_back(print(x));
	return "[" + join(values, ", ") + "]";
}

inline auto print(std::vector<Identifier> const& ids) -> std::string
{
	std::vector<std::string> values;
	for (auto&& x : ids)
		values.push_back(print(x));
	return "[" + join(values, ", ") + "]";
}

inline auto print(AxesLengths const& values) -> std::string
{
	std::string result;
	for (auto&& value : values)
		result += print(std::get<0>(value)) 
				+ print(std::get<1>(value));
	return result;
}

// few other helpers

inline auto values(Identifiers const& identifiers) -> std::vector<int64_t>
{
	std::vector<int64_t> axes;
	for (auto&& v : identifiers)
		axes.push_back(std::stoll(print(v)));
	return axes;
}

inline auto values(IdentifiersMap const& identifiers) -> std::vector<int64_t>
{
	std::vector<int64_t> axes;
	for (auto&& [k, v] : identifiers)
		axes.push_back(v);
	return axes;
}

inline auto list(Composition const& composition) -> BracketGroup
{
	BracketGroup output;
	for (auto&& comp : composition)
	{
		if (comp.index() == 0)
		{
			auto&& comp_axis = std::get<0>(comp);
			for (auto&& axis : comp_axis)
				output.push_back(axis);
		}
		else
		{
			auto&& axis = std::get<1>(comp);
			output.push_back(axis);
		}
	}
	return output;
}

inline auto difference(Identifiers const& lhs, Identifiers const& rhs) -> std::vector<Identifier>
{
	std::vector<Identifier> difference;
	std::set_difference(lhs.begin(), lhs.end(),
						rhs.begin(), rhs.end(),
						std::back_inserter(difference));
	return difference;
}

inline auto symmetric_difference(Identifiers const& lhs, Identifiers const& rhs) -> std::vector<Identifier>
{
	std::vector<Identifier> difference;
	std::set_symmetric_difference(lhs.begin(), lhs.end(),
								  rhs.begin(), rhs.end(),
								  std::back_inserter(difference));
	return difference;
}

} // namespace einops::implementation
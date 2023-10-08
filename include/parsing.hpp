#pragma once

#include <extension/python.hpp>

namespace einops {
namespace implementation {

class ParsedExpression
{
public:
	Composition composition;
	Identifiers identifiers;
	bool has_ellipsis{ false };
	bool has_ellipsis_parenthesized{ false };
	bool has_non_unitary_anonymous_axes{ false };

public:
	ParsedExpression(std::string expression, bool allow_underscore = false, bool allow_duplicates = false)
	{
		if (contains(expression, '.'))
		{
			if (!contains(expression, "..."))
				throw Exception("Expression may contain dots only inside ellipsis (...)");

			if (count(expression, "...") != 1 || count(expression, ".") != 3)
				throw Exception("Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor");

			expression = replace(expression, "...", _ellipsis);
			has_ellipsis = true;
		}

		std::optional<BracketGroup> bracket_group;

		auto add_axis_name = [this, &allow_underscore, &allow_duplicates, &bracket_group](std::string const& x)
		{
			if (identifiers.contains(x))
				if (!(allow_underscore && x == "_") && !allow_duplicates)
					throw Exception(format("Indexing expression contains duplicate dimension \"{}\"", x));

			if (x == _ellipsis)
			{
				identifiers.insert(_ellipsis);

				if (!bracket_group.has_value())
				{
					composition.push_back(_ellipsis);
					has_ellipsis_parenthesized = false;
				}
				else
				{
					bracket_group.value().push_back(_ellipsis);
					has_ellipsis_parenthesized = true;
				}
			}
			else
			{
				auto is_number = isdecimal(x);

				if (is_number && std::stoll(x) == 1)
				{
					if (!bracket_group.has_value())
						composition.push_back({});
			
					return;
				} 
				auto [is_axis_name, reason] = TEST_axis_name_return_reason(x, allow_underscore);

				if (!(is_number || is_axis_name))
					throw Exception(format("Invalid axis identifier: {}\n{}", x, reason));

				if (is_number)
				{
					auto ax = AnonymousAxis(x);

					identifiers.insert(ax);

					has_non_unitary_anonymous_axes = true;

					if (bracket_group.has_value())
						bracket_group.value().push_back(ax);
					else
						composition.push_back(BracketGroup{ ax });
				}
				else
				{
					identifiers.insert(x);

					if (bracket_group.has_value())
						bracket_group.value().push_back(x);
					else
						composition.push_back(BracketGroup{ x });
				}
			}
		};

		std::string current_identifier;

		for (auto car : expression)
		{
			if (car == '(' || car == ')' || car == ' ')
			{
				if (!current_identifier.empty())
					add_axis_name(current_identifier);

				current_identifier = "";

				if (car == '(')
				{
					if (bracket_group.has_value())
						throw Exception("Axis composition is one-level (brackets inside brackets not allowed)");

					bracket_group = BracketGroup();
				}
				else
				if (car == ')')
				{
					if (!bracket_group.has_value())
						throw Exception("Brackets are not balanced");

					composition.push_back(bracket_group.value());
					bracket_group = std::nullopt;
				}
			}
			else
			if (isalnum(car) || (car == '_' || car == _ellipsis.front()))
				current_identifier += car;
			else
				throw Exception(format("Unknown character '{}'", car));
		}

		if (bracket_group.has_value())
			throw Exception(format("Imbalanced parentheses in expression: \"{}\"", expression));

		if (!current_identifier.empty())
			add_axis_name(current_identifier);
	}

	static auto TEST_axis_name_return_reason(std::string const& name, bool allow_underscore = false) -> std::tuple<bool, std::string>
	{
		if (!isidentifier(name))
		{
			return std::make_tuple(false, "not a valid identifier");
		}
		else
		if (name.front() == '_' || name.back() == '_')
		{
			if (name == "_" && allow_underscore)
				return std::make_tuple(true, "");
			else
				return std::make_tuple(false, "axis name should not start or end with underscore");
		}
		else
		{
			if (contains(python_keyword, name))
				std::cout << format("It is discouraged to use axes names that are keywords: {}", name) << std::endl;

			if (name == "axis")
				std::cout << "It is discouraged to use 'axis' as an axis name and will raise an error in future" << std::endl;

			return std::make_tuple(true, "");
		}
	}

	static auto TEST_axis_name(Identifier const& name) -> bool
	{
		auto [is_valid, reason] = ParsedExpression::TEST_axis_name_return_reason(name.index() == 1 ? std::get<1>(name).to_string() 
																									: std::get<0>(name));
		return is_valid;
	}
};

} // namespace implementation
} // namespace einops
#pragma once

#include <parsing.hpp>
using namespace einops::implementation;

class AnonymousAxisPlaceholder
{
public:
	AnonymousAxisPlaceholder(int64_t value)
		: value (value)
	{}

	bool operator== (const AnonymousAxis& other) const
	{
		return value == other.to_integer();
	}

	operator std::string() const
	{
		return std::to_string(value).append("-axis");
	}

	int64_t value;
};

inline bool operator==(AnonymousAxis const& lhs, AnonymousAxisPlaceholder const& rhs)
{
	return lhs.to_integer() == rhs.value;
}

inline bool operator!=(AnonymousAxis const& lhs, AnonymousAxisPlaceholder const& rhs)
{
	return !(lhs == rhs);
}

using aap = AnonymousAxisPlaceholder;
using vos = std::vector<std::string>;

class ParsingTest : public UnitTest
{
public:
	ParsingTest() 
		: UnitTest("Parsing")
	{}

	void test_anonymous_axes()
	{
		auto a = AnonymousAxis("2");
		auto b = AnonymousAxis("2");
		TESTB(a != b);
		auto c = AnonymousAxisPlaceholder(2);
		auto d = AnonymousAxisPlaceholder(3);
		TESTB(a == c && b == c);
		TESTB(a != d && b != d);
	}

	void test_elementary_axis_name()
	{
		for (auto&& name : vos{ "a", "b", "h", "dx", "h1", "zz", "i9123", "somelongname", "Alex", "camelCase", "u_n_d_e_r_score", "unreasonablyLongAxisName" })
			TESTB(ParsedExpression::check_axis_name(name));

		for (auto&& name : vos{ "", "2b", "12", "_startWithUnderscore", "endWithUnderscore_", "_", "...",  _ellipsis })
			TESTB(!ParsedExpression::check_axis_name(name));
	}

	void test_invalid_expressions()
	{
		auto test_raises_expression = [](std::string const& expression) -> bool
		{
			try { auto _ = ParsedExpression(expression, false, false); }
			catch (Exception const&) { return true; } return false;
		};
		{
			// double ellipsis should raise an error
			auto _ = ParsedExpression("... a b c d");
			TESTB(test_raises_expression("... a b c d ..."));
			TESTB(test_raises_expression("... a b c (d ...)"));
			TESTB(test_raises_expression("(... a) b c (d ...)"));
		}
		{
			// double/missing/enclosed parenthesis
			auto _ = ParsedExpression("(a)b c(d ...)");
			TESTB(test_raises_expression("(a)) b c (d ...)"));
			TESTB(test_raises_expression("(a b c (d ...)"));
			TESTB(test_raises_expression("(a) (()) b c (d ...)"));
			TESTB(test_raises_expression("(a) ((b c) (d ...))"));
		}
		{
			// invalid identifiers
			auto _ = ParsedExpression("camelCase under_scored cApiTaLs ..."); // TODO: unicode 'ß'
			TESTB(test_raises_expression("1a"));
			TESTB(test_raises_expression("_pre"));
			TESTB(test_raises_expression("...pre"));
			TESTB(test_raises_expression("pre..."));
		}
	}

	void test_parse_expression()
	{
		{
			auto parsed = ParsedExpression("a1  b1   c1    d1");
			TESTB(compare(parsed.identifiers, Identifiers{ "a1", "b1", "c1", "d1" }));
			TESTB(compare(parsed.composition, Composition{ vos{ "a1" }, vos{ "b1" }, vos{ "c1" }, vos{ "d1" } }));
			TESTB(!parsed.has_non_unitary_anonymous_axes);
			TESTB(!parsed.has_ellipsis);
		}
		{
			auto parsed = ParsedExpression("() () () ()");
			TESTB(compare(parsed.identifiers, Identifiers{}));
			TESTB(compare(parsed.composition, Composition{ {}, {}, {}, {} }));
			TESTB(!parsed.has_non_unitary_anonymous_axes);
			TESTB(!parsed.has_ellipsis);
		}
		{
			auto parsed = ParsedExpression("1 1 1 ()");
			TESTB(compare(parsed.identifiers, Identifiers{}));
			TESTB(compare(parsed.composition, Composition{ {}, {}, {}, {} }));
			TESTB(!parsed.has_non_unitary_anonymous_axes);
			TESTB(!parsed.has_ellipsis);
		}
		{
			auto parsed = ParsedExpression("5 (3 4)");
			TESTB(parsed.identifiers.size() == 3);
			TESTS(print(values(parsed.identifiers)), print(std::vector<int64_t>{ 3, 4, 5 }));
			TESTB(compare(parsed.composition, Composition{ vos{ aap(5) }, vos{ aap(3), aap(4) } }));
			TESTB(parsed.has_non_unitary_anonymous_axes);
			TESTB(!parsed.has_ellipsis);
		}
		{
			auto parsed = ParsedExpression("5 1 (1 4) 1");
			TESTB(parsed.identifiers.size() == 2);
			TESTS(print(values(parsed.identifiers)), print(std::vector<int64_t>{ 4, 5 }));
			TESTB(compare(parsed.composition, Composition{ vos{ aap(5) }, {}, vos{ aap(4) }, {} }));
		}
		{
			auto parsed = ParsedExpression("name1 ... a1 12 (name2 14)");
			TESTB(parsed.identifiers.size() == 6);
			TESTB(difference(parsed.identifiers, Identifiers{ "name1", _ellipsis, "a1", "name2" }).size() == 2);
			TESTB(compare(parsed.composition, Composition{ vos{ "name1" }, _ellipsis, vos{ "a1" }, vos{ aap(12) }, vos{ "name2" , aap(14) } }));
			TESTB(parsed.has_non_unitary_anonymous_axes);
			TESTB(parsed.has_ellipsis);
			TESTB(!parsed.has_ellipsis_parenthesized);
		}
		{
			auto parsed = ParsedExpression("(name1 ... a1 12) name2 14");
			TESTB(parsed.identifiers.size() == 6);
			TESTB(difference(parsed.identifiers, Identifiers{ "name1", _ellipsis, "a1", "name2" }).size() == 2);
			TESTB(compare(parsed.composition, Composition{ vos{ "name1", _ellipsis, "a1", aap(12) }, vos{ "name2" }, vos{ aap(14) } }));
			TESTB(parsed.has_non_unitary_anonymous_axes);
			TESTB(parsed.has_ellipsis);
			TESTB(parsed.has_ellipsis_parenthesized);
		}
	}

	void test_list() final
	{
		test_anonymous_axes();
		test_elementary_axis_name();
		test_invalid_expressions();
		test_parse_expression();
	}
};
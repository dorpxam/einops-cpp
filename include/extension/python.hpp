#pragma once

#include <extension/alias.hpp>
#include <extension/cache.hpp>
#include <extension/hash.hpp>
#include <extension/tools.hpp>

namespace einops::implementation {

/*
	einops use '…' character instead of '...' 
	for simplify the expression parser for ellipsis. 
	
	This character is encoded as Unicode:

		U+2026 Horizontal Ellipsis Unicode Character '…'

	The UTF-8 "support" in C++20 is an unusable mess.

	Fortunatly, all compilers have options to disable this:
		- MSVC:			Zc:char8_t-
		- gcc/clang:	-fno-char8_t

	For simplify the C++ process in this version, we will
	use '@' symbol instead '…', that is Ascii compatible.
*/

const auto _ellipsis = std::string("@"); // '@' instead of '…' in python

// follow the rules of the original python code, except for unicode.
// need to investigate this point for a greater compatibility when
// user need to port a python code in C++ (TODO: unicode support)

inline auto isidentifier(std::string const& name) -> bool
{
	if (name.empty())
		return false;
	else
	if (contains(name, _ellipsis))
		return false;
	else
	{
		// Python 2.x: the uppercase and lowercase letters A through Z, 
		//			   the underscore _ and, except for the first character, 
		//			   the digits 0 through 9.
		// TODO:  3.x: add alpha numeric unicode support
		for (auto&& [i, c] : iter::enumerate(name))
			if (!((i == 0 ? std::iswalpha(c) 
						  : std::iswalnum(c)) || c == '_'))
				return false;
	}
	return true;
};

// list of python keyword used in the 'check_axis_name' method as warning
// but not really usefull in C++, but keep that for now (TODO: remove)

const std::vector<std::string> python_keyword =
{
	"False",	"class",      "finally",	"is",         "return",
	"None",		"continue",   "for",		"lambda",     "try",
	"True",		"def",        "from",		"nonlocal",   "while",
	"and",		"del",        "global",		"not",        "with",
	"as",		"elif",       "if",			"or",         "yield",
	"assert",	"else",       "import",		"pass",
	"break ",	"except",     "in",			"raise"
};

// Some strings for ctype-style character classification
const auto ascii_lowercase = std::string("abcdefghijklmnopqrstuvwxyz");
const auto ascii_uppercase = std::string("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
const auto ascii_letters = ascii_lowercase + ascii_uppercase;

} // namespace einops::implementation
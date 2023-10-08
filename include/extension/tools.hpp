#pragma once

#include <algorithm>
#include <chrono>
#include <cwctype>
#include <format>
#include <functional>
#include <iostream>
#include <locale>
#include <map>
#include <numeric>
#include <optional>
#include <ranges>
#include <regex>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <extension/iterators.hpp>

template <typename T, class... Types>
inline bool operator==(const T& t, const std::variant<Types...>& v) 
{
	const T* c = std::get_if<T>(&v);
	return c && *c == t;
}

template <typename T, class... Types>
inline bool operator==(const std::variant<Types...>& v, const T& t) 
{
	return t == v;
}

template <typename first_type, typename tuple_type, size_t ...index>
inline auto to_vector_helper(const tuple_type &t, std::index_sequence<index...>)
{
    return std::vector<first_type>{
        std::get<index>(t)...
            };
}

template <typename first_type, typename ...others>
inline auto to_vector(const std::tuple<first_type, others...> &t)
{
    typedef typename std::remove_reference<decltype(t)>::type tuple_type;

    constexpr auto s =
        std::tuple_size<tuple_type>::value;

    return to_vector_helper<first_type, tuple_type>
        (t, std::make_index_sequence<s>{});
}

template <typename T>
inline auto compare(T const& lhs, T const& rhs) -> bool
{
	return lhs.size() == rhs.size() && std::equal(rhs.begin(), rhs.end(), lhs.begin());
}

template <typename T>
inline auto compare(std::vector<T> const& lhs, std::vector<T> const& rhs) -> bool
{
	if (lhs.size() != rhs.size())
		return false;
	else
	{
		for (auto&& [lval, rval] : iter::zip(lhs, rhs))
			if (lval != rval)
				return false;
	}
	return true;
}

template <typename T>
inline auto contains(std::string const& lhs, T rhs) -> bool
{
	return lhs.find(rhs) != std::string::npos;
}

template <typename T>
inline auto contains(std::vector<T> const& lhs, T value) -> bool
{
	for (auto& it : lhs)
		if (it == value)
			return true;
	return false;
}

template <typename T>
inline auto contains(std::vector<std::vector<T>> const& lhs, std::vector<T> const& value) -> bool
{
	for (auto& it : lhs)
		if (it == value)
			return true;
	return false;
}

template <typename T>
inline auto index(std::vector<T> const& lhs, T const& rhs) -> int64_t
{
	auto it = std::find(lhs.begin(), lhs.end(), rhs);
	return it != lhs.end() ? it - lhs.begin() : -1;
}

template <typename T>
inline auto index(std::vector<std::vector<T>> const& lhs, std::vector<T> const& value) -> int64_t
{
	for (auto&& [i, it] : iter::enumerate(lhs))
		if (it == value)
			return i;
	return false;
}

inline auto split(std::string const& str, std::string const& delimiter) -> std::tuple<std::string, std::string>
{
	return { str.substr(0, str.find(delimiter)),
			 str.substr(str.find(delimiter) + delimiter.length()) };
}

inline auto splits(std::string const& str, std::string const& delimiter) -> std::vector<std::string>
{
	std::size_t pos_start = 0, pos_end, delim_len = delimiter.length();
	std::string token;
	std::vector<std::string> res;

	while ((pos_end = str.find(delimiter, pos_start)) != std::string::npos)
	{
		token = str.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}

	res.push_back(str.substr(pos_start));
	return res;
}

inline auto join(std::vector<std::string> const& strs, std::string const& delimiter) -> std::string
{
	return std::accumulate(std::begin(strs), std::end(strs), std::string(),
		[=](std::string const& ss, std::string const& s)
		{
			return ss.empty() ? s : ss + delimiter + s;
		});
}

inline auto count(std::string const& str, std::string const& target) -> int
{
	int occurrences = 0;
	std::string::size_type pos = 0;
	while ((pos = str.find(target, pos)) != std::string::npos)
	{
		++occurrences;
		pos += target.length();
	}
	return occurrences;
}

inline auto replace(std::string data, std::string const& to_search, std::string const& replace_str) -> std::string
{
	size_t pos = data.find(to_search);
	while (pos != std::string::npos)
	{
		data.replace(pos, to_search.size(), replace_str);
		pos = data.find(to_search, pos + replace_str.size());
	}
	return data;
}

inline auto isdecimal(std::string const& str_num) -> bool
{
	int i = 0;
	int str_len = str_num.size();
	while (i < str_len && str_num[i] == ' ')
		i++;
	if (i < str_len && (str_num[i] == '+' || str_num[i] == '-'))
		i++;
	int digits = 0, dots = 0;
	while (i < str_len && ((str_num[i] >= '0' && str_num[i] <= '9') || (str_num[i] == '.')))
	{
		if (str_num[i] >= '0' && str_num[i] <= '9')
			digits++;
		else if (str_num[i] == '.')
			dots++;
		i++;
	}
	if (digits == 0 || dots > 1)
		return false;
	if (i < str_len && str_num[i] == 'e') 
	{
		if (++i < str_len && (str_num[i] == '+' || str_num[i] == '-'))
			i++;
		if (i == str_len || (str_num[i] < '0' || str_num[i]>'9'))
			return false;
		while (i < str_len && (str_num[i] >= '0' && str_num[i] <= '9'))
			i++;
	}
	while (i < str_len && str_num[i] == ' ')
		i++;
	return (i == str_len);
}
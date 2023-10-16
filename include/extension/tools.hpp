#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cwctype>
#include <functional>
#include <iostream>
#include <locale>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_set>
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

// Convert a vector<T> to initializer_list<T>
// https://stackoverflow.com/questions/18895583/convert-a-vectort-to-initializer-listt

namespace implementation {

constexpr size_t DEFAULT_MAX_LENGTH = 128;

template <typename V> struct backingValue { static V value; };
template <typename V> V backingValue<V>::value;

template <typename V, typename... Vcount> struct backingList { static std::initializer_list<V> list; };
template <typename V, typename... Vcount>
std::initializer_list<V> backingList<V, Vcount...>::list = { (Vcount)backingValue<V>::value... };

template <size_t maxLength, typename It, typename V = typename It::value_type, typename... Vcount>
static typename std::enable_if< sizeof...(Vcount) >= maxLength,
	std::initializer_list<V> >::type generate_n(It begin, It end, It current)
{
	throw std::length_error("More than maxLength elements in range.");
}

template <size_t maxLength = DEFAULT_MAX_LENGTH, typename It, typename V = typename It::value_type, typename... Vcount>
static typename std::enable_if < sizeof...(Vcount) < maxLength,
	std::initializer_list<V> > ::type generate_n(It begin, It end, It current)
{
	if (current != end)
		return generate_n<maxLength, It, V, V, Vcount...>(begin, end, ++current);

	current = begin;
	for (auto it = backingList<V, Vcount...>::list.begin();
		it != backingList<V, Vcount...>::list.end();
		++current, ++it)
		*const_cast<V*>(&*it) = *current;

	return backingList<V, Vcount...>::list;
}

} // namespace implementation

template <typename Iterator>
inline auto to_initializer_list(Iterator begin, Iterator end) -> std::initializer_list<typename Iterator::value_type>
{
	return implementation::generate_n(begin, end, begin);
}

template <typename Type>
inline auto to_initializer_list(std::vector<Type> const& vector) -> std::initializer_list<Type>
{
	return to_initializer_list(vector.begin(), vector.end());
}

template<class T, class... Rest>
inline constexpr bool are_all_same = (std::is_same_v<T, Rest> && ...);

template <typename ...Args>
inline auto from_map(std::map<std::string, int64_t> const& map) -> std::vector<std::tuple<std::string, int64_t>>
{
	std::vector<std::tuple<std::string, int64_t>> vec;
	for (auto&& [k, v] : map)
		vec.push_back(std::make_tuple(k, v));
	return vec;
}

template <typename T>
inline auto sort_and_reverse(std::vector<T> const& in) -> std::vector<T>
{
	auto out = std::vector<T>(std::begin(in), std::end(in));
	std::sort(std::begin(out), std::end(out));
	std::reverse(std::begin(out), std::end(out));
	return out;
}

template <typename Type, typename Container, typename Lambda>
inline auto processor(Container const& container, Lambda const& condition) -> std::vector<Type>
{
	std::vector<Type> output;
	for (auto&& value : container)
		output.push_back(condition(value));
	return output;
}

template <typename Output, typename Type>
inline auto sum(std::vector<Type> const& vector) -> Output
{
	Output result = 0;
	for (auto&& value : vector)
		result += value;
	return result;
}

template <typename Type>
inline auto prod(std::vector<Type> const& vector) -> Type
{
	Type result = 1;
	for (auto&& value : vector)
		result *= value;
	return result;
}

template <typename Type>
inline auto reverse(std::vector<Type>&& vector) -> std::vector<Type>
{
	std::reverse(vector.begin(), vector.end());
	return vector;
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
		for (auto&& [lval, rval] : iters::zip(lhs, rhs))
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

template <typename T1, typename T2>
inline void insert(std::vector<T1>& vec, size_t index, T2 value)
{
	vec.insert(vec.begin() + index, value);
}

template <typename T>
inline void remove(std::vector<T>& vec, size_t index)
{
	vec.erase(vec.begin() + index);
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
	for (auto&& [i, it] : iters::enumerate(lhs))
		if (it == value)
			return i;
	return false;
}

inline auto divide(std::string const& str, std::string const& delimiter) -> std::tuple<std::string, std::string>
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
	size_t i = 0;
	size_t str_len = str_num.size();
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
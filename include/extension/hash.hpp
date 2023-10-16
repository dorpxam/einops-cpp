#pragma once

#include <cmath>
#include <map>
#include <string>
#include <type_traits>

using Hash = std::size_t;
using HashMap = std::map<std::string, Hash>;

template <typename T, typename... Rest>
void hash_combine(Hash& seed, const T& v, const Rest&... rest)
{
	seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	(hash_combine(seed, rest), ...);
}

struct HashBuilder
{
	template <typename... Args>
	Hash operator()(const Args&... args) const noexcept
	{
		Hash hash = 0;
		hash_combine(hash, args...);
		return hash;
	}
	
	template <typename... Args>
	Hash operator()(Hash hash, const Args&... args) const noexcept
	{
		hash_combine(hash, args...);
		return hash;
	}
};

namespace python {

namespace {

// standards madness
using shash =  intptr_t; // ssize_t
using uhash = uintptr_t; //  size_t

constexpr auto sptr = sizeof(void*);
constexpr auto sizeofuhash = sizeof(uhash);
constexpr auto bits = (sptr >= 8) ? 61 : 31;
constexpr auto modulus = ((size_t)1 << bits) - 1;
constexpr auto multiplier = 1000003UL;
constexpr auto inf = 314159;

union
{
	unsigned char uc[24];
	struct 
	{
		shash prefix;
		shash suffix;
	} 
	fnv;
}
secret;

inline auto hashfunc(const void* source, intptr_t length)
{
	shash x;
	union 
	{
		uhash value;
		unsigned char bytes[sizeofuhash];
	} 
	block;

	const unsigned char* ptr = reinterpret_cast<const unsigned char*>(source);

	auto remainder = length % sizeofuhash;
	if (remainder == 0)
		remainder = sizeofuhash;

	auto blocks = (length - remainder) / sizeofuhash;

	x  = (uhash)secret.fnv.prefix;
	x ^= (uhash)*ptr << 7;
	
	while (blocks--)
	{
		std::memcpy(block.bytes, ptr, sizeofuhash);
		x = (multiplier * x) ^ block.value;
		ptr += sizeofuhash;
	}

	for (; remainder > 0; remainder--)
		x = (multiplier * x) ^ (uhash)*ptr++;

	x ^= (uhash)length;
	x ^= (uhash)secret.fnv.suffix;

	if (x == (uhash)-1)
		x =  (uhash)-2;

	return x;
}

template <typename Object>
inline auto id(Object const& object)
{
	return reinterpret_cast<const void*>(std::addressof(object));
}

} // namespace
	
inline shash hash(int val)
{
	return val; // seriously?
}

inline shash hash(double value)
{
	//if (!std::isfinite(value))
	//	if (std::isinf(value))
	//		return value > 0 ? inf : -inf;
	//	else
	//		hash(id(value));

	int e;
	double m = frexp(value, &e);

	int sign = 1;
	if (m < 0) {
		sign = -1;
		m = -m;
	}

	uhash x = 0, y;
	while (m) 
	{
		x = ((x << 28) & modulus) | x >> (bits - 28);
		m *= 268435456.0;
		e -= 28;
		y = (uhash)m;
		m -= y;
		x += y;
		if (x >= modulus)
			x -= modulus;
	}

	e = e >= 0 ? e % bits : bits - 1 - ((-1 - e) % bits);
	x = ((x << e) & modulus) | x >> (bits - e);

	x = x * sign;
	if (x == (uhash)-1)
		x =  (uhash)-2;

	return (shash)x;
}

inline shash hash(const void* src, intptr_t len)
{
	if (len == 0)
		return 0;

	auto hash = hashfunc(src, len);

	return hash == -1 ? -2 : hash;
}

inline shash hash(std::string const& str)
{
	return hash(str.data(), str.size());
}

inline shash hash(const void* ptr)
{
	auto hash = (size_t)ptr;
	hash = (hash >> 4) | (hash << (8 * sizeof(ptr) - 4));
	return hash == -1 ? -2 : hash;
}

} // namespace python
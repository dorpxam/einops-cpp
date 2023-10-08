#pragma once

#include <extension/exception.hpp>

namespace einops {
	
class AnonymousAxis
{
public:
	AnonymousAxis(int64_t axis)
		: value(axis)
		, uuid(++_UUID)
	{
		check();
	}

	AnonymousAxis(std::string const& axis)
		: value(std::stoll(axis))
		, uuid(++_UUID)
	{
		check();
	}

	AnonymousAxis(AnonymousAxis const& other)
		: value(other.value)
		, uuid(other.uuid)
	{}

	AnonymousAxis& operator=(AnonymousAxis const& other)
	{
		value = other.value;
		uuid = other.uuid;
		return *this;
	}

	bool operator< (const AnonymousAxis& other) const
	{
		return value < other.value;
	}

	bool operator== (const AnonymousAxis& other) const
	{
		return uuid == other.uuid;
	}

	int64_t to_integer() const
	{
		return value;
	}

	std::string to_string() const
	{
		return std::to_string(value).append("-axis");
	}

	operator std::string() const
	{
		return to_string();
	}

private:
	int64_t value;
	uint64_t uuid;

	inline void check()
	{
		if (value <= 1)
		{
			if (value == 1)
				throw Exception("No need to create anonymous axis of length 1. Report this as an issue");
			else
				throw Exception(std::format("Anonymous axis should have positive length, not {}", value));
		}
	}

private:
	static uint64_t _UUID;
};

uint64_t AnonymousAxis::_UUID { 0 };

} // namespace einops
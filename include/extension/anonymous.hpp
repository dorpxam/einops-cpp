#pragma once

#include <extension/exception.hpp>
#include <extension/format.hpp>

namespace einops {
namespace implementation {
	
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

	const void* to_uuid() const
	{
		return reinterpret_cast<const void*>(&uuid);
	}

	friend bool operator==(AnonymousAxis const& lhs, AnonymousAxis const& rhs);
	friend bool operator!=(AnonymousAxis const& lhs, AnonymousAxis const& rhs);

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
				throw Exception(format("Anonymous axis should have positive length, not {}", value));
		}
	}

private:
	static uint64_t _UUID;
};

uint64_t AnonymousAxis::_UUID { 0 };

inline bool operator==(AnonymousAxis const& lhs, AnonymousAxis const& rhs)
{
	return lhs.uuid == rhs.uuid;
}

inline bool operator!=(AnonymousAxis const& lhs, AnonymousAxis const& rhs)
{
	return !(lhs == rhs);
}

} // namespace implementation
} // namespace einops
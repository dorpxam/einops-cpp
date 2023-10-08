#pragma once

#include <tuple>
#include <iterator>

namespace iter {

// Chaining function for range-based for loop
// https://codereview.stackexchange.com/questions/98680/chaining-function-for-range-based-for-loop

namespace details {

template <typename T>
struct ref_or_copy
{
    ref_or_copy(T&& rval)
        : storage(new T(std::move(rval)))
        , val(*storage) 
    {}

    ref_or_copy(T& lval) 
        : val(lval)
    {}

    std::unique_ptr<T> storage;
    T& val;
};

template <typename C1, typename C2>
struct chain_obj_struct 
{
    ref_or_copy<std::remove_reference_t<C1>> c1;
    ref_or_copy<std::remove_reference_t<C2>> c2;

    struct iterator
    {
        decltype(std::begin(c1.val)) it1;
        decltype(std::begin(c1.val)) it1_end;
        decltype(std::begin(c2.val)) it2;
        bool in_first;

        iterator& operator++()
        {
            if (in_first) 
            {
                ++it1;
                if (it1 == it1_end)
                    in_first = false;
            }
            else
                ++it2;

            return *this;
        }

        typename std::conditional<
            std::is_const<std::remove_reference_t<decltype(*it1)>>::value,
            decltype(*it1),
            decltype(*it2)>::type
            operator*()
        {
            if (in_first) return *it1;
            return *it2;
        }

        bool operator==(const iterator& i2)
        {
            if (in_first != i2.in_first) return false;
            if (in_first)
                return it1 == i2.it1;
            else
                return it2 == i2.it2;
        }

        bool operator!=(const iterator& i2) 
        {
            return !this->operator==(i2);
        }
    };

    iterator begin() 
    {
        return iterator{ std::begin(c1.val), std::end(c1.val), std::begin(c2.val), true };
    }

    iterator end()
    {
        return iterator{ std::end(c1.val), std::end(c1.val), std::end(c2.val), false };
    }
};

template <typename C1, typename C2>
chain_obj_struct<C1, C2> chain_obj(C1&& c1, C2&& c2) 
{
    return chain_obj_struct<C1, C2>{std::forward<C1>(c1), std::forward<C2>(c2)};
}

} // namespace details

template <typename C>
auto chain(C&& c) 
{ 
    return std::forward<C>(c);
}

template <typename C1, typename C2, typename... Cs>
auto chain(C1&& c1, C2&& c2)
{
    return details::chain_obj(std::forward<C1>(c1), std::forward<C2>(c2));
}

template <typename C1, typename... Cs>
auto chain(C1&& c1, Cs&&... cs)
{
    return details::chain_obj(std::forward<C1>(c1), chain(std::forward<Cs>(cs)...));
}

namespace details {

// Python-like loop enumeration in C++
// https://stackoverflow.com/questions/11328264/python-like-loop-enumeration-in-c

template <typename Iterable>
class enumerate_object
{
private:
    Iterable _iter;
    std::size_t _size;
    decltype(std::begin(_iter)) _begin;
    const decltype(std::end(_iter)) _end;

public:
    enumerate_object(Iterable iter)
        : _iter(iter)
        , _size(0)
        , _begin(std::begin(iter))
        , _end(std::end(iter))
    {}

    const enumerate_object& begin() const { return *this; }
    const enumerate_object& end()   const { return *this; }

    bool operator!=(const enumerate_object&) const
    {
        return _begin != _end;
    }

    void operator++()
    {
        ++_begin;
        ++_size;
    }

    auto operator*() const
        -> std::pair<std::size_t, decltype(*_begin)>
    {
        return { _size, *_begin };
    }
};

} // namespace details

template <typename Iterable>
auto enumerate(Iterable&& iter)
    -> details::enumerate_object<Iterable>
{
    return { std::forward<Iterable>(iter) };
}

// C++ Range
// https://github.com/whoshuu/cpp_range

namespace details {

template <typename T>
class Range 
{
public:
    Range(const T& start, const T& stop, const T& step) : start_{ start }, stop_{ stop }, step_{ step }
    {
        if (step_ == 0) 
            throw std::invalid_argument("Range step argument must not be zero");
        else 
        {
            if ((start_ > stop_ && step_ > 0) || (start_ < stop_ && step_ < 0)) 
                throw std::invalid_argument("Range arguments must result in termination");
        }
    }

    class iterator 
    {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef T value_type;
        typedef T& reference;
        typedef T* pointer;

        iterator(value_type value, value_type step, value_type boundary) 
            : value_{ value }
            , step_{ step }
            , boundary_{ boundary }
            , positive_step_(step_ > 0) 
        {}

        iterator operator++() { value_ += step_; return *this; }
        reference operator*() { return value_; }
        const pointer operator->() { return &value_; }

        bool operator==(const iterator& rhs) 
        {
            return positive_step_ 
                ? (value_ >= rhs.value_ && value_ > boundary_)
                : (value_ <= rhs.value_ && value_ < boundary_);
        }

        bool operator!=(const iterator& rhs) 
        {
            return positive_step_ 
                ? (value_ < rhs.value_ && value_ >= boundary_)
                : (value_ > rhs.value_ && value_ <= boundary_);
        }

    private:
        value_type value_;
        const T step_;
        const T boundary_;
        const bool positive_step_;
    };

    iterator begin() const 
    {
        return iterator(start_, step_, start_);
    }

    iterator end() const
    {
        return iterator(stop_, step_, start_);
    }

    std::vector<T> vec() const
    {
        std::vector<T> result;
        for (auto&& value : *this)
            result.push_back(value);
        return result;
    }

private:
    const T start_;
    const T stop_;
    const T step_;
};

} // namespace details

template <typename T>
details::Range<T> range(const T& stop) 
{
    return details::Range<T>(T{ 0 }, stop, T{ 1 });
}

template <typename T>
details::Range<T> range(const T& start, const T& stop) 
{
    return details::Range<T>(start, stop, T{ 1 });
}

template <typename T>
details::Range<T> range(const T& start, const T& stop, const T& step) 
{
    return details::Range<T>(start, stop, step);
}

// Implementing Python’s zip in C++
// https://debashish-ghosh.medium.com/lets-iterate-together-fd7f5e49672b

namespace details {

template <typename... _Tp>
bool variadic_or(_Tp &&...args)
{
    return (... || args);
}

template <typename Tuple, std::size_t... I>
bool any_equals(Tuple&& t1, Tuple&& t2, std::index_sequence<I...>)
{
    return variadic_or(std::get<I>(std::forward<Tuple>(t1)) == std::get<I>(std::forward<Tuple>(t2))...);
}

} // namespace details

template <typename... T>
struct zip
{
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = std::tuple<std::iter_value_t<std::ranges::iterator_t<T>>...>;
        using reference = std::tuple<std::iter_reference_t<std::ranges::iterator_t<T>>...>;
        using difference_type = std::tuple<std::iter_difference_t<std::ranges::iterator_t<T>>...>;
        using pointer = std::tuple<typename std::iterator_traits<std::ranges::iterator_t<T>>::pointer...>;

        reference operator*()
        {
            return std::apply(
                []<typename... _Tp>(_Tp && ...e) { return std::forward_as_tuple(*std::forward<_Tp>(e)...); }, data_);
        }

        iterator& operator++()
        {
            std::apply(
                [this]<typename... _Tp>(_Tp && ...e) { data_ = std::make_tuple(++std::forward<_Tp>(e)...); }, data_);
            return *this;
        }

        auto operator!=(const iterator& iter) const
        {
            return !details::any_equals(data_, iter.data_, std::index_sequence_for<T...>{});
        }

        std::tuple<std::ranges::iterator_t<T>...> data_;
    };

    zip(T& ...args) 
        : data(std::forward_as_tuple(std::forward<T>(args)...)) 
    {}

    auto begin()
    {
        return iterator{ std::apply(
            []<typename... _Tp>(_Tp && ...e) { return std::make_tuple(std::begin(std::forward<_Tp>(e))...); }, data) };
    }

    auto end()
    {
        return iterator{ std::apply(
            []<typename... _Tp>(_Tp && ...e) { return std::make_tuple(std::end(std::forward<_Tp>(e))...); }, data) };
    }

    std::tuple<T &...> data;
};

} // namespace iter
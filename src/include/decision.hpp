#pragma once

#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

enum missing_type_flag : std::uint8_t
{
    None = 0b00,
    Zero = 0b01,
    NaN = 0b10,
};

enum decision_type_flag : std::uint8_t
{
    // decision_type_categorical = 0b01,
    decision_type_default_left = 0b10,
};

struct decision_tree_data
{
    std::vector<std::uint16_t> feature_index;
    std::vector<std::uint8_t> flag;
    std::vector<double> threshold;
    std::vector<int> left_node;
    std::vector<int> right_node;
    std::vector<double> leaf_value;
};

struct decision_tree_data_v2
{
    struct alignas(32) node_t
    {
        double threshold;
        int left, right;
        std::uint16_t feature_index;
        std::uint8_t flag;
    };

    std::vector<node_t> nodes;
    std::vector<double> leaf_value;
};

struct decision_tree_data_v3
{
    struct internal_data;

    std::shared_ptr<internal_data> data;
};

inline std::uint8_t get_missing_type(std::uint8_t flag) noexcept
{
    return (flag >> 2) & 0b11;
}
inline std::uint8_t set_missing_type(std::uint8_t flag, std::uint8_t missing_type) noexcept
{
    missing_type &= 0b11;
    flag &= ~(0b11 << 2);
    flag |= missing_type << 2;
    return flag;
}
inline bool get_decision_type(std::uint8_t flag, std::uint8_t mask) noexcept
{
    return flag & mask;
}
inline std::uint8_t set_decision_type(std::uint8_t flag, std::uint8_t mask, bool on_off) noexcept
{
    return on_off ? flag | mask : flag & ~mask;
}

constexpr double ZERO_THRESHOLD = 1e-35f;
inline bool is_zero(double f)
{
    return -ZERO_THRESHOLD <= f && f <= ZERO_THRESHOLD;
}

std::optional<std::ostringstream> decision_tree_source(
    decision_tree_data const &decision_tree,
    std::string_view function_name,
    std::string_view feature_type,
    std::string_view features_name);
bool decision_tree_validate(
    decision_tree_data const &decision_tree) noexcept;
double
decision_tree_run(
    float const *features,
    decision_tree_data const &decision_tree);
double
decision_tree_run_nocheck(
    float const *features,
    decision_tree_data const &decision_tree);

std::optional<decision_tree_data_v2>
decision_tree_convert(decision_tree_data const &from);

bool decision_tree_v2_validate(
    decision_tree_data_v2 const &decision_tree) noexcept;

double
decision_tree_v2_run(
    float const *features,
    decision_tree_data_v2 const &decision_tree);
double
decision_tree_v2_run_nocheck(
    float const *features,
    decision_tree_data_v2 const &decision_tree);
double
decision_tree_v2_run_v2(
    float const *features,
    decision_tree_data_v2 const &decision_tree);
double
decision_tree_v2_run_nocheck_v2(
    float const *features,
    decision_tree_data_v2 const &decision_tree);

std::optional<decision_tree_data_v3>
decision_tree_v2_convert(decision_tree_data_v2 const &from);

double
decision_tree_v3_run(
    float const *features,
    decision_tree_data_v3 const &decision_tree);

from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class FeatureSpec:
    name: str
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    exclude_features: List[str] = field(default_factory=list)


def get_all_feature_lists_one_leg():
    feature_specs = [
        original_feature,
        initial_feature,
        segment_feature,
        time_feature,
        rank_feature,
        price_feature,
        cabin_feature,
        company_feature,
        carrier_feature,
        flyer_feature,
        profile_feature,
        airport_feature,
        corporateTariffCode_feature,
        # additional_feature,
        group_feature,
        # aircraft_feature,
        # label_feature,
        # option_feature,
        route_feature,
    ]

    numerical = []
    categorical = []
    exclude = []
    for spec in feature_specs:
        numerical.extend(spec.numerical_features)
        categorical.extend(spec.categorical_features)
        exclude.extend(spec.exclude_features)

    # Exclude columns with large missing ratio
    for leg in [0, 1]:
        for seg in [0, 1]:
            if seg == 0:
                suffixes = [
                    "seatsAvailable",
                    # "cabinClass",
                    # "flightNumber",
                    # "aircraft_code",
                    # "arrivalTo_airport_city_iata",
                    # "arrivalTo_airport_iata",
                    # "departureFrom_airport_iata",
                    # "operatingCarrier_code",
                ]
            else:
                suffixes = [
                    # Missing
                    "cabinClass",
                    "seatsAvailable",
                    "baggageAllowance_quantity",
                    "baggageAllowance_weightMeasurementType",
                    "aircraft_code",
                    "arrivalTo_airport_city_iata",
                    "arrivalTo_airport_iata",
                    "departureFrom_airport_iata",
                    "flightNumber",
                    "marketingCarrier_code",
                    "operatingCarrier_code",
                ]
            for suffix in suffixes:
                exclude.append(f"legs{leg}_segments{seg}_{suffix}")

    # Exclude segment 2-3 columns (>98% missing)
    for leg in [0, 1]:
        base_seg = [2, 3]
        if leg == 1:
            base_seg += [0, 1]
        for seg in [2, 3]:
            for suffix in [
                "aircraft_code",
                "arrivalTo_airport_city_iata",
                "arrivalTo_airport_iata",
                "baggageAllowance_quantity",
                "baggageAllowance_weightMeasurementType",
                "cabinClass",
                "departureFrom_airport_iata",
                "duration",
                "flightNumber",
                "marketingCarrier_code",
                "operatingCarrier_code",
                "seatsAvailable",
            ]:
                exclude.append(f"legs{leg}_segments{seg}_{suffix}")
    return numerical, categorical, exclude


original_feature = FeatureSpec(
    name="original",
    numerical_features=[
        "legs0_segments0_seatsAvailable",
        # "legs1_segments0_seatsAvailable",
        "legs0_segments0_baggageAllowance_quantity",
        # "legs1_segments0_baggageAllowance_quantity",
        "legs0_duration",
        "legs0_segments0_duration",
        "legs0_segments1_duration",
        # "legs1_duration",
        # "legs1_segments0_duration",
        # "legs1_segments1_duration",
        "taxes",
        "totalPrice",
    ],
    categorical_features=[
        "bySelf",
        "companyID",
        "corporateTariffCode",
        "nationality",
        "isAccess3D",
        "isVip",
        "legs0_segments0_cabinClass",
        "legs0_segments0_flightNumber",
        "legs0_segments0_aircraft_code",
        "legs0_segments0_arrivalTo_airport_city_iata",
        "legs0_segments0_arrivalTo_airport_iata",
        "legs0_segments0_departureFrom_airport_iata",
        "legs0_segments0_baggageAllowance_weightMeasurementType",
        "legs0_segments0_marketingCarrier_code",
        "legs0_segments0_operatingCarrier_code",
        # "legs1_segments0_cabinClass",
        # "legs1_segments0_flightNumber",
        # "legs1_segments0_aircraft_code",
        # "legs1_segments0_arrivalTo_airport_city_iata",
        # "legs1_segments0_arrivalTo_airport_iata",
        # "legs1_segments0_departureFrom_airport_iata",
        # "legs1_segments0_baggageAllowance_weightMeasurementType",
        # "legs1_segments0_marketingCarrier_code",
        # "legs1_segments0_operatingCarrier_code",
        "pricingInfo_isAccessTP",
        "miniRules0_statusInfos",
        "miniRules1_statusInfos",
        "sex",
    ],
    exclude_features=[
        "Id",
        "ranker_id",
        "profileId",
        "selected",
        "requestDate",
        "legs0_departureAt",
        "legs0_arrivalAt",
        # "legs1_departureAt",
        # "legs1_arrivalAt",
        "miniRules0_percentage",
        "miniRules1_percentage",
        "frequentFlyer",
        "pricingInfo_passengerCount",
        "miniRules0_monetaryAmount",
        "miniRules1_monetaryAmount",
        "taxes",
        "totalPrice",
        # "legs0_segments1_duration",
        # "legs1_segments1_duration",
        # "bySelf",
        # For lightgbm
        # "companyID",
        # "legs0_segments0_arrivalTo_airport_city_iata",
        # "legs0_segments0_arrivalTo_airport_iata",
        # "legs0_segments0_departureFrom_airport_iata",
        # "legs1_segments0_arrivalTo_airport_city_iata",
        # "legs1_segments0_arrivalTo_airport_iata",
        # "legs1_segments0_departureFrom_airport_iata",
    ],
)


initial_feature = FeatureSpec(
    name="initial",
    numerical_features=[
        "price_per_tax",
        "tax_rate",
        "log_taxes",
        "log_price",
        "total_duration",
        "duration_ratio",
        "l0_seg",
        "n_ff_programs",
        "fee_ratio_rule0",
        "fee_ratio_rule1",
        # "price_duration_rat",
        "group_size",
        "group_size_log",
        # "unique_carrier_count_legs0",
        # "unique_carrier_count_legs1",
        # "unique_carrier_count_total",
    ],
    categorical_features=[
        "is_zero_tax",
        "is_shortest_duration",
        "is_one_way",
        "has_ff_program",
        # "has_corporate_tariff",
        "corporate_policy_compliant",
        "corporate_vip_flag",
        "free_cancel",
        "free_exchange",
        "is_popular_route",
        "contains_capitials",
        "is_popular_flight",
        "is_codeshare_leg0_seg0",
        "is_codeshare_leg1_seg0",
        "fee_ratio_rule0_is_missing",
        "fee_ratio_rule1_is_missing",
        "both_direct",
        "is_vip_freq",
        # "legs0_seat_enough",
        # "legs1_seat_enough",
        # "single_carrier_legs0",
        # "single_carrier_legs1",
        # "single_carrier_total",
    ],
    exclude_features=[
        "group_size",
        "price_per_tax",
        "l0_seg",
        "is_shortest_duration",
        "is_popular_flight",
        "total_duration",
        "both_direct",
        "is_zero_tax",
        # "is_one_way",
    ],
)


segment_feature = FeatureSpec(
    name="segment",
    numerical_features=[
        "n_segments_leg0",
        "n_segments_leg1",
    ],
    categorical_features=[
        "is_direct_leg0",
        "is_direct_leg1",
        "is_min_segments",
        "is_min_segments_leg0",
        "is_min_segments_leg1",
        "is_direct_shortest_legs0",
        "is_direct_shortest_legs1",
        # "is_both_legs_shortest",
    ],
    exclude_features=["is_min_segments"],
)


time_feature = FeatureSpec(
    name="time",
    numerical_features=[
        "stay_duration_hours",
        "stay_duration_hours_log",
        "hours_to_departure",
    ],
    categorical_features=[
        "legs0_departureAt_hour",
        "legs0_arrivalAt_hour",
        # "legs1_departureAt_hour",
        # "legs1_arrivalAt_hour",
        "legs0_departureAt_minute_of_day",
        "legs0_arrivalAt_minute_of_day",
        # "legs1_departureAt_minute_of_day",
        # "legs1_arrivalAt_minute_of_day",
        "legs0_departureAt_weekday",
        "legs0_arrivalAt_weekday",
        # "legs1_departureAt_weekday",
        # "legs1_arrivalAt_weekday",
        "legs0_departureAt_business_time",
        "legs0_arrivalAt_business_time",
        # "legs1_departureAt_business_time",
        # "legs1_arrivalAt_business_time",
        "legs0_departureAt_time_bin",
        "legs0_arrivalAt_time_bin",
        # "legs1_departureAt_time_bin",
        # "legs1_arrivalAt_time_bin",
        "legs0_dep_arr_bin_combo",
        "legs0_is_business_friendly",
        "legs0_is_red_eye",
        # "legs1_dep_arr_bin_combo",
        # "legs1_is_business_friendly",
        # "legs1_is_red_eye",
        "is_short_trip",
        "stay_duration_bin",
        "is_last_minute_booking",
        "legs0_departureAt_peak_morning",
        "legs0_departureAt_peak_evening",
        "legs0_arrivalAt_peak_morning",
        "legs0_arrivalAt_peak_evening",
        # "legs1_departureAt_peak_morning",
        # "legs1_arrivalAt_peak_morning",
        # "legs1_arrivalAt_peak_evening",
        "req_hour",
        "req_hour_bin",
    ],
    exclude_features=[
        # "legs0_departureAt_hour",
        # "legs0_arrivalAt_hour",
        # "legs1_departureAt_hour",
        # "legs1_arrivalAt_hour",
        "legs0_departureAt_business_time",
        "legs0_arrivalAt_business_time",
        "legs1_departureAt_business_time",
        "legs1_arrivalAt_business_time",
        "stay_duration_hours",
        # "legs0_arrivalAt_time_bin",
        # "legs0_departureAt_time_bin",
        # "legs1_arrivalAt_time_bin",
        # "legs1_departureAt_time_bin",
        # "legs0_dep_arr_bin_combo",
        # "legs1_dep_arr_bin_combo",
        "stay_duration_bin",
        "is_last_minute_booking",
        # Peak
        "legs0_departureAt_peak_morning",
        "legs0_departureAt_peak_evening",
        "legs0_arrivalAt_peak_morning",
        "legs0_arrivalAt_peak_evening",
        "legs1_departureAt_peak_morning",
        "legs1_arrivalAt_peak_morning",
        "legs1_arrivalAt_peak_evening",
        "req_hour",
        "req_hour_bin",
        # "legs0_is_red_eye",
        # "legs1_is_red_eye",
        # "legs0_is_business_friendly",
        "legs1_is_business_friendly",
        "legs0_departureAt_weekday",
        "legs0_arrivalAt_weekday",
    ],
)


rank_feature = FeatureSpec(
    name="rank",
    numerical_features=[
        "price_quantile_rank",
        "duration_quantile_rank",
        "leg0_duration_quantile_rank",
        # "leg1_duration_quantile_rank",
        "rank_interaction_mul",
        "rank_interaction_ratio",
        "rank_interaction_sum",
        "rank_interaction_sub",
        # "leg_dur_interaction_mul",
        # "leg_dur_interaction_ratio",
        # "leg_dur_interaction_sub",
    ],
    categorical_features=[],
    exclude_features=[
        "rank_interaction_ratio",
        "rank_interaction_sum",
        "leg_dur_interaction_mul",
        "leg_dur_interaction_ratio",
        "leg_dur_interaction_sub",
    ],
)


price_feature = FeatureSpec(
    name="price",
    numerical_features=[
        "price_zscore_from_median",
        "price_relative_to_min",
    ],
    categorical_features=[
        "is_top3_cheapest",
        "is_cheaper_than_avg",
        "is_expensive_outlier",
        "is_direct_cheapest_legs0",
        # "is_direct_cheapest_legs1",
        # "is_both_legs_cheapest",
        # "BestPriceTravelPolicy",
        # "BestPriceCorporateTariff",
        # "MinTime",
    ],
    exclude_features=[
        # "price_relative_to_min",
        # "is_cheaper_than_avg",
        "is_expensive_outlier",
    ],
)


cabin_feature = FeatureSpec(
    name="cabin",
    numerical_features=[
        "avg_cabin_class_legs0",
        # "avg_cabin_class_legs1",
        # "cabin_class_diff",
    ],
    categorical_features=[
        "legs0_has_first_class",
        # "legs1_has_first_class",
        "legs0_max_cabin_level",
        # "legs1_max_cabin_level",
        # "global_max_cabin_level",
        "legs0_all_cabin_level_1",
        # "legs1_all_cabin_level_1",
    ],
    exclude_features=[
        "has_business_class",
        "vip_first_class",
    ],
)


company_feature = FeatureSpec(
    name="company",
    numerical_features=[
        "company_ocurrence",
        "avg_selected_price",
        "std_selected_price",
        "selected_legs0_direct_ratio",
        # "selected_legs1_direct_ratio",
        "selected_direct_ratio",
        "selected_legs0_night_ratio",
        # "selected_legs1_night_ratio",
        "company_avg_pct",
        "company_avg_dct",
        "company_policy_rate",
        "z_price_vs_company_selected",
    ],
    categorical_features=[],
    exclude_features=[
        "selected_direct_ratio",
        "selected_legs0_night_ratio",
        "selected_legs1_night_ratio",
        "z_price_vs_company_selected",
    ],
)


carrier_feature = FeatureSpec(
    name="carrier",
    numerical_features=[
        "legs0_Carrier_code_cabin1_select_ratio",
        "legs0_Carrier_code_cabin2_select_ratio",
        "legs0_Carrier_code_cabin3_select_ratio",
        "legs0_Carrier_code_cabin4_select_ratio",
        "legs0_Carrier_code_avg_price_rank",
        "legs0_Carrier_code_avg_duration_rank",
        "legs0_Carrier_code_avg_dep_hour",
        "legs0_Carrier_code_night_ratio",
        "legs0_Carrier_code_company_diversity",
        "legs0_Carrier_code_user_diversity",
        # "legs1_Carrier_code_cabin1_select_ratio",
        # "legs1_Carrier_code_cabin2_select_ratio",
        # "legs1_Carrier_code_cabin3_select_ratio",
        # "legs1_Carrier_code_cabin4_select_ratio",
        # "legs1_Carrier_code_avg_price_rank",
        # "legs1_Carrier_code_avg_duration_rank",
        # "legs1_Carrier_code_avg_dep_hour",
        # "legs1_Carrier_code_night_ratio",
        # "legs1_Carrier_code_company_diversity",
        # "legs1_Carrier_code_user_diversity",
        "legs0_Carrier_code_selection_rate",
        "legs0_Carrier_code_log_total_count",
        "legs0_Carrier_code_log_selected_count",
        # "legs1_Carrier_code_selection_rate",
        # "legs1_Carrier_code_log_total_count",
        # "legs1_Carrier_code_log_selected_count",
        # "carrier_pop_prod",
    ],
    categorical_features=[
        "legs0_Carrier_code_selected_rank_bin",
        # "legs1_Carrier_code_selected_rank_bin",
    ],
    exclude_features=[
        "legs0_Carrier_code_selected_rank_bin",
        "legs1_Carrier_code_selected_rank_bin",
        "legs0_Carrier_code_avg_dep_hour",
        "legs1_Carrier_code_avg_dep_hour",
        "legs0_Carrier_code_cabin3_select_ratio",
        "legs1_Carrier_code_cabin3_select_ratio",
        "legs0_Carrier_code_cabin4_select_ratio",
        "legs1_Carrier_code_cabin4_select_ratio",
        "legs0_Carrier_code_selection_rate",
        "legs1_Carrier_code_selection_rate",
        "legs0_Carrier_code_log_total_count",
        "legs1_Carrier_code_log_total_count",
        "legs0_Carrier_code_night_ratio",
        "legs1_Carrier_code_night_ratio",
        # "carrier_pop_prod",
    ],
)


flyer_feature = FeatureSpec(
    name="flyer",
    numerical_features=[],
    categorical_features=[
        "legs0_segments0_marketingCarrier_code_in_frequentFlyer",
        "legs0_segments0_marketingCarrier_code_is_only_frequentFlyer",
        "is_major_carrier_0_0",
        "legs0_segments0_marketingCarrier_code_ff_and_economic",
    ],
    exclude_features=[
        "legs0_segments0_marketingCarrier_code_ff_and_economic",
        "legs0_segments0_marketingCarrier_code_is_only_frequentFlyer",
    ],
)


profile_feature = FeatureSpec(
    name="profile",
    numerical_features=[
        "user_selected_count",
    ],
    categorical_features=[],
    exclude_features=[
        # "user_selected_count",
    ],
)


airport_feature = FeatureSpec(
    name="airport",
    numerical_features=[
        "geo_distance_km",
        "price_distance_mul",
    ],
    categorical_features=[
        "legs0_departure_airport_UTC_Offset_Hours",
        "legs0_arrival_airport_UTC_Offset_Hours",
        # "legs1_departure_airport_UTC_Offset_Hours",
        # "legs1_arrival_airport_UTC_Offset_Hours",
        "legs0_departure_airport_Country_CodeA2",
        "legs0_arrival_airport_Country_CodeA2",
        # "legs1_departure_airport_Country_CodeA2",
        # "legs1_arrival_airport_Country_CodeA2",
        "legs0_is_cross_country",
        "legs0_is_cross_timezone",
        # "legs1_is_cross_country",
        # "legs1_is_cross_timezone",
        # "airport_same_outbound",
        # "airport_same_return",
    ],
    exclude_features=[
        "geo_distance_km",
        "price_distance_mul",
        "legs0_departure_airport_UTC_Offset_Hours",
        "legs0_arrival_airport_UTC_Offset_Hours",
        "legs1_departure_airport_UTC_Offset_Hours",
        "legs1_arrival_airport_UTC_Offset_Hours",
        "legs0_departure_airport_Country_CodeA2",
        "legs0_arrival_airport_Country_CodeA2",
        "legs1_departure_airport_Country_CodeA2",
        "legs1_arrival_airport_Country_CodeA2",
        "legs0_is_cross_timezone",
        "legs1_is_cross_timezone",
    ],
)


corporateTariffCode_feature = FeatureSpec(
    name="corporateTariffCode",
    numerical_features=[
        "corporateTariffCode_hotness",
        "corporateTariffCode_company_count",
        # "code_ratio_in_company",
        # "code_rank_in_company",
    ],
    categorical_features=[
        # "corporateTariffCode_unique_to_company",
        # "is_top1_code_in_company",
        "popular_corporateTariffCode",
    ],
    exclude_features=[],
)


additional_feature = FeatureSpec(
    name="additional",
    numerical_features=[
        "age",
    ],
    categorical_features=[
        "searchType",
        "hasAssistant",
        "age_bin_fixed",
    ],
    exclude_features=[],
)


group_feature = FeatureSpec(
    name="group",
    numerical_features=[
        "group_price_mean",
        "group_price_std",
        "group_price_range",
        "group_dur_mean",
        "group_dur_std",
        "group_dur_range",
        # "group_position",
        # "group_n_carriers",
        # "legs0_carrier_count_in_group",
        # "legs1_carrier_count_in_group",
        # "legs0_segment_count_in_group",
        # "legs1_segment_count_in_group",
        # "group_legs0_n_segments",
        # "group_legs1_n_segments",
        "group_n_leg_combos",
        "group_option_price_rank_ratio",
    ],
    categorical_features=[],
    exclude_features=[
        "group_position",
    ],
)


aircraft_feature = FeatureSpec(
    name="aircraft",
    numerical_features=[
        "route_aircraft_ratio",
        "aircraft_avg_pct",
        "aircraft_avg_dct",
    ],
    exclude_features=[
        "aircraft_avg_pct",
        "aircraft_avg_dct",
    ],
)


label_feature = FeatureSpec(
    name="label",
    numerical_features=[
        # "labels_count",
    ],
    categorical_features=[
        "has_any_label",
        # "label_BestPrice",
        # "label_BestPriceTravelPolicy",
        # "label_BestPriceDirect",
        # "label_Convenience",
        # "label_MinTime",
        # "label_BestPriceCorporateTariff",
        "category",
    ],
    exclude_features=[],
)


option_feature = FeatureSpec(
    name="option",
    numerical_features=[
        "flight_option_count",
        "price_to_flight_min_ratio",
        "flight_price_percentile",
    ],
    categorical_features=[
        "is_flight_only_option",
        "is_flight_cheapest",
    ],
    exclude_features=[
        "is_flight_cheapest",
        "flight_price_percentile",
    ],
)


route_feature = FeatureSpec(
    name="route",
    numerical_features=[
        "outbound_route_hotness",
        # "return_route_hotness",
        "outbound_route_avg_price",
        "outbound_route_avg_duration",
        "outbound_route_policy_rate",
        "outbound_route_direct_ratio",
        # "return_route_avg_price",
        # "return_route_avg_duration",
        # "return_route_policy_rate",
        # "return_route_direct_ratio",
    ],
    categorical_features=[
        # "outbound_route",
        # "return_route",
        "outbound_origin",
        "outbound_destination",
        "return_origin",
        "return_destination",
        "is_exact_round_trip",
    ],
    exclude_features=[
        "outbound_origin",
        "outbound_destination",
        "return_origin",
        "return_destination",
    ],
)

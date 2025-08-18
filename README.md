<h2 align="center"> FlightRank 2025: Aeroclub RecSys Cup </h2>

### Link
https://www.kaggle.com/competitions/aeroclub-recsys-2025

### Data
#### Identifiers and Metadata
- `Id` - Unique identifier for each flight option
- `ranker_id` - Group identifier for each search session (key grouping variable for ranking)
- `profileId` - User identifier
- `companyID` - Company identifier

#### User Information
- `sex` - User gender
- `nationality` - User nationality/citizenship
- `frequentFlyer` - Frequent flyer program status
- `isVip` - VIP status indicator
- `bySelf` - Whether user books flights independently
- `isAccess3D` - Binary marker for internal feature

#### Company Information
- `corporateTariffCode` - Corporate tariff code for business travel policies

#### Search and Route Information
- `searchRoute` - Flight route: single direction without "/" or round trip with "/"
- `requestDate` - Date and time when search was performed

#### Pricing Information
- `totalPrice` - Total ticket price
- `taxes` - Taxes and fees component

#### Flight Timing and Duration
- `legs0_departureAt` - Departure time for outbound flight
- `legs0_arrivalAt` - Arrival time for outbound flight
- `legs0_duration` - Duration of outbound flight
- `legs1_departureAt` - Departure time for return flight
- `legs1_arrivalAt` - Arrival time for return flight
- `legs1_duration` - Duration of return flight

#### Flight Segments

Each flight leg (legs0/legs1) can consist of multiple segments (segments0-3) when there are connections. Each segment contains:

##### Geography and Route
- `legs*_segments*_departureFrom_airport_iata` - Departure airport code
- `legs*_segments*_arrivalTo_airport_iata` - Arrival airport code
- `legs*_segments*_arrivalTo_airport_city_iata` - Arrival city code

##### Airline and Flight Details
- `legs*_segments*_marketingCarrier_code` - Marketing airline code
- `legs*_segments*_operatingCarrier_code` - Operating airline code (actual carrier)
- `legs*_segments*_aircraft_code` - Aircraft type code
- `legs*_segments*_flightNumber` - Flight number
- `legs*_segments*_duration`- Segment duration

##### Service Characteristics
- `legs*_segments*_baggageAllowance_quantity` - Baggage allowance: small numbers indicate piece count, large numbers indicate weight in kg
- `legs*_segments*_baggageAllowance_weightMeasurementType` - Type of baggage measurement
- `legs*_segments*_cabinClass` - Service class: 1.0 = economy, 2.0 = business, 4.0 = premium
- `legs*_segments*_seatsAvailable` - Number of available seats

#### Cancellation and Exchange Rules
##### Rule 0 (Cancellation)
- `miniRules0_monetaryAmount` - Monetary penalty for cancellation
- `miniRules0_percentage` - Percentage penalty for cancellation
- `miniRules0_statusInfos` - Cancellation rule status (0 = no cancellation allowed)
##### Rule 1 (Exchange)
- `miniRules1_monetaryAmount` - Monetary penalty for exchange
- `miniRules1_percentage` - Percentage penalty for exchange
- `miniRules1_statusInfos` - Exchange rule status
##### Pricing Policy Information
- `pricingInfo_isAccessTP` - Compliance with corporate Travel Policy
- `pricingInfo_passengerCount` - Number of passengers

#### Target Variable
- `selected` - In training data: binary variable (0 = not selected, 1 = selected). In submission: ranks within ranker_id groups

### Discussion


### Submission

| 版本 | 说明 | 本地分数 | Kaggle 分数 |
| --- | --- | --- | --- |
| XGBoost | [XGBoost baseline](https://www.kaggle.com/code/ka1242/xgboost-ranker-with-polars) | 0.505 | 0.47699|
| 20250706_015853 | `optuna` 调整超参数 | 0.498 | 0.46451 |
| 20250706_043754 | 调整训练轮次 | 0.500 | 0.47139 |
| 20250706_070608 | 用全部训练集训练 | 0.503 / 0.685 | **0.47966**  |
| 20250706_075631 | 增加凌晨航班/是否周末 | 0.501 / 0.682 | × |
| 20250706_085549 | 添加 round_trip | 0.504 / 0.684 | × |
| 20250706_123233 | 添加 rquestDate, miniRules0_percentage | 0.498 |  |
| 20250706_135351 | 优化 rank 特征，训练轮数750 | 0.503 / 0.728 | **0.48416** |
| 20250707_013041 | 添加训练轮数到1000， 补充rank特征交互 | 0.496 / 0.762 | 0.48195 |
| 20250707_035000 | 丰富 cabin 特征信息，时间分桶 | 0.508 / 0.725 | **0.48627** |
| 20250707_082340 | 增加行李，直达耗时最短 | 0.502 / 0.740  | 0.48021 |
| 20250707_092907 | 优化舱位特征，行李 | 0.504 / 0.743 | 0.48131  |
| 20250707_120651 | 删除类别多的特征 flightNumber | 0.506 / 0.745 | 0.48526  |
| 20250708_021747 | 删除 frequentFlyer 特征 | 0.508 / 0.742 | 0.48159  |
| 20250708_041915 | 增加公司特征，删除缺失值多的特征 | 0.507 / 0.84915 | **0.49205** |
| 20250708_072251 | 加入退改、旅程类型特征（短、中、长）| 0.511 / 0.84821 | 0.49077 |
| 20250708_080713 | 加入航班号特征 | 0.518 / 0.85096  | 0.48921 |
| 20250708_121124 | 用 optuna 自动选超参数 | 0.515 / 0.84938  | 0.48875 |
| 20250709_004208 | 将模型优化目标由 ndcg 变为 map | 0.515 /   | 0.48792 |
| 20250709_033059 | 加入航司特征 | 0.519 / 0.85046 | 0.49187  |
| 20250709_034731 | 删除航班号特征 | 0.513 / 0.848x | **0.50004**|
| 20250709_062805 | 增加返程航司特征，增加 FF 特征 | 0.512 / 0.84979 | 0.49903 |
| 20250709_064539 | 删除与 leg1 相关的特征（有56%缺失值）| 0.508 / 0.84954 | 0.49545 | 
| 20250710_001936 | add profile features, adjust former feature | 0.514 / 0.84898 | 0.49967 |
| 20250710_021128 | add layover time, national group | 0.514 / 0.84887 | 0.49628 |
| 20250710_081426 | remove bad features | 0.517 / 0.84899 | 0.49857  |
| 20250710_113737 | handle categorial missing values | 0.515 / 0.84962 | **0.50179** |
| 20250711_045734 | handle missing values in miniRules and baggageAllowance, tune hyperparam, | 0.522 / 0.85005 | 0.50179 |
| 20250711_104207 | 调优参数，修复 segment 段数错误 | 0.511 / 0.847 | 0.50179 |
| 20250711_121212 | 修改为最便宜的三个 | 0.514 / 0.84951 | 0.50013  |
| 20250711_132125 | 超参数调优 | xxx / xxx | **0.50518** |
| 20250712_033050 | 删除座位数，轮次调为 750 | 0.520 / 0.850xx | 0.50289 |
| 20250712_044822 | 还原座位数特征和轮次，加入 top3 duration 特征 | 0.526 / 0.85053 | **0.50564**  |
| 20250712_093037 | 加入各个 leg 最短段数，直飞最短/最便宜加入 top3 | 0.521 / 0.85100 | 0.50096 |
| 20250713_083737 | rank 归一化，修复公司特征的问题，删除一些特征 | 0.52635 / 0.85180 | **0.50711**  |
| 20250714_104736 | 加入城市特征，尝试重构航班特征 | 0.53062 / 0.85043 | 0.50693  |
| 20250714_143227 | 加入各个航司舱位比例 | 0.53276 / 0.85085 | **0.50876** |
| 20250716_132706 | 加入机场特征，筛选优化特征 | 0.53817 / 0.85304 | **0.51381** |
| 20250718_093425 | lightgbm 特征 | 0.51644 / 0.849622 | 0.50169  |
| 20250718_083308 | 修复 group 的 bug | 0.54133 / 0.50700 | **0.51822** |
| 20250719_002505 | xgboost 调参 | 0.54582 / 0.51342 | **0.51859**  |
| 20250719_120239 | lightgbm | 0.52400 / 0.49600 | 0.50583 |
| 20250720_003111 | lightgbm调参 | 0.53873 / 0.505589 | 0.50693  |
| 20250721_025740 | xgboost 特征调整，加回 leg1_seg0 的原始信息 | 0.54154 / 0.51205 | **0.51960** |
| 20250721_083807 | 增加小时的分箱，增加 leg1 机场特征 | 0.54751 / 0.51350 | **0.52244**  |
| 20250722_050939 | xgboost 加入 legs1 的其他特征 | 0.54774 / 0.512xx | **0.52070**  |
| 20250723_145130 | xgboost 加入 group 特征，调参 | 0.55213 / 0.51573 | 0.51758  |
| 20250724_032338 | lightgbm | 0.5403 / 0.50823 | 0.51345  |
| 20250724_141914 | xgboost 加标签 | 0.54684 / 0.51265 | **0.52098**  |
| 20250725_040223 | 加入 label 特征（部分顺序有问题），加入重排序 | 0.54853 / 0.51424 | **0.52309** |
| 20250725_083055 | 调整训练轮次和重排序参数 | 0.54931 / 0.51606 | **0.52391**  |
| 20250727_084025 | 加入 group 和 route 特征 | 0.54800 / 0.51600 | **0.52795**  |
| 20250728_094305 | 删除公司 z_price 特征 | 0.54988 / 0.51708 | **0.52492**  |
| 20250729_084249 | 删除 importance 小的特征 | 0.55510 / 0.51802 | 0.51822   |
| 20250731_122023 | 加回 additional 特征，价格排名用 log | 0.54948 / 0.51575 | 0.52015 |
| 20250801_103435 | 删除红眼、高基数类别特征，加入舱位差特征 | 0.54858 / 0.51540 | 0.51317 |
| 20250802_030514 | 加回一些特征 | 0.55098 / 0.51520 | 0.51703 |
| 20250802_074816 | 回滚 0.52795，删除增加一些特征 | 0.54824 / 0.51682 | 0.52603 |
| 20250803_063026 | 加强正则化，label 特征根据 parquet 构造 | 0.55280 / 0.51807 | 0.52474 |
| 20250804_001151 | lightgbm 参数调优 | 0.54177 / 0.509246 | 0.51244  |
| 20250805_032352 | 增加机场特征，避免直接用机场作为类别特征 | 0.54222 / 0.508265 | 0.51143  |
| 20250807_032439 | xgboost，删除一些类别特征，机场统计特征 | 0.55213 / 0.51641 | 0.52538  |
| 20250809_093033 | 最优版本加上舱位等级，机场特征 | 0.53484 / 0.50587 | 0.52612 |

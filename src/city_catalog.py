from __future__ import annotations

"""Global city catalog used by the data pipeline."""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from .utils import DATA_RAW


@dataclass(frozen=True)
class CityRecord:
    city_id: str
    city_name: str
    country: str
    iso3: str
    continent: str
    latitude: float
    longitude: float


def _records() -> List[CityRecord]:
    rows = [
        # Asia
        ("beijing_cn", "Beijing", "China", "CHN", "Asia", 39.9042, 116.4074),
        ("shanghai_cn", "Shanghai", "China", "CHN", "Asia", 31.2304, 121.4737),
        ("shenzhen_cn", "Shenzhen", "China", "CHN", "Asia", 22.5431, 114.0579),
        ("guangzhou_cn", "Guangzhou", "China", "CHN", "Asia", 23.1291, 113.2644),
        ("chengdu_cn", "Chengdu", "China", "CHN", "Asia", 30.5728, 104.0668),
        ("wuhan_cn", "Wuhan", "China", "CHN", "Asia", 30.5928, 114.3055),
        ("hong_kong_hk", "Hong Kong", "Hong Kong", "HKG", "Asia", 22.3193, 114.1694),
        ("taipei_tw", "Taipei", "Taiwan", "TWN", "Asia", 25.0330, 121.5654),
        ("kaohsiung_tw", "Kaohsiung", "Taiwan", "TWN", "Asia", 22.6273, 120.3014),
        ("tokyo_jp", "Tokyo", "Japan", "JPN", "Asia", 35.6762, 139.6503),
        ("osaka_jp", "Osaka", "Japan", "JPN", "Asia", 34.6937, 135.5023),
        ("nagoya_jp", "Nagoya", "Japan", "JPN", "Asia", 35.1815, 136.9066),
        ("fukuoka_jp", "Fukuoka", "Japan", "JPN", "Asia", 33.5902, 130.4017),
        ("sapporo_jp", "Sapporo", "Japan", "JPN", "Asia", 43.0618, 141.3545),
        ("yokohama_jp", "Yokohama", "Japan", "JPN", "Asia", 35.4437, 139.6380),
        ("seoul_kr", "Seoul", "South Korea", "KOR", "Asia", 37.5665, 126.9780),
        ("busan_kr", "Busan", "South Korea", "KOR", "Asia", 35.1796, 129.0756),
        ("incheon_kr", "Incheon", "South Korea", "KOR", "Asia", 37.4563, 126.7052),
        ("singapore_sg", "Singapore", "Singapore", "SGP", "Asia", 1.3521, 103.8198),
        ("bangkok_th", "Bangkok", "Thailand", "THA", "Asia", 13.7563, 100.5018),
        ("jakarta_id", "Jakarta", "Indonesia", "IDN", "Asia", -6.2088, 106.8456),
        ("manila_ph", "Manila", "Philippines", "PHL", "Asia", 14.5995, 120.9842),
        ("kuala_lumpur_my", "Kuala Lumpur", "Malaysia", "MYS", "Asia", 3.1390, 101.6869),
        ("hanoi_vn", "Hanoi", "Vietnam", "VNM", "Asia", 21.0278, 105.8342),
        ("ho_chi_minh_city_vn", "Ho Chi Minh City", "Vietnam", "VNM", "Asia", 10.8231, 106.6297),
        ("phnom_penh_kh", "Phnom Penh", "Cambodia", "KHM", "Asia", 11.5564, 104.9282),
        ("vientiane_la", "Vientiane", "Laos", "LAO", "Asia", 17.9757, 102.6331),
        ("delhi_in", "Delhi", "India", "IND", "Asia", 28.6139, 77.2090),
        ("mumbai_in", "Mumbai", "India", "IND", "Asia", 19.0760, 72.8777),
        ("bengaluru_in", "Bengaluru", "India", "IND", "Asia", 12.9716, 77.5946),
        ("hyderabad_in", "Hyderabad", "India", "IND", "Asia", 17.3850, 78.4867),
        ("chennai_in", "Chennai", "India", "IND", "Asia", 13.0827, 80.2707),
        ("dhaka_bd", "Dhaka", "Bangladesh", "BGD", "Asia", 23.8103, 90.4125),
        ("karachi_pk", "Karachi", "Pakistan", "PAK", "Asia", 24.8607, 67.0011),
        ("lahore_pk", "Lahore", "Pakistan", "PAK", "Asia", 31.5204, 74.3587),
        ("colombo_lk", "Colombo", "Sri Lanka", "LKA", "Asia", 6.9271, 79.8612),
        ("kathmandu_np", "Kathmandu", "Nepal", "NPL", "Asia", 27.7172, 85.3240),
        ("dubai_ae", "Dubai", "United Arab Emirates", "ARE", "Asia", 25.2048, 55.2708),
        ("abu_dhabi_ae", "Abu Dhabi", "United Arab Emirates", "ARE", "Asia", 24.4539, 54.3773),
        ("riyadh_sa", "Riyadh", "Saudi Arabia", "SAU", "Asia", 24.7136, 46.6753),
        ("jeddah_sa", "Jeddah", "Saudi Arabia", "SAU", "Asia", 21.4858, 39.1925),
        ("doha_qa", "Doha", "Qatar", "QAT", "Asia", 25.2854, 51.5310),
        ("kuwait_city_kw", "Kuwait City", "Kuwait", "KWT", "Asia", 29.3759, 47.9774),
        ("muscat_om", "Muscat", "Oman", "OMN", "Asia", 23.5880, 58.3829),
        ("tehran_ir", "Tehran", "Iran", "IRN", "Asia", 35.6892, 51.3890),
        ("baghdad_iq", "Baghdad", "Iraq", "IRQ", "Asia", 33.3152, 44.3661),
        ("tel_aviv_il", "Tel Aviv", "Israel", "ISR", "Asia", 32.0853, 34.7818),
        ("jerusalem_il", "Jerusalem", "Israel", "ISR", "Asia", 31.7683, 35.2137),
        ("amman_jo", "Amman", "Jordan", "JOR", "Asia", 31.9539, 35.9106),
        ("beirut_lb", "Beirut", "Lebanon", "LBN", "Asia", 33.8938, 35.5018),
        # North/Central Asia
        ("novosibirsk_ru", "Novosibirsk", "Russia", "RUS", "Asia", 55.0084, 82.9357),
        ("yekaterinburg_ru", "Yekaterinburg", "Russia", "RUS", "Asia", 56.8389, 60.6057),
        ("vladivostok_ru", "Vladivostok", "Russia", "RUS", "Asia", 43.1155, 131.8855),
        ("astana_kz", "Astana", "Kazakhstan", "KAZ", "Asia", 51.1694, 71.4491),
        ("almaty_kz", "Almaty", "Kazakhstan", "KAZ", "Asia", 43.2220, 76.8512),
        ("tashkent_uz", "Tashkent", "Uzbekistan", "UZB", "Asia", 41.2995, 69.2401),
        ("bishkek_kg", "Bishkek", "Kyrgyzstan", "KGZ", "Asia", 42.8746, 74.5698),
        ("ulaanbaatar_mn", "Ulaanbaatar", "Mongolia", "MNG", "Asia", 47.8864, 106.9057),
        ("istanbul_tr", "Istanbul", "Turkey", "TUR", "Europe", 41.0082, 28.9784),
        # Europe
        ("london_gb", "London", "United Kingdom", "GBR", "Europe", 51.5072, -0.1276),
        ("manchester_gb", "Manchester", "United Kingdom", "GBR", "Europe", 53.4808, -2.2426),
        ("paris_fr", "Paris", "France", "FRA", "Europe", 48.8566, 2.3522),
        ("lyon_fr", "Lyon", "France", "FRA", "Europe", 45.7640, 4.8357),
        ("marseille_fr", "Marseille", "France", "FRA", "Europe", 43.2965, 5.3698),
        ("berlin_de", "Berlin", "Germany", "DEU", "Europe", 52.5200, 13.4050),
        ("munich_de", "Munich", "Germany", "DEU", "Europe", 48.1351, 11.5820),
        ("hamburg_de", "Hamburg", "Germany", "DEU", "Europe", 53.5511, 9.9937),
        ("frankfurt_de", "Frankfurt", "Germany", "DEU", "Europe", 50.1109, 8.6821),
        ("madrid_es", "Madrid", "Spain", "ESP", "Europe", 40.4168, -3.7038),
        ("barcelona_es", "Barcelona", "Spain", "ESP", "Europe", 41.3874, 2.1686),
        ("valencia_es", "Valencia", "Spain", "ESP", "Europe", 39.4699, -0.3763),
        ("seville_es", "Seville", "Spain", "ESP", "Europe", 37.3891, -5.9845),
        ("rome_it", "Rome", "Italy", "ITA", "Europe", 41.9028, 12.4964),
        ("milan_it", "Milan", "Italy", "ITA", "Europe", 45.4642, 9.1900),
        ("naples_it", "Naples", "Italy", "ITA", "Europe", 40.8518, 14.2681),
        ("turin_it", "Turin", "Italy", "ITA", "Europe", 45.0703, 7.6869),
        ("florence_it", "Florence", "Italy", "ITA", "Europe", 43.7696, 11.2558),
        ("amsterdam_nl", "Amsterdam", "Netherlands", "NLD", "Europe", 52.3676, 4.9041),
        ("brussels_be", "Brussels", "Belgium", "BEL", "Europe", 50.8503, 4.3517),
        ("zurich_ch", "Zurich", "Switzerland", "CHE", "Europe", 47.3769, 8.5417),
        ("geneva_ch", "Geneva", "Switzerland", "CHE", "Europe", 46.2044, 6.1432),
        ("basel_ch", "Basel", "Switzerland", "CHE", "Europe", 47.5596, 7.5886),
        ("vienna_at", "Vienna", "Austria", "AUT", "Europe", 48.2082, 16.3738),
        ("stockholm_se", "Stockholm", "Sweden", "SWE", "Europe", 59.3293, 18.0686),
        ("oslo_no", "Oslo", "Norway", "NOR", "Europe", 59.9139, 10.7522),
        ("copenhagen_dk", "Copenhagen", "Denmark", "DNK", "Europe", 55.6761, 12.5683),
        ("helsinki_fi", "Helsinki", "Finland", "FIN", "Europe", 60.1699, 24.9384),
        ("dublin_ie", "Dublin", "Ireland", "IRL", "Europe", 53.3498, -6.2603),
        ("lisbon_pt", "Lisbon", "Portugal", "PRT", "Europe", 38.7223, -9.1393),
        ("porto_pt", "Porto", "Portugal", "PRT", "Europe", 41.1579, -8.6291),
        ("prague_cz", "Prague", "Czechia", "CZE", "Europe", 50.0755, 14.4378),
        ("warsaw_pl", "Warsaw", "Poland", "POL", "Europe", 52.2297, 21.0122),
        ("krakow_pl", "Krakow", "Poland", "POL", "Europe", 50.0647, 19.9450),
        ("budapest_hu", "Budapest", "Hungary", "HUN", "Europe", 47.4979, 19.0402),
        ("athens_gr", "Athens", "Greece", "GRC", "Europe", 37.9838, 23.7275),
        ("thessaloniki_gr", "Thessaloniki", "Greece", "GRC", "Europe", 40.6401, 22.9444),
        ("bucharest_ro", "Bucharest", "Romania", "ROU", "Europe", 44.4268, 26.1025),
        ("sofia_bg", "Sofia", "Bulgaria", "BGR", "Europe", 42.6977, 23.3219),
        ("belgrade_rs", "Belgrade", "Serbia", "SRB", "Europe", 44.7866, 20.4489),
        ("zagreb_hr", "Zagreb", "Croatia", "HRV", "Europe", 45.8150, 15.9819),
        ("ljubljana_si", "Ljubljana", "Slovenia", "SVN", "Europe", 46.0569, 14.5058),
        ("vilnius_lt", "Vilnius", "Lithuania", "LTU", "Europe", 54.6872, 25.2797),
        ("riga_lv", "Riga", "Latvia", "LVA", "Europe", 56.9496, 24.1052),
        ("tallinn_ee", "Tallinn", "Estonia", "EST", "Europe", 59.4370, 24.7536),
        ("kyiv_ua", "Kyiv", "Ukraine", "UKR", "Europe", 50.4501, 30.5234),
        ("minsk_by", "Minsk", "Belarus", "BLR", "Europe", 53.9006, 27.5590),
        ("moscow_ru", "Moscow", "Russia", "RUS", "Europe", 55.7558, 37.6173),
        ("saint_petersburg_ru", "Saint Petersburg", "Russia", "RUS", "Europe", 59.9343, 30.3351),
        # North America
        ("new_york_us", "New York", "United States", "USA", "North America", 40.7128, -74.0060),
        ("los_angeles_us", "Los Angeles", "United States", "USA", "North America", 34.0522, -118.2437),
        ("san_francisco_us", "San Francisco", "United States", "USA", "North America", 37.7749, -122.4194),
        ("chicago_us", "Chicago", "United States", "USA", "North America", 41.8781, -87.6298),
        ("washington_dc_us", "Washington", "United States", "USA", "North America", 38.9072, -77.0369),
        ("boston_us", "Boston", "United States", "USA", "North America", 42.3601, -71.0589),
        ("houston_us", "Houston", "United States", "USA", "North America", 29.7604, -95.3698),
        ("miami_us", "Miami", "United States", "USA", "North America", 25.7617, -80.1918),
        ("atlanta_us", "Atlanta", "United States", "USA", "North America", 33.7490, -84.3880),
        ("dallas_us", "Dallas", "United States", "USA", "North America", 32.7767, -96.7970),
        ("seattle_us", "Seattle", "United States", "USA", "North America", 47.6062, -122.3321),
        ("philadelphia_us", "Philadelphia", "United States", "USA", "North America", 39.9526, -75.1652),
        ("detroit_us", "Detroit", "United States", "USA", "North America", 42.3314, -83.0458),
        ("toronto_ca", "Toronto", "Canada", "CAN", "North America", 43.6532, -79.3832),
        ("vancouver_ca", "Vancouver", "Canada", "CAN", "North America", 49.2827, -123.1207),
        ("montreal_ca", "Montreal", "Canada", "CAN", "North America", 45.5017, -73.5673),
        ("calgary_ca", "Calgary", "Canada", "CAN", "North America", 51.0447, -114.0719),
        ("ottawa_ca", "Ottawa", "Canada", "CAN", "North America", 45.4215, -75.6972),
        ("quebec_city_ca", "Quebec City", "Canada", "CAN", "North America", 46.8139, -71.2080),
        ("mexico_city_mx", "Mexico City", "Mexico", "MEX", "North America", 19.4326, -99.1332),
        ("guadalajara_mx", "Guadalajara", "Mexico", "MEX", "North America", 20.6597, -103.3496),
        ("monterrey_mx", "Monterrey", "Mexico", "MEX", "North America", 25.6866, -100.3161),
        ("tijuana_mx", "Tijuana", "Mexico", "MEX", "North America", 32.5149, -117.0382),
        ("puebla_mx", "Puebla", "Mexico", "MEX", "North America", 19.0414, -98.2063),
        ("havana_cu", "Havana", "Cuba", "CUB", "North America", 23.1136, -82.3666),
        ("santo_domingo_do", "Santo Domingo", "Dominican Republic", "DOM", "North America", 18.4861, -69.9312),
        ("san_jose_cr", "San Jose", "Costa Rica", "CRI", "North America", 9.9281, -84.0907),
        ("panama_city_pa", "Panama City", "Panama", "PAN", "North America", 8.9824, -79.5199),
        ("guatemala_city_gt", "Guatemala City", "Guatemala", "GTM", "North America", 14.6349, -90.5069),
        # South America
        ("sao_paulo_br", "Sao Paulo", "Brazil", "BRA", "South America", -23.5505, -46.6333),
        ("rio_de_janeiro_br", "Rio de Janeiro", "Brazil", "BRA", "South America", -22.9068, -43.1729),
        ("brasilia_br", "Brasilia", "Brazil", "BRA", "South America", -15.7939, -47.8828),
        ("belo_horizonte_br", "Belo Horizonte", "Brazil", "BRA", "South America", -19.9167, -43.9345),
        ("porto_alegre_br", "Porto Alegre", "Brazil", "BRA", "South America", -30.0346, -51.2177),
        ("recife_br", "Recife", "Brazil", "BRA", "South America", -8.0476, -34.8770),
        ("curitiba_br", "Curitiba", "Brazil", "BRA", "South America", -25.4284, -49.2733),
        ("buenos_aires_ar", "Buenos Aires", "Argentina", "ARG", "South America", -34.6037, -58.3816),
        ("cordoba_ar", "Cordoba", "Argentina", "ARG", "South America", -31.4201, -64.1888),
        ("rosario_ar", "Rosario", "Argentina", "ARG", "South America", -32.9442, -60.6505),
        ("santiago_cl", "Santiago", "Chile", "CHL", "South America", -33.4489, -70.6693),
        ("valparaiso_cl", "Valparaiso", "Chile", "CHL", "South America", -33.0472, -71.6127),
        ("lima_pe", "Lima", "Peru", "PER", "South America", -12.0464, -77.0428),
        ("arequipa_pe", "Arequipa", "Peru", "PER", "South America", -16.4090, -71.5375),
        ("bogota_co", "Bogota", "Colombia", "COL", "South America", 4.7110, -74.0721),
        ("medellin_co", "Medellin", "Colombia", "COL", "South America", 6.2442, -75.5812),
        ("quito_ec", "Quito", "Ecuador", "ECU", "South America", -0.1807, -78.4678),
        ("guayaquil_ec", "Guayaquil", "Ecuador", "ECU", "South America", -2.1709, -79.9224),
        ("la_paz_bo", "La Paz", "Bolivia", "BOL", "South America", -16.4897, -68.1193),
        ("santa_cruz_bo", "Santa Cruz", "Bolivia", "BOL", "South America", -17.7833, -63.1821),
        ("asuncion_py", "Asuncion", "Paraguay", "PRY", "South America", -25.2637, -57.5759),
        ("montevideo_uy", "Montevideo", "Uruguay", "URY", "South America", -34.9011, -56.1645),
        ("caracas_ve", "Caracas", "Venezuela", "VEN", "South America", 10.4806, -66.9036),
        ("maracaibo_ve", "Maracaibo", "Venezuela", "VEN", "South America", 10.6545, -71.6500),
        # Africa
        ("cairo_eg", "Cairo", "Egypt", "EGY", "Africa", 30.0444, 31.2357),
        ("alexandria_eg", "Alexandria", "Egypt", "EGY", "Africa", 31.2001, 29.9187),
        ("lagos_ng", "Lagos", "Nigeria", "NGA", "Africa", 6.5244, 3.3792),
        ("abuja_ng", "Abuja", "Nigeria", "NGA", "Africa", 9.0765, 7.3986),
        ("johannesburg_za", "Johannesburg", "South Africa", "ZAF", "Africa", -26.2041, 28.0473),
        ("cape_town_za", "Cape Town", "South Africa", "ZAF", "Africa", -33.9249, 18.4241),
        ("nairobi_ke", "Nairobi", "Kenya", "KEN", "Africa", -1.2921, 36.8219),
        ("addis_ababa_et", "Addis Ababa", "Ethiopia", "ETH", "Africa", 8.9806, 38.7578),
        ("accra_gh", "Accra", "Ghana", "GHA", "Africa", 5.6037, -0.1870),
        ("dar_es_salaam_tz", "Dar es Salaam", "Tanzania", "TZA", "Africa", -6.7924, 39.2083),
        ("kampala_ug", "Kampala", "Uganda", "UGA", "Africa", 0.3476, 32.5825),
        ("kigali_rw", "Kigali", "Rwanda", "RWA", "Africa", -1.9441, 30.0619),
        ("casablanca_ma", "Casablanca", "Morocco", "MAR", "Africa", 33.5731, -7.5898),
        ("rabat_ma", "Rabat", "Morocco", "MAR", "Africa", 34.0209, -6.8416),
        ("algiers_dz", "Algiers", "Algeria", "DZA", "Africa", 36.7538, 3.0588),
        ("tunis_tn", "Tunis", "Tunisia", "TUN", "Africa", 36.8065, 10.1815),
        ("tripoli_ly", "Tripoli", "Libya", "LBY", "Africa", 32.8872, 13.1913),
        ("dakar_sn", "Dakar", "Senegal", "SEN", "Africa", 14.7167, -17.4677),
        ("luanda_ao", "Luanda", "Angola", "AGO", "Africa", -8.8390, 13.2894),
        ("maputo_mz", "Maputo", "Mozambique", "MOZ", "Africa", -25.9692, 32.5732),
        ("harare_zw", "Harare", "Zimbabwe", "ZWE", "Africa", -17.8252, 31.0335),
        ("lusaka_zm", "Lusaka", "Zambia", "ZMB", "Africa", -15.3875, 28.3228),
        ("gaborone_bw", "Gaborone", "Botswana", "BWA", "Africa", -24.6282, 25.9231),
        ("kinshasa_cd", "Kinshasa", "Democratic Republic of the Congo", "COD", "Africa", -4.4419, 15.2663),
        ("abidjan_ci", "Abidjan", "Cote d'Ivoire", "CIV", "Africa", 5.3600, -4.0083),
        ("khartoum_sd", "Khartoum", "Sudan", "SDN", "Africa", 15.5007, 32.5599),
        ("antananarivo_mg", "Antananarivo", "Madagascar", "MDG", "Africa", -18.8792, 47.5079),
        # Oceania
        ("sydney_au", "Sydney", "Australia", "AUS", "Oceania", -33.8688, 151.2093),
        ("melbourne_au", "Melbourne", "Australia", "AUS", "Oceania", -37.8136, 144.9631),
        ("brisbane_au", "Brisbane", "Australia", "AUS", "Oceania", -27.4698, 153.0251),
        ("perth_au", "Perth", "Australia", "AUS", "Oceania", -31.9505, 115.8605),
        ("adelaide_au", "Adelaide", "Australia", "AUS", "Oceania", -34.9285, 138.6007),
        ("canberra_au", "Canberra", "Australia", "AUS", "Oceania", -35.2809, 149.1300),
        ("hobart_au", "Hobart", "Australia", "AUS", "Oceania", -42.8821, 147.3272),
        ("gold_coast_au", "Gold Coast", "Australia", "AUS", "Oceania", -28.0167, 153.4000),
        ("auckland_nz", "Auckland", "New Zealand", "NZL", "Oceania", -36.8509, 174.7645),
        ("wellington_nz", "Wellington", "New Zealand", "NZL", "Oceania", -41.2866, 174.7756),
        ("christchurch_nz", "Christchurch", "New Zealand", "NZL", "Oceania", -43.5321, 172.6362),
        # Additional coverage: Asia / North Asia / Central Asia
        ("kolkata_in", "Kolkata", "India", "IND", "Asia", 22.5726, 88.3639),
        ("pune_in", "Pune", "India", "IND", "Asia", 18.5204, 73.8567),
        ("surat_in", "Surat", "India", "IND", "Asia", 21.1702, 72.8311),
        ("ahmedabad_in", "Ahmedabad", "India", "IND", "Asia", 23.0225, 72.5714),
        ("mashhad_ir", "Mashhad", "Iran", "IRN", "Asia", 36.2605, 59.6168),
        ("tabriz_ir", "Tabriz", "Iran", "IRN", "Asia", 38.0962, 46.2738),
        ("baku_az", "Baku", "Azerbaijan", "AZE", "Asia", 40.4093, 49.8671),
        ("tbilisi_ge", "Tbilisi", "Georgia", "GEO", "Asia", 41.7151, 44.8271),
        ("yerevan_am", "Yerevan", "Armenia", "ARM", "Asia", 40.1792, 44.4991),
        ("dushanbe_tj", "Dushanbe", "Tajikistan", "TJK", "Asia", 38.5598, 68.7870),
        ("ashgabat_tm", "Ashgabat", "Turkmenistan", "TKM", "Asia", 37.9601, 58.3261),
        ("khabarovsk_ru", "Khabarovsk", "Russia", "RUS", "Asia", 48.4808, 135.0928),
        ("krasnoyarsk_ru", "Krasnoyarsk", "Russia", "RUS", "Asia", 56.0153, 92.8932),
        ("irkutsk_ru", "Irkutsk", "Russia", "RUS", "Asia", 52.2869, 104.3050),
        ("omsk_ru", "Omsk", "Russia", "RUS", "Asia", 54.9885, 73.3242),
        ("yakutsk_ru", "Yakutsk", "Russia", "RUS", "Asia", 62.0355, 129.6755),
        # Additional coverage: Europe
        ("birmingham_gb", "Birmingham", "United Kingdom", "GBR", "Europe", 52.4862, -1.8904),
        ("glasgow_gb", "Glasgow", "United Kingdom", "GBR", "Europe", 55.8642, -4.2518),
        ("liverpool_gb", "Liverpool", "United Kingdom", "GBR", "Europe", 53.4084, -2.9916),
        ("cologne_de", "Cologne", "Germany", "DEU", "Europe", 50.9375, 6.9603),
        ("stuttgart_de", "Stuttgart", "Germany", "DEU", "Europe", 48.7758, 9.1829),
        ("dusseldorf_de", "Dusseldorf", "Germany", "DEU", "Europe", 51.2277, 6.7735),
        ("toulouse_fr", "Toulouse", "France", "FRA", "Europe", 43.6047, 1.4442),
        ("nice_fr", "Nice", "France", "FRA", "Europe", 43.7102, 7.2620),
        ("bilbao_es", "Bilbao", "Spain", "ESP", "Europe", 43.2630, -2.9350),
        ("malaga_es", "Malaga", "Spain", "ESP", "Europe", 36.7213, -4.4214),
        ("bologna_it", "Bologna", "Italy", "ITA", "Europe", 44.4949, 11.3426),
        ("genoa_it", "Genoa", "Italy", "ITA", "Europe", 44.4056, 8.9463),
        ("rotterdam_nl", "Rotterdam", "Netherlands", "NLD", "Europe", 51.9244, 4.4777),
        ("antwerp_be", "Antwerp", "Belgium", "BEL", "Europe", 51.2194, 4.4025),
        ("bratislava_sk", "Bratislava", "Slovakia", "SVK", "Europe", 48.1486, 17.1077),
        ("sarajevo_ba", "Sarajevo", "Bosnia and Herzegovina", "BIH", "Europe", 43.8563, 18.4131),
        ("tirana_al", "Tirana", "Albania", "ALB", "Europe", 41.3275, 19.8187),
        # Additional coverage: North America and Central America
        ("phoenix_us", "Phoenix", "United States", "USA", "North America", 33.4484, -112.0740),
        ("denver_us", "Denver", "United States", "USA", "North America", 39.7392, -104.9903),
        ("minneapolis_us", "Minneapolis", "United States", "USA", "North America", 44.9778, -93.2650),
        ("charlotte_us", "Charlotte", "United States", "USA", "North America", 35.2271, -80.8431),
        ("san_diego_us", "San Diego", "United States", "USA", "North America", 32.7157, -117.1611),
        ("las_vegas_us", "Las Vegas", "United States", "USA", "North America", 36.1699, -115.1398),
        ("edmonton_ca", "Edmonton", "Canada", "CAN", "North America", 53.5461, -113.4938),
        ("winnipeg_ca", "Winnipeg", "Canada", "CAN", "North America", 49.8951, -97.1384),
        ("leon_mx", "Leon", "Mexico", "MEX", "North America", 21.1220, -101.6840),
        ("merida_mx", "Merida", "Mexico", "MEX", "North America", 20.9674, -89.5926),
        ("queretaro_mx", "Queretaro", "Mexico", "MEX", "North America", 20.5888, -100.3899),
        ("tegucigalpa_hn", "Tegucigalpa", "Honduras", "HND", "North America", 14.0723, -87.1921),
        ("managua_ni", "Managua", "Nicaragua", "NIC", "North America", 12.1140, -86.2362),
        ("san_salvador_sv", "San Salvador", "El Salvador", "SLV", "North America", 13.6929, -89.2182),
        ("port_au_prince_ht", "Port-au-Prince", "Haiti", "HTI", "North America", 18.5944, -72.3074),
        # Additional coverage: South America
        ("salvador_br", "Salvador", "Brazil", "BRA", "South America", -12.9777, -38.5016),
        ("fortaleza_br", "Fortaleza", "Brazil", "BRA", "South America", -3.7319, -38.5267),
        ("manaus_br", "Manaus", "Brazil", "BRA", "South America", -3.1190, -60.0217),
        ("belem_br", "Belem", "Brazil", "BRA", "South America", -1.4558, -48.4902),
        ("mendoza_ar", "Mendoza", "Argentina", "ARG", "South America", -32.8895, -68.8458),
        ("la_plata_ar", "La Plata", "Argentina", "ARG", "South America", -34.9205, -57.9536),
        ("concepcion_cl", "Concepcion", "Chile", "CHL", "South America", -36.8201, -73.0444),
        ("trujillo_pe", "Trujillo", "Peru", "PER", "South America", -8.1118, -79.0288),
        ("cusco_pe", "Cusco", "Peru", "PER", "South America", -13.5319, -71.9675),
        ("cali_co", "Cali", "Colombia", "COL", "South America", 3.4516, -76.5320),
        ("barranquilla_co", "Barranquilla", "Colombia", "COL", "South America", 10.9639, -74.7964),
        ("cartagena_co", "Cartagena", "Colombia", "COL", "South America", 10.3910, -75.4794),
        ("georgetown_gy", "Georgetown", "Guyana", "GUY", "South America", 6.8013, -58.1553),
        ("paramaribo_sr", "Paramaribo", "Suriname", "SUR", "South America", 5.8520, -55.2038),
        ("valencia_ve", "Valencia", "Venezuela", "VEN", "South America", 10.1579, -67.9972),
        # Additional coverage: Africa
        ("ibadan_ng", "Ibadan", "Nigeria", "NGA", "Africa", 7.3775, 3.9470),
        ("kano_ng", "Kano", "Nigeria", "NGA", "Africa", 12.0022, 8.5920),
        ("port_harcourt_ng", "Port Harcourt", "Nigeria", "NGA", "Africa", 4.8156, 7.0498),
        ("durban_za", "Durban", "South Africa", "ZAF", "Africa", -29.8587, 31.0218),
        ("pretoria_za", "Pretoria", "South Africa", "ZAF", "Africa", -25.7479, 28.2293),
        ("mombasa_ke", "Mombasa", "Kenya", "KEN", "Africa", -4.0435, 39.6682),
        ("kisumu_ke", "Kisumu", "Kenya", "KEN", "Africa", -0.0917, 34.7680),
        ("kumasi_gh", "Kumasi", "Ghana", "GHA", "Africa", 6.6885, -1.6244),
        ("douala_cm", "Douala", "Cameroon", "CMR", "Africa", 4.0511, 9.7679),
        ("yaounde_cm", "Yaounde", "Cameroon", "CMR", "Africa", 3.8480, 11.5021),
        ("brazzaville_cg", "Brazzaville", "Republic of the Congo", "COG", "Africa", -4.2634, 15.2429),
        ("conakry_gn", "Conakry", "Guinea", "GIN", "Africa", 9.6412, -13.5784),
        ("niamey_ne", "Niamey", "Niger", "NER", "Africa", 13.5116, 2.1254),
        ("bamako_ml", "Bamako", "Mali", "MLI", "Africa", 12.6392, -8.0029),
        ("nouakchott_mr", "Nouakchott", "Mauritania", "MRT", "Africa", 18.0735, -15.9582),
        ("djibouti_city_dj", "Djibouti City", "Djibouti", "DJI", "Africa", 11.5721, 43.1456),
        ("mogadishu_so", "Mogadishu", "Somalia", "SOM", "Africa", 2.0469, 45.3182),
        ("bujumbura_bi", "Bujumbura", "Burundi", "BDI", "Africa", -3.3614, 29.3599),
        ("lilongwe_mw", "Lilongwe", "Malawi", "MWI", "Africa", -13.9626, 33.7741),
        ("windhoek_na", "Windhoek", "Namibia", "NAM", "Africa", -22.5609, 17.0658),
        ("freetown_sl", "Freetown", "Sierra Leone", "SLE", "Africa", 8.4657, -13.2317),
        ("monrovia_lr", "Monrovia", "Liberia", "LBR", "Africa", 6.3156, -10.8074),
        ("ouagadougou_bf", "Ouagadougou", "Burkina Faso", "BFA", "Africa", 12.3714, -1.5197),
        ("bobo_dioulasso_bf", "Bobo-Dioulasso", "Burkina Faso", "BFA", "Africa", 11.1771, -4.2979),
        ("mwanza_tz", "Mwanza", "Tanzania", "TZA", "Africa", -2.5164, 32.9175),
        # Additional coverage: Oceania and Pacific
        ("darwin_au", "Darwin", "Australia", "AUS", "Oceania", -12.4634, 130.8456),
        ("newcastle_au", "Newcastle", "Australia", "AUS", "Oceania", -32.9283, 151.7817),
        ("geelong_au", "Geelong", "Australia", "AUS", "Oceania", -38.1499, 144.3617),
        ("hamilton_nz", "Hamilton", "New Zealand", "NZL", "Oceania", -37.7870, 175.2793),
        ("dunedin_nz", "Dunedin", "New Zealand", "NZL", "Oceania", -45.8788, 170.5028),
        ("suva_fj", "Suva", "Fiji", "FJI", "Oceania", -18.1248, 178.4501),
        ("port_moresby_pg", "Port Moresby", "Papua New Guinea", "PNG", "Oceania", -9.4438, 147.1803),
        ("lae_pg", "Lae", "Papua New Guinea", "PNG", "Oceania", -6.7333, 146.9961),
    ]

    return [CityRecord(*row) for row in rows]


def _country_round_robin(group: pd.DataFrame) -> list[dict]:
    """Interleave rows by country to improve country diversity."""
    rows = group.to_dict(orient="records")
    buckets: dict[str, list[dict]] = {}
    country_order: list[str] = []
    for row in rows:
        country = str(row["country"])
        if country not in buckets:
            buckets[country] = []
            country_order.append(country)
        buckets[country].append(row)

    ordered: list[dict] = []
    while True:
        added = False
        for country in country_order:
            bucket = buckets[country]
            if not bucket:
                continue
            ordered.append(bucket.pop(0))
            added = True
        if not added:
            break

    return ordered


def _balanced_sample(data: pd.DataFrame, max_cities: int) -> pd.DataFrame:
    """Select cities with broad continental balance under a size cap."""
    if max_cities >= len(data):
        return data.copy()

    continent_order = list(dict.fromkeys(data["continent"].tolist()))
    queues: dict[str, list[dict]] = {}
    for continent in continent_order:
        group = data[data["continent"] == continent].copy()
        queues[continent] = _country_round_robin(group)

    pointers = {continent: 0 for continent in continent_order}
    selected_rows: list[dict] = []
    selected_by_continent: Counter[str] = Counter()

    # Pass 1: ensure each continent is represented when possible.
    for continent in continent_order:
        if len(selected_rows) >= max_cities:
            break
        if pointers[continent] < len(queues[continent]):
            selected_rows.append(queues[continent][pointers[continent]])
            pointers[continent] += 1
            selected_by_continent[continent] += 1

    # Pass 2: fill remaining slots by minimizing selected share.
    while len(selected_rows) < max_cities:
        candidates = [c for c in continent_order if pointers[c] < len(queues[c])]
        if not candidates:
            break
        chosen = min(
            candidates,
            key=lambda c: (
                selected_by_continent[c] / max(1, len(queues[c])),
                selected_by_continent[c],
                continent_order.index(c),
            ),
        )
        selected_rows.append(queues[chosen][pointers[chosen]])
        pointers[chosen] += 1
        selected_by_continent[chosen] += 1

    return pd.DataFrame(selected_rows, columns=data.columns)


def load_city_catalog(max_cities: int | None = None) -> pd.DataFrame:
    """Return a curated city catalog with broad global coverage."""
    records: Iterable[CityRecord] = _records()
    data = pd.DataFrame([r.__dict__ for r in records])
    override_path = DATA_RAW / "city_catalog.csv"
    if override_path.exists():
        try:
            raw = pd.read_csv(override_path)
        except Exception:
            raw = pd.DataFrame()
        if (not raw.empty) and ("city_id" in raw.columns):
            raw = raw.copy()
            raw["city_id"] = raw["city_id"].astype(str)
            data = data.copy()
            data["city_id"] = data["city_id"].astype(str)
            base = data.set_index("city_id")
            override = raw.set_index("city_id")
            union_index = base.index.union(override.index)
            base = base.reindex(union_index)
            override = override.reindex(union_index)
            for col in override.columns:
                if col in base.columns:
                    base[col] = override[col].combine_first(base[col])
                else:
                    base[col] = override[col]
            data = base.reset_index()
    if max_cities is None or max_cities <= 0:
        return data.copy()
    return _balanced_sample(data, max_cities=max_cities)

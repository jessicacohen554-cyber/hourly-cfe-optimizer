// ARCHIVED — only consumed by scenario_comparison.html (now in archive/)
// Auto-generated from step5d MAC queue: consequential_queue.json
// Cross-regional sequential deployment queue sorted by marginal MAC
const MAC_QUEUE_DATA = [
  {
    "iso": "MISO",
    "zone_label": "99\u219299.5%",
    "threshold_start": 99,
    "threshold_end": 99.5,
    "marginal_mac": 0.0,
    "co2_displaced_mt": 2.9081,
    "delta_cost_per_mwh": 0.0,
    "delta_resources": {
      "hydro": 2.01
    },
    "demand_twh": 1112.668,
    "target_year": 2049,
    "gas_backup_mw_end": 64941,
    "delta_gas_mw": -768,
    "new_gas_mw_end": 0,
    "queue_position": 1
  },
  {
    "iso": "MISO",
    "zone_label": "99.9\u219299.99%",
    "threshold_start": 99.9,
    "threshold_end": 99.99,
    "marginal_mac": 0.0,
    "co2_displaced_mt": 0.3992,
    "delta_cost_per_mwh": 0,
    "delta_resources": {},
    "demand_twh": 1137.146,
    "target_year": 2050,
    "gas_backup_mw_end": 68262,
    "delta_gas_mw": 0,
    "new_gas_mw_end": 262,
    "queue_position": 2
  },
  {
    "iso": "SPP",
    "zone_label": "85\u219287.5%",
    "threshold_start": 85,
    "threshold_end": 87.5,
    "marginal_mac": 7.36,
    "co2_displaced_mt": 3.8022,
    "delta_cost_per_mwh": 24.73,
    "delta_resources": {
      "hydro": 0.64
    },
    "demand_twh": 379.98,
    "target_year": 2039,
    "gas_backup_mw_end": 39592,
    "delta_gas_mw": 1380,
    "new_gas_mw_end": 7592,
    "queue_position": 3
  },
  {
    "iso": "SPP",
    "zone_label": "50\u219255%",
    "threshold_start": 50,
    "threshold_end": 55,
    "marginal_mac": 48.65,
    "co2_displaced_mt": 14.4923,
    "delta_cost_per_mwh": 37.15,
    "delta_resources": {
      "wind": 16.02,
      "hydro": 2.95
    },
    "demand_twh": 329.442,
    "target_year": 2031,
    "gas_backup_mw_end": 37672,
    "delta_gas_mw": -188,
    "new_gas_mw_end": 5672,
    "queue_position": 4
  },
  {
    "iso": "MISO",
    "zone_label": "23.19\u219250%",
    "threshold_start": 23.19,
    "threshold_end": 50,
    "marginal_mac": 50.79,
    "co2_displaced_mt": 129.7137,
    "delta_cost_per_mwh": 50.2,
    "delta_resources": {
      "wind": 125.29,
      "clean_firm": 2.94,
      "hydro": 3.03
    },
    "demand_twh": 735.865,
    "target_year": 2030,
    "gas_backup_mw_end": 83008,
    "delta_gas_mw": 15008,
    "new_gas_mw_end": 15008,
    "queue_position": 5
  },
  {
    "iso": "SPP",
    "zone_label": "55\u219260%",
    "threshold_start": 55,
    "threshold_end": 60,
    "marginal_mac": 55.48,
    "co2_displaced_mt": 18.0748,
    "delta_cost_per_mwh": 47.91,
    "delta_resources": {
      "solar": 2.28,
      "wind": 16.31,
      "clean_firm": 2.35
    },
    "demand_twh": 341.408,
    "target_year": 2033,
    "gas_backup_mw_end": 38654,
    "delta_gas_mw": 982,
    "new_gas_mw_end": 6654,
    "queue_position": 6
  },
  {
    "iso": "SPP",
    "zone_label": "31.05\u219250%",
    "threshold_start": 31.05,
    "threshold_end": 50,
    "marginal_mac": 68.79,
    "co2_displaced_mt": 9.8817,
    "delta_cost_per_mwh": 41.86,
    "delta_resources": {
      "wind": 15.45,
      "hydro": 0.79
    },
    "demand_twh": 323.616,
    "target_year": 2030,
    "gas_backup_mw_end": 37860,
    "delta_gas_mw": 5860,
    "new_gas_mw_end": 5860,
    "queue_position": 7
  },
  {
    "iso": "PJM",
    "zone_label": "50\u219255%",
    "threshold_start": 50,
    "threshold_end": 55,
    "marginal_mac": 83.17,
    "co2_displaced_mt": 42.3648,
    "delta_cost_per_mwh": 62.79,
    "delta_resources": {
      "wind": 46.47,
      "clean_firm": 4.8,
      "hydro": 4.27
    },
    "demand_twh": 972.294,
    "target_year": 2031,
    "gas_backup_mw_end": 93745,
    "delta_gas_mw": 1985,
    "new_gas_mw_end": 18745,
    "queue_position": 8
  },
  {
    "iso": "PJM",
    "zone_label": "34.53\u219250%",
    "threshold_start": 34.53,
    "threshold_end": 50,
    "marginal_mac": 99.72,
    "co2_displaced_mt": 85.2967,
    "delta_cost_per_mwh": 74.42,
    "delta_resources": {
      "solar": 5.11,
      "wind": 79.12,
      "clean_firm": 11.06,
      "offshore_wind": 18.99
    },
    "demand_twh": 949.506,
    "target_year": 2030,
    "gas_backup_mw_end": 91760,
    "delta_gas_mw": 16760,
    "new_gas_mw_end": 16760,
    "queue_position": 9
  },
  {
    "iso": "ERCOT",
    "zone_label": "30.27\u219250%",
    "threshold_start": 30.27,
    "threshold_end": 50,
    "marginal_mac": 106.23,
    "co2_displaced_mt": 28.1655,
    "delta_cost_per_mwh": 44.36,
    "delta_resources": {
      "wind": 65.04,
      "hydro": 2.41
    },
    "demand_twh": 579.615,
    "target_year": 2030,
    "gas_backup_mw_end": 67171,
    "delta_gas_mw": 12171,
    "new_gas_mw_end": 12171,
    "queue_position": 10
  },
  {
    "iso": "PJM",
    "zone_label": "95\u219297.5%",
    "threshold_start": 95,
    "threshold_end": 97.5,
    "marginal_mac": 120.68,
    "co2_displaced_mt": 19.5446,
    "delta_cost_per_mwh": 75.0,
    "delta_resources": {
      "clean_firm": 31.45
    },
    "demand_twh": 1455.114,
    "target_year": 2048,
    "gas_backup_mw_end": 153919,
    "delta_gas_mw": 19152,
    "new_gas_mw_end": 78919,
    "queue_position": 11
  },
  {
    "iso": "NEISO",
    "zone_label": "87.5\u219290%",
    "threshold_start": 87.5,
    "threshold_end": 90,
    "marginal_mac": 127.83,
    "co2_displaced_mt": 1.0247,
    "delta_cost_per_mwh": 58.54,
    "delta_resources": {
      "clean_firm": 0.91,
      "hydro": 0.53,
      "offshore_wind": 0.8
    },
    "demand_twh": 150.724,
    "target_year": 2040,
    "gas_backup_mw_end": 13808,
    "delta_gas_mw": 474,
    "new_gas_mw_end": 0,
    "queue_position": 12
  },
  {
    "iso": "PJM",
    "zone_label": "80\u219285%",
    "threshold_start": 80,
    "threshold_end": 85,
    "marginal_mac": 133.09,
    "co2_displaced_mt": 26.9417,
    "delta_cost_per_mwh": 72.46,
    "delta_resources": {
      "wind": 26.52,
      "offshore_wind": 22.96
    },
    "demand_twh": 1147.883,
    "target_year": 2038,
    "gas_backup_mw_end": 102635,
    "delta_gas_mw": 2812,
    "new_gas_mw_end": 27635,
    "queue_position": 13
  },
  {
    "iso": "NYISO",
    "zone_label": "85\u219287.5%",
    "threshold_start": 85,
    "threshold_end": 87.5,
    "marginal_mac": 145.71,
    "co2_displaced_mt": 2.857,
    "delta_cost_per_mwh": 88.39,
    "delta_resources": {
      "clean_firm": 4.71
    },
    "demand_twh": 200.032,
    "target_year": 2039,
    "gas_backup_mw_end": 20177,
    "delta_gas_mw": 321,
    "new_gas_mw_end": 2177,
    "queue_position": 14
  },
  {
    "iso": "NYISO",
    "zone_label": "87.5\u219290%",
    "threshold_start": 87.5,
    "threshold_end": 90,
    "marginal_mac": 147.15,
    "co2_displaced_mt": 2.6116,
    "delta_cost_per_mwh": 80.0,
    "delta_resources": {
      "clean_firm": 4.8
    },
    "demand_twh": 204.032,
    "target_year": 2040,
    "gas_backup_mw_end": 20503,
    "delta_gas_mw": 326,
    "new_gas_mw_end": 2503,
    "queue_position": 15
  },
  {
    "iso": "PJM",
    "zone_label": "85\u219287.5%",
    "threshold_start": 85,
    "threshold_end": 87.5,
    "marginal_mac": 169.95,
    "co2_displaced_mt": 12.291,
    "delta_cost_per_mwh": 75.48,
    "delta_resources": {
      "clean_firm": 27.67
    },
    "demand_twh": 1175.432,
    "target_year": 2039,
    "gas_backup_mw_end": 104907,
    "delta_gas_mw": 2272,
    "new_gas_mw_end": 29907,
    "queue_position": 16
  },
  {
    "iso": "NYISO",
    "zone_label": "75\u219280%",
    "threshold_start": 75,
    "threshold_end": 80,
    "marginal_mac": 170.32,
    "co2_displaced_mt": 3.7869,
    "delta_cost_per_mwh": 66.71,
    "delta_resources": {
      "wind": 1.37,
      "clean_firm": 1.26,
      "hydro": 3.49,
      "offshore_wind": 6.07
    },
    "demand_twh": 192.264,
    "target_year": 2037,
    "gas_backup_mw_end": 19801,
    "delta_gas_mw": -561,
    "new_gas_mw_end": 1801,
    "queue_position": 17
  },
  {
    "iso": "NYISO",
    "zone_label": "80\u219285%",
    "threshold_start": 80,
    "threshold_end": 85,
    "marginal_mac": 176.99,
    "co2_displaced_mt": 4.3077,
    "delta_cost_per_mwh": 89.28,
    "delta_resources": {
      "clean_firm": 4.62,
      "offshore_wind": 3.92
    },
    "demand_twh": 196.109,
    "target_year": 2038,
    "gas_backup_mw_end": 19856,
    "delta_gas_mw": 55,
    "new_gas_mw_end": 1856,
    "queue_position": 18
  },
  {
    "iso": "SPP",
    "zone_label": "60\u219265%",
    "threshold_start": 60,
    "threshold_end": 65,
    "marginal_mac": 178.15,
    "co2_displaced_mt": 4.0914,
    "delta_cost_per_mwh": 43.34,
    "delta_resources": {
      "wind": 16.29
    },
    "demand_twh": 347.554,
    "target_year": 2034,
    "gas_backup_mw_end": 39587,
    "delta_gas_mw": 933,
    "new_gas_mw_end": 7587,
    "queue_position": 19
  },
  {
    "iso": "ERCOT",
    "zone_label": "70\u219275%",
    "threshold_start": 70,
    "threshold_end": 75,
    "marginal_mac": 191.86,
    "co2_displaced_mt": 14.1963,
    "delta_cost_per_mwh": 46.0,
    "delta_resources": {
      "wind": 59.21
    },
    "demand_twh": 712.494,
    "target_year": 2036,
    "gas_backup_mw_end": 86429,
    "delta_gas_mw": 2461,
    "new_gas_mw_end": 31429,
    "queue_position": 20
  },
  {
    "iso": "NYISO",
    "zone_label": "33.69\u219255%",
    "threshold_start": 33.69,
    "threshold_end": 55,
    "marginal_mac": 199.32,
    "co2_displaced_mt": 10.9886,
    "delta_cost_per_mwh": 80.67,
    "delta_resources": {
      "solar": 19.12,
      "wind": 3.13,
      "hydro": 4.9
    },
    "demand_twh": 170.725,
    "target_year": 2031,
    "gas_backup_mw_end": 18335,
    "delta_gas_mw": 335,
    "new_gas_mw_end": 335,
    "queue_position": 21
  },
  {
    "iso": "PJM",
    "zone_label": "97.5\u219299%",
    "threshold_start": 97.5,
    "threshold_end": 99,
    "marginal_mac": 204.44,
    "co2_displaced_mt": 12.3514,
    "delta_cost_per_mwh": 75.0,
    "delta_resources": {
      "clean_firm": 33.67
    },
    "demand_twh": 1490.037,
    "target_year": 2049,
    "gas_backup_mw_end": 157019,
    "delta_gas_mw": 3100,
    "new_gas_mw_end": 82019,
    "queue_position": 22
  },
  {
    "iso": "MISO",
    "zone_label": "50\u219255%",
    "threshold_start": 50,
    "threshold_end": 55,
    "marginal_mac": 217.25,
    "co2_displaced_mt": 14.7862,
    "delta_cost_per_mwh": 53.74,
    "delta_resources": {
      "wind": 49.98,
      "clean_firm": 9.49
    },
    "demand_twh": 752.054,
    "target_year": 2031,
    "gas_backup_mw_end": 84599,
    "delta_gas_mw": 1591,
    "new_gas_mw_end": 16599,
    "queue_position": 23
  },
  {
    "iso": "NYISO",
    "zone_label": "92.5\u219295%",
    "threshold_start": 92.5,
    "threshold_end": 95,
    "marginal_mac": 218.38,
    "co2_displaced_mt": 2.8634,
    "delta_cost_per_mwh": 82.71,
    "delta_resources": {
      "solar": 5.61,
      "hydro": 1.96
    },
    "demand_twh": 225.268,
    "target_year": 2045,
    "gas_backup_mw_end": 23785,
    "delta_gas_mw": 1803,
    "new_gas_mw_end": 5785,
    "queue_position": 24
  },
  {
    "iso": "MISO",
    "zone_label": "85\u219287.5%",
    "threshold_start": 85,
    "threshold_end": 87.5,
    "marginal_mac": 220.57,
    "co2_displaced_mt": 8.7641,
    "delta_cost_per_mwh": 73.27,
    "delta_resources": {
      "solar": 24.4,
      "hydro": 1.99
    },
    "demand_twh": 895.069,
    "target_year": 2039,
    "gas_backup_mw_end": 88527,
    "delta_gas_mw": 2342,
    "new_gas_mw_end": 20527,
    "queue_position": 25
  },
  {
    "iso": "PJM",
    "zone_label": "65\u219270%",
    "threshold_start": 65,
    "threshold_end": 70,
    "marginal_mac": 235.12,
    "co2_displaced_mt": 22.074,
    "delta_cost_per_mwh": 74.91,
    "delta_resources": {
      "solar": 46.36,
      "wind": 2.46,
      "clean_firm": 19.96,
      "offshore_wind": 0.5
    },
    "demand_twh": 1069.049,
    "target_year": 2035,
    "gas_backup_mw_end": 102014,
    "delta_gas_mw": -2304,
    "new_gas_mw_end": 27014,
    "queue_position": 26
  },
  {
    "iso": "PJM",
    "zone_label": "87.5\u219290%",
    "threshold_start": 87.5,
    "threshold_end": 90,
    "marginal_mac": 236.39,
    "co2_displaced_mt": 15.9045,
    "delta_cost_per_mwh": 72.46,
    "delta_resources": {
      "wind": 27.81,
      "offshore_wind": 24.07
    },
    "demand_twh": 1203.642,
    "target_year": 2040,
    "gas_backup_mw_end": 107854,
    "delta_gas_mw": 2947,
    "new_gas_mw_end": 32854,
    "queue_position": 27
  },
  {
    "iso": "ERCOT",
    "zone_label": "65\u219270%",
    "threshold_start": 65,
    "threshold_end": 70,
    "marginal_mac": 244.22,
    "co2_displaced_mt": 15.9623,
    "delta_cost_per_mwh": 46.0,
    "delta_resources": {
      "wind": 84.75
    },
    "demand_twh": 688.4,
    "target_year": 2035,
    "gas_backup_mw_end": 83968,
    "delta_gas_mw": -1643,
    "new_gas_mw_end": 28968,
    "queue_position": 28
  },
  {
    "iso": "NEISO",
    "zone_label": "28.560000000000002\u219250%",
    "threshold_start": 28.560000000000002,
    "threshold_end": 50,
    "marginal_mac": 245.68,
    "co2_displaced_mt": 10.126,
    "delta_cost_per_mwh": 104.77,
    "delta_resources": {
      "solar": 5.07,
      "clean_firm": 2.19,
      "offshore_wind": 16.39
    },
    "demand_twh": 126.097,
    "target_year": 2030,
    "gas_backup_mw_end": 13784,
    "delta_gas_mw": -216,
    "new_gas_mw_end": 0,
    "queue_position": 29
  },
  {
    "iso": "PJM",
    "zone_label": "70\u219275%",
    "threshold_start": 70,
    "threshold_end": 75,
    "marginal_mac": 248.91,
    "co2_displaced_mt": 23.7662,
    "delta_cost_per_mwh": 77.97,
    "delta_resources": {
      "wind": 29.07,
      "clean_firm": 24.39,
      "offshore_wind": 22.41
    },
    "demand_twh": 1094.706,
    "target_year": 2036,
    "gas_backup_mw_end": 100712,
    "delta_gas_mw": -1302,
    "new_gas_mw_end": 25712,
    "queue_position": 30
  },
  {
    "iso": "ERCOT",
    "zone_label": "99\u219299.5%",
    "threshold_start": 99,
    "threshold_end": 99.5,
    "marginal_mac": 257.51,
    "co2_displaced_mt": 2.7179,
    "delta_cost_per_mwh": 49.25,
    "delta_resources": {
      "solar": 4.19,
      "wind": 10.02
    },
    "demand_twh": 1114.31,
    "target_year": 2049,
    "gas_backup_mw_end": 125381,
    "delta_gas_mw": -111,
    "new_gas_mw_end": 70381,
    "queue_position": 31
  },
  {
    "iso": "NEISO",
    "zone_label": "75\u219280%",
    "threshold_start": 75,
    "threshold_end": 80,
    "marginal_mac": 258.2,
    "co2_displaced_mt": 2.7611,
    "delta_cost_per_mwh": 87.29,
    "delta_resources": {
      "solar": 1.23,
      "clean_firm": 2.1,
      "offshore_wind": 4.84
    },
    "demand_twh": 142.869,
    "target_year": 2037,
    "gas_backup_mw_end": 13558,
    "delta_gas_mw": -414,
    "new_gas_mw_end": 0,
    "queue_position": 32
  },
  {
    "iso": "SPP",
    "zone_label": "87.5\u219290%",
    "threshold_start": 87.5,
    "threshold_end": 90,
    "marginal_mac": 261.48,
    "co2_displaced_mt": 3.8554,
    "delta_cost_per_mwh": 43.84,
    "delta_resources": {
      "wind": 21.84
    },
    "demand_twh": 386.82,
    "target_year": 2040,
    "gas_backup_mw_end": 39711,
    "delta_gas_mw": 119,
    "new_gas_mw_end": 7711,
    "queue_position": 33
  },
  {
    "iso": "NEISO",
    "zone_label": "65\u219270%",
    "threshold_start": 65,
    "threshold_end": 70,
    "marginal_mac": 264.05,
    "co2_displaced_mt": 2.8701,
    "delta_cost_per_mwh": 90.41,
    "delta_resources": {
      "wind": 5.9,
      "clean_firm": 0.95,
      "offshore_wind": 1.82
    },
    "demand_twh": 137.861,
    "target_year": 2035,
    "gas_backup_mw_end": 14061,
    "delta_gas_mw": 236,
    "new_gas_mw_end": 61,
    "queue_position": 34
  },
  {
    "iso": "ERCOT",
    "zone_label": "75\u219280%",
    "threshold_start": 75,
    "threshold_end": 80,
    "marginal_mac": 264.65,
    "co2_displaced_mt": 15.7402,
    "delta_cost_per_mwh": 51.05,
    "delta_resources": {
      "solar": 2.98,
      "wind": 60.01,
      "clean_firm": 18.61
    },
    "demand_twh": 737.432,
    "target_year": 2037,
    "gas_backup_mw_end": 86202,
    "delta_gas_mw": -227,
    "new_gas_mw_end": 31202,
    "queue_position": 35
  },
  {
    "iso": "CAISO",
    "zone_label": "34.32\u219250%",
    "threshold_start": 34.32,
    "threshold_end": 50,
    "marginal_mac": 268.38,
    "co2_displaced_mt": 10.4333,
    "delta_cost_per_mwh": 65.81,
    "delta_resources": {
      "solar": 23.76,
      "wind": 0.74,
      "clean_firm": 2.86,
      "hydro": 2.88,
      "geothermal": 12.31
    },
    "demand_twh": 246.147,
    "target_year": 2030,
    "gas_backup_mw_end": 40936,
    "delta_gas_mw": 3936,
    "new_gas_mw_end": 3936,
    "queue_position": 36
  },
  {
    "iso": "ERCOT",
    "zone_label": "50\u219265%",
    "threshold_start": 50,
    "threshold_end": 65,
    "marginal_mac": 270.65,
    "co2_displaced_mt": 69.885,
    "delta_cost_per_mwh": 57.15,
    "delta_resources": {
      "solar": 324.5,
      "clean_firm": 6.42
    },
    "demand_twh": 665.121,
    "target_year": 2034,
    "gas_backup_mw_end": 85611,
    "delta_gas_mw": 18440,
    "new_gas_mw_end": 30611,
    "queue_position": 37
  },
  {
    "iso": "CAISO",
    "zone_label": "92.5\u219295%",
    "threshold_start": 92.5,
    "threshold_end": 95,
    "marginal_mac": 270.91,
    "co2_displaced_mt": 4.4155,
    "delta_cost_per_mwh": 64.75,
    "delta_resources": {
      "wind": 6.83,
      "clean_firm": 9.05,
      "hydro": 2.59
    },
    "demand_twh": 326.443,
    "target_year": 2045,
    "gas_backup_mw_end": 38959,
    "delta_gas_mw": 867,
    "new_gas_mw_end": 1959,
    "queue_position": 38
  },
  {
    "iso": "SPP",
    "zone_label": "80\u219285%",
    "threshold_start": 80,
    "threshold_end": 85,
    "marginal_mac": 285.26,
    "co2_displaced_mt": 7.7186,
    "delta_cost_per_mwh": 45.15,
    "delta_resources": {
      "solar": 1.51,
      "wind": 45.3,
      "clean_firm": 1.96
    },
    "demand_twh": 373.261,
    "target_year": 2038,
    "gas_backup_mw_end": 38212,
    "delta_gas_mw": -1582,
    "new_gas_mw_end": 6212,
    "queue_position": 39
  },
  {
    "iso": "ERCOT",
    "zone_label": "87.5\u219290%",
    "threshold_start": 87.5,
    "threshold_end": 90,
    "marginal_mac": 287.46,
    "co2_displaced_mt": 8.1055,
    "delta_cost_per_mwh": 50.17,
    "delta_resources": {
      "wind": 26.08,
      "clean_firm": 20.37
    },
    "demand_twh": 817.604,
    "target_year": 2040,
    "gas_backup_mw_end": 91583,
    "delta_gas_mw": 1475,
    "new_gas_mw_end": 36583,
    "queue_position": 40
  },
  {
    "iso": "NEISO",
    "zone_label": "80\u219285%",
    "threshold_start": 80,
    "threshold_end": 85,
    "marginal_mac": 288.42,
    "co2_displaced_mt": 3.3093,
    "delta_cost_per_mwh": 83.89,
    "delta_resources": {
      "solar": 4.53,
      "wind": 3.94,
      "offshore_wind": 2.91
    },
    "demand_twh": 145.441,
    "target_year": 2038,
    "gas_backup_mw_end": 13790,
    "delta_gas_mw": 232,
    "new_gas_mw_end": 0,
    "queue_position": 41
  },
  {
    "iso": "NEISO",
    "zone_label": "70\u219275%",
    "threshold_start": 70,
    "threshold_end": 75,
    "marginal_mac": 290.11,
    "co2_displaced_mt": 2.9057,
    "delta_cost_per_mwh": 83.28,
    "delta_resources": {
      "solar": 0.75,
      "wind": 4.46,
      "offshore_wind": 4.68
    },
    "demand_twh": 140.343,
    "target_year": 2036,
    "gas_backup_mw_end": 13972,
    "delta_gas_mw": -89,
    "new_gas_mw_end": 0,
    "queue_position": 42
  },
  {
    "iso": "PJM",
    "zone_label": "60\u219265%",
    "threshold_start": 60,
    "threshold_end": 65,
    "marginal_mac": 291.55,
    "co2_displaced_mt": 23.3633,
    "delta_cost_per_mwh": 71.25,
    "delta_resources": {
      "wind": 92.76,
      "clean_firm": 0.97,
      "hydro": 1.38
    },
    "demand_twh": 1043.993,
    "target_year": 2034,
    "gas_backup_mw_end": 104318,
    "delta_gas_mw": 4366,
    "new_gas_mw_end": 29318,
    "queue_position": 43
  },
  {
    "iso": "NEISO",
    "zone_label": "50\u219255%",
    "threshold_start": 50,
    "threshold_end": 55,
    "marginal_mac": 295.17,
    "co2_displaced_mt": 2.1959,
    "delta_cost_per_mwh": 89.57,
    "delta_resources": {
      "wind": 1.96,
      "hydro": 1.05,
      "offshore_wind": 4.15
    },
    "demand_twh": 128.366,
    "target_year": 2031,
    "gas_backup_mw_end": 13713,
    "delta_gas_mw": -71,
    "new_gas_mw_end": 0,
    "queue_position": 44
  },
  {
    "iso": "CAISO",
    "zone_label": "65\u219270%",
    "threshold_start": 65,
    "threshold_end": 70,
    "marginal_mac": 308.94,
    "co2_displaced_mt": 5.6872,
    "delta_cost_per_mwh": 98.62,
    "delta_resources": {
      "solar": 16.38,
      "hydro": 0.71
    },
    "demand_twh": 270.437,
    "target_year": 2035,
    "gas_backup_mw_end": 43487,
    "delta_gas_mw": 859,
    "new_gas_mw_end": 6487,
    "queue_position": 45
  },
  {
    "iso": "PJM",
    "zone_label": "92.5\u219295%",
    "threshold_start": 92.5,
    "threshold_end": 95,
    "marginal_mac": 310.6,
    "co2_displaced_mt": 14.9617,
    "delta_cost_per_mwh": 71.02,
    "delta_resources": {
      "wind": 28.19,
      "clean_firm": 34.9,
      "hydro": 2.35
    },
    "demand_twh": 1355.181,
    "target_year": 2045,
    "gas_backup_mw_end": 134767,
    "delta_gas_mw": 8188,
    "new_gas_mw_end": 59767,
    "queue_position": 46
  },
  {
    "iso": "NYISO",
    "zone_label": "60\u219265%",
    "threshold_start": 60,
    "threshold_end": 65,
    "marginal_mac": 328.76,
    "co2_displaced_mt": 3.9981,
    "delta_cost_per_mwh": 94.75,
    "delta_resources": {
      "wind": 9.18,
      "clean_firm": 0.53,
      "offshore_wind": 3.66
    },
    "demand_twh": 181.175,
    "target_year": 2034,
    "gas_backup_mw_end": 19123,
    "delta_gas_mw": 88,
    "new_gas_mw_end": 1123,
    "queue_position": 47
  },
  {
    "iso": "PJM",
    "zone_label": "90\u219292.5%",
    "threshold_start": 90,
    "threshold_end": 92.5,
    "marginal_mac": 335.96,
    "co2_displaced_mt": 16.2496,
    "delta_cost_per_mwh": 68.05,
    "delta_resources": {
      "solar": 77.99,
      "hydro": 2.24
    },
    "demand_twh": 1292.401,
    "target_year": 2043,
    "gas_backup_mw_end": 126579,
    "delta_gas_mw": 18725,
    "new_gas_mw_end": 51579,
    "queue_position": 48
  },
  {
    "iso": "MISO",
    "zone_label": "80\u219285%",
    "threshold_start": 80,
    "threshold_end": 85,
    "marginal_mac": 337.6,
    "co2_displaced_mt": 18.6766,
    "delta_cost_per_mwh": 55.1,
    "delta_resources": {
      "solar": 3.54,
      "wind": 88.78,
      "clean_firm": 22.11
    },
    "demand_twh": 875.801,
    "target_year": 2038,
    "gas_backup_mw_end": 86185,
    "delta_gas_mw": -2623,
    "new_gas_mw_end": 18185,
    "queue_position": 49
  },
  {
    "iso": "PJM",
    "zone_label": "75\u219280%",
    "threshold_start": 75,
    "threshold_end": 80,
    "marginal_mac": 342.98,
    "co2_displaced_mt": 22.4849,
    "delta_cost_per_mwh": 73.15,
    "delta_resources": {
      "solar": 48.53,
      "wind": 18.94,
      "clean_firm": 36.9,
      "offshore_wind": 1.05
    },
    "demand_twh": 1120.979,
    "target_year": 2037,
    "gas_backup_mw_end": 99823,
    "delta_gas_mw": -889,
    "new_gas_mw_end": 24823,
    "queue_position": 50
  },
  {
    "iso": "MISO",
    "zone_label": "75\u219280%",
    "threshold_start": 75,
    "threshold_end": 80,
    "marginal_mac": 356.49,
    "co2_displaced_mt": 20.0455,
    "delta_cost_per_mwh": 55.35,
    "delta_resources": {
      "solar": 3.46,
      "wind": 104.01,
      "clean_firm": 21.63
    },
    "demand_twh": 856.948,
    "target_year": 2037,
    "gas_backup_mw_end": 88808,
    "delta_gas_mw": -4614,
    "new_gas_mw_end": 20808,
    "queue_position": 51
  },
  {
    "iso": "PJM",
    "zone_label": "55\u219260%",
    "threshold_start": 55,
    "threshold_end": 60,
    "marginal_mac": 357.89,
    "co2_displaced_mt": 17.5228,
    "delta_cost_per_mwh": 83.14,
    "delta_resources": {
      "wind": 43.36,
      "clean_firm": 31.12,
      "offshore_wind": 0.94
    },
    "demand_twh": 1019.525,
    "target_year": 2033,
    "gas_backup_mw_end": 99952,
    "delta_gas_mw": 6207,
    "new_gas_mw_end": 24952,
    "queue_position": 52
  },
  {
    "iso": "MISO",
    "zone_label": "87.5\u219290%",
    "threshold_start": 87.5,
    "threshold_end": 90,
    "marginal_mac": 359.84,
    "co2_displaced_mt": 8.9569,
    "delta_cost_per_mwh": 52.62,
    "delta_resources": {
      "solar": 2.2,
      "wind": 51.34,
      "clean_firm": 6.98,
      "hydro": 0.73
    },
    "demand_twh": 914.76,
    "target_year": 2040,
    "gas_backup_mw_end": 89997,
    "delta_gas_mw": 1470,
    "new_gas_mw_end": 21997,
    "queue_position": 53
  },
  {
    "iso": "SPP",
    "zone_label": "65\u219280%",
    "threshold_start": 65,
    "threshold_end": 80,
    "marginal_mac": 360.95,
    "co2_displaced_mt": 22.2137,
    "delta_cost_per_mwh": 59.64,
    "delta_resources": {
      "solar": 133.64,
      "hydro": 0.81
    },
    "demand_twh": 366.661,
    "target_year": 2037,
    "gas_backup_mw_end": 39794,
    "delta_gas_mw": 207,
    "new_gas_mw_end": 7794,
    "queue_position": 54
  },
  {
    "iso": "ERCOT",
    "zone_label": "80\u219285%",
    "threshold_start": 80,
    "threshold_end": 85,
    "marginal_mac": 373.73,
    "co2_displaced_mt": 15.4992,
    "delta_cost_per_mwh": 47.74,
    "delta_resources": {
      "wind": 107.37,
      "clean_firm": 13.64
    },
    "demand_twh": 763.242,
    "target_year": 2038,
    "gas_backup_mw_end": 84740,
    "delta_gas_mw": -1462,
    "new_gas_mw_end": 29740,
    "queue_position": 55
  },
  {
    "iso": "MISO",
    "zone_label": "55\u219275%",
    "threshold_start": 55,
    "threshold_end": 75,
    "marginal_mac": 381.49,
    "co2_displaced_mt": 66.0935,
    "delta_cost_per_mwh": 66.0,
    "delta_resources": {
      "solar": 382.03
    },
    "demand_twh": 838.501,
    "target_year": 2036,
    "gas_backup_mw_end": 93422,
    "delta_gas_mw": 8823,
    "new_gas_mw_end": 25422,
    "queue_position": 56
  },
  {
    "iso": "ERCOT",
    "zone_label": "97.5\u219299%",
    "threshold_start": 97.5,
    "threshold_end": 99,
    "marginal_mac": 387.22,
    "co2_displaced_mt": 8.0222,
    "delta_cost_per_mwh": 54.73,
    "delta_resources": {
      "solar": 29.67,
      "wind": 9.29,
      "clean_firm": 17.79
    },
    "demand_twh": 1114.31,
    "target_year": 2049,
    "gas_backup_mw_end": 125492,
    "delta_gas_mw": 4004,
    "new_gas_mw_end": 70492,
    "queue_position": 57
  },
  {
    "iso": "NYISO",
    "zone_label": "55\u219260%",
    "threshold_start": 55,
    "threshold_end": 60,
    "marginal_mac": 392.66,
    "co2_displaced_mt": 3.706,
    "delta_cost_per_mwh": 84.47,
    "delta_resources": {
      "solar": 4.0,
      "wind": 7.79,
      "clean_firm": 2.65,
      "hydro": 1.02,
      "offshore_wind": 1.78
    },
    "demand_twh": 177.622,
    "target_year": 2033,
    "gas_backup_mw_end": 19035,
    "delta_gas_mw": 700,
    "new_gas_mw_end": 1035,
    "queue_position": 58
  },
  {
    "iso": "CAISO",
    "zone_label": "50\u219255%",
    "threshold_start": 50,
    "threshold_end": 55,
    "marginal_mac": 414.21,
    "co2_displaced_mt": 4.9539,
    "delta_cost_per_mwh": 133.12,
    "delta_resources": {
      "solar": 1.4,
      "wind": 12.93
    },
    "demand_twh": 250.824,
    "target_year": 2031,
    "gas_backup_mw_end": 41222,
    "delta_gas_mw": 286,
    "new_gas_mw_end": 4222,
    "queue_position": 59
  },
  {
    "iso": "CAISO",
    "zone_label": "55\u219260%",
    "threshold_start": 55,
    "threshold_end": 60,
    "marginal_mac": 428.7,
    "co2_displaced_mt": 5.1032,
    "delta_cost_per_mwh": 112.69,
    "delta_resources": {
      "solar": 2.88,
      "wind": 14.3,
      "clean_firm": 0.8,
      "hydro": 0.94
    },
    "demand_twh": 260.446,
    "target_year": 2033,
    "gas_backup_mw_end": 42420,
    "delta_gas_mw": 1198,
    "new_gas_mw_end": 5420,
    "queue_position": 60
  },
  {
    "iso": "NYISO",
    "zone_label": "65\u219270%",
    "threshold_start": 65,
    "threshold_end": 70,
    "marginal_mac": 437.47,
    "co2_displaced_mt": 4.101,
    "delta_cost_per_mwh": 92.59,
    "delta_resources": {
      "solar": 6.45,
      "wind": 4.18,
      "hydro": 0.89,
      "offshore_wind": 7.5
    },
    "demand_twh": 184.798,
    "target_year": 2035,
    "gas_backup_mw_end": 19119,
    "delta_gas_mw": -4,
    "new_gas_mw_end": 1119,
    "queue_position": 61
  },
  {
    "iso": "CAISO",
    "zone_label": "95\u219297.5%",
    "threshold_start": 95,
    "threshold_end": 97.5,
    "marginal_mac": 438.17,
    "co2_displaced_mt": 3.6567,
    "delta_cost_per_mwh": 73.0,
    "delta_resources": {
      "clean_firm": 21.95
    },
    "demand_twh": 345.406,
    "target_year": 2048,
    "gas_backup_mw_end": 39677,
    "delta_gas_mw": 718,
    "new_gas_mw_end": 2677,
    "queue_position": 62
  },
  {
    "iso": "NEISO",
    "zone_label": "55\u219260%",
    "threshold_start": 55,
    "threshold_end": 60,
    "marginal_mac": 448.2,
    "co2_displaced_mt": 2.0517,
    "delta_cost_per_mwh": 105.28,
    "delta_resources": {
      "solar": 0.78,
      "wind": 3.53,
      "clean_firm": 2.35,
      "offshore_wind": 2.08
    },
    "demand_twh": 133.029,
    "target_year": 2033,
    "gas_backup_mw_end": 14319,
    "delta_gas_mw": 606,
    "new_gas_mw_end": 319,
    "queue_position": 63
  },
  {
    "iso": "NYISO",
    "zone_label": "95\u219297.5%",
    "threshold_start": 95,
    "threshold_end": 97.5,
    "marginal_mac": 460.88,
    "co2_displaced_mt": 2.6368,
    "delta_cost_per_mwh": 80.0,
    "delta_resources": {
      "clean_firm": 15.19
    },
    "demand_twh": 239.056,
    "target_year": 2048,
    "gas_backup_mw_end": 25119,
    "delta_gas_mw": 1334,
    "new_gas_mw_end": 7119,
    "queue_position": 64
  },
  {
    "iso": "ERCOT",
    "zone_label": "85\u219287.5%",
    "threshold_start": 85,
    "threshold_end": 87.5,
    "marginal_mac": 471.36,
    "co2_displaced_mt": 7.7692,
    "delta_cost_per_mwh": 57.0,
    "delta_resources": {
      "solar": 64.25
    },
    "demand_twh": 789.955,
    "target_year": 2039,
    "gas_backup_mw_end": 90108,
    "delta_gas_mw": 5368,
    "new_gas_mw_end": 35108,
    "queue_position": 65
  },
  {
    "iso": "NYISO",
    "zone_label": "90\u219292.5%",
    "threshold_start": 90,
    "threshold_end": 92.5,
    "marginal_mac": 492.26,
    "co2_displaced_mt": 2.6004,
    "delta_cost_per_mwh": 76.09,
    "delta_resources": {
      "wind": 5.11,
      "clean_firm": 5.61,
      "hydro": 1.78,
      "offshore_wind": 4.33
    },
    "demand_twh": 216.521,
    "target_year": 2043,
    "gas_backup_mw_end": 21982,
    "delta_gas_mw": 1479,
    "new_gas_mw_end": 3982,
    "queue_position": 66
  },
  {
    "iso": "CAISO",
    "zone_label": "60\u219265%",
    "threshold_start": 60,
    "threshold_end": 65,
    "marginal_mac": 510.99,
    "co2_displaced_mt": 5.5531,
    "delta_cost_per_mwh": 94.35,
    "delta_resources": {
      "solar": 14.75,
      "wind": 14.18
    },
    "demand_twh": 265.394,
    "target_year": 2034,
    "gas_backup_mw_end": 42628,
    "delta_gas_mw": 208,
    "new_gas_mw_end": 5628,
    "queue_position": 67
  },
  {
    "iso": "CAISO",
    "zone_label": "97.5\u219299%",
    "threshold_start": 97.5,
    "threshold_end": 99,
    "marginal_mac": 536.84,
    "co2_displaced_mt": 3.0413,
    "delta_cost_per_mwh": 73.0,
    "delta_resources": {
      "clean_firm": 22.37
    },
    "demand_twh": 351.969,
    "target_year": 2049,
    "gas_backup_mw_end": 37722,
    "delta_gas_mw": -1955,
    "new_gas_mw_end": 722,
    "queue_position": 68
  },
  {
    "iso": "CAISO",
    "zone_label": "70\u219275%",
    "threshold_start": 70,
    "threshold_end": 75,
    "marginal_mac": 537.67,
    "co2_displaced_mt": 5.626,
    "delta_cost_per_mwh": 95.09,
    "delta_resources": {
      "solar": 14.74,
      "wind": 2.37,
      "geothermal": 14.03
    },
    "demand_twh": 275.575,
    "target_year": 2036,
    "gas_backup_mw_end": 44435,
    "delta_gas_mw": 948,
    "new_gas_mw_end": 7435,
    "queue_position": 69
  },
  {
    "iso": "SPP",
    "zone_label": "90\u219292.5%",
    "threshold_start": 90,
    "threshold_end": 92.5,
    "marginal_mac": 538.45,
    "co2_displaced_mt": 4.0033,
    "delta_cost_per_mwh": 46.82,
    "delta_resources": {
      "solar": 4.23,
      "wind": 35.28,
      "clean_firm": 6.53
    },
    "demand_twh": 408.086,
    "target_year": 2043,
    "gas_backup_mw_end": 41784,
    "delta_gas_mw": 2073,
    "new_gas_mw_end": 9784,
    "queue_position": 70
  },
  {
    "iso": "SPP",
    "zone_label": "92.5\u219295%",
    "threshold_start": 92.5,
    "threshold_end": 95,
    "marginal_mac": 540.7,
    "co2_displaced_mt": 4.1487,
    "delta_cost_per_mwh": 44.59,
    "delta_resources": {
      "solar": 1.3,
      "wind": 38.16,
      "clean_firm": 9.08,
      "hydro": 1.76
    },
    "demand_twh": 422.909,
    "target_year": 2045,
    "gas_backup_mw_end": 42293,
    "delta_gas_mw": 509,
    "new_gas_mw_end": 10293,
    "queue_position": 71
  },
  {
    "iso": "CAISO",
    "zone_label": "75\u219280%",
    "threshold_start": 75,
    "threshold_end": 80,
    "marginal_mac": 573.03,
    "co2_displaced_mt": 5.6121,
    "delta_cost_per_mwh": 168.46,
    "delta_resources": {
      "solar": 2.35,
      "wind": 1.22,
      "hydro": 0.51,
      "geothermal": 14.57
    },
    "demand_twh": 280.811,
    "target_year": 2037,
    "gas_backup_mw_end": 45410,
    "delta_gas_mw": 975,
    "new_gas_mw_end": 8410,
    "queue_position": 72
  },
  {
    "iso": "NYISO",
    "zone_label": "97.5\u219299%",
    "threshold_start": 97.5,
    "threshold_end": 99,
    "marginal_mac": 589.7,
    "co2_displaced_mt": 2.102,
    "delta_cost_per_mwh": 80.0,
    "delta_resources": {
      "clean_firm": 5.74,
      "offshore_wind": 9.75
    },
    "demand_twh": 243.837,
    "target_year": 2049,
    "gas_backup_mw_end": 24865,
    "delta_gas_mw": -254,
    "new_gas_mw_end": 6865,
    "queue_position": 73
  },
  {
    "iso": "NYISO",
    "zone_label": "70\u219275%",
    "threshold_start": 70,
    "threshold_end": 75,
    "marginal_mac": 667.45,
    "co2_displaced_mt": 4.0892,
    "delta_cost_per_mwh": 97.94,
    "delta_resources": {
      "solar": 23.21,
      "wind": 2.51,
      "clean_firm": -1.26,
      "hydro": -1.26,
      "offshore_wind": 2.14
    },
    "demand_twh": 188.494,
    "target_year": 2036,
    "gas_backup_mw_end": 20362,
    "delta_gas_mw": 1243,
    "new_gas_mw_end": 2362,
    "queue_position": 74
  },
  {
    "iso": "MISO",
    "zone_label": "90\u219292.5%",
    "threshold_start": 90,
    "threshold_end": 92.5,
    "marginal_mac": 684.97,
    "co2_displaced_mt": 9.523,
    "delta_cost_per_mwh": 57.06,
    "delta_resources": {
      "wind": 15.22,
      "clean_firm": 97.97,
      "hydro": 1.12
    },
    "demand_twh": 976.473,
    "target_year": 2043,
    "gas_backup_mw_end": 87254,
    "delta_gas_mw": -2743,
    "new_gas_mw_end": 19254,
    "queue_position": 75
  },
  {
    "iso": "SPP",
    "zone_label": "95\u219297.5%",
    "threshold_start": 95,
    "threshold_end": 97.5,
    "marginal_mac": 760.23,
    "co2_displaced_mt": 4.3768,
    "delta_cost_per_mwh": 45.8,
    "delta_resources": {
      "wind": 52.77,
      "clean_firm": 18.8,
      "hydro": 1.09
    },
    "demand_twh": 446.16,
    "target_year": 2048,
    "gas_backup_mw_end": 42983,
    "delta_gas_mw": 690,
    "new_gas_mw_end": 10983,
    "queue_position": 76
  },
  {
    "iso": "ERCOT",
    "zone_label": "90\u219292.5%",
    "threshold_start": 90,
    "threshold_end": 92.5,
    "marginal_mac": 771.55,
    "co2_displaced_mt": 8.9153,
    "delta_cost_per_mwh": 51.53,
    "delta_resources": {
      "solar": 2.79,
      "wind": 56.27,
      "clean_firm": 74.44
    },
    "demand_twh": 906.492,
    "target_year": 2043,
    "gas_backup_mw_end": 96103,
    "delta_gas_mw": 4520,
    "new_gas_mw_end": 41103,
    "queue_position": 77
  },
  {
    "iso": "NEISO",
    "zone_label": "85\u219287.5%",
    "threshold_start": 85,
    "threshold_end": 87.5,
    "marginal_mac": 789.57,
    "co2_displaced_mt": 1.5129,
    "delta_cost_per_mwh": 82.12,
    "delta_resources": {
      "solar": 3.35,
      "wind": 3.92,
      "clean_firm": 1.49,
      "offshore_wind": 5.79
    },
    "demand_twh": 148.059,
    "target_year": 2039,
    "gas_backup_mw_end": 13334,
    "delta_gas_mw": -456,
    "new_gas_mw_end": 0,
    "queue_position": 78
  },
  {
    "iso": "NEISO",
    "zone_label": "90\u219292.5%",
    "threshold_start": 90,
    "threshold_end": 92.5,
    "marginal_mac": 810.83,
    "co2_displaced_mt": 1.5822,
    "delta_cost_per_mwh": 78.9,
    "delta_resources": {
      "solar": 1.55,
      "wind": 2.05,
      "clean_firm": 10.18,
      "offshore_wind": 2.49
    },
    "demand_twh": 159.01,
    "target_year": 2043,
    "gas_backup_mw_end": 14195,
    "delta_gas_mw": 387,
    "new_gas_mw_end": 195,
    "queue_position": 79
  },
  {
    "iso": "CAISO",
    "zone_label": "80\u219285%",
    "threshold_start": 80,
    "threshold_end": 85,
    "marginal_mac": 829.49,
    "co2_displaced_mt": 6.3342,
    "delta_cost_per_mwh": 146.65,
    "delta_resources": {
      "solar": 3.69,
      "wind": 0.75,
      "clean_firm": 29.55,
      "hydro": 1.04,
      "geothermal": 0.8
    },
    "demand_twh": 286.146,
    "target_year": 2038,
    "gas_backup_mw_end": 42197,
    "delta_gas_mw": -3213,
    "new_gas_mw_end": 5197,
    "queue_position": 80
  },
  {
    "iso": "NEISO",
    "zone_label": "60\u219265%",
    "threshold_start": 60,
    "threshold_end": 65,
    "marginal_mac": 975.94,
    "co2_displaced_mt": 1.2749,
    "delta_cost_per_mwh": 107.87,
    "delta_resources": {
      "solar": 3.37,
      "wind": -0.51,
      "clean_firm": 5.85,
      "hydro": 0.55,
      "offshore_wind": 1.76
    },
    "demand_twh": 135.424,
    "target_year": 2034,
    "gas_backup_mw_end": 13825,
    "delta_gas_mw": -494,
    "new_gas_mw_end": 0,
    "queue_position": 81
  },
  {
    "iso": "MISO",
    "zone_label": "92.5\u219295%",
    "threshold_start": 92.5,
    "threshold_end": 95,
    "marginal_mac": 1008.43,
    "co2_displaced_mt": 10.5435,
    "delta_cost_per_mwh": 60.91,
    "delta_resources": {
      "solar": 30.64,
      "wind": 58.24,
      "clean_firm": 85.68
    },
    "demand_twh": 1019.91,
    "target_year": 2045,
    "gas_backup_mw_end": 82197,
    "delta_gas_mw": -5057,
    "new_gas_mw_end": 14197,
    "queue_position": 82
  },
  {
    "iso": "CAISO",
    "zone_label": "85\u219287.5%",
    "threshold_start": 85,
    "threshold_end": 87.5,
    "marginal_mac": 1088.51,
    "co2_displaced_mt": 3.6036,
    "delta_cost_per_mwh": 208.25,
    "delta_resources": {
      "solar": 1.15,
      "wind": 1.76,
      "clean_firm": 0.51,
      "geothermal": 15.4
    },
    "demand_twh": 291.583,
    "target_year": 2039,
    "gas_backup_mw_end": 43314,
    "delta_gas_mw": 1117,
    "new_gas_mw_end": 6314,
    "queue_position": 83
  },
  {
    "iso": "MISO",
    "zone_label": "99.5\u219299.9%",
    "threshold_start": 99.5,
    "threshold_end": 99.9,
    "marginal_mac": 1106.86,
    "co2_displaced_mt": 1.8188,
    "delta_cost_per_mwh": 67.84,
    "delta_resources": {
      "solar": 15.69,
      "wind": 2.07,
      "clean_firm": 11.91
    },
    "demand_twh": 1137.146,
    "target_year": 2050,
    "gas_backup_mw_end": 68262,
    "delta_gas_mw": 3321,
    "new_gas_mw_end": 262,
    "queue_position": 84
  },
  {
    "iso": "SPP",
    "zone_label": "99\u219299.5%",
    "threshold_start": 99,
    "threshold_end": 99.5,
    "marginal_mac": 1181.02,
    "co2_displaced_mt": 1.1406,
    "delta_cost_per_mwh": 60.41,
    "delta_resources": {
      "solar": 5.95,
      "clean_firm": 16.35
    },
    "demand_twh": 454.191,
    "target_year": 2049,
    "gas_backup_mw_end": 40100,
    "delta_gas_mw": -2453,
    "new_gas_mw_end": 8100,
    "queue_position": 85
  },
  {
    "iso": "NEISO",
    "zone_label": "95\u219297.5%",
    "threshold_start": 95,
    "threshold_end": 97.5,
    "marginal_mac": 1192.05,
    "co2_displaced_mt": 1.7963,
    "delta_cost_per_mwh": 77.91,
    "delta_resources": {
      "solar": 1.28,
      "wind": 1.7,
      "clean_firm": 21.79,
      "offshore_wind": 2.72
    },
    "demand_twh": 173.846,
    "target_year": 2048,
    "gas_backup_mw_end": 10405,
    "delta_gas_mw": -1159,
    "new_gas_mw_end": 0,
    "queue_position": 86
  },
  {
    "iso": "CAISO",
    "zone_label": "87.5\u219290%",
    "threshold_start": 87.5,
    "threshold_end": 90,
    "marginal_mac": 1238.15,
    "co2_displaced_mt": 3.9393,
    "delta_cost_per_mwh": 228.92,
    "delta_resources": {
      "solar": 2.49,
      "wind": 1.29,
      "clean_firm": 15.87,
      "hydro": 0.54,
      "geothermal": 1.11
    },
    "demand_twh": 297.123,
    "target_year": 2040,
    "gas_backup_mw_end": 42206,
    "delta_gas_mw": -1108,
    "new_gas_mw_end": 5206,
    "queue_position": 87
  },
  {
    "iso": "NEISO",
    "zone_label": "92.5\u219295%",
    "threshold_start": 92.5,
    "threshold_end": 95,
    "marginal_mac": 1249.58,
    "co2_displaced_mt": 1.8793,
    "delta_cost_per_mwh": 77.52,
    "delta_resources": {
      "solar": 0.81,
      "wind": 1.08,
      "clean_firm": 26.66,
      "offshore_wind": 1.73
    },
    "demand_twh": 164.786,
    "target_year": 2045,
    "gas_backup_mw_end": 11564,
    "delta_gas_mw": -2631,
    "new_gas_mw_end": 0,
    "queue_position": 88
  },
  {
    "iso": "SPP",
    "zone_label": "99.5\u219299.9%",
    "threshold_start": 99.5,
    "threshold_end": 99.9,
    "marginal_mac": 1278.73,
    "co2_displaced_mt": 0.7439,
    "delta_cost_per_mwh": 54.75,
    "delta_resources": {
      "solar": 5.24,
      "wind": 8.18,
      "clean_firm": 3.96
    },
    "demand_twh": 462.366,
    "target_year": 2050,
    "gas_backup_mw_end": 41060,
    "delta_gas_mw": 960,
    "new_gas_mw_end": 9060,
    "queue_position": 89
  },
  {
    "iso": "ERCOT",
    "zone_label": "95\u219297.5%",
    "threshold_start": 95,
    "threshold_end": 97.5,
    "marginal_mac": 1307.93,
    "co2_displaced_mt": 12.3675,
    "delta_cost_per_mwh": 58.06,
    "delta_resources": {
      "solar": 263.97,
      "clean_firm": 14.62
    },
    "demand_twh": 1076.628,
    "target_year": 2048,
    "gas_backup_mw_end": 121488,
    "delta_gas_mw": 15619,
    "new_gas_mw_end": 66488,
    "queue_position": 90
  },
  {
    "iso": "ERCOT",
    "zone_label": "92.5\u219295%",
    "threshold_start": 92.5,
    "threshold_end": 95,
    "marginal_mac": 1424.41,
    "co2_displaced_mt": 10.0852,
    "delta_cost_per_mwh": 50.74,
    "delta_resources": {
      "solar": 121.88,
      "wind": 161.26
    },
    "demand_twh": 971.057,
    "target_year": 2045,
    "gas_backup_mw_end": 105869,
    "delta_gas_mw": 9766,
    "new_gas_mw_end": 50869,
    "queue_position": 91
  },
  {
    "iso": "PJM",
    "zone_label": "99\u219299.5%",
    "threshold_start": 99,
    "threshold_end": 99.5,
    "marginal_mac": 1536.62,
    "co2_displaced_mt": 3.128,
    "delta_cost_per_mwh": 74.08,
    "delta_resources": {
      "clean_firm": 35.08,
      "offshore_wind": 29.8
    },
    "demand_twh": 1490.037,
    "target_year": 2049,
    "gas_backup_mw_end": 147061,
    "delta_gas_mw": -9958,
    "new_gas_mw_end": 72061,
    "queue_position": 92
  },
  {
    "iso": "MISO",
    "zone_label": "95\u219297.5%",
    "threshold_start": 95,
    "threshold_end": 97.5,
    "marginal_mac": 1572.96,
    "co2_displaced_mt": 10.7876,
    "delta_cost_per_mwh": 56.55,
    "delta_resources": {
      "solar": 31.04,
      "wind": 224.79,
      "clean_firm": 44.26
    },
    "demand_twh": 1088.716,
    "target_year": 2048,
    "gas_backup_mw_end": 86006,
    "delta_gas_mw": 3809,
    "new_gas_mw_end": 18006,
    "queue_position": 93
  },
  {
    "iso": "PJM",
    "zone_label": "99.5\u219299.9%",
    "threshold_start": 99.5,
    "threshold_end": 99.9,
    "marginal_mac": 1597.18,
    "co2_displaced_mt": 2.9405,
    "delta_cost_per_mwh": 68.53,
    "delta_resources": {
      "clean_firm": 32.92,
      "hydro": 5.1,
      "offshore_wind": 30.52
    },
    "demand_twh": 1525.798,
    "target_year": 2050,
    "gas_backup_mw_end": 143524,
    "delta_gas_mw": -3537,
    "new_gas_mw_end": 68524,
    "queue_position": 94
  },
  {
    "iso": "SPP",
    "zone_label": "99.9\u219299.99%",
    "threshold_start": 99.9,
    "threshold_end": 99.99,
    "marginal_mac": 1650.01,
    "co2_displaced_mt": 0.1633,
    "delta_cost_per_mwh": 60.0,
    "delta_resources": {
      "solar": 4.49
    },
    "demand_twh": 462.366,
    "target_year": 2050,
    "gas_backup_mw_end": 41060,
    "delta_gas_mw": 0,
    "new_gas_mw_end": 9060,
    "queue_position": 95
  },
  {
    "iso": "SPP",
    "zone_label": "97.5\u219299%",
    "threshold_start": 97.5,
    "threshold_end": 99,
    "marginal_mac": 1660.45,
    "co2_displaced_mt": 2.7625,
    "delta_cost_per_mwh": 45.95,
    "delta_resources": {
      "solar": 11.19,
      "wind": 86.95,
      "clean_firm": 1.67
    },
    "demand_twh": 454.191,
    "target_year": 2049,
    "gas_backup_mw_end": 42553,
    "delta_gas_mw": -430,
    "new_gas_mw_end": 10553,
    "queue_position": 96
  },
  {
    "iso": "CAISO",
    "zone_label": "90\u219292.5%",
    "threshold_start": 90,
    "threshold_end": 92.5,
    "marginal_mac": 1761.29,
    "co2_displaced_mt": 4.2742,
    "delta_cost_per_mwh": 78.09,
    "delta_resources": {
      "wind": 44.99,
      "clean_firm": 32.55,
      "offshore_wind": 18.86,
      "geothermal": -59.42
    },
    "demand_twh": 314.383,
    "target_year": 2043,
    "gas_backup_mw_end": 38092,
    "delta_gas_mw": -4114,
    "new_gas_mw_end": 1092,
    "queue_position": 97
  },
  {
    "iso": "CAISO",
    "zone_label": "99\u219299.5%",
    "threshold_start": 99,
    "threshold_end": 99.5,
    "marginal_mac": 1952.53,
    "co2_displaced_mt": 1.3628,
    "delta_cost_per_mwh": 63.0,
    "delta_resources": {
      "solar": 42.24
    },
    "demand_twh": 351.969,
    "target_year": 2049,
    "gas_backup_mw_end": 37722,
    "delta_gas_mw": 0,
    "new_gas_mw_end": 722,
    "queue_position": 98
  },
  {
    "iso": "NEISO",
    "zone_label": "97.5\u219299%",
    "threshold_start": 97.5,
    "threshold_end": 99,
    "marginal_mac": 2077.14,
    "co2_displaced_mt": 1.0295,
    "delta_cost_per_mwh": 79.76,
    "delta_resources": {
      "wind": 9.43,
      "clean_firm": 10.69,
      "offshore_wind": 6.25
    },
    "demand_twh": 176.975,
    "target_year": 2049,
    "gas_backup_mw_end": 8652,
    "delta_gas_mw": -1753,
    "new_gas_mw_end": 0,
    "queue_position": 99
  },
  {
    "iso": "ERCOT",
    "zone_label": "99.5\u219299.9%",
    "threshold_start": 99.5,
    "threshold_end": 99.9,
    "marginal_mac": 2154.4,
    "co2_displaced_mt": 2.1778,
    "delta_cost_per_mwh": 51.08,
    "delta_resources": {
      "solar": 42.39,
      "wind": 49.47
    },
    "demand_twh": 1153.311,
    "target_year": 2050,
    "gas_backup_mw_end": 131880,
    "delta_gas_mw": 6499,
    "new_gas_mw_end": 76880,
    "queue_position": 100
  },
  {
    "iso": "ERCOT",
    "zone_label": "99.9\u219299.99%",
    "threshold_start": 99.9,
    "threshold_end": 99.99,
    "marginal_mac": 2322.79,
    "co2_displaced_mt": 0.4083,
    "delta_cost_per_mwh": 47.85,
    "delta_resources": {
      "solar": 3.34,
      "wind": 16.48
    },
    "demand_twh": 1153.311,
    "target_year": 2050,
    "gas_backup_mw_end": 131697,
    "delta_gas_mw": -183,
    "new_gas_mw_end": 76697,
    "queue_position": 101
  },
  {
    "iso": "NYISO",
    "zone_label": "99\u219299.5%",
    "threshold_start": 99,
    "threshold_end": 99.5,
    "marginal_mac": 3490.45,
    "co2_displaced_mt": 0.8299,
    "delta_cost_per_mwh": 99.0,
    "delta_resources": {
      "solar": 29.26
    },
    "demand_twh": 243.837,
    "target_year": 2049,
    "gas_backup_mw_end": 24865,
    "delta_gas_mw": 0,
    "new_gas_mw_end": 6865,
    "queue_position": 102
  },
  {
    "iso": "MISO",
    "zone_label": "97.5\u219299%",
    "threshold_start": 97.5,
    "threshold_end": 99,
    "marginal_mac": 3568.83,
    "co2_displaced_mt": 7.4657,
    "delta_cost_per_mwh": 68.97,
    "delta_resources": {
      "solar": 222.21,
      "clean_firm": 164.08
    },
    "demand_twh": 1112.668,
    "target_year": 2049,
    "gas_backup_mw_end": 65709,
    "delta_gas_mw": -20297,
    "new_gas_mw_end": 0,
    "queue_position": 103
  },
  {
    "iso": "CAISO",
    "zone_label": "99.5\u219299.9%",
    "threshold_start": 99.5,
    "threshold_end": 99.9,
    "marginal_mac": 3944.92,
    "co2_displaced_mt": 0.6873,
    "delta_cost_per_mwh": 63.0,
    "delta_resources": {
      "solar": 43.04
    },
    "demand_twh": 358.656,
    "target_year": 2050,
    "gas_backup_mw_end": 38913,
    "delta_gas_mw": 1191,
    "new_gas_mw_end": 1913,
    "queue_position": 104
  },
  {
    "iso": "NEISO",
    "zone_label": "99\u219299.5%",
    "threshold_start": 99,
    "threshold_end": 99.5,
    "marginal_mac": 3993.47,
    "co2_displaced_mt": 0.359,
    "delta_cost_per_mwh": 81.0,
    "delta_resources": {
      "wind": 8.85,
      "clean_firm": 8.85
    },
    "demand_twh": 176.975,
    "target_year": 2049,
    "gas_backup_mw_end": 7340,
    "delta_gas_mw": -1312,
    "new_gas_mw_end": 0,
    "queue_position": 105
  },
  {
    "iso": "NYISO",
    "zone_label": "99.5\u219299.9%",
    "threshold_start": 99.5,
    "threshold_end": 99.9,
    "marginal_mac": 5844.96,
    "co2_displaced_mt": 0.5055,
    "delta_cost_per_mwh": 99.0,
    "delta_resources": {
      "solar": 29.84
    },
    "demand_twh": 248.714,
    "target_year": 2050,
    "gas_backup_mw_end": 26154,
    "delta_gas_mw": 1289,
    "new_gas_mw_end": 8154,
    "queue_position": 106
  },
  {
    "iso": "NEISO",
    "zone_label": "99.5\u219299.9%",
    "threshold_start": 99.5,
    "threshold_end": 99.9,
    "marginal_mac": 7487.83,
    "co2_displaced_mt": 0.3342,
    "delta_cost_per_mwh": 208.04,
    "delta_resources": {
      "clean_firm": 12.03
    },
    "demand_twh": 180.16,
    "target_year": 2050,
    "gas_backup_mw_end": 6445,
    "delta_gas_mw": -895,
    "new_gas_mw_end": 0,
    "queue_position": 107
  },
  {
    "iso": "NEISO",
    "zone_label": "99.9\u219299.99%",
    "threshold_start": 99.9,
    "threshold_end": 99.99,
    "marginal_mac": 8061.71,
    "co2_displaced_mt": 0.0621,
    "delta_cost_per_mwh": 538.44,
    "delta_resources": {
      "hydro": 0.93
    },
    "demand_twh": 180.16,
    "target_year": 2050,
    "gas_backup_mw_end": 6445,
    "delta_gas_mw": 0,
    "new_gas_mw_end": 0,
    "queue_position": 108
  },
  {
    "iso": "CAISO",
    "zone_label": "99.9\u219299.99%",
    "threshold_start": 99.9,
    "threshold_end": 99.99,
    "marginal_mac": 21477.88,
    "co2_displaced_mt": 0.1262,
    "delta_cost_per_mwh": 63.0,
    "delta_resources": {
      "solar": 43.04
    },
    "demand_twh": 358.656,
    "target_year": 2050,
    "gas_backup_mw_end": 37393,
    "delta_gas_mw": -1520,
    "new_gas_mw_end": 393,
    "queue_position": 109
  },
  {
    "iso": "PJM",
    "zone_label": "99.9\u219299.99%",
    "threshold_start": 99.9,
    "threshold_end": 99.99,
    "marginal_mac": 23956.83,
    "co2_displaced_mt": 0.5401,
    "delta_cost_per_mwh": 70.67,
    "delta_resources": {
      "solar": 122.06,
      "wind": 61.03
    },
    "demand_twh": 1525.798,
    "target_year": 2050,
    "gas_backup_mw_end": 140271,
    "delta_gas_mw": -3253,
    "new_gas_mw_end": 65271,
    "queue_position": 110
  }
];

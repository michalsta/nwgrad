# Benchmarks: every cell

Generated from `tools/fleetbench/results/` by `tools/fleetbench/build_doc.py`; the method, the hosts and the reading of these numbers are in [benchmarks.md](benchmarks.md).

Mcell/s; each tool's best arm whose scores all matched nwgrad's float64 reference. (x) = nwgrad default / tool: above 1, nwgrad is faster. `-` no implementation, `>10x` stopped at 10x nwgrad's time, `wrong` scores disagreed, `err` failed; `sat` a saturating integer cascade, `‡` harness-parallel. The second column is the best of nwgrad's non-default variants.

<details><summary><b>nighthaven</b> — i5-12500 (Alder Lake), AVX2 — x86-64-v3, the primary target: every cell</summary>


#### nighthaven · dna_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **4959** | - | 4506 (1.10x) | 7261 (0.68x) | 2630 (1.89x) | 9786 (0.51x) | 8772 (0.57x) | - | - | - | - | - | - | 511 (9.71x) sat | - |
| global affine score int32 | **4011** | - | 3710 (1.08x) | 4022 (1.00x) | 2902 (1.38x) | 5112 (0.78x) | 4619 (0.87x) | - | 834 (4.81x) | 994 (4.03x) | - | - | - | - | - |
| global affine score int64 | **1389** | - | 1427 (0.97x) | 1388 (1.00x) | >10x | 653 (2.13x) | 918 (1.51x) | - | - | - | - | - | - | - | - |
| global affine score float32 | **3888** | kernel=sse4: 2310 | 3654 (1.06x) | 3850 (1.01x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| global affine score float64 | **2019** | fill=rowwise: 1704 | 1988 (1.02x) | 2018 (1.00x) | - | 325 (6.21x) | 338 (5.97x) | - | - | - | - | 372 (5.43x) | - | - | - |
| global affine path int16 | **3113** | - | 3039 (1.02x) | 3244 (0.96x) | 1784 (1.75x) | 2214 (1.41x) | 3532 (0.88x) | - | - | - | - | - | - | 372 (8.38x) sat | - |
| global affine path int32 | **2069** | - | 2018 (1.03x) | 2125 (0.97x) | 1522 (1.36x) | 998 (2.07x) | 2227 (0.93x) | - | 767 (2.70x) | 978 (2.12x) | - | - | - | - | - |
| global affine path int64 | **958** | - | 981 (0.98x) | 960 (1.00x) | 113 (8.49x) | 260 (3.69x) | 323 (2.97x) | - | - | - | - | - | - | - | - |
| global affine path float32 | **2042** | traceback=pointers: 2058 | 1943 (1.05x) | 2077 (0.98x) | - | 218 (9.36x) | 293 (6.97x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **1217** | traceback=scores: 1257 | 1124 (1.08x) | 1224 (0.99x) | - | 220 (5.55x) | 285 (4.27x) | - | - | - | - | >10x | 124 (9.80x) | - | - |
| global affine hard int16 | **4134** | - | 3680 (1.12x) | 4129 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **2441** | - | 2408 (1.01x) | 2446 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **1020** | - | 1073 (0.95x) | 1025 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **2359** | traceback=hirschberg_pmax: 2422 | 2283 (1.03x) | 2433 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **1342** | traceback=grid: 1361 | 1254 (1.07x) | 1324 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **203** | api=functions: 258 | 199 (1.02x) | 208 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **5854** | - | 4924 (1.19x) | 9299 (0.63x) | 2628 (2.23x) | 9765 (0.60x) | 12063 (0.49x) | - | - | - | - | - | - | >10x | - |
| global linear score int32 | **5164** | - | 4383 (1.18x) | 5170 (1.00x) | 2907 (1.78x) | 4833 (1.07x) | 6544 (0.79x) | - | 863 (5.98x) | 11954 (0.43x) | 5308 (0.97x) | - | - | >10x | - |
| global linear score int64 | **1385** | - | 1865 (0.74x) | 1384 (1.00x) | >10x | 653 (2.12x) | 950 (1.46x) | - | - | - | - | - | - | - | - |
| global linear score float32 | **4137** | fill=striped: 2606 | 4468 (0.93x) | 4805 (0.86x) | - | >10x | 494 (8.38x) | - | - | - | - | - | - | - | - |
| global linear score float64 (load!) | **2494** | fill=striped: 2036 | 2312 (1.08x) | 2494 (1.00x) | - | 325 (7.68x) | 495 (5.04x) | - | - | - | - | 726 (3.43x) | - | - | - |
| global linear path int16 (load!) | **3970** | - | 3380 (1.17x) | 4067 (0.98x) | 1770 (2.24x) | 2121 (1.87x) | 3974 (1.00x) | - | - | - | - | - | - | >10x | - |
| global linear path int32 (load!) | **2927** | - | 2656 (1.10x) | 3015 (0.97x) | 1506 (1.94x) | 951 (3.08x) | 2489 (1.18x) | - | 799 (3.67x) | 8846 (0.33x) | 1560 (1.88x) | - | - | - | - |
| global linear path int64 (load!) | **1547** | - | 1511 (1.02x) | 1547 (1.00x) | >10x | 246 (6.28x) | 949 (1.63x) | - | - | - | - | - | - | - | - |
| global linear path float32 (load!) | **2832** | traceback=scores: 2837 | 2696 (1.05x) | 2800 (1.01x) | - | >10x | 444 (6.38x) | - | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **1837** | traceback=scores: 1837 | 1672 (1.10x) | 1830 (1.00x) | - | 208 (8.83x) | 455 (4.03x) | - | - | - | - | >10x | >10x | - | - |
| global linear hard int16 (load!) | **4075** | - | 5480 (0.74x) | 6235 (0.65x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int32 (load!) | **4044** | - | 3437 (1.18x) | 4027 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **1795** | - | 1727 (1.04x) | 1799 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **3776** | traceback=scores: 3817 | 3561 (1.06x) | 3755 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **2185** | traceback=scores: 2190 | 2034 (1.07x) | 2185 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **358** | api=functions: 452 | 316 (1.13x) | 359 (1.00x) | - | - | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 (load!) | **6593** | - | 4217 (1.56x) | 6627 (0.99x) | 2992 (2.20x) | 3849 (1.71x) | 6903 (0.96x) | 2794 (2.36x) sat | - | - | - | - | - | >10x | - |
| local affine score int32 (load!) | **3589** | - | 3320 (1.08x) | 3587 (1.00x) | 2873 (1.25x) | 2065 (1.74x) | 3647 (0.98x) | - | - | - | - | - | - | - | - |
| local affine score int64 (load!) | **1061** | - | 1154 (0.92x) | 1059 (1.00x) | 108 (9.79x) | 443 (2.39x) | 868 (1.22x) | - | - | - | - | - | - | - | - |
| local affine score float32 (load!) | **3439** | fill=striped: 2516 | 3337 (1.03x) | 3438 (1.00x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| local affine score float64 (load!) | **1765** | fill=rowwise: 2015 | 1717 (1.03x) | 1763 (1.00x) | - | 187 (9.46x) | 258 (6.83x) | - | - | - | - | 227 (7.76x) | - | - | - |
| local affine path int16 (load!) | **2844** | - | 2452 (1.16x) | 2909 (0.98x) | 1631 (1.74x) | 2031 (1.40x) | 2002 (1.42x) | 579 (4.91x) sat | - | - | - | - | - | >10x | - |
| local affine path int32 (load!) | **1855** | - | 1803 (1.03x) | 1872 (0.99x) | 1358 (1.37x) | 921 (2.01x) | 1110 (1.67x) | - | - | - | - | - | - | - | - |
| local affine path int64 (load!) | **646** | - | 697 (0.93x) | 646 (1.00x) | 98 (6.59x) | 314 (2.06x) | 233 (2.78x) | - | - | - | - | - | - | - | - |
| local affine path float32 (load!) | **1669** | traceback=hirschberg_pmax: 1708 | 1744 (0.96x) | 1708 (0.98x) | - | >10x | 178 (9.39x) | - | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **967** | traceback=grid: 969 | 1022 (0.95x) | 974 (0.99x) | - | 129 (7.51x) | 177 (5.47x) | - | - | - | - | >10x | 124 (7.79x) | - | - |
| local affine hard int16 (load!) | **3657** | - | 3277 (1.12x) | 3680 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int32 (load!) | **2105** | - | 2097 (1.00x) | 2158 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **679** | - | 733 (0.93x) | 679 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1484** | traceback=hirschberg_pmax: 1955 | 2046 (0.73x) | 1935 (0.77x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **1040** | traceback=hirschberg: 1060 | 1121 (0.93x) | 1033 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **191** | api=functions: 250 | 180 (1.06x) | 191 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear score int16 (load!) | **4913** | - | 5091 (0.96x) | 7960 (0.62x) | 2385 (2.06x) | 3953 (1.24x) | 5122 (0.96x) | wrong | - | - | - | - | - | 504 (9.74x) sat | - |
| local linear score int32 (load!) | **4435** | - | 3850 (1.15x) | 4415 (1.00x) | 2333 (1.90x) | 2066 (2.15x) | 2661 (1.67x) | - | - | - | - | - | - | - | - |
| local linear score int64 (load!) | **1146** | - | 1268 (0.90x) | 1143 (1.00x) | >10x | 442 (2.59x) | 1102 (1.04x) | - | - | - | - | - | - | - | - |
| local linear score float32 (load!) | **3084** | fill=striped: 2677 | 2942 (1.05x) | 3072 (1.00x) | - | >10x | 344 (8.97x) | - | - | - | - | - | - | - | - |
| local linear score float64 (load!) | **1561** | fill=striped: 2017 | 1496 (1.04x) | 1576 (0.99x) | - | 186 (8.39x) | 344 (4.54x) | - | - | - | - | 462 (3.38x) | - | - | - |
| local linear path int16 (load!) | **3389** | - | 3152 (1.07x) | 3500 (0.97x) | 1593 (2.13x) | 2044 (1.66x) | 2915 (1.16x) | wrong | - | - | - | - | - | 409 (8.28x) sat | - |
| local linear path int32 (load!) | **2433** | - | 2213 (1.10x) | 2453 (0.99x) | 1340 (1.82x) | 923 (2.64x) | 1690 (1.44x) | - | - | - | - | - | - | - | - |
| local linear path int64 (load!) | **950** | - | 998 (0.95x) | 954 (1.00x) | 98 (9.69x) | 312 (3.04x) | 233 (4.08x) | - | - | - | - | - | - | - | - |
| local linear path float32 | **2058** | traceback=pointers: 2057 | 2031 (1.01x) | 2094 (0.98x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| local linear path float64 | **1243** | traceback=pointers: 1243 | 1220 (1.02x) | 1244 (1.00x) | - | 130 (9.59x) | 185 (6.73x) | - | - | - | - | >10x | 172 (7.21x) | - | - |
| local linear hard int16 | **4567** | - | 4497 (1.02x) | 4968 (0.92x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **3106** | - | 2796 (1.11x) | 3111 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **1034** | - | 1091 (0.95x) | 1039 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **2522** | traceback=pointers: 2520 | 2495 (1.01x) | 2531 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **1398** | traceback=pointers: 1397 | 1369 (1.02x) | 1385 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **318** | api=functions: 426 | 278 (1.14x) | 317 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |

#### nighthaven · dna_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| global affine score int32 | **3939** | - | 3735 (1.05x) | 5507 (0.72x) | 4744 (0.83x) | 5312 (0.74x) | 4542 (0.87x) | - | 1446 (2.72x) | 3462 (1.14x) | - | - | - | - | - |
| global affine score int64 | **1574** | - | 1673 (0.94x) | 1574 (1.00x) | >10x | 641 (2.46x) | 915 (1.72x) | - | - | - | - | - | - | - | - |
| global affine score float32 | **3979** | fill=striped: 4005 | 4072 (0.98x) | 4029 (0.99x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| global affine score float64 | **1676** | fill=rowwise: 1677 | 1682 (1.00x) | 1673 (1.00x) | - | 284 (5.89x) | 291 (5.75x) | - | - | - | - | 385 (4.35x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| global affine path int32 | **1938** | - | 2357 (0.82x) | 1938 (1.00x) | 1612 (1.20x) | 452 (4.28x) | 725 (2.67x) | - | 1360 (1.42x) | 2042 (0.95x) | - | - | - | - | - |
| global affine path int64 | **573** | - | 590 (0.97x) | 576 (0.99x) | 115 (4.98x) | 315 (1.82x) | 472 (1.21x) | - | - | - | - | - | - | - | - |
| global affine path float32 | **2574** | fill=striped: 2572 | 2525 (1.02x) | 2572 (1.00x) | - | 262 (9.82x) | 398 (6.47x) | - | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **1545** | traceback=grid: 1557 | 1555 (0.99x) | 1547 (1.00x) | - | 265 (5.83x) | 378 (4.09x) | - | - | - | - | >10x | >10x | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| global affine hard int32 (load!) | **1944** | - | 2379 (0.82x) | 1957 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **574** | - | 591 (0.97x) | 578 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **2584** | fill=striped: 2592 | 2550 (1.01x) | 2579 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1559** | traceback=grid: 1565 | 1561 (1.00x) | 1558 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **87** | fill=striped: 87 | 104 (0.84x) | 87 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear score int16 (load!) | **6736** | - | 7303 (0.92x) | 9653 (0.70x) | 11090 (0.61x) | 10792 (0.62x) | 15542 (0.43x) | - | - | - | - | - | - | >10x | - |
| global linear score int32 (load!) | **4045** | - | 3800 (1.06x) | 5508 (0.73x) | 7062 (0.57x) | 5362 (0.75x) | 7360 (0.55x) | - | 1422 (2.84x) | 54523 (0.07x) | 44772 (0.09x) | - | - | - | - |
| global linear score int64 | **1574** | - | 1672 (0.94x) | 1577 (1.00x) | >10x | 641 (2.45x) | 921 (1.71x) | - | - | - | - | - | - | - | - |
| global linear score float32 | **4962** | fill=striped: 4904 | 4965 (1.00x) | 6718 (0.74x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| global linear score float64 (load!) | **1888** | fill=striped: 1883 | 1880 (1.00x) | 1895 (1.00x) | - | 284 (6.64x) | 392 (4.82x) | - | - | - | - | 668 (2.83x) | - | - | - |
| global linear path int16 (load!) | **4049** | - | 6675 (0.61x) | 5798 (0.70x) | 3738 (1.08x) | 887 (4.57x) | 1663 (2.43x) | - | - | - | - | - | - | >10x | - |
| global linear path int32 (load!) | **3354** | - | 3458 (0.97x) | 3333 (1.01x) | 1973 (1.70x) | 457 (7.33x) | 1190 (2.82x) | - | 1359 (2.47x) | 36375 (0.09x) | 10301 (0.33x) | - | - | >10x | - |
| global linear path int64 (load!) | **777** | - | 854 (0.91x) | 768 (1.01x) | 115 (6.76x) | 260 (2.98x) | 1218 (0.64x) | - | - | - | - | - | - | - | - |
| global linear path float32 (load!) | **2944** | traceback=scores: 3758 | 2939 (1.00x) | 2935 (1.00x) | - | >10x | 583 (5.05x) | - | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **1605** | traceback=scores: 1969 | 1606 (1.00x) | 1598 (1.00x) | - | 216 (7.44x) | 591 (2.71x) | - | - | - | - | 166 (9.66x) | 285 (5.63x) | - | - |
| global linear hard int16 (load!) | **4014** | - | 4737 (0.85x) | 5950 (0.67x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int32 (load!) | **3421** | - | 3452 (0.99x) | 3466 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **778** | - | 860 (0.91x) | 782 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **3004** | traceback=scores: 3893 | 3026 (0.99x) | 3000 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1622** | traceback=scores: 2009 | 1626 (1.00x) | 1611 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **235** | fill=striped: 236 | 230 (1.02x) | 236 (1.00x) | - | - | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| local affine score int32 | **4317** | - | 3952 (1.09x) | 4765 (0.91x) | 6127 (0.70x) | 2170 (1.99x) | 3686 (1.17x) | - | - | - | - | - | - | - | - |
| local affine score int64 | **1233** | - | 1377 (0.90x) | 1247 (0.99x) | >10x | 457 (2.70x) | 874 (1.41x) | - | - | - | - | - | - | - | - |
| local affine score float32 | **3798** | fill=striped: 3846 | 3770 (1.01x) | 5293 (0.72x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| local affine score float64 | **1753** | fill=rowwise: 1838 | 1833 (0.96x) | 1841 (0.95x) | - | 181 (9.68x) | 228 (7.69x) | - | - | - | - | 250 (7.01x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| local affine path int32 | **1194** | - | 1222 (0.98x) | 1195 (1.00x) | 1544 (0.77x) | 425 (2.81x) | 536 (2.23x) | - | - | - | - | - | - | - | - |
| local affine path int64 | **428** | - | 466 (0.92x) | 429 (1.00x) | 100 (4.29x) | 309 (1.38x) | 255 (1.68x) | - | - | - | - | - | - | - | - |
| local affine path float32 | **1370** | traceback=scores: 1599 | 1384 (0.99x) | 1368 (1.00x) | - | 141 (9.74x) | 188 (7.30x) | - | - | - | - | - | - | - | - |
| local affine path float64 | **721** | traceback=grid: 719 | 678 (1.06x) | 716 (1.01x) | - | 142 (5.08x) | 187 (3.86x) | - | - | - | - | 82 (8.83x) | 170 (4.24x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| local affine hard int32 | **1210** | - | 1227 (0.99x) | 1210 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **430** | - | 465 (0.92x) | 429 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **1385** | traceback=scores: 1678 | 1403 (0.99x) | 1382 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **720** | traceback=grid: 729 | 701 (1.03x) | 722 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **32** | fill=striped: 32 | 44 (0.72x) | 32 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **5627** | - | 6850 (0.82x) | 9393 (0.60x) | 14205 (0.40x) | 3886 (1.45x) | 6510 (0.86x) | 6063 (0.93x) sat | - | - | - | - | - | 641 (8.77x) sat | - |
| local linear score int32 | **4378** | - | 3617 (1.21x) | 4736 (0.92x) | 7779 (0.56x) | 2175 (2.01x) | 3036 (1.44x) | - | - | - | - | - | - | - | - |
| local linear score int64 | **1254** | - | 1377 (0.91x) | 1256 (1.00x) | >10x | 457 (2.75x) | 1136 (1.10x) | - | - | - | - | - | - | - | - |
| local linear score float32 | **4419** | api=functions: 4391 | 4450 (0.99x) | 6206 (0.71x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| local linear score float64 | **1888** | fill=striped: 1859 | 1905 (0.99x) | 1885 (1.00x) | - | >10x | 292 (6.47x) | - | - | - | - | 634 (2.98x) | - | - | - |
| local linear path int16 | **4612** | - | 3876 (1.19x) | 4589 (1.00x) | 3396 (1.36x) | 776 (5.95x) | 1327 (3.48x) | 816 (5.65x) sat | - | - | - | - | - | 539 (8.55x) sat | - |
| local linear path int32 | **2255** | - | 2605 (0.87x) | 2442 (0.92x) | 1854 (1.22x) | 392 (5.75x) | 659 (3.42x) | - | - | - | - | - | - | - | - |
| local linear path int64 | **472** | - | 506 (0.93x) | 480 (0.98x) | 101 (4.69x) | 310 (1.52x) | 245 (1.93x) | - | - | - | - | - | - | - | - |
| local linear path float32 | **1896** | traceback=scores: 2393 | 1778 (1.07x) | 1898 (1.00x) | - | >10x | 193 (9.80x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **955** | traceback=scores: 1239 | 899 (1.06x) | 942 (1.01x) | - | 136 (7.03x) | 194 (4.93x) | - | - | - | - | 111 (8.61x) | 271 (3.52x) | - | - |
| local linear hard int16 | **4532** | - | 4114 (1.10x) | 4994 (0.91x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **2488** | - | 2632 (0.95x) | 2483 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **472** | - | 505 (0.93x) | 473 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1923** | traceback=scores: 2407 | 1804 (1.07x) | 1925 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **966** | traceback=scores: 1254 | 906 (1.07x) | 943 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **247** | fill=striped: 246 | 244 (1.01x) | 246 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |

#### nighthaven · dna_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global affine score int32 | **2769** | - | 2812 (0.98x) | 2778 (1.00x) | 4265 (0.65x) | 1116 (2.48x) | 917 (3.02x) | 1565 (1.77x) | 5263 (0.53x) | - | - | - | - |
| global affine score int64 | **1261** | - | 1311 (0.96x) | 1261 (1.00x) | >10x | 635 (1.99x) | 916 (1.38x) | - | - | - | - | - | - |
| global affine score float32 | **3061** | fill=striped: 3064 | 3074 (1.00x) | 3059 (1.00x) | - | >10x | >10x | - | - | - | - | - | - |
| global affine score float64 | **1637** | api=functions: 1683 | 1652 (0.99x) | 1650 (0.99x) | - | 281 (5.83x) | 287 (5.71x) | - | - | - | 380 (4.30x) | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global affine path int32 | **1343** | - | 1312 (1.02x) | 1345 (1.00x) | 733 (1.83x) | 266 (5.05x) | 313 (4.29x) | 908 (1.48x) | 2037 (0.66x) | - | - | - | - |
| global affine path int64 | **524** | - | 524 (1.00x) | 522 (1.00x) | 112 (4.66x) | 277 (1.89x) | 318 (1.65x) | - | - | - | - | - | - |
| global affine path float32 | **3631** | fill=striped: 3628 | 3632 (1.00x) | 3628 (1.00x) | - | >10x | >10x | - | - | - | - | - | - |
| global affine path float64 | **2102** | fill=rowwise: 2138 | 2146 (0.98x) | 2130 (0.99x) | - | 236 (8.92x) | 285 (7.36x) | - | - | - | >10x | >10x | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global affine hard int32 | **1344** | - | 1385 (0.97x) | 1346 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **522** | - | 525 (1.00x) | 523 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **3635** | fill=striped: 3631 | 3642 (1.00x) | 3636 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **2142** | traceback=grid: 2135 | 2129 (1.01x) | 2033 (1.05x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **137** | fill=rowwise: 140 | 150 (0.92x) | 139 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global linear score int32 | **2772** | - | 2814 (0.99x) | 2766 (1.00x) | 5077 (0.55x) | 1147 (2.42x) | 2221 (1.25x) | 1548 (1.79x) | 82507 (0.03x) | 146368 (0.02x) | - | - | - |
| global linear score int64 | **1261** | - | 1307 (0.96x) | 1244 (1.01x) | >10x | 640 (1.97x) | 917 (1.37x) | - | - | - | - | - | - |
| global linear score float32 | **3497** | api=functions: 3510 | 3511 (1.00x) | 3504 (1.00x) | - | >10x | 383 (9.13x) | - | - | - | - | - | - |
| global linear score float64 (load!) | **1722** | fill=striped: 1773 | 1573 (1.09x) | 1730 (1.00x) | - | 281 (6.13x) | 383 (4.50x) | - | - | - | 657 (2.62x) | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global linear path int32 | **1541** | - | 1506 (1.02x) | 1544 (1.00x) | 733 (2.10x) | 219 (7.03x) | 957 (1.61x) | 914 (1.69x) | 54728 (0.03x) | 35942 (0.04x) | - | - | - |
| global linear path int64 | **714** | - | 748 (0.95x) | 709 (1.01x) | 112 (6.35x) | 236 (3.03x) | 962 (0.74x) | - | - | - | - | - | - |
| global linear path float32 | **2274** | traceback=pointers: 2272 | 2233 (1.02x) | 2300 (0.99x) | - | >10x | 389 (5.85x) | - | - | - | - | - | - |
| global linear path float64 | **1143** | fill=striped: 1152 | 1131 (1.01x) | 1152 (0.99x) | - | 195 (5.88x) | 398 (2.87x) | - | - | - | 156 (7.31x) | 132 (8.69x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global linear hard int32 | **1525** | - | 1508 (1.01x) | 1539 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **714** | - | 749 (0.95x) | 707 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **2277** | fill=striped: 2273 | 2239 (1.02x) | 2304 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **1111** | traceback=pointers: 1153 | 1132 (0.98x) | 1151 (0.97x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **204** | fill=striped: 201 | 202 (1.01x) | 200 (1.02x) | - | - | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local affine score int32 | **2609** | - | 2640 (0.99x) | 2614 (1.00x) | 4741 (0.55x) | 524 (4.98x) | 976 (2.67x) | - | - | - | - | - | - |
| local affine score int64 | **1076** | - | 1218 (0.88x) | 1201 (0.90x) | 121 (8.90x) | 446 (2.41x) | 856 (1.26x) | - | - | - | - | - | - |
| local affine score float32 | **3388** | api=functions: 3320 | 3402 (1.00x) | 3332 (1.02x) | - | >10x | >10x | - | - | - | - | - | - |
| local affine score float64 | **1692** | fill=striped: 1700 | 1470 (1.15x) | 1690 (1.00x) | - | 178 (9.49x) | 227 (7.45x) | - | - | - | 252 (6.72x) | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local affine path int32 | **565** | - | 573 (0.99x) | 565 (1.00x) | 620 (0.91x) | 286 (1.98x) | 247 (2.29x) | - | - | - | - | - | - |
| local affine path int64 | **211** | - | 210 (1.01x) | 211 (1.00x) | 98 (2.15x) | 284 (0.74x) | 246 (0.86x) | - | - | - | - | - | - |
| local affine path float32 | **522** | traceback=pointers: 1321 | 519 (1.01x) | 524 (1.00x) | - | 140 (3.74x) | 179 (2.92x) | - | - | - | - | - | - |
| local affine path float64 | **680** | traceback=pointers: 693 | 671 (1.01x) | 687 (0.99x) | - | 141 (4.82x) | 179 (3.79x) | - | - | - | 82 (8.34x) | >10x | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local affine hard int32 | **493** | - | 574 (0.86x) | 565 (0.87x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **210** | - | 212 (0.99x) | 210 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **524** | traceback=grid: 1326 | 520 (1.01x) | 524 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **703** | traceback=grid: 700 | 665 (1.06x) | 705 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **83** | fill=striped: 83 | 98 (0.85x) | 83 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local linear score int32 | **2512** | - | 2647 (0.95x) | 2614 (0.96x) | 5014 (0.50x) | 554 (4.53x) | 1527 (1.65x) | - | - | - | - | - | - |
| local linear score int64 | **1203** | - | 1228 (0.98x) | 1197 (1.01x) | 123 (9.82x) | 446 (2.70x) | 1144 (1.05x) | - | - | - | - | - | - |
| local linear score float32 | **3511** | fill=striped: 3577 | 3513 (1.00x) | 3508 (1.00x) | - | >10x | >10x | - | - | - | - | - | - |
| local linear score float64 | **1731** | api=functions: 1735 | 1738 (1.00x) | 1728 (1.00x) | - | 181 (9.58x) | 283 (6.12x) | - | - | - | 652 (2.65x) | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local linear path int32 | **1308** | - | 1375 (0.95x) | 1382 (0.95x) | 635 (2.06x) | 286 (4.57x) | 231 (5.66x) | - | - | - | - | - | - |
| local linear path int64 | **462** | - | 479 (0.97x) | 461 (1.00x) | 98 (4.71x) | 284 (1.63x) | 232 (2.00x) | - | - | - | - | - | - |
| local linear path float32 | **1678** | traceback=pointers: 1678 | 1550 (1.08x) | 1682 (1.00x) | - | >10x | 188 (8.94x) | - | - | - | - | - | - |
| local linear path float64 | **844** | traceback=pointers: 844 | 775 (1.09x) | 846 (1.00x) | - | 135 (6.26x) | 188 (4.50x) | - | - | - | 111 (7.62x) | 119 (7.08x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local linear hard int32 | **1381** | - | 1371 (1.01x) | 1392 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **461** | - | 479 (0.96x) | 462 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1683** | fill=striped: 1680 | 1553 (1.08x) | 1684 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **844** | traceback=pointers: 847 | 757 (1.11x) | 829 (1.02x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **191** | fill=striped: 191 | 192 (1.00x) | 192 (0.99x) | - | - | - | - | - | - | - | - | - |

#### nighthaven · mirna · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **4858** | - | 5300 (0.92x) | 5705 (0.85x) | 1183 (4.11x) | 6189 (0.78x) | 6855 (0.71x) | - | - | - | - | - | - | >10x | - |
| global affine score int32 | **3455** | - | 3028 (1.14x) | 3360 (1.03x) | 1205 (2.87x) | 3910 (0.88x) | 3909 (0.88x) | - | 535 (6.45x) | >10x | - | - | - | >10x | - |
| global affine score int64 | **1290** | - | 1286 (1.00x) | 1296 (1.00x) | >10x | 680 (1.90x) | 857 (1.50x) | - | - | - | - | - | - | - | - |
| global affine score float32 | **3251** | kernel=sse4: 2091 | 3114 (1.04x) | 3436 (0.95x) | - | 386 (8.43x) | 382 (8.50x) | - | - | - | - | - | - | - | - |
| global affine score float64 | **1800** | fill=rowwise: 1103 | 1680 (1.07x) | 1873 (0.96x) | - | 386 (4.66x) | 379 (4.75x) | - | - | - | - | 209 (8.62x) | - | - | - |
| global affine path int16 (load!) | **1087** | - | 1085 (1.00x) | 1123 (0.97x) | 534 (2.03x) | 2110 (0.52x) | 1330 (0.82x) | - | - | - | - | - | - | >10x | - |
| global affine path int32 (load!) | **921** | - | 908 (1.01x) | 929 (0.99x) | 642 (1.44x) | 1179 (0.78x) | 1064 (0.87x) | - | 454 (2.03x) | >10x | - | - | - | - | - |
| global affine path int64 (load!) | **593** | - | 593 (1.00x) | 597 (0.99x) | 99 (5.99x) | 205 (2.89x) | 170 (3.48x) | - | - | - | - | - | - | - | - |
| global affine path float32 (load!) | **922** | traceback=scores: 918 | 910 (1.01x) | 941 (0.98x) | - | 166 (5.57x) | 140 (6.58x) | - | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **691** | traceback=pointers: 696 | 685 (1.01x) | 706 (0.98x) | - | 168 (4.11x) | 137 (5.06x) | - | - | - | - | >10x | >10x | - | - |
| global affine hard int16 | **2002** | - | 2050 (0.98x) | 2018 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **1493** | - | 1501 (0.99x) | 1501 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **792** | - | 808 (0.98x) | 795 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1516** | traceback=pointers: 1518 | 1529 (0.99x) | 1515 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **989** | traceback=grid: 985 | 962 (1.03x) | 1016 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **272** | api=functions: 178 | 226 (1.20x) | 271 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **5319** | - | 4990 (1.07x) | 6929 (0.77x) | 1178 (4.52x) | 6244 (0.85x) | 8673 (0.61x) | - | - | - | - | - | - | >10x | - |
| global linear score int32 | **4110** | - | 3420 (1.20x) | 4342 (0.95x) | 1208 (3.40x) | 3913 (1.05x) | 5157 (0.80x) | - | 619 (6.64x) | 808 (5.09x) | 858 (4.79x) | - | - | >10x | - |
| global linear score int64 | **1366** | - | 1669 (0.82x) | 1358 (1.01x) | >10x | 681 (2.01x) | 1000 (1.37x) | - | - | - | - | - | - | - | - |
| global linear score float32 | **4248** | kernel=sse4: 2513 | 3521 (1.21x) | 4238 (1.00x) | - | >10x | 508 (8.36x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **2387** | fill=striped: 1169 | 1945 (1.23x) | 2378 (1.00x) | - | 386 (6.18x) | 509 (4.69x) | - | - | - | - | 288 (8.30x) | - | - | - |
| global linear path int16 | **811** | - | 850 (0.95x) | 879 (0.92x) | 528 (1.54x) | 2102 (0.39x) | 1212 (0.67x) | - | - | - | - | - | - | 110 (7.37x) sat | - |
| global linear path int32 | **765** | - | 787 (0.97x) | 825 (0.93x) | 616 (1.24x) | 1085 (0.70x) | 959 (0.80x) | - | 499 (1.53x) | 694 (1.10x) | 269 (2.84x) | - | - | - | - |
| global linear path int64 | **613** | - | 640 (0.96x) | 651 (0.94x) | 99 (6.17x) | 210 (2.92x) | 510 (1.20x) | - | - | - | - | - | - | - | - |
| global linear path float32 | **737** | traceback=pointers: 745 | 759 (0.97x) | 745 (0.99x) | - | 170 (4.34x) | 272 (2.71x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **661** | traceback=pointers: 659 | 647 (1.02x) | 671 (0.99x) | - | 173 (3.83x) | 279 (2.37x) | - | - | - | - | 79 (8.33x) | >10x | - | - |
| global linear hard int16 | **1989** | - | 1962 (1.01x) | 1998 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **1675** | - | 1603 (1.05x) | 1716 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **1107** | - | 910 (1.22x) | 1015 (1.09x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1620** | traceback=pointers: 1622 | 1577 (1.03x) | 1545 (1.05x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **1256** | traceback=pointers: 1248 | 1146 (1.10x) | 1214 (1.04x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **408** | api=functions: 289 | 384 (1.06x) | 406 (1.01x) | - | - | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 (load!) | **5305** | - | 5018 (1.06x) | 5333 (0.99x) | 2051 (2.59x) | 2937 (1.81x) | 5660 (0.94x) | 1782 (2.98x) sat | - | - | - | - | - | >10x | - |
| local affine score int32 (load!) | **3226** | - | 2851 (1.13x) | 3136 (1.03x) | 1445 (2.23x) | 1783 (1.81x) | 3193 (1.01x) | - | - | - | - | - | - | >10x | - |
| local affine score int64 (load!) | **1009** | - | 1078 (0.94x) | 1007 (1.00x) | >10x | 387 (2.60x) | 792 (1.27x) | - | - | - | - | - | - | - | - |
| local affine score float32 (load!) | **3049** | kernel=sse4: 1821 | 2857 (1.07x) | 3067 (0.99x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| local affine score float64 (load!) | **1653** | fill=rowwise: 1458 | 1500 (1.10x) | 1644 (1.01x) | - | 193 (8.58x) | 288 (5.74x) | - | - | - | - | >10x | - | - | - |
| local affine path int16 (load!) | **2585** | - | 2522 (1.02x) | 2598 (1.00x) | 619 (4.17x) sat | 1859 (1.39x) | 1360 (1.90x) | 1152 (2.24x) sat | - | - | - | - | - | >10x | - |
| local affine path int32 (load!) | **1658** | - | 1647 (1.01x) | 1671 (0.99x) | 761 (2.18x) | 1065 (1.56x) | 913 (1.82x) | - | - | - | - | - | - | >10x | - |
| local affine path int64 (load!) | **624** | - | 652 (0.96x) | 625 (1.00x) | 88 (7.06x) | 275 (2.27x) | 192 (3.26x) | - | - | - | - | - | - | - | - |
| local affine path float32 (load!) | **1588** | traceback=scores: 1576 | 1599 (0.99x) | 1579 (1.01x) | - | >10x | 167 (9.52x) | - | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **915** | traceback=hirschberg_pmax: 925 | 936 (0.98x) | 918 (1.00x) | - | 117 (7.83x) | 166 (5.53x) | - | - | - | - | >10x | >10x | - | - |
| local affine hard int16 (load!) | **2866** | - | 2826 (1.01x) | 2902 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int32 (load!) | **1777** | - | 1770 (1.00x) | 1798 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **642** | - | 674 (0.95x) | 642 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1681** | traceback=pointers: 1692 | 1733 (0.97x) | 1704 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **959** | traceback=grid: 961 | 983 (0.98x) | 955 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **283** | api=functions: 184 | 260 (1.09x) | 283 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear score int16 (load!) | **6203** | - | 5654 (1.10x) | 6149 (1.01x) | 926 (6.70x) | 3179 (1.95x) | 4454 (1.39x) | wrong | - | - | - | - | - | >10x | - |
| local linear score int32 (load!) | **3909** | - | 3216 (1.22x) | 3922 (1.00x) | 994 (3.93x) | 1465 (2.67x) | 2391 (1.64x) | - | - | - | - | - | - | >10x | - |
| local linear score int64 (load!) | **1123** | - | 1159 (0.97x) | 1102 (1.02x) | >10x | 358 (3.14x) | 1005 (1.12x) | - | - | - | - | - | - | - | - |
| local linear score float32 (load!) | **2905** | kernel=sse4: 1562 | 2551 (1.14x) | 2954 (0.98x) | - | >10x | 400 (7.26x) | - | - | - | - | - | - | - | - |
| local linear score float64 (load!) | **1575** | fill=striped: 1405 | 1340 (1.18x) | 1579 (1.00x) | - | 192 (8.21x) | 400 (3.94x) | - | - | - | - | 192 (8.19x) | - | - | - |
| local linear path int16 (load!) | **2023** | - | 1950 (1.04x) | 2113 (0.96x) | 576 (3.52x) | 1857 (1.09x) | 1577 (1.28x) | wrong | - | - | - | - | - | >10x | - |
| local linear path int32 (load!) | **1473** | - | 1501 (0.98x) | 1705 (0.86x) | 700 (2.10x) | 1064 (1.38x) | 1114 (1.32x) | - | - | - | - | - | - | >10x | - |
| local linear path int64 (load!) | **789** | - | 801 (0.98x) | 802 (0.98x) | 88 (8.98x) | 263 (3.00x) | 219 (3.60x) | - | - | - | - | - | - | - | - |
| local linear path float32 (load!) | **1457** | traceback=scores: 1463 | 1424 (1.02x) | 1488 (0.98x) | - | >10x | 164 (8.89x) | - | - | - | - | - | - | - | - |
| local linear path float64 (load!) | **979** | traceback=pointers: 977 | 936 (1.05x) | 982 (1.00x) | - | 116 (8.47x) | 165 (5.92x) | - | - | - | - | >10x | >10x | - | - |
| local linear hard int16 (load!) | **3129** | - | 2825 (1.11x) | 3096 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int32 (load!) | **2312** | - | 1750 (1.32x) | 2329 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **917** | - | 924 (0.99x) | 919 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1944** | traceback=pointers: 1947 | 1838 (1.06x) | 1954 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **1190** | traceback=pointers: 1187 | 1099 (1.08x) | 1183 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **370** | api=functions: 277 | 343 (1.08x) | 369 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |

#### nighthaven · protein_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **3519** | - | 3745 (0.94x) | 3469 (1.01x) | 2010 (1.75x) | 2425 (1.45x) | 4401 (0.80x) | - | - | - | - | >10x | - |
| global affine score int32 | **2931** | - | 3522 (0.83x) | 2887 (1.02x) | 1923 (1.52x) | 2234 (1.31x) | 3253 (0.90x) | - | 811 (3.62x) | - | - | >10x | - |
| global affine score int64 | **1009** | - | 1295 (0.78x) | 1185 (0.85x) | 116 (8.67x) | 758 (1.33x) | 913 (1.11x) | - | - | - | - | - | - |
| global affine score float32 | **1486** | fill=striped: 1484 | 1535 (0.97x) | 1452 (1.02x) | - | 321 (4.63x) | 343 (4.33x) | - | - | - | - | - | - |
| global affine score float64 | **1446** | fill=rowwise: 1435 | 1500 (0.96x) | 1437 (1.01x) | - | 321 (4.50x) | 343 (4.21x) | - | - | 371 (3.89x) | - | - | - |
| global affine path int16 | **1554** | - | 1450 (1.07x) | 1582 (0.98x) | 1406 (1.11x) | 1600 (0.97x) | 2531 (0.61x) | - | - | - | - | 212 (7.34x) sat | - |
| global affine path int32 | **1300** | - | 1162 (1.12x) | 1307 (1.00x) | 1177 (1.10x) | 1065 (1.22x) | 1897 (0.69x) | - | 755 (1.72x) | - | - | - | - |
| global affine path int64 | **865** | - | 862 (1.00x) | 873 (0.99x) | 111 (7.80x) | 455 (1.90x) | 300 (2.89x) | - | - | - | - | - | - |
| global affine path float32 | **1250** | traceback=scores: 1290 | 1123 (1.11x) | 1240 (1.01x) | - | 211 (5.93x) | 261 (4.80x) | - | - | - | - | - | - |
| global affine path float64 | **910** | traceback=grid: 940 | 879 (1.04x) | 916 (0.99x) | - | 206 (4.41x) | 261 (3.49x) | - | - | 99 (9.18x) | 124 (7.32x) | - | - |
| global affine hard int16 | **1655** | - | 1518 (1.09x) | 1657 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **1355** | - | 1240 (1.09x) | 1346 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **895** | - | 896 (1.00x) | 888 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1288** | traceback=scores: 1326 | 1191 (1.08x) | 1289 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **938** | traceback=scores: 944 | 911 (1.03x) | 926 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **192** | api=functions: 251 | 198 (0.97x) | 199 (0.96x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **3526** | - | 3511 (1.00x) | 3550 (0.99x) | 2003 (1.76x) | 2581 (1.37x) | 5336 (0.66x) | - | - | - | - | >10x | - |
| global linear score int32 | **2921** | - | 3498 (0.84x) | 2940 (0.99x) | 1946 (1.50x) | 2429 (1.20x) | 4734 (0.62x) | - | 834 (3.50x) | - | - | - | - |
| global linear score int64 | **1184** | - | 1297 (0.91x) | 1183 (1.00x) | >10x | 782 (1.51x) | 930 (1.27x) | - | - | - | - | - | - |
| global linear score float32 | **1781** | fill=striped: 1800 | 1997 (0.89x) | 1831 (0.97x) | - | 321 (5.55x) | 519 (3.43x) | - | - | - | - | - | - |
| global linear score float64 | **1631** | fill=striped: 1612 | 1721 (0.95x) | 1631 (1.00x) | - | 321 (5.08x) | 519 (3.14x) | - | - | 721 (2.26x) | - | - | - |
| global linear path int16 (load!) | **2177** | - | 2178 (1.00x) | 2174 (1.00x) | 1417 (1.54x) | 1591 (1.37x) | 3202 (0.68x) | - | - | - | - | 257 (8.47x) sat | - |
| global linear path int32 (load!) | **1979** | - | 2172 (0.91x) | 1996 (0.99x) | 1173 (1.69x) | 1076 (1.84x) | 2273 (0.87x) | - | 772 (2.56x) | - | - | - | - |
| global linear path int64 (load!) | **1237** | - | 1290 (0.96x) | 1233 (1.00x) | >10x | 456 (2.71x) | 1032 (1.20x) | - | - | - | - | - | - |
| global linear path float32 (load!) | **1543** | traceback=scores: 1532 | 1341 (1.15x) | 1537 (1.00x) | - | 237 (6.51x) | 725 (2.13x) | - | - | - | - | - | - |
| global linear path float64 (load!) | **1202** | traceback=scores: 1199 | 1024 (1.17x) | 1214 (0.99x) | - | 235 (5.12x) | 726 (1.66x) | - | - | 199 (6.04x) | 174 (6.91x) | - | - |
| global linear hard int16 (load!) | **2559** | - | 2509 (1.02x) | 2542 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard int32 (load!) | **2352** | - | 2597 (0.91x) | 2289 (1.03x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **1321** | - | 1389 (0.95x) | 1320 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **1690** | traceback=pointers: 1713 | 1456 (1.16x) | 1689 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1299** | traceback=pointers: 1291 | 1211 (1.07x) | 1316 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **260** | api=functions: 316 | 234 (1.11x) | 261 (1.00x) | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 (load!) | **3306** | - | 3488 (0.95x) | 3323 (1.00x) | 1801 (1.84x) | 2140 (1.55x) | 3459 (0.96x) | 873 (3.79x) sat | - | - | - | >10x | - |
| local affine score int32 (load!) | **2754** | - | 3012 (0.91x) | 2718 (1.01x) | 1649 (1.67x) | 1581 (1.74x) | 2412 (1.14x) | - | - | - | - | >10x | - |
| local affine score int64 (load!) | **1001** | - | 1176 (0.85x) | 996 (1.01x) | >10x | 524 (1.91x) | 664 (1.51x) | - | - | - | - | - | - |
| local affine score float32 (load!) | **1635** | fill=striped: 1624 | 1650 (0.99x) | 1628 (1.00x) | - | 184 (8.88x) | 260 (6.28x) | - | - | - | - | - | - |
| local affine score float64 (load!) | **1501** | fill=striped: 1481 | 1566 (0.96x) | 1497 (1.00x) | - | 184 (8.15x) | 260 (5.78x) | - | - | 252 (5.96x) | - | - | - |
| local affine path int16 (load!) | **1556** | - | 1434 (1.09x) | 1569 (0.99x) | 1305 (1.19x) | 1514 (1.03x) | 1493 (1.04x) | 350 (4.45x) sat | - | - | - | 163 (9.55x) sat | - |
| local affine path int32 (load!) | **1211** | - | 1171 (1.03x) | 1270 (0.95x) | 1066 (1.14x) | 923 (1.31x) | 941 (1.29x) | - | - | - | - | - | - |
| local affine path int64 (load!) | **606** | - | 625 (0.97x) | 605 (1.00x) | 97 (6.27x) | 424 (1.43x) | 312 (1.94x) | - | - | - | - | - | - |
| local affine path float32 (load!) | **1221** | traceback=scores: 1229 | 1106 (1.10x) | 1225 (1.00x) | - | 156 (7.83x) | 196 (6.24x) | - | - | - | - | - | - |
| local affine path float64 (load!) | **889** | traceback=scores: 897 | 893 (1.00x) | 889 (1.00x) | - | 158 (5.63x) | 196 (4.54x) | - | - | >10x | 124 (7.16x) | - | - |
| local affine hard int16 (load!) | **1590** | - | 1542 (1.03x) | 1673 (0.95x) | - | - | - | - | - | - | - | - | - |
| local affine hard int32 (load!) | **1341** | - | 1240 (1.08x) | 1333 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **620** | - | 644 (0.96x) | 616 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1280** | traceback=scores: 1297 | 1174 (1.09x) | 1276 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **917** | traceback=scores: 923 | 899 (1.02x) | 913 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **179** | api=functions: 245 | 171 (1.05x) | 176 (1.02x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **3307** | - | 3453 (0.96x) | 3317 (1.00x) | 1806 (1.83x) | 2133 (1.55x) | 4623 (0.72x) | wrong | - | - | - | >10x | - |
| local linear score int32 | **2743** | - | 2854 (0.96x) | 2733 (1.00x) | 1641 (1.67x) | 1583 (1.73x) | 2286 (1.20x) | - | - | - | - | >10x | - |
| local linear score int64 | **998** | - | 1173 (0.85x) | 998 (1.00x) | >10x | 521 (1.92x) | 1218 (0.82x) | - | - | - | - | - | - |
| local linear score float32 | **2013** | fill=striped: 2006 | 2190 (0.92x) | 2007 (1.00x) | - | >10x | 345 (5.84x) | - | - | - | - | - | - |
| local linear score float64 | **1680** | fill=striped: 1678 | 1794 (0.94x) | 1690 (0.99x) | - | 183 (9.17x) | 345 (4.87x) | - | - | 360 (4.66x) | - | - | - |
| local linear path int16 | **1746** | - | 1539 (1.13x) | 1809 (0.97x) | 1295 (1.35x) | 1512 (1.15x) | 2026 (0.86x) | wrong | - | - | - | 203 (8.59x) sat | - |
| local linear path int32 | **1612** | - | 1318 (1.22x) | 1606 (1.00x) | 1065 (1.51x) | 911 (1.77x) | 1362 (1.18x) | - | - | - | - | - | - |
| local linear path int64 | **770** | - | 809 (0.95x) | 771 (1.00x) | 81 (9.51x) | 424 (1.82x) | 256 (3.01x) | - | - | - | - | - | - |
| local linear path float32 | **1240** | traceback=pointers: 1239 | 1119 (1.11x) | 1251 (0.99x) | - | 165 (7.53x) | 206 (6.03x) | - | - | - | - | - | - |
| local linear path float64 | **907** | traceback=pointers: 905 | 890 (1.02x) | 916 (0.99x) | - | 165 (5.49x) | 200 (4.54x) | - | - | 117 (7.72x) | 177 (5.13x) | - | - |
| local linear hard int16 | **1977** | - | 1673 (1.18x) | 1975 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **1760** | - | 1400 (1.26x) | 1751 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **799** | - | 850 (0.94x) | 796 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **1329** | traceback=hirschberg: 1342 | 1211 (1.10x) | 1344 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **950** | traceback=scores: 953 | 928 (1.02x) | 955 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **294** | api=functions: 353 | 265 (1.11x) | 291 (1.01x) | - | - | - | - | - | - | - | - | - |

#### nighthaven · protein_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global affine score int32 | **3889** | - | 3818 (1.02x) | 5263 (0.74x) | 4299 (0.90x) | 2347 (1.66x) | 3475 (1.12x) | 1480 (2.63x) | - | - | - | - |
| global affine score int64 | **1304** | - | 1606 (0.81x) | 1520 (0.86x) | >10x | 728 (1.79x) | 915 (1.43x) | - | - | - | - | - |
| global affine score float32 | **2822** | fill=striped: 2818 | 2955 (0.95x) | 2829 (1.00x) | - | >10x | 292 (9.67x) | - | - | - | - | - |
| global affine score float64 (load!) | **1461** | fill=rowwise: 1479 | 1474 (0.99x) | 1458 (1.00x) | - | 282 (5.19x) | 291 (5.02x) | - | 385 (3.80x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global affine path int32 (load!) | **1792** | - | 2101 (0.85x) | 1797 (1.00x) | 1877 (0.95x) | 385 (4.66x) | 591 (3.03x) | 1341 (1.34x) | - | - | - | - |
| global affine path int64 (load!) | **572** | - | 587 (0.98x) | 573 (1.00x) | 115 (4.98x) | 275 (2.08x) | 340 (1.68x) | - | - | - | - | - |
| global affine path float32 (load!) | **2034** | traceback=grid: 2026 | 1976 (1.03x) | 2022 (1.01x) | - | 228 (8.93x) | 293 (6.95x) | - | - | - | - | - |
| global affine path float64 (load!) | **1360** | fill=striped: 1363 | 1357 (1.00x) | 1364 (1.00x) | - | 225 (6.05x) | 292 (4.65x) | - | >10x | >10x | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global affine hard int32 (load!) | **1789** | - | 2107 (0.85x) | 1797 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **573** | - | 590 (0.97x) | 574 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **2037** | fill=striped: 2042 | 1996 (1.02x) | 2031 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1368** | fill=striped: 1369 | 1361 (1.01x) | 1367 (1.00x) | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **96** | fill=rowwise: 96 | 113 (0.85x) | 97 (1.00x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global linear score int32 | **4041** | - | 3835 (1.05x) | 5288 (0.76x) | 6242 (0.65x) | 2339 (1.73x) | 4433 (0.91x) | 1419 (2.85x) | - | - | - | - |
| global linear score int64 | **1522** | - | 1405 (1.08x) | 1517 (1.00x) | >10x | 727 (2.09x) | 922 (1.65x) | - | - | - | - | - |
| global linear score float32 (load!) | **4332** | fill=striped: 4329 | 4395 (0.99x) | 5640 (0.77x) | - | >10x | >10x | - | - | - | - | - |
| global linear score float64 (load!) | **1827** | fill=striped: 1841 | 1818 (1.01x) | 1776 (1.03x) | - | 274 (6.67x) | 384 (4.76x) | - | 671 (2.73x) | - | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global linear path int32 (load!) | **3211** | - | 3340 (0.96x) | 3258 (0.99x) | 1751 (1.83x) | 371 (8.65x) | 1401 (2.29x) | 1257 (2.55x) | - | - | >10x | - |
| global linear path int64 (load!) | **772** | - | 858 (0.90x) | 774 (1.00x) | 82 (9.38x) | 321 (2.41x) | 1404 (0.55x) | - | - | - | - | - |
| global linear path float32 | **2779** | traceback=scores: 3427 | 2811 (0.99x) | 2771 (1.00x) | - | >10x | 804 (3.45x) | - | - | - | - | - |
| global linear path float64 (load!) | **1527** | traceback=scores: 1891 | 1489 (1.03x) | 1524 (1.00x) | - | 261 (5.85x) | 797 (1.92x) | - | 230 (6.64x) | 285 (5.36x) | - | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global linear hard int32 (load!) | **3296** | - | 3370 (0.98x) | 3347 (0.98x) | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **773** | - | 865 (0.89x) | 781 (0.99x) | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **2838** | traceback=scores: 3546 | 2817 (1.01x) | 2830 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1537** | traceback=scores: 1910 | 1538 (1.00x) | 1543 (1.00x) | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **199** | fill=striped: 208 | 200 (0.99x) | 204 (0.97x) | - | - | - | - | - | - | - | >10x |
| local affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local affine score int32 (load!) | **3757** | - | 3721 (1.01x) | 4574 (0.82x) | 3552 (1.06x) | 1449 (2.59x) | 2360 (1.59x) | - | - | - | - | - |
| local affine score int64 (load!) | **1221** | - | 1363 (0.90x) | 1242 (0.98x) | >10x | 500 (2.44x) | 684 (1.79x) | - | - | - | - | - |
| local affine score float32 (load!) | **3248** | fill=striped: 3126 | 3336 (0.97x) | 3256 (1.00x) | - | >10x | >10x | - | - | - | - | - |
| local affine score float64 (load!) | **1581** | fill=striped: 1578 | 1593 (0.99x) | 1562 (1.01x) | - | 176 (9.01x) | 232 (6.81x) | - | 282 (5.61x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local affine path int32 (load!) | **1171** | - | 1189 (0.99x) | 1184 (0.99x) | 1653 (0.71x) | 349 (3.36x) | 441 (2.66x) | - | - | - | - | - |
| local affine path int64 (load!) | **427** | - | 462 (0.92x) | 428 (1.00x) | 100 (4.26x) | 310 (1.38x) | 362 (1.18x) | - | - | - | - | - |
| local affine path float32 (load!) | **1114** | traceback=scores: 1378 | 1147 (0.97x) | 1110 (1.00x) | - | 173 (6.43x) | 202 (5.52x) | - | - | - | - | - |
| local affine path float64 (load!) | **652** | traceback=grid: 653 | 652 (1.00x) | 655 (1.00x) | - | 175 (3.72x) | 202 (3.23x) | - | 71 (9.20x) | 128 (5.11x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local affine hard int32 (load!) | **1189** | - | 1204 (0.99x) | 1188 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **428** | - | 439 (0.98x) | 428 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1120** | traceback=scores: 1344 | 1154 (0.97x) | 1117 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **656** | traceback=pointers: 660 | 637 (1.03x) | 655 (1.00x) | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **84** | fill=striped: 84 | 103 (0.81x) | 84 (1.00x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local linear score int32 (load!) | **4161** | - | 3653 (1.14x) | 4478 (0.93x) | 6828 (0.61x) | 1448 (2.87x) | 2374 (1.75x) | - | - | - | - | - |
| local linear score int64 (load!) | **1243** | - | 1365 (0.91x) | 1245 (1.00x) | >10x | 501 (2.48x) | 1238 (1.00x) | - | - | - | - | - |
| local linear score float32 (load!) | **4118** | fill=striped: 4130 | 4110 (1.00x) | 5707 (0.72x) | - | >10x | >10x | - | - | - | - | - |
| local linear score float64 (load!) | **1824** | fill=striped: 1825 | 1832 (1.00x) | 1828 (1.00x) | - | >10x | 292 (6.25x) | - | 544 (3.35x) | - | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local linear path int32 (load!) | **2377** | - | 2468 (0.96x) | 2425 (0.98x) | 1695 (1.40x) | 351 (6.77x) | 531 (4.47x) | - | - | - | - | - |
| local linear path int64 (load!) | **445** | - | 505 (0.88x) | 452 (0.99x) | 99 (4.50x) | 309 (1.44x) | 269 (1.66x) | - | - | - | - | - |
| local linear path float32 (load!) | **1810** | traceback=scores: 2303 | 1734 (1.04x) | 1816 (1.00x) | - | >10x | 216 (8.37x) | - | - | - | - | - |
| local linear path float64 (load!) | **937** | traceback=scores: 1211 | 894 (1.05x) | 932 (1.01x) | - | 164 (5.73x) | 202 (4.64x) | - | 134 (7.00x) | 278 (3.37x) | - | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local linear hard int32 (load!) | **2428** | - | 2579 (0.94x) | 2427 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **448** | - | 504 (0.89x) | 447 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **1838** | traceback=scores: 2307 | 1758 (1.05x) | 1840 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **945** | traceback=scores: 1216 | 897 (1.05x) | 940 (1.01x) | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **181** | fill=striped: 181 | 178 (1.01x) | 181 (1.00x) | - | - | - | - | - | - | - | - |

#### nighthaven · protein_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global affine score int32 (load!) | **2671** | - | 2820 (0.95x) | 2777 (0.96x) | 3154 (0.85x) | 1017 (2.63x) | 1070 (2.50x) | 1526 (1.75x) | - | - | - |
| global affine score int64 (load!) | **1240** | - | 1306 (0.95x) | 1252 (0.99x) | >10x | 613 (2.02x) | 916 (1.35x) | - | - | - | - |
| global affine score float32 (load!) | **2226** | fill=striped: 2256 | 2256 (0.99x) | 2225 (1.00x) | - | 278 (8.02x) | 287 (7.75x) | - | - | - | - |
| global affine score float64 (load!) | **1402** | fill=rowwise: 1401 | 1408 (1.00x) | 1390 (1.01x) | - | 278 (5.05x) | 287 (4.88x) | - | 380 (3.69x) | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global affine path int32 | **1321** | - | 1285 (1.03x) | 1323 (1.00x) | 735 (1.80x) | 261 (5.06x) | 316 (4.18x) | 908 (1.45x) | - | - | - |
| global affine path int64 | **519** | - | 501 (1.04x) | 519 (1.00x) | 112 (4.62x) | 253 (2.05x) | 308 (1.69x) | - | - | - | - |
| global affine path float32 | **2479** | fill=striped: 2483 | 2488 (1.00x) | 2481 (1.00x) | - | >10x | 267 (9.28x) | - | - | - | - |
| global affine path float64 | **1623** | traceback=grid: 1701 | 1707 (0.95x) | 1689 (0.96x) | - | 216 (7.52x) | 267 (6.07x) | - | >10x | >10x | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global affine hard int32 | **1322** | - | 1359 (0.97x) | 1323 (1.00x) | - | - | - | - | - | - | - |
| global affine hard int64 | **519** | - | 518 (1.00x) | 519 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float32 | **2479** | fill=striped: 2480 | 2489 (1.00x) | 2478 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float64 | **1698** | traceback=grid: 1704 | 1706 (1.00x) | 1680 (1.01x) | - | - | - | - | - | - | - |
| global affine soft float64 | **133** | fill=striped: 133 | 140 (0.95x) | 132 (1.01x) | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global linear score int32 | **2775** | - | 2821 (0.98x) | 2778 (1.00x) | 4954 (0.56x) | 1014 (2.74x) | 1530 (1.81x) | 1584 (1.75x) | - | - | 787 (3.53x) sat |
| global linear score int64 | **1255** | - | 1309 (0.96x) | 1244 (1.01x) | >10x | 615 (2.04x) | 917 (1.37x) | - | - | - | - |
| global linear score float32 | **3454** | api=functions: 3455 | 3458 (1.00x) | 3456 (1.00x) | - | >10x | 383 (9.01x) | - | - | - | - |
| global linear score float64 | **1691** | api=functions: 1734 | 1739 (0.97x) | 1731 (0.98x) | - | 278 (6.08x) | 383 (4.42x) | - | 657 (2.57x) | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global linear path int32 | **1536** | - | 1460 (1.05x) | 1516 (1.01x) | 632 (2.43x) | 294 (5.23x) | 1065 (1.44x) | 910 (1.69x) | - | - | 206 (7.47x) sat |
| global linear path int64 | **696** | - | 743 (0.94x) | 695 (1.00x) | 71 (9.75x) | 292 (2.39x) | 1078 (0.65x) | - | - | - | - |
| global linear path float32 | **2245** | fill=striped: 2253 | 2214 (1.01x) | 2282 (0.98x) | - | 243 (9.25x) | 677 (3.32x) | - | - | - | - |
| global linear path float64 | **1139** | fill=striped: 1137 | 1121 (1.02x) | 1138 (1.00x) | - | 240 (4.75x) | 677 (1.68x) | - | 213 (5.35x) | 171 (6.67x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global linear hard int32 | **1538** | - | 1511 (1.02x) | 1541 (1.00x) | - | - | - | - | - | - | - |
| global linear hard int64 | **695** | - | 746 (0.93x) | 696 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float32 | **2252** | fill=striped: 2260 | 2222 (1.01x) | 2286 (0.99x) | - | - | - | - | - | - | - |
| global linear hard float64 | **1141** | fill=striped: 1138 | 1145 (1.00x) | 1139 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 | **260** | fill=striped: 261 | 264 (0.98x) | 260 (1.00x) | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local affine score int32 | **2598** | - | 2623 (0.99x) | 2605 (1.00x) | 2639 (0.98x) | 462 (5.62x) | 963 (2.70x) | - | - | - | - |
| local affine score int64 | **1193** | - | 1211 (0.98x) | 1196 (1.00x) | >10x | 446 (2.68x) | 685 (1.74x) | - | - | - | - |
| local affine score float32 | **2494** | fill=striped: 2488 | 2491 (1.00x) | 2489 (1.00x) | - | >10x | >10x | - | - | - | - |
| local affine score float64 | **1528** | fill=rowwise: 1500 | 1502 (1.02x) | 1499 (1.02x) | - | 176 (8.71x) | 230 (6.65x) | - | 317 (4.82x) | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local affine path int32 | **560** | - | 567 (0.99x) | 560 (1.00x) | 620 (0.90x) | 286 (1.96x) | 361 (1.55x) | - | - | - | - |
| local affine path int64 | **210** | - | 214 (0.98x) | 210 (1.00x) | 98 (2.14x) | 284 (0.74x) | 366 (0.57x) | - | - | - | - |
| local affine path float32 | **519** | traceback=grid: 1082 | 519 (1.00x) | 520 (1.00x) | - | 170 (3.05x) | 194 (2.68x) | - | - | - | - |
| local affine path float64 | **634** | traceback=grid: 643 | 579 (1.09x) | 642 (0.99x) | - | 171 (3.71x) | 194 (3.27x) | - | 68 (9.33x) | >10x | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local affine hard int32 | **546** | - | 568 (0.96x) | 560 (0.98x) | - | - | - | - | - | - | - |
| local affine hard int64 | **210** | - | 213 (0.98x) | 210 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float32 | **520** | traceback=grid: 1083 | 517 (1.01x) | 520 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 | **639** | fill=striped: 643 | 573 (1.11x) | 645 (0.99x) | - | - | - | - | - | - | - |
| local affine soft float64 | **105** | fill=striped: 105 | 114 (0.93x) | 105 (1.00x) | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local linear score int32 | **2603** | - | 2644 (0.98x) | 2608 (1.00x) | 3295 (0.79x) | 467 (5.58x) | 1527 (1.70x) | - | - | - | 522 (4.98x) sat |
| local linear score int64 | **1196** | - | 1219 (0.98x) | 1192 (1.00x) | 122 (9.77x) | 446 (2.68x) | 1246 (0.96x) | - | - | - | - |
| local linear score float32 | **3463** | api=functions: 3463 | 3454 (1.00x) | 3463 (1.00x) | - | >10x | >10x | - | - | - | - |
| local linear score float64 | **1739** | fill=striped: 1737 | 1729 (1.01x) | 1551 (1.12x) | - | 176 (9.90x) | 287 (6.06x) | - | 646 (2.69x) | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local linear path int32 | **1370** | - | 1373 (1.00x) | 1387 (0.99x) | 657 (2.08x) | 286 (4.79x) | 245 (5.60x) | - | - | - | 441 (3.11x) sat |
| local linear path int64 | **452** | - | 476 (0.95x) | 455 (0.99x) | 86 (5.26x) | 285 (1.59x) | 246 (1.83x) | - | - | - | - |
| local linear path float32 | **1673** | fill=striped: 1675 | 1534 (1.09x) | 1673 (1.00x) | - | >10x | 200 (8.34x) | - | - | - | - |
| local linear path float64 | **823** | fill=striped: 838 | 771 (1.07x) | 837 (0.98x) | - | 149 (5.52x) | 195 (4.23x) | - | 129 (6.38x) | 162 (5.08x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local linear hard int32 | **1379** | - | 1373 (1.00x) | 1377 (1.00x) | - | - | - | - | - | - | - |
| local linear hard int64 | **453** | - | 477 (0.95x) | 455 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **1673** | traceback=pointers: 1677 | 1537 (1.09x) | 1672 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **836** | fill=striped: 840 | 762 (1.10x) | 830 (1.01x) | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **228** | fill=striped: 228 | 228 (1.00x) | 228 (1.00x) | - | - | - | - | - | - | - |

#### nighthaven · protein_L300 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| global affine score int32 (load!) | **4580** | - | 4799 (0.95x) | 4605 (0.99x) | 3351 (1.37x) | 2525 (1.81x) | 4162 (1.10x) | 1197 (3.83x) | - | - | - | - |
| global affine score int64 | **1441** | - | 1554 (0.93x) | 1440 (1.00x) | >10x | 785 (1.84x) | 910 (1.58x) | - | - | - | - | - |
| global affine score float32 | **2262** | fill=striped: 2262 | 2397 (0.94x) | 2256 (1.00x) | - | 290 (7.80x) | 304 (7.45x) | - | - | - | - | - |
| global affine score float64 | **1898** | fill=striped: 1895 | 1975 (0.96x) | 1896 (1.00x) | - | 290 (6.54x) | 304 (6.25x) | - | 388 (4.90x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| global affine path int32 | **1629** | - | 1511 (1.08x) | 1652 (0.99x) | 1766 (0.92x) | 877 (1.86x) | 2033 (0.80x) | 1117 (1.46x) | - | - | - | - |
| global affine path int64 | **577** | - | 632 (0.91x) | 577 (1.00x) | 114 (5.09x) | 406 (1.42x) | 324 (1.78x) | - | - | - | - | - |
| global affine path float32 | **1086** | traceback=scores: 1479 | 1222 (0.89x) | 1099 (0.99x) | - | 220 (4.94x) | 279 (3.90x) | - | - | - | - | - |
| global affine path float64 | **804** | traceback=scores: 911 | 872 (0.92x) | 803 (1.00x) | - | 215 (3.73x) | 278 (2.89x) | - | 109 (7.38x) | 175 (4.58x) | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| global affine hard int32 | **1707** | - | 1891 (0.90x) | 1697 (1.01x) | - | - | - | - | - | - | - | - |
| global affine hard int64 | **584** | - | 642 (0.91x) | 585 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1134** | traceback=scores: 1537 | 1265 (0.90x) | 1119 (1.01x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **816** | traceback=scores: 933 | 897 (0.91x) | 821 (0.99x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **107** | fill=striped: 107 | 120 (0.89x) | 107 (1.00x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| global linear score int32 | **4585** | - | 5042 (0.91x) | 4616 (0.99x) | 4419 (1.04x) | 2520 (1.82x) | 5156 (0.89x) | 1216 (3.77x) | - | - | - | - |
| global linear score int64 | **1441** | - | 1553 (0.93x) | 1440 (1.00x) | >10x | 784 (1.84x) | 933 (1.54x) | - | - | - | - | - |
| global linear score float32 | **3684** | fill=striped: 3572 | 3885 (0.95x) | 3716 (0.99x) | - | >10x | 420 (8.76x) | - | - | - | - | - |
| global linear score float64 | **2640** | fill=striped: 2641 | 2570 (1.03x) | 2639 (1.00x) | - | 290 (9.09x) | 422 (6.26x) | - | 699 (3.78x) | - | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| global linear path int32 | **2682** | - | 2665 (1.01x) | 2687 (1.00x) | 1767 (1.52x) | 873 (3.07x) | 2486 (1.08x) | 1116 (2.40x) | - | - | - | - |
| global linear path int64 | **1418** | - | 1414 (1.00x) | 1416 (1.00x) | >10x | 408 (3.47x) | 1266 (1.12x) | - | - | - | - | - |
| global linear path float32 | **2468** | fill=striped: 2464 | 2616 (0.94x) | 2473 (1.00x) | - | 253 (9.74x) | 779 (3.17x) | - | - | - | - | - |
| global linear path float64 | **1593** | fill=striped: 1593 | 1721 (0.93x) | 1595 (1.00x) | - | 250 (6.36x) | 778 (2.05x) | - | 228 (7.00x) | 275 (5.80x) | - | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| global linear hard int32 | **2908** | - | 2934 (0.99x) | 2848 (1.02x) | - | - | - | - | - | - | - | - |
| global linear hard int64 | **1467** | - | 1462 (1.00x) | 1468 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **2699** | fill=striped: 2681 | 2856 (0.95x) | 2674 (1.01x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **1679** | fill=striped: 1678 | 1818 (0.92x) | 1672 (1.00x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **150** | api=functions: 160 | 144 (1.04x) | 150 (1.00x) | - | - | - | - | - | - | - | >10x |
| local affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| local affine score int32 | **4005** | - | 4182 (0.96x) | 4082 (0.98x) | 2769 (1.45x) | 1630 (2.46x) | 2490 (1.61x) | - | - | - | - | - |
| local affine score int64 | **1167** | - | 1360 (0.86x) | 1165 (1.00x) | >10x | 533 (2.19x) | 681 (1.71x) | - | - | - | - | - |
| local affine score float32 | **2553** | fill=striped: 2595 | 2664 (0.96x) | 2596 (0.98x) | - | >10x | >10x | - | - | - | - | - |
| local affine score float64 | **2043** | fill=rowwise: 2054 | 2097 (0.97x) | 2045 (1.00x) | - | >10x | 239 (8.55x) | - | 276 (7.40x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| local affine path int32 | **1378** | - | 1407 (0.98x) | 1362 (1.01x) | 1577 (0.87x) | 692 (1.99x) | 1006 (1.37x) | - | - | - | - | - |
| local affine path int64 | **427** | - | 463 (0.92x) | 426 (1.00x) | 99 (4.31x) | 384 (1.11x) | 344 (1.24x) | - | - | - | - | - |
| local affine path float32 | **972** | traceback=scores: 1293 | 983 (0.99x) | 969 (1.00x) | - | 166 (5.86x) | 199 (4.89x) | - | - | - | - | - |
| local affine path float64 | **631** | traceback=scores: 788 | 649 (0.97x) | 631 (1.00x) | - | 167 (3.79x) | 200 (3.16x) | - | 72 (8.82x) | 173 (3.64x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| local affine hard int32 | **1415** | - | 1456 (0.97x) | 1415 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard int64 | **430** | - | 468 (0.92x) | 428 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **993** | traceback=scores: 1328 | 929 (1.07x) | 988 (1.01x) | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **641** | traceback=scores: 791 | 659 (0.97x) | 639 (1.00x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **43** | fill=striped: 44 | 61 (0.71x) | 43 (1.00x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| local linear score int32 | **4065** | - | 3474 (1.17x) | 4071 (1.00x) | 4711 (0.86x) | 1629 (2.49x) | 2308 (1.76x) | - | - | - | - | - |
| local linear score int64 | **1167** | - | 1359 (0.86x) | 1166 (1.00x) | >10x | 534 (2.19x) | 1213 (0.96x) | - | - | - | - | - |
| local linear score float32 | **3779** | fill=striped: 3791 | 4050 (0.93x) | 3791 (1.00x) | - | >10x | >10x | - | - | - | - | - |
| local linear score float64 | **2597** | fill=striped: 2597 | 2641 (0.98x) | 2594 (1.00x) | - | >10x | 304 (8.53x) | - | 459 (5.65x) | - | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| local linear path int32 | **2452** | - | 2493 (0.98x) | 2437 (1.01x) | 1597 (1.54x) | 692 (3.54x) | 1500 (1.63x) | - | - | - | - | - |
| local linear path int64 | **1140** | - | 579 (1.97x) | 1109 (1.03x) | >10x | 375 (3.04x) | 265 (4.30x) | - | - | - | - | - |
| local linear path float32 | **1785** | fill=striped: 1793 | 1743 (1.02x) | 1799 (0.99x) | - | >10x | 211 (8.46x) | - | - | - | - | - |
| local linear path float64 | **1096** | traceback=scores: 1116 | 1097 (1.00x) | 1096 (1.00x) | - | 164 (6.67x) | 203 (5.40x) | - | 131 (8.36x) | 269 (4.08x) | - | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| local linear hard int32 | **2606** | - | 2676 (0.97x) | 2619 (0.99x) | - | - | - | - | - | - | - | - |
| local linear hard int64 | **1174** | - | 586 (2.00x) | 1160 (1.01x) | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **1891** | fill=striped: 1898 | 1855 (1.02x) | 1895 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **1134** | fill=striped: 1135 | 1129 (1.00x) | 1133 (1.00x) | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **101** | api=functions: 109 | 98 (1.03x) | 101 (1.00x) | - | - | - | - | - | - | - | - |

#### nighthaven · protein_L3000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global affine score int32 (load!) | **2902** | - | 2996 (0.97x) | 2910 (1.00x) | 3851 (0.75x) | 1022 (2.84x) | 1234 (2.35x) | 1582 (1.83x) | - | - | - | - |
| global affine score int64 (load!) | **1241** | - | 1308 (0.95x) | 1239 (1.00x) | >10x | 612 (2.03x) | 914 (1.36x) | - | - | - | - | - |
| global affine score float32 (load!) | **2162** | api=functions: 2151 | 2194 (0.99x) | 2158 (1.00x) | - | 279 (7.75x) | 288 (7.49x) | - | - | - | - | - |
| global affine score float64 (load!) | **1375** | fill=rowwise: 1398 | 1378 (1.00x) | 1372 (1.00x) | - | 279 (4.93x) | 288 (4.77x) | - | 381 (3.60x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global affine path int32 (load!) | **1495** | - | 1591 (0.94x) | 1493 (1.00x) | 887 (1.69x) | 266 (5.63x) | 351 (4.26x) | 1435 (1.04x) | - | - | - | - |
| global affine path int64 (load!) | **562** | - | 537 (1.05x) | 561 (1.00x) | 115 (4.87x) | 264 (2.13x) | 341 (1.65x) | - | - | - | - | - |
| global affine path float32 (load!) | **2216** | traceback=grid: 2241 | 2187 (1.01x) | 2201 (1.01x) | - | 223 (9.94x) | 294 (7.54x) | - | - | - | - | - |
| global affine path float64 (load!) | **1522** | fill=striped: 1541 | 1520 (1.00x) | 1520 (1.00x) | - | 220 (6.92x) | 289 (5.26x) | - | >10x | >10x | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global affine hard int32 (load!) | **1494** | - | 1589 (0.94x) | 1494 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **564** | - | 538 (1.05x) | 563 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **2213** | fill=striped: 2214 | 2200 (1.01x) | 2205 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1522** | fill=rowwise: 1525 | 1526 (1.00x) | 1524 (1.00x) | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **103** | fill=striped: 103 | 118 (0.87x) | 102 (1.01x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global linear score int32 (load!) | **2897** | - | 3003 (0.96x) | 2900 (1.00x) | 6478 (0.45x) | 1023 (2.83x) | 1858 (1.56x) | 1591 (1.82x) | - | - | - | - |
| global linear score int64 (load!) | **1238** | - | 1309 (0.95x) | 1240 (1.00x) | >10x | 613 (2.02x) | 916 (1.35x) | - | - | - | - | - |
| global linear score float32 (load!) | **3149** | fill=striped: 3145 | 3117 (1.01x) | 3385 (0.93x) | - | >10x | 385 (8.17x) | - | - | - | - | - |
| global linear score float64 (load!) | **1719** | api=functions: 1717 | 1714 (1.00x) | 1715 (1.00x) | - | 279 (6.16x) | 386 (4.45x) | - | 661 (2.60x) | - | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global linear path int32 (load!) | **1725** | - | 1720 (1.00x) | 1732 (1.00x) | 801 (2.15x) | 311 (5.55x) | 1451 (1.19x) | 1388 (1.24x) | - | - | - | - |
| global linear path int64 (load!) | **711** | - | 757 (0.94x) | 711 (1.00x) | 71 (9.96x) | 304 (2.34x) | 1447 (0.49x) | - | - | - | - | - |
| global linear path float32 (load!) | **2377** | traceback=scores: 2481 | 2300 (1.03x) | 2440 (0.97x) | - | 253 (9.40x) | 816 (2.91x) | - | - | - | - | - |
| global linear path float64 (load!) | **1133** | traceback=pointers: 1134 | 1119 (1.01x) | 1135 (1.00x) | - | 250 (4.54x) | 816 (1.39x) | - | 229 (4.94x) | 258 (4.39x) | - | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global linear hard int32 (load!) | **1740** | - | 1719 (1.01x) | 1734 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **711** | - | 756 (0.94x) | 711 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **2393** | traceback=scores: 2492 | 2312 (1.04x) | 2452 (0.98x) | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1137** | fill=striped: 1137 | 1123 (1.01x) | 1135 (1.00x) | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **235** | fill=striped: 235 | 234 (1.00x) | 235 (1.00x) | - | - | - | - | - | - | - | >10x |
| local affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local affine score int32 (load!) | **2701** | - | 2684 (1.01x) | 2701 (1.00x) | 3438 (0.79x) | 597 (4.53x) | 979 (2.76x) | - | - | - | - | - |
| local affine score int64 (load!) | **1179** | - | 1224 (0.96x) | 1180 (1.00x) | >10x | 449 (2.63x) | 639 (1.85x) | - | - | - | - | - |
| local affine score float32 (load!) | **2422** | fill=striped: 2422 | 2426 (1.00x) | 2410 (1.01x) | - | >10x | >10x | - | - | - | - | - |
| local affine score float64 (load!) | **1464** | fill=rowwise: 1496 | 1471 (1.00x) | 1463 (1.00x) | - | 176 (8.34x) | 230 (6.36x) | - | 281 (5.20x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local affine path int32 (load!) | **1014** | - | 1053 (0.96x) | 1011 (1.00x) | 740 (1.37x) | 291 (3.48x) | 361 (2.81x) | - | - | - | - | - |
| local affine path int64 (load!) | **423** | - | 404 (1.05x) | 424 (1.00x) | 100 (4.21x) | 287 (1.47x) | 368 (1.15x) | - | - | - | - | - |
| local affine path float32 (load!) | **1080** | traceback=pointers: 1076 | 1105 (0.98x) | 1080 (1.00x) | - | 170 (6.34x) | 202 (5.34x) | - | - | - | - | - |
| local affine path float64 (load!) | **631** | traceback=pointers: 630 | 577 (1.09x) | 630 (1.00x) | - | 172 (3.66x) | 202 (3.12x) | - | 73 (8.70x) | 118 (5.34x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local affine hard int32 (load!) | **1018** | - | 1052 (0.97x) | 1017 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **425** | - | 405 (1.05x) | 425 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1079** | traceback=grid: 1083 | 1118 (0.97x) | 1081 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **631** | traceback=pointers: 632 | 579 (1.09x) | 629 (1.00x) | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **75** | fill=striped: 75 | 89 (0.85x) | 74 (1.01x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local linear score int32 (load!) | **2623** | - | 2788 (0.94x) | 2702 (0.97x) | 5691 (0.46x) | 617 (4.25x) | 1524 (1.72x) | - | - | - | - | - |
| local linear score int64 (load!) | **1178** | - | 1183 (1.00x) | 1180 (1.00x) | 122 (9.64x) | 449 (2.63x) | 1241 (0.95x) | - | - | - | - | - |
| local linear score float32 (load!) | **3114** | fill=striped: 3120 | 3180 (0.98x) | 3392 (0.92x) | - | >10x | >10x | - | - | - | - | - |
| local linear score float64 (load!) | **1714** | fill=striped: 1749 | 1715 (1.00x) | 1718 (1.00x) | - | 176 (9.76x) | 288 (5.94x) | - | 607 (2.82x) | - | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local linear path int32 (load!) | **1391** | - | 1464 (0.95x) | 1444 (0.96x) | 781 (1.78x) | 292 (4.77x) | 268 (5.19x) | - | - | - | - | - |
| local linear path int64 (load!) | **448** | - | 485 (0.92x) | 454 (0.99x) | 85 (5.29x) | 287 (1.56x) | 270 (1.66x) | - | - | - | - | - |
| local linear path float32 (load!) | **1706** | traceback=scores: 1824 | 1585 (1.08x) | 1698 (1.00x) | - | >10x | 216 (7.88x) | - | - | - | - | - |
| local linear path float64 (load!) | **840** | fill=striped: 844 | 776 (1.08x) | 847 (0.99x) | - | 157 (5.36x) | 205 (4.09x) | - | 136 (6.15x) | 158 (5.30x) | - | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local linear hard int32 (load!) | **1400** | - | 1472 (0.95x) | 1439 (0.97x) | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **450** | - | 486 (0.93x) | 449 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **1711** | traceback=scores: 1845 | 1581 (1.08x) | 1707 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **845** | fill=striped: 845 | 795 (1.06x) | 836 (1.01x) | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **212** | fill=striped: 207 | 211 (1.00x) | 211 (1.00x) | - | - | - | - | - | - | - | - |

#### nighthaven · dna_L100 · 6 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **21054** | fill=striped: 21357 | 19394 (1.09x) | 21107 (1.00x) | 15078 (1.40x) | 23953 (0.88x) | 6694 (3.15x) | 4407 (4.78x) | 5237 (4.02x) | - | - | - | - |
| global affine score float32 | **20936** | schedule=sorted: 20828 | 20649 (1.01x) | 20998 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **10922** | schedule=sorted: 10883 | 10746 (1.02x) | 10909 (1.00x) | - | - | - | - | - | - | 1928 (5.67x) | - | - |
| global affine path int32 | **11069** | schedule=sorted: 11093 | 10414 (1.06x) | 11056 (1.00x) | 7971 (1.39x) | 7076 (1.56x) | 2250 (4.92x) | 4067 (2.72x) | 5236 (2.11x) | - | - | - | - |
| global affine path float32 | **11209** | schedule=sorted: 11217 | 10800 (1.04x) | 11308 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 | **6641** | schedule=sorted: 6675 | 6243 (1.06x) | 6667 (1.00x) | - | - | - | - | - | - | >10x | >10x | - |
| global affine hard int32 | **12493** | schedule=sorted: 12693 | 12397 (1.01x) | 12267 (1.02x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **12724** | schedule=sorted: 13121 | 12806 (0.99x) | 13145 (0.97x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **7289** | schedule=sorted: 7404 | 6755 (1.08x) | 7265 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **1134** | fill=striped: 1138 | 1115 (1.02x) | 1127 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 | **27031** | schedule=sorted: 27044 | 22274 (1.21x) | 27057 (1.00x) | 15271 (1.77x) | 24420 (1.11x) | 6416 (4.21x) | 4574 (5.91x) | 60844 (0.44x) | 18771 (1.44x) | - | - | >10x |
| global linear score float32 | **25010** | schedule=sorted: 25226 | 23963 (1.04x) | 25196 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **13147** | schedule=sorted: 13164 | 12485 (1.05x) | 13189 (1.00x) | - | - | - | - | - | - | 3597 (3.65x) | - | - |
| global linear path int32 | **15424** | schedule=sorted: 15370 | 14026 (1.10x) | 15482 (1.00x) | 8005 (1.93x) | 7066 (2.18x) | 3609 (4.27x) | 4214 (3.66x) | 43705 (0.35x) | 6687 (2.31x) | - | - | - |
| global linear path float32 | **14725** | schedule=sorted: 14714 | 14486 (1.02x) | 14519 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 | **9685** | schedule=sorted: 9700 | 9348 (1.04x) | 9609 (1.01x) | - | - | - | - | - | - | >10x | >10x | - |
| global linear hard int32 | **20234** | schedule=sorted: 20200 | 17683 (1.14x) | 20458 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **19527** | schedule=sorted: 19621 | 18875 (1.03x) | 19391 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **11320** | schedule=sorted: 11324 | 10756 (1.05x) | 11352 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **1989** | fill=striped: 2012 | 1725 (1.15x) | 2006 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **19090** | fill=striped: 19648 | 17670 (1.08x) | 19176 (1.00x) | 15001 (1.27x) | 9984 (1.91x) | 2522 (7.57x) | - | - | - | - | - | - |
| local affine score float32 | **18320** | schedule=sorted: 18273 | 18091 (1.01x) | 18350 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **9409** | fill=striped: 10879 | 9388 (1.00x) | 9385 (1.00x) | - | - | - | - | - | - | 1185 (7.94x) | - | - |
| local affine path int32 | **9929** | schedule=sorted: 9958 | 9603 (1.03x) | 9937 (1.00x) | 7204 (1.38x) | 6059 (1.64x) | 1090 (9.11x) | - | - | - | - | - | - |
| local affine path float32 | **9145** | schedule=sorted: 9120 | 9549 (0.96x) | 9215 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **5264** | schedule=sorted: 5245 | 5538 (0.95x) | 5253 (1.00x) | - | - | - | - | - | - | >10x | 659 (7.99x) | - |
| local affine hard int32 | **11359** | schedule=sorted: 11353 | 10701 (1.06x) | 11348 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **10177** | schedule=sorted: 10174 | 10973 (0.93x) | 10255 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **5592** | schedule=sorted: 5606 | 6056 (0.92x) | 5609 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **1036** | schedule=sorted: 1040 | 968 (1.07x) | 912 (1.14x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 | **23591** | schedule=sorted: 23573 | 20475 (1.15x) | 23469 (1.01x) | 12128 (1.95x) | 9939 (2.37x) | 2429 (9.71x) | - | - | - | - | - | - |
| local linear score float32 | **16329** | schedule=sorted: 16315 | 15509 (1.05x) | 16355 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **8373** | fill=striped: 10890 | 7935 (1.06x) | 8382 (1.00x) | - | - | - | - | - | - | 2370 (3.53x) | - | - |
| local linear path int32 | **12692** | schedule=sorted: 12918 | 11549 (1.10x) | 13025 (0.97x) | 7060 (1.80x) | 6202 (2.05x) | 1673 (7.59x) | - | - | - | - | - | - |
| local linear path float32 | **10922** | schedule=sorted: 10913 | 10718 (1.02x) | 10969 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **6599** | schedule=sorted: 6599 | 6461 (1.02x) | 6609 (1.00x) | - | - | - | - | - | - | >10x | 922 (7.16x) | - |
| local linear hard int32 | **16062** | schedule=sorted: 16157 | 14348 (1.12x) | 16169 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **13111** | schedule=sorted: 13112 | 13086 (1.00x) | 13171 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **7319** | schedule=sorted: 7312 | 7172 (1.02x) | 7321 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **1715** | schedule=sorted: 1725 | 1510 (1.14x) | 1715 (1.00x) | - | - | - | - | - | - | - | - | - |

#### nighthaven · dna_L1000 · 6 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **29565** | fill=striped: 29385 | 35724 (0.83x) | 29669 (1.00x) | 24271 (1.22x) | 26119 (1.13x) | 5742 (5.15x) | 7632 (3.87x) | 18900 (1.56x) | - | - | - | - |
| global affine score float32 | **21419** | schedule=sorted: 21523 | 22107 (0.97x) | 21623 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **8946** | fill=striped: 8892 | 8927 (1.00x) | 8810 (1.02x) | - | - | - | - | - | - | 1940 (4.61x) | - | - |
| global affine path int32 | **10474** | schedule=sorted: 10474 | 12684 (0.83x) | 10488 (1.00x) | 7636 (1.37x) | 1119 (9.36x) | >10x | 7023 (1.49x) | 6884 (1.52x) | - | - | - | >10x |
| global affine path float32 | **13775** | schedule=sorted: 13853 | 13495 (1.02x) | 13419 (1.03x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 | **8384** | schedule=sorted: 8366 | 8347 (1.00x) | 8330 (1.01x) | - | - | - | - | - | - | >10x | >10x | - |
| global affine hard int32 | **10520** | schedule=sorted: 10510 | 12787 (0.82x) | 10510 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **13777** | schedule=sorted: 13871 | 13522 (1.02x) | 13808 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **8481** | fill=striped: 8427 | 8413 (1.01x) | 8384 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **272** | fill=striped: 270 | 282 (0.96x) | 273 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 | **29462** | schedule=sorted: 29747 | 35723 (0.82x) | 29754 (0.99x) | 37000 (0.80x) | 26325 (1.12x) | 7295 (4.04x) | 7596 (3.88x) | 273949 (0.11x) | 171906 (0.17x) | - | - | - |
| global linear score float32 | **35714** | fill=striped: 35878 | 36035 (0.99x) | 35898 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **9803** | fill=striped: 9943 | 9939 (0.99x) | 10051 (0.98x) | - | - | - | - | - | - | 3375 (2.90x) | - | - |
| global linear path int32 | **17895** | fill=striped: 17774 | 18178 (0.98x) | 17791 (1.01x) | 7544 (2.37x) | >10x | >10x | 6972 (2.57x) | 174975 (0.10x) | 62322 (0.29x) | - | - | >10x |
| global linear path float32 | **15844** | schedule=sorted: 15881 | 15872 (1.00x) | 15757 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 | **8578** | fill=striped: 8522 | 8585 (1.00x) | 8592 (1.00x) | - | - | - | - | - | - | >10x | 1312 (6.54x) | - |
| global linear hard int32 | **18255** | fill=striped: 18162 | 18785 (0.97x) | 18378 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **16203** | schedule=sorted: 16382 | 16142 (1.00x) | 15994 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **8626** | schedule=sorted: 8638 | 8752 (0.99x) | 8631 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **932** | schedule=sorted: 938 | 936 (1.00x) | 932 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **24798** | fill=striped: 25153 | 25862 (0.96x) | 24761 (1.00x) | 32255 (0.77x) | 11067 (2.24x) | 2781 (8.92x) | - | - | - | - | - | - |
| local affine score float32 | **28189** | schedule=sorted: 28490 | 28962 (0.97x) | 28639 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **9834** | schedule=sorted: 9799 | 9773 (1.01x) | 9799 (1.00x) | - | - | - | - | - | - | 1290 (7.62x) | - | - |
| local affine path int32 | **6273** | fill=striped: 6353 | 6358 (0.99x) | 6356 (0.99x) | 7529 (0.83x) | 1115 (5.62x) | >10x | - | - | - | - | - | - |
| local affine path float32 | **7344** | fill=striped: 7428 | 7454 (0.99x) | 7439 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **3835** | schedule=sorted: 3855 | 3644 (1.05x) | 3865 (0.99x) | - | - | - | - | - | - | 414 (9.27x) | >10x | - |
| local affine hard int32 | **6301** | fill=striped: 6426 | 6383 (0.99x) | 6371 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **7494** | fill=striped: 7500 | 7546 (0.99x) | 7486 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **3887** | schedule=sorted: 3858 | 3723 (1.04x) | 3908 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **160** | fill=striped: 160 | 207 (0.77x) | 160 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 | **25159** | fill=striped: 25286 | 25916 (0.97x) | 24995 (1.01x) | 40749 (0.62x) | 11137 (2.26x) | 2952 (8.52x) | - | - | - | - | - | - |
| local linear score float32 | **33465** | schedule=sorted: 32964 | 33620 (1.00x) | 33484 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **10123** | fill=striped: 10050 | 10111 (1.00x) | 9942 (1.02x) | - | - | - | - | - | - | 3190 (3.17x) | - | - |
| local linear path int32 | **12815** | schedule=sorted: 13157 | 13791 (0.93x) | 12462 (1.03x) | 7670 (1.67x) | >10x | >10x | - | - | - | - | - | - |
| local linear path float32 | **10057** | schedule=sorted: 10042 | 9637 (1.04x) | 10088 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **5001** | fill=striped: 5009 | 4821 (1.04x) | 4997 (1.00x) | - | - | - | - | - | - | 562 (8.90x) | 951 (5.26x) | - |
| local linear hard int32 | **13404** | schedule=sorted: 13230 | 13918 (0.96x) | 13219 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **10228** | schedule=sorted: 10197 | 9792 (1.04x) | 10227 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **5042** | fill=striped: 5068 | 4897 (1.03x) | 5035 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **934** | fill=striped: 938 | 942 (0.99x) | 939 (1.00x) | - | - | - | - | - | - | - | - | - |

#### nighthaven · mirna · 6 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **14353** | schedule=sorted: 14958 | 12690 (1.13x) | 14293 (1.00x) | 5592 (2.57x) | 7438 (1.93x) | 5167 (2.78x) | 2862 (5.01x) | >10x | - | - | - | >10x |
| global affine score float32 | **14674** | schedule=sorted: 14162 | 12281 (1.19x) | 14345 (1.02x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **9526** | schedule=sorted: 7620 | 7254 (1.31x) | 9530 (1.00x) | - | - | - | - | - | - | 1015 (9.39x) | - | - |
| global affine path int32 | **4647** | schedule=sorted: 4663 | 4629 (1.00x) | 4713 (0.99x) | 3447 (1.35x) | 4944 (0.94x) | 997 (4.66x) | 2412 (1.93x) | >10x | - | - | - | - |
| global affine path float32 | **4810** | schedule=sorted: 4788 | 4746 (1.01x) | 4805 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 | **3642** | schedule=sorted: 3658 | 3589 (1.01x) | 3671 (0.99x) | - | - | - | - | - | - | >10x | >10x | - |
| global affine hard int32 | **6859** | schedule=sorted: 6952 | 6998 (0.98x) | 6944 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **6333** | schedule=sorted: 7200 | 7309 (0.87x) | 7211 (0.88x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **5014** | schedule=sorted: 5015 | 4939 (1.02x) | 5013 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **1418** | schedule=sorted: 1416 | 1300 (1.09x) | 1416 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 | **19829** | schedule=sorted: 20311 | 17096 (1.16x) | 20495 (0.97x) | 5579 (3.55x) | 8333 (2.38x) | 5113 (3.88x) | 3287 (6.03x) | 4221 (4.70x) | 3212 (6.17x) | - | - | >10x |
| global linear score float32 | **19277** | schedule=sorted: 19248 | 16835 (1.15x) | 19906 (0.97x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **11315** | schedule=sorted: 11645 | 9874 (1.15x) | 11457 (0.99x) | - | - | - | - | - | - | 1357 (8.34x) | - | - |
| global linear path int32 | **4015** | schedule=sorted: 4017 | 4117 (0.98x) | 4247 (0.95x) | 3313 (1.21x) | 5059 (0.79x) | 979 (4.10x) | 2624 (1.53x) | 3693 (1.09x) | 1246 (3.22x) | - | - | - |
| global linear path float32 | **3849** | schedule=sorted: 3874 | 4030 (0.95x) | 3844 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 | **3469** | schedule=sorted: 3459 | 3481 (1.00x) | 3494 (0.99x) | - | - | - | - | - | - | 416 (8.33x) | >10x | - |
| global linear hard int32 | **7894** | schedule=sorted: 7909 | 7544 (1.05x) | 7951 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **7678** | schedule=sorted: 7510 | 7425 (1.03x) | 7285 (1.05x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **5727** | schedule=sorted: 5882 | 5484 (1.04x) | 5807 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **2161** | schedule=sorted: 2161 | 2005 (1.08x) | 2150 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **13138** | schedule=sorted: 15557 | 12313 (1.07x) | 14112 (0.93x) | 7724 (1.70x) | 6309 (2.08x) | 2287 (5.75x) | - | - | - | - | - | >10x |
| local affine score float32 | **12795** | schedule=sorted: 12884 | 13373 (0.96x) | 13075 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **7325** | fill=striped: 7217 | 6247 (1.17x) | 6545 (1.12x) | - | - | - | - | - | - | >10x | - | - |
| local affine path int32 | **8077** | schedule=sorted: 8092 | 7955 (1.02x) | 8119 (0.99x) | 4039 (2.00x) | 4433 (1.82x) | 874 (9.24x) | - | - | - | - | - | >10x |
| local affine path float32 | **7853** | schedule=sorted: 7842 | 7978 (0.98x) | 7835 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **4744** | schedule=sorted: 4744 | 4825 (0.98x) | 4707 (1.01x) | - | - | - | - | - | - | >10x | >10x | - |
| local affine hard int32 | **7952** | schedule=sorted: 7996 | 8015 (0.99x) | 8211 (0.97x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **7910** | schedule=sorted: 8000 | 8153 (0.97x) | 7842 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **4807** | schedule=sorted: 4806 | 4948 (0.97x) | 4760 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **1496** | schedule=sorted: 1497 | 1358 (1.10x) | 1488 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 | **18866** | schedule=sorted: 18438 | 15667 (1.20x) | 18687 (1.01x) | 5316 (3.55x) | 7021 (2.69x) | 2387 (7.90x) | - | - | - | - | - | >10x |
| local linear score float32 | **14706** | schedule=sorted: 13983 | 12481 (1.18x) | 14524 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **7836** | schedule=sorted: 7919 | 6822 (1.15x) | 7964 (0.98x) | - | - | - | - | - | - | 953 (8.22x) | - | - |
| local linear path int32 | **8051** | schedule=sorted: 7990 | 7420 (1.09x) | 8251 (0.98x) | 3752 (2.15x) | 4306 (1.87x) | 1057 (7.62x) | - | - | - | - | - | >10x |
| local linear path float32 | **7259** | schedule=sorted: 7258 | 7172 (1.01x) | 7271 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **4958** | schedule=sorted: 4967 | 4810 (1.03x) | 4997 (0.99x) | - | - | - | - | - | - | >10x | >10x | - |
| local linear hard int32 | **10008** | schedule=sorted: 9976 | 8928 (1.12x) | 9874 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **8679** | schedule=sorted: 8853 | 8435 (1.03x) | 8544 (1.02x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **5608** | schedule=sorted: 5555 | 5247 (1.07x) | 5474 (1.02x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **1962** | schedule=sorted: 1963 | 1819 (1.08x) | 1962 (1.00x) | - | - | - | - | - | - | - | - | - |

#### nighthaven · protein_L100 · 6 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **15763** | schedule=sorted: 15705 | 18733 (0.84x) | 15791 (1.00x) | 10271 (1.53x) | 11474 (1.37x) | 3899 (4.04x) | 4311 (3.66x) | - | - | >10x |
| global affine score float32 | **7950** | schedule=sorted: 7918 | 8381 (0.95x) | 7911 (1.00x) | - | - | - | - | - | - | - |
| global affine score float64 | **7792** | schedule=sorted: 7833 | 8126 (0.96x) | 7789 (1.00x) | - | - | - | - | 1945 (4.01x) | - | - |
| global affine path int32 | **7063** | schedule=sorted: 6988 | 6419 (1.10x) | 7128 (0.99x) | 6286 (1.12x) | 4752 (1.49x) | 1870 (3.78x) | 4006 (1.76x) | - | - | - |
| global affine path float32 | **6832** | schedule=sorted: 6849 | 5867 (1.16x) | 6777 (1.01x) | - | - | - | - | - | - | - |
| global affine path float64 | **5022** | schedule=sorted: 4907 | 4901 (1.02x) | 4963 (1.01x) | - | - | - | - | 527 (9.54x) | 656 (7.65x) | - |
| global affine hard int32 | **6325** | schedule=sorted: 6349 | 5750 (1.10x) | 6392 (0.99x) | - | - | - | - | - | - | - |
| global affine hard float32 | **6168** | schedule=sorted: 6181 | 5635 (1.09x) | 6179 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float64 | **4618** | schedule=sorted: 4667 | 4499 (1.03x) | 4639 (1.00x) | - | - | - | - | - | - | - |
| global affine soft float64 | **1068** | fill=striped: 1058 | 1028 (1.04x) | 1072 (1.00x) | - | - | - | - | - | - | - |
| global linear score int32 | **15455** | schedule=sorted: 15656 | 18252 (0.85x) | 15724 (0.98x) | 10324 (1.50x) | 11754 (1.31x) | 4901 (3.15x) | 4435 (3.49x) | - | - | - |
| global linear score float32 | **9667** | schedule=sorted: 9696 | 10562 (0.92x) | 9656 (1.00x) | - | - | - | - | - | - | - |
| global linear score float64 | **8741** | schedule=sorted: 8803 | 9284 (0.94x) | 8804 (0.99x) | - | - | - | - | 3668 (2.38x) | - | - |
| global linear path int32 | **10656** | fill=striped: 10559 | 11463 (0.93x) | 10428 (1.02x) | 6220 (1.71x) | 5088 (2.09x) | 2718 (3.92x) | 4095 (2.60x) | - | - | - |
| global linear path float32 | **8117** | schedule=sorted: 8046 | 6833 (1.19x) | 8119 (1.00x) | - | - | - | - | - | - | - |
| global linear path float64 | **6519** | schedule=sorted: 6485 | 6072 (1.07x) | 6433 (1.01x) | - | - | - | - | 1034 (6.31x) | 910 (7.16x) | - |
| global linear hard int32 | **9623** | schedule=sorted: 9736 | 10456 (0.92x) | 9757 (0.99x) | - | - | - | - | - | - | - |
| global linear hard float32 | **7459** | schedule=sorted: 7465 | 6528 (1.14x) | 7510 (0.99x) | - | - | - | - | - | - | - |
| global linear hard float64 | **6119** | schedule=sorted: 6080 | 5675 (1.08x) | 6099 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 | **1377** | fill=striped: 1389 | 1237 (1.11x) | 1378 (1.00x) | - | - | - | - | - | - | - |
| local affine score int32 | **14766** | fill=striped: 14727 | 15764 (0.94x) | 14598 (1.01x) | 8590 (1.72x) | 7545 (1.96x) | 2397 (6.16x) | - | - | - | >10x |
| local affine score float32 | **8700** | schedule=sorted: 8675 | 9174 (0.95x) | 8679 (1.00x) | - | - | - | - | - | - | - |
| local affine score float64 | **8114** | schedule=sorted: 8088 | 8461 (0.96x) | 8114 (1.00x) | - | - | - | - | 1319 (6.15x) | - | - |
| local affine path int32 | **6826** | schedule=sorted: 6964 | 6286 (1.09x) | 6786 (1.01x) | 5652 (1.21x) | 4540 (1.50x) | 928 (7.36x) | - | - | - | - |
| local affine path float32 | **6573** | schedule=sorted: 6538 | 6046 (1.09x) | 6573 (1.00x) | - | - | - | - | - | - | - |
| local affine path float64 | **4761** | schedule=sorted: 4698 | 4697 (1.01x) | 4768 (1.00x) | - | - | - | - | >10x | 654 (7.28x) | - |
| local affine hard int32 | **6276** | schedule=sorted: 6376 | 5651 (1.11x) | 6281 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float32 | **6003** | schedule=sorted: 6014 | 5456 (1.10x) | 6050 (0.99x) | - | - | - | - | - | - | - |
| local affine hard float64 | **4469** | schedule=sorted: 4469 | 4474 (1.00x) | 4486 (1.00x) | - | - | - | - | - | - | - |
| local affine soft float64 | **958** | fill=striped: 964 | 910 (1.05x) | 847 (1.13x) | - | - | - | - | - | - | - |
| local linear score int32 | **14762** | schedule=sorted: 14764 | 15823 (0.93x) | 14545 (1.01x) | 8721 (1.69x) | 6353 (2.32x) | 2447 (6.03x) | - | - | - | >10x |
| local linear score float32 | **10737** | schedule=sorted: 10717 | 11587 (0.93x) | 10718 (1.00x) | - | - | - | - | - | - | - |
| local linear score float64 | **9141** | schedule=sorted: 9152 | 9659 (0.95x) | 9102 (1.00x) | - | - | - | - | 1885 (4.85x) | - | - |
| local linear path int32 | **8601** | fill=striped: 8726 | 6994 (1.23x) | 8591 (1.00x) | 5552 (1.55x) | 4520 (1.90x) | 1346 (6.39x) | - | - | - | - |
| local linear path float32 | **6600** | schedule=sorted: 6559 | 5910 (1.12x) | 6638 (0.99x) | - | - | - | - | - | - | - |
| local linear path float64 | **4807** | schedule=sorted: 4821 | 4551 (1.06x) | 4858 (0.99x) | - | - | - | - | 622 (7.72x) | 904 (5.32x) | - |
| local linear hard int32 | **7804** | fill=striped: 8094 | 6554 (1.19x) | 7808 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float32 | **6166** | schedule=sorted: 6169 | 5656 (1.09x) | 6239 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float64 | **4615** | schedule=sorted: 4592 | 4271 (1.08x) | 4615 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **1545** | fill=striped: 1553 | 1399 (1.10x) | 1534 (1.01x) | - | - | - | - | - | - | - |

#### nighthaven · protein_L1000 · 6 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **28221** | schedule=sorted: 28359 | 33642 (0.84x) | 27785 (1.02x) | 22100 (1.28x) | 12273 (2.30x) | 3858 (7.32x) | 7605 (3.71x) | - | - | - |
| global affine score float32 | **15287** | schedule=sorted: 15268 | 16053 (0.95x) | 15145 (1.01x) | - | - | - | - | - | - | - |
| global affine score float64 | **7652** | schedule=sorted: 7811 | 7639 (1.00x) | 7783 (0.98x) | - | - | - | - | 1963 (3.90x) | - | - |
| global affine path int32 | **9655** | fill=striped: 9657 | 11346 (0.85x) | 9690 (1.00x) | 6487 (1.49x) | 988 (9.77x) | >10x | 7043 (1.37x) | - | - | >10x |
| global affine path float32 | **10994** | fill=striped: 10988 | 10709 (1.03x) | 10761 (1.02x) | - | - | - | - | - | - | - |
| global affine path float64 | **7377** | fill=striped: 7393 | 7332 (1.01x) | 7409 (1.00x) | - | - | - | - | >10x | 811 (9.10x) | - |
| global affine hard int32 | **9694** | fill=striped: 9689 | 11406 (0.85x) | 9688 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float32 | **10946** | fill=striped: 11009 | 10812 (1.01x) | 10846 (1.01x) | - | - | - | - | - | - | - |
| global affine hard float64 | **7410** | fill=striped: 7403 | 7395 (1.00x) | 7412 (1.00x) | - | - | - | - | - | - | - |
| global affine soft float64 | **283** | fill=striped: 283 | 285 (0.99x) | 282 (1.00x) | - | - | - | - | - | - | - |
| global linear score int32 | **28003** | schedule=sorted: 28439 | 33868 (0.83x) | 27793 (1.01x) | 22350 (1.25x) | 12234 (2.29x) | 5030 (5.57x) | 7593 (3.69x) | - | - | - |
| global linear score float32 | **31998** | fill=striped: 31905 | 32865 (0.97x) | 32040 (1.00x) | - | - | - | - | - | - | - |
| global linear score float64 | **9691** | schedule=sorted: 9756 | 9803 (0.99x) | 9818 (0.99x) | - | - | - | - | 3418 (2.84x) | - | - |
| global linear path int32 | **17469** | schedule=sorted: 17459 | 17908 (0.98x) | 17255 (1.01x) | 6343 (2.75x) | >10x | >10x | 7010 (2.49x) | - | - | >10x |
| global linear path float32 | **14975** | fill=striped: 14983 | 15034 (1.00x) | 14846 (1.01x) | - | - | - | - | - | - | - |
| global linear path float64 | **8211** | fill=striped: 8195 | 8167 (1.01x) | 8217 (1.00x) | - | - | - | - | 1158 (7.09x) | 1422 (5.77x) | - |
| global linear hard int32 | **17738** | fill=striped: 17773 | 18292 (0.97x) | 16720 (1.06x) | - | - | - | - | - | - | - |
| global linear hard float32 | **15208** | schedule=sorted: 15286 | 15189 (1.00x) | 15074 (1.01x) | - | - | - | - | - | - | - |
| global linear hard float64 | **8289** | schedule=sorted: 8308 | 8278 (1.00x) | 8231 (1.01x) | - | - | - | - | - | - | - |
| global linear soft float64 | **869** | schedule=sorted: 868 | 864 (1.01x) | 869 (1.00x) | - | - | - | - | - | - | - |
| local affine score int32 | **23838** | fill=striped: 24432 | 24371 (0.98x) | 24541 (0.97x) | 18438 (1.29x) | 7265 (3.28x) | 2516 (9.48x) | - | - | - | - |
| local affine score float32 | **17530** | fill=striped: 17620 | 18065 (0.97x) | 17554 (1.00x) | - | - | - | - | - | - | - |
| local affine score float64 | **8412** | schedule=sorted: 8324 | 8410 (1.00x) | 8411 (1.00x) | - | - | - | - | 1441 (5.84x) | - | - |
| local affine path int32 | **6213** | schedule=sorted: 6295 | 6261 (0.99x) | 6298 (0.99x) | 6347 (0.98x) | 991 (6.27x) | >10x | - | - | - | - |
| local affine path float32 | **6039** | schedule=sorted: 6033 | 6199 (0.97x) | 6021 (1.00x) | - | - | - | - | - | - | - |
| local affine path float64 | **3521** | fill=striped: 3504 | 3426 (1.03x) | 3475 (1.01x) | - | - | - | - | 362 (9.74x) | 749 (4.70x) | - |
| local affine hard int32 | **6191** | schedule=sorted: 6346 | 6256 (0.99x) | 6200 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float32 | **6063** | fill=striped: 6068 | 6230 (0.97x) | 6053 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 | **3487** | fill=striped: 3526 | 3494 (1.00x) | 3526 (0.99x) | - | - | - | - | - | - | - |
| local affine soft float64 | **277** | schedule=sorted: 279 | 287 (0.97x) | 277 (1.00x) | - | - | - | - | - | - | - |
| local linear score int32 | **24481** | fill=striped: 24481 | 24245 (1.01x) | 24568 (1.00x) | 18629 (1.31x) | 7194 (3.40x) | 2589 (9.45x) | - | - | - | - |
| local linear score float32 | **30732** | schedule=sorted: 30326 | 31236 (0.98x) | 30905 (0.99x) | - | - | - | - | - | - | - |
| local linear score float64 | **9717** | fill=striped: 9727 | 9701 (1.00x) | 9728 (1.00x) | - | - | - | - | 2749 (3.53x) | - | - |
| local linear path int32 | **12567** | schedule=sorted: 12896 | 13400 (0.94x) | 12927 (0.97x) | 6625 (1.90x) | >10x | >10x | - | - | - | - |
| local linear path float32 | **9776** | fill=striped: 9777 | 9397 (1.04x) | 9812 (1.00x) | - | - | - | - | - | - | - |
| local linear path float64 | **4914** | fill=striped: 4926 | 4767 (1.03x) | 4925 (1.00x) | - | - | - | - | 684 (7.19x) | 1341 (3.66x) | - |
| local linear hard int32 | **13105** | schedule=sorted: 13210 | 13622 (0.96x) | 13030 (1.01x) | - | - | - | - | - | - | - |
| local linear hard float32 | **9896** | fill=striped: 9916 | 9532 (1.04x) | 9924 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float64 | **4944** | fill=striped: 4974 | 4799 (1.03x) | 4950 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **795** | fill=striped: 796 | 789 (1.01x) | 794 (1.00x) | - | - | - | - | - | - | - |

#### nighthaven · protein_L300 · 6 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **24697** | fill=striped: 24676 | 28736 (0.86x) | 24324 (1.02x) | 17441 (1.42x) | 13241 (1.87x) | 4166 (5.93x) | 6287 (3.93x) | - | - | - |
| global affine score float32 | **12244** | fill=striped: 12282 | 12857 (0.95x) | 12281 (1.00x) | - | - | - | - | - | - | - |
| global affine score float64 | **10298** | fill=striped: 10303 | 10786 (0.95x) | 10283 (1.00x) | - | - | - | - | 2027 (5.08x) | - | - |
| global affine path int32 | **8762** | fill=striped: 8756 | 9616 (0.91x) | 8795 (1.00x) | 9390 (0.93x) | 2995 (2.93x) | 2166 (4.04x) | 5887 (1.49x) | - | - | - |
| global affine path float32 | **6029** | schedule=sorted: 6032 | 6594 (0.91x) | 5976 (1.01x) | - | - | - | - | - | - | - |
| global affine path float64 | **4339** | schedule=sorted: 4365 | 4751 (0.91x) | 4393 (0.99x) | - | - | - | - | 574 (7.55x) | 926 (4.69x) | - |
| global affine hard int32 | **8723** | schedule=sorted: 8836 | 9815 (0.89x) | 8840 (0.99x) | - | - | - | - | - | - | - |
| global affine hard float32 | **6068** | schedule=sorted: 6077 | 6748 (0.90x) | 5998 (1.01x) | - | - | - | - | - | - | - |
| global affine hard float64 | **4387** | schedule=sorted: 4385 | 4818 (0.91x) | 4423 (0.99x) | - | - | - | - | - | - | - |
| global affine soft float64 | **582** | fill=striped: 580 | 602 (0.97x) | 542 (1.07x) | - | - | - | - | - | - | - |
| global linear score int32 | **24828** | fill=striped: 24854 | 28704 (0.86x) | 24763 (1.00x) | 17583 (1.41x) | 13208 (1.88x) | 5389 (4.61x) | 6364 (3.90x) | - | - | - |
| global linear score float32 | **19847** | fill=striped: 20085 | 21464 (0.92x) | 20063 (0.99x) | - | - | - | - | - | - | - |
| global linear score float64 | **14279** | schedule=sorted: 14352 | 14635 (0.98x) | 14301 (1.00x) | - | - | - | - | 3398 (4.20x) | - | - |
| global linear path int32 | **14677** | schedule=sorted: 14759 | 14941 (0.98x) | 14689 (1.00x) | 9385 (1.56x) | 2916 (5.03x) | 3147 (4.66x) | 5925 (2.48x) | - | - | - |
| global linear path float32 | **13239** | schedule=sorted: 13253 | 14213 (0.93x) | 13298 (1.00x) | - | - | - | - | - | - | - |
| global linear path float64 | **8670** | schedule=sorted: 8682 | 9323 (0.93x) | 8683 (1.00x) | - | - | - | - | 1210 (7.17x) | 1442 (6.01x) | - |
| global linear hard int32 | **15158** | schedule=sorted: 14930 | 15315 (0.99x) | 15360 (0.99x) | - | - | - | - | - | - | - |
| global linear hard float32 | **13765** | fill=striped: 13801 | 14754 (0.93x) | 13904 (0.99x) | - | - | - | - | - | - | - |
| global linear hard float64 | **8844** | schedule=sorted: 8931 | 9598 (0.92x) | 8883 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 | **815** | schedule=sorted: 815 | 786 (1.04x) | 814 (1.00x) | - | - | - | - | - | - | - |
| local affine score int32 | **21434** | fill=striped: 22021 | 21994 (0.97x) | 22051 (0.97x) | 14527 (1.48x) | 8147 (2.63x) | 2473 (8.67x) | - | - | - | - |
| local affine score float32 | **14094** | fill=striped: 14095 | 14525 (0.97x) | 14076 (1.00x) | - | - | - | - | - | - | - |
| local affine score float64 | **11125** | schedule=sorted: 11149 | 11403 (0.98x) | 11102 (1.00x) | - | - | - | - | 1455 (7.65x) | - | - |
| local affine path int32 (load!) | **7447** | schedule=sorted: 7474 | 7429 (1.00x) | 7431 (1.00x) | 8458 (0.88x) | 2917 (2.55x) | 1011 (7.36x) | - | - | - | - |
| local affine path float32 (load!) | **5286** | fill=striped: 5286 | 5348 (0.99x) | 5287 (1.00x) | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **3437** | schedule=sorted: 3436 | 3535 (0.97x) | 3440 (1.00x) | - | - | - | - | 378 (9.10x) | 916 (3.75x) | - |
| local affine hard int32 | **7560** | fill=striped: 7545 | 7584 (1.00x) | 7560 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float32 | **5307** | schedule=sorted: 5325 | 5440 (0.98x) | 5315 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 | **3457** | fill=striped: 3456 | 3565 (0.97x) | 3465 (1.00x) | - | - | - | - | - | - | - |
| local affine soft float64 | **241** | fill=striped: 238 | 321 (0.75x) | 234 (1.03x) | - | - | - | - | - | - | - |
| local linear score int32 | **21840** | schedule=sorted: 21999 | 22180 (0.98x) | 21950 (0.99x) | 14472 (1.51x) | 8363 (2.61x) | 2572 (8.49x) | - | - | - | - |
| local linear score float32 | **20610** | fill=striped: 20576 | 21995 (0.94x) | 20481 (1.01x) | - | - | - | - | - | - | - |
| local linear score float64 | **14054** | fill=striped: 14109 | 14337 (0.98x) | 14042 (1.00x) | - | - | - | - | 2397 (5.86x) | - | - |
| local linear path int32 | **13289** | fill=striped: 13226 | 12718 (1.04x) | 13178 (1.01x) | 8417 (1.58x) | 2922 (4.55x) | 1506 (8.82x) | - | - | - | - |
| local linear path float32 | **9527** | fill=striped: 9538 | 9500 (1.00x) | 9604 (0.99x) | - | - | - | - | - | - | - |
| local linear path float64 | **5966** | fill=striped: 5970 | 5905 (1.01x) | 5978 (1.00x) | - | - | - | - | 695 (8.59x) | 1413 (4.22x) | - |
| local linear hard int32 | **13152** | fill=striped: 13660 | 13670 (0.96x) | 13319 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float32 | **9790** | schedule=sorted: 9828 | 9822 (1.00x) | 9863 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float64 | **6066** | schedule=sorted: 6068 | 6036 (1.00x) | 6082 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **554** | fill=striped: 555 | 537 (1.03x) | 555 (1.00x) | - | - | - | - | - | - | - |

#### nighthaven · protein_L3000 · 6 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **14912** | fill=striped: 14789 | 14705 (1.01x) | 14493 (1.03x) | 18926 (0.79x) | 8830 (1.69x) | 3289 (4.53x) | 7644 (1.95x) | - | - | - |
| global affine score float32 | **10978** | schedule=sorted: 11201 | 11253 (0.98x) | 10867 (1.01x) | - | - | - | - | - | - | - |
| global affine score float64 | **7105** | schedule=sorted: 7151 | 7158 (0.99x) | 7056 (1.01x) | - | - | - | - | 1938 (3.67x) | - | - |
| global affine path int32 | **7697** | fill=striped: 7701 | 8248 (0.93x) | 7669 (1.00x) | 2268 (3.39x) | >10x | >10x | 6534 (1.18x) | - | - | >10x |
| global affine path float32 | **11405** | schedule=sorted: 11409 | 11391 (1.00x) | 11353 (1.00x) | - | - | - | - | - | - | - |
| global affine path float64 | **7773** | fill=striped: 7763 | 7813 (0.99x) | 7764 (1.00x) | - | - | - | - | >10x | >10x | - |
| global affine hard int32 | **7763** | schedule=sorted: 7645 | 8150 (0.95x) | 7678 (1.01x) | - | - | - | - | - | - | - |
| global affine hard float32 | **11379** | fill=striped: 11424 | 11351 (1.00x) | 11351 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float64 | **7779** | fill=striped: 7781 | 7807 (1.00x) | 7719 (1.01x) | - | - | - | - | - | - | - |
| global affine soft float64 | **301** | schedule=sorted: 298 | 305 (0.99x) | 300 (1.00x) | - | - | - | - | - | - | - |
| global linear score int32 | **14783** | schedule=sorted: 15006 | 14988 (0.99x) | 14868 (0.99x) | 32576 (0.45x) | 8746 (1.69x) | 4402 (3.36x) | 7556 (1.96x) | - | - | - |
| global linear score float32 | **16876** | schedule=sorted: 17366 | 17219 (0.98x) | 17334 (0.97x) | - | - | - | - | - | - | - |
| global linear score float64 | **8668** | fill=striped: 8948 | 8927 (0.97x) | 8892 (0.97x) | - | - | - | - | 3357 (2.58x) | - | - |
| global linear path int32 | **8186** | fill=striped: 8661 | 8465 (0.97x) | 8723 (0.94x) | 2251 (3.64x) | >10x | >10x | 6527 (1.25x) | - | - | >10x |
| global linear path float32 | **12000** | schedule=sorted: 12045 | 11833 (1.01x) | 12091 (0.99x) | - | - | - | - | - | - | - |
| global linear path float64 | **5664** | schedule=sorted: 5861 | 5775 (0.98x) | 5870 (0.96x) | - | - | - | - | 1129 (5.02x) | 695 (8.15x) | - |
| global linear hard int32 | **8582** | fill=striped: 8717 | 8517 (1.01x) | 8708 (0.99x) | - | - | - | - | - | - | - |
| global linear hard float32 | **12072** | schedule=sorted: 12037 | 11908 (1.01x) | 12318 (0.98x) | - | - | - | - | - | - | - |
| global linear hard float64 | **5888** | fill=striped: 5930 | 5831 (1.01x) | 5879 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 | **846** | schedule=sorted: 853 | 858 (0.99x) | 851 (0.99x) | - | - | - | - | - | - | - |
| local affine score int32 | **13723** | fill=striped: 13751 | 14149 (0.97x) | 13724 (1.00x) | 17113 (0.80x) | 5133 (2.67x) | 2260 (6.07x) | - | - | - | - |
| local affine score float32 | **12075** | schedule=sorted: 12419 | 12591 (0.96x) | 12262 (0.98x) | - | - | - | - | - | - | - |
| local affine score float64 | **7658** | schedule=sorted: 7642 | 7542 (1.02x) | 7583 (1.01x) | - | - | - | - | 1440 (5.32x) | - | - |
| local affine path int32 | **5108** | fill=striped: 5033 | 5111 (1.00x) | 5032 (1.02x) | 2238 (2.28x) | 735 (6.95x) | >10x | - | - | - | - |
| local affine path float32 | **5463** | schedule=sorted: 5363 | 5345 (1.02x) | 5440 (1.00x) | - | - | - | - | - | - | - |
| local affine path float64 | **3196** | fill=striped: 3222 | 2820 (1.13x) | 3190 (1.00x) | - | - | - | - | 361 (8.86x) | >10x | - |
| local affine hard int32 | **5068** | fill=striped: 5074 | 5228 (0.97x) | 4998 (1.01x) | - | - | - | - | - | - | - |
| local affine hard float32 | **5556** | fill=striped: 5514 | 5600 (0.99x) | 5537 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 | **3183** | fill=striped: 3236 | 2879 (1.11x) | 3236 (0.98x) | - | - | - | - | - | - | - |
| local affine soft float64 | **262** | schedule=sorted: 263 | 273 (0.96x) | 262 (1.00x) | - | - | - | - | - | - | - |
| local linear score int32 | **13413** | schedule=sorted: 13730 | 13999 (0.96x) | 13486 (0.99x) | 32182 (0.42x) | 5160 (2.60x) | 2402 (5.58x) | - | - | - | - |
| local linear score float32 | **16709** | fill=striped: 16960 | 17356 (0.96x) | 17126 (0.98x) | - | - | - | - | - | - | - |
| local linear score float64 | **8849** | schedule=sorted: 8881 | 8957 (0.99x) | 8878 (1.00x) | - | - | - | - | 3071 (2.88x) | - | - |
| local linear path int32 | **6869** | schedule=sorted: 6952 | 6866 (1.00x) | 6995 (0.98x) | 2230 (3.08x) | 734 (9.36x) | >10x | - | - | - | - |
| local linear path float32 | **8575** | fill=striped: 8607 | 8012 (1.07x) | 8600 (1.00x) | - | - | - | - | - | - | - |
| local linear path float64 | **4294** | schedule=sorted: 4308 | 4045 (1.06x) | 4286 (1.00x) | - | - | - | - | 682 (6.30x) | 639 (6.72x) | - |
| local linear hard int32 | **6994** | fill=striped: 6989 | 6850 (1.02x) | 7015 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float32 | **8623** | fill=striped: 8555 | 8047 (1.07x) | 8668 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float64 | **4276** | schedule=sorted: 4279 | 3988 (1.07x) | 4286 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **810** | fill=striped: 810 | 805 (1.01x) | 806 (1.00x) | - | - | - | - | - | - | - |

</details>

<details><summary><b>oldsolace</b> — i7-7700HQ (Kaby Lake), AVX2 — x86-64-v3: every cell</summary>


#### oldsolace · dna_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **3652** | - | 3327 (1.10x) | 4113 (0.89x) | 1756 (2.08x) | 6278 (0.58x) | 453 (8.06x) | - | - | - | - | - | - | >10x | - |
| global affine score int32 | **2303** | - | 1784 (1.29x) | 2344 (0.98x) | 1719 (1.34x) | 3171 (0.73x) | 488 (4.72x) | - | 427 (5.39x) | 431 (5.34x) | - | - | - | - | - |
| global affine score int64 | **841** | - | 764 (1.10x) | 843 (1.00x) | 100 (8.37x) | 465 (1.81x) | 479 (1.76x) | - | - | - | - | - | - | - | - |
| global affine score float32 | **1982** | kernel=sse4: 1099 | 1659 (1.19x) | 2036 (0.97x) | - | 236 (8.40x) | 245 (8.10x) | - | - | - | - | - | - | - | - |
| global affine score float64 | **1024** | fill=striped: 834 | 875 (1.17x) | 1023 (1.00x) | - | 242 (4.23x) | 242 (4.23x) | - | - | - | - | 224 (4.57x) | - | - | - |
| global affine path int16 | **1706** | - | 1506 (1.13x) | 1665 (1.02x) | 1119 (1.52x) | 1731 (0.99x) | >10x | - | - | - | - | - | - | 204 (8.37x) sat | - |
| global affine path int32 | **1069** | - | 1001 (1.07x) | 1204 (0.89x) | 872 (1.23x) | 742 (1.44x) | 205 (5.22x) | - | 409 (2.62x) | 454 (2.36x) | - | - | - | - | - |
| global affine path int64 | **506** | - | 449 (1.13x) | 540 (0.94x) | 82 (6.18x) | 127 (4.00x) | 166 (3.05x) | - | - | - | - | - | - | - | - |
| global affine path float32 | **1044** | traceback=hirschberg_pmax: 1140 | 995 (1.05x) | 1132 (0.92x) | - | 133 (7.84x) | 156 (6.70x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **669** | traceback=grid: 674 | 620 (1.08x) | 675 (0.99x) | - | 133 (5.03x) | 165 (4.05x) | - | - | - | - | >10x | >10x | - | - |
| global affine hard int16 | **2163** | - | 2119 (1.02x) | 2291 (0.94x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **1295** | - | 1148 (1.13x) | 1185 (1.09x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **588** | - | 527 (1.12x) | 621 (0.95x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1275** | traceback=grid: 1329 | 1167 (1.09x) | 1303 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **713** | traceback=pointers: 718 | 630 (1.13x) | 744 (0.96x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **117** | api=functions: 137 | 127 (0.92x) | 117 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **4675** | - | 3807 (1.23x) | 4965 (0.94x) | 1771 (2.64x) | 6427 (0.73x) | 646 (7.23x) | - | - | - | - | - | - | >10x | - |
| global linear score int32 | **2602** | - | 2181 (1.19x) | 2860 (0.91x) | 1770 (1.47x) | 3190 (0.82x) | 635 (4.09x) | - | 488 (5.33x) | 6217 (0.42x) | 2487 (1.05x) | - | - | >10x | - |
| global linear score int64 | **1058** | - | 1069 (0.99x) | 1155 (0.92x) | >10x | 439 (2.41x) | 620 (1.71x) | - | - | - | - | - | - | - | - |
| global linear score float32 | **2859** | kernel=sse4: 1484 | 2045 (1.40x) | 2870 (1.00x) | - | >10x | 334 (8.55x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **1468** | fill=striped: 1144 | 1011 (1.45x) | 1494 (0.98x) | - | 242 (6.07x) | 328 (4.48x) | - | - | - | - | 360 (4.08x) | - | - | - |
| global linear path int16 | **2003** | - | 1766 (1.13x) | 1928 (1.04x) | 1077 (1.86x) | 1700 (1.18x) | 267 (7.50x) | - | - | - | - | - | - | >10x | - |
| global linear path int32 | **1605** | - | 1337 (1.20x) | 1617 (0.99x) | 928 (1.73x) | 729 (2.20x) | 263 (6.10x) | - | 430 (3.73x) | 5351 (0.30x) | 758 (2.12x) | - | - | - | - |
| global linear path int64 | **816** | - | 811 (1.01x) | 901 (0.91x) | 87 (9.36x) | 129 (6.34x) | 245 (3.33x) | - | - | - | - | - | - | - | - |
| global linear path float32 | **1510** | traceback=pointers: 1498 | 1351 (1.12x) | 1519 (0.99x) | - | >10x | 296 (5.11x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **1002** | traceback=pointers: 1001 | 865 (1.16x) | 1033 (0.97x) | - | 127 (7.92x) | 276 (3.63x) | - | - | - | - | 110 (9.12x) | >10x | - | - |
| global linear hard int16 | **3251** | - | 2595 (1.25x) | 3455 (0.94x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **2236** | - | 1586 (1.41x) | 2060 (1.09x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **1077** | - | 971 (1.11x) | 1074 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **2137** | traceback=scores: 2144 | 1811 (1.18x) | 2192 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **1271** | traceback=pointers: 1273 | 1031 (1.23x) | 1281 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **209** | api=functions: 240 | 193 (1.08x) | 210 (1.00x) | - | - | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 | **3454** | - | 3017 (1.14x) | 3468 (1.00x) | 2080 (1.66x) | 2727 (1.27x) | 419 (8.24x) | 1694 (2.04x) sat | - | - | - | - | - | >10x | - |
| local affine score int32 | **1985** | - | 1699 (1.17x) | 1991 (1.00x) | 1687 (1.18x) | 1475 (1.35x) | 462 (4.29x) | - | - | - | - | - | - | - | - |
| local affine score int64 | **698** | - | 702 (0.99x) | 725 (0.96x) | 93 (7.54x) | 266 (2.62x) | 418 (1.67x) | - | - | - | - | - | - | - | - |
| local affine score float32 | **1720** | fill=striped: 1367 | 1512 (1.14x) | 1912 (0.90x) | - | >10x | 177 (9.73x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **852** | fill=striped: 1055 | 869 (0.98x) | 851 (1.00x) | - | 136 (6.27x) | 175 (4.85x) | - | - | - | - | 136 (6.27x) | - | - | - |
| local affine path int16 | **1492** | - | 1401 (1.06x) | 1532 (0.97x) | 1026 (1.45x) | 1512 (0.99x) | >10x | 392 (3.81x) sat | - | - | - | - | - | >10x | - |
| local affine path int32 | **972** | - | 955 (1.02x) | 994 (0.98x) | 819 (1.19x) | 674 (1.44x) | 106 (9.14x) | - | - | - | - | - | - | - | - |
| local affine path int64 | **425** | - | 429 (0.99x) | 416 (1.02x) | 75 (5.67x) | 92 (4.60x) | 117 (3.64x) | - | - | - | - | - | - | - | - |
| local affine path float32 | **837** | traceback=pointers: 906 | 893 (0.94x) | 924 (0.91x) | - | 93 (9.01x) | 101 (8.26x) | - | - | - | - | - | - | - | - |
| local affine path float64 | **527** | traceback=grid: 529 | 498 (1.06x) | 549 (0.96x) | - | 81 (6.50x) | 104 (5.07x) | - | - | - | - | >10x | >10x | - | - |
| local affine hard int16 | **1764** | - | 1831 (0.96x) | 1946 (0.91x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int32 | **1091** | - | 1127 (0.97x) | 1082 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **450** | - | 459 (0.98x) | 450 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **1045** | traceback=scores: 1044 | 1041 (1.00x) | 1060 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **572** | traceback=hirschberg_pmax: 572 | 588 (0.97x) | 574 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **112** | api=functions: 145 | 118 (0.94x) | 111 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **4089** | - | 3587 (1.14x) | 3712 (1.10x) | 1554 (2.63x) | 2756 (1.48x) | >10x | wrong | - | - | - | - | - | >10x | - |
| local linear score int32 | **2396** | - | 2191 (1.09x) | 2044 (1.17x) | 1416 (1.69x) | 1421 (1.69x) | 542 (4.42x) | - | - | - | - | - | - | - | - |
| local linear score int64 | **946** | - | 901 (1.05x) | 932 (1.01x) | >10x | 261 (3.62x) | 607 (1.56x) | - | - | - | - | - | - | - | - |
| local linear score float32 | **2023** | fill=striped: 1563 | 1866 (1.08x) | 2083 (0.97x) | - | >10x | 231 (8.74x) | - | - | - | - | - | - | - | - |
| local linear score float64 | **1060** | fill=striped: 1112 | 958 (1.11x) | 1080 (0.98x) | - | 134 (7.91x) | 236 (4.49x) | - | - | - | - | 228 (4.65x) | - | - | - |
| local linear path int16 | **1757** | - | 1676 (1.05x) | 1797 (0.98x) | 986 (1.78x) | 1496 (1.17x) | >10x | wrong | - | - | - | - | - | 209 (8.41x) sat | - |
| local linear path int32 | **1287** | - | 1078 (1.19x) | 1301 (0.99x) | 798 (1.61x) | 714 (1.80x) | 159 (8.07x) | - | - | - | - | - | - | - | - |
| local linear path int64 | **652** | - | 624 (1.04x) | 665 (0.98x) | 74 (8.79x) | 88 (7.40x) | 165 (3.96x) | - | - | - | - | - | - | - | - |
| local linear path float32 | **1179** | traceback=pointers: 1178 | 1098 (1.07x) | 1195 (0.99x) | - | >10x | 137 (8.61x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **675** | traceback=hirschberg: 727 | 681 (0.99x) | 722 (0.93x) | - | 86 (7.84x) | 135 (4.99x) | - | - | - | - | >10x | >10x | - | - |
| local linear hard int16 | **2600** | - | 2278 (1.14x) | 2587 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **1639** | - | 1330 (1.23x) | 1689 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **743** | - | 703 (1.06x) | 745 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1492** | traceback=scores: 1497 | 1380 (1.08x) | 1483 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **822** | traceback=scores: 829 | 753 (1.09x) | 854 (0.96x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **183** | api=functions: 219 | 173 (1.06x) | 179 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |

#### oldsolace · dna_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| global affine score int32 | **2928** | - | 3280 (0.89x) | 2921 (1.00x) | 2951 (0.99x) | 3384 (0.87x) | 553 (5.29x) | - | 708 (4.14x) | 1852 (1.58x) | - | - | - | - | - |
| global affine score int64 | **910** | - | 974 (0.93x) | 907 (1.00x) | 103 (8.86x) | 478 (1.90x) | 543 (1.68x) | - | - | - | - | - | - | - | - |
| global affine score float32 | **1682** | fill=striped: 1545 | 1721 (0.98x) | 1671 (1.01x) | - | 237 (7.10x) | 231 (7.27x) | - | - | - | - | - | - | - | - |
| global affine score float64 | **888** | api=functions: 942 | 899 (0.99x) | 978 (0.91x) | - | 236 (3.76x) | 237 (3.74x) | - | - | - | - | 259 (3.42x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| global affine path int32 | **1013** | - | 1137 (0.89x) | 1029 (0.98x) | 1249 (0.81x) | 678 (1.49x) | 229 (4.42x) | - | 633 (1.60x) | 1253 (0.81x) | - | - | - | - | - |
| global affine path int64 | **403** | - | 378 (1.07x) | 374 (1.08x) | 89 (4.55x) | 145 (2.78x) | 177 (2.28x) | - | - | - | - | - | - | - | - |
| global affine path float32 | **1167** | fill=striped: 1159 | 1157 (1.01x) | 1136 (1.03x) | - | 157 (7.45x) | 181 (6.43x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **840** | fill=rowwise: 842 | 779 (1.08x) | 804 (1.04x) | - | 157 (5.35x) | 191 (4.39x) | - | - | - | - | >10x | 146 (5.74x) | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| global affine hard int32 | **1116** | - | 1230 (0.91x) | 1088 (1.03x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **394** | - | 398 (0.99x) | 393 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1086** | traceback=grid: 1153 | 1122 (0.97x) | 1150 (0.94x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **828** | fill=striped: 844 | 835 (0.99x) | 849 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **56** | fill=rowwise: 59 | 68 (0.82x) | 59 (0.95x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **5184** | - | 6650 (0.78x) | 5844 (0.89x) | 7777 (0.67x) | 6940 (0.75x) | 711 (7.30x) | - | - | - | - | - | - | >10x | - |
| global linear score int32 | **2682** | - | 2963 (0.91x) | 3021 (0.89x) | 4502 (0.60x) | 3162 (0.85x) | 673 (3.99x) | - | 700 (3.83x) | 28506 (0.09x) | 22241 (0.12x) | - | - | - | - |
| global linear score int64 | **829** | - | 884 (0.94x) | 839 (0.99x) | 101 (8.22x) | 479 (1.73x) | 714 (1.16x) | - | - | - | - | - | - | - | - |
| global linear score float32 | **3125** | fill=striped: 3253 | 3074 (1.02x) | 3227 (0.97x) | - | >10x | 317 (9.84x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **1143** | fill=striped: 1151 | 1107 (1.03x) | 1203 (0.95x) | - | 231 (4.94x) | 299 (3.82x) | - | - | - | - | 447 (2.56x) | - | - | - |
| global linear path int16 | **3232** | - | 3409 (0.95x) | 3330 (0.97x) | 2447 (1.32x) | 1353 (2.39x) | >10x | - | - | - | - | - | - | >10x | - |
| global linear path int32 | **1702** | - | 1558 (1.09x) | 1647 (1.03x) | 1293 (1.32x) | 702 (2.43x) | 285 (5.98x) | - | 664 (2.56x) | 23386 (0.07x) | 5679 (0.30x) | - | - | >10x | - |
| global linear path int64 | **461** | - | 467 (0.99x) | 467 (0.99x) | 90 (5.12x) | 133 (3.46x) | 290 (1.59x) | - | - | - | - | - | - | - | - |
| global linear path float32 | **1528** | traceback=scores: 2100 | 1675 (0.91x) | 1622 (0.94x) | - | >10x | 349 (4.38x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **759** | traceback=scores: 1012 | 750 (1.01x) | 710 (1.07x) | - | 137 (5.56x) | 331 (2.29x) | - | - | - | - | 132 (5.74x) | 226 (3.36x) | - | - |
| global linear hard int16 | **3694** | - | 3656 (1.01x) | 3966 (0.93x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **1742** | - | 1746 (1.00x) | 1814 (0.96x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **461** | - | 520 (0.89x) | 438 (1.05x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1531** | traceback=scores: 1997 | 1602 (0.96x) | 1696 (0.90x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **758** | traceback=scores: 992 | 714 (1.06x) | 771 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **139** | api=functions: 155 | 140 (0.99x) | 139 (1.00x) | - | - | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| local affine score int32 | **2559** | - | 2654 (0.96x) | 2697 (0.95x) | 3117 (0.82x) | 1355 (1.89x) | 473 (5.41x) | - | - | - | - | - | - | - | - |
| local affine score int64 | **785** | - | 823 (0.95x) | 869 (0.90x) | 99 (7.90x) | 265 (2.96x) | 460 (1.71x) | - | - | - | - | - | - | - | - |
| local affine score float32 | **2231** | api=functions: 2306 | 2207 (1.01x) | 2379 (0.94x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| local affine score float64 | **1064** | fill=striped: 1038 | 1028 (1.03x) | 1050 (1.01x) | - | 135 (7.86x) | 177 (6.01x) | - | - | - | - | 153 (6.97x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| local affine path int32 | **670** | - | 714 (0.94x) | 697 (0.96x) | 1125 (0.60x) | 578 (1.16x) | 122 (5.47x) | - | - | - | - | - | - | - | - |
| local affine path int64 | **312** | - | 305 (1.02x) | 317 (0.98x) | 75 (4.14x) | 94 (3.31x) | 134 (2.32x) | - | - | - | - | - | - | - | - |
| local affine path float32 | **774** | traceback=scores: 883 | 751 (1.03x) | 792 (0.98x) | - | 94 (8.22x) | 116 (6.68x) | - | - | - | - | - | - | - | - |
| local affine path float64 | **429** | traceback=pointers: 430 | 428 (1.00x) | 425 (1.01x) | - | 88 (4.85x) | 112 (3.82x) | - | - | - | - | 45 (9.62x) | 131 (3.26x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| local affine hard int32 | **684** | - | 756 (0.90x) | 732 (0.93x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **307** | - | 314 (0.98x) | 286 (1.07x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **782** | traceback=scores: 870 | 786 (0.99x) | 767 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **473** | fill=striped: 463 | 448 (1.06x) | 469 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **21** | fill=striped: 22 | 30 (0.72x) | 22 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **4897** | - | 5819 (0.84x) | 5123 (0.96x) | 9744 (0.50x) | 3021 (1.62x) | >10x | 3262 (1.50x) sat | - | - | - | - | - | >10x | - |
| local linear score int32 | **2468** | - | 2951 (0.84x) | 2639 (0.94x) | 4974 (0.50x) | 1504 (1.64x) | 573 (4.31x) | - | - | - | - | - | - | - | - |
| local linear score int64 | **839** | - | 897 (0.94x) | 848 (0.99x) | 105 (7.96x) | 265 (3.16x) | 637 (1.32x) | - | - | - | - | - | - | - | - |
| local linear score float32 | **3077** | fill=striped: 2861 | 3094 (0.99x) | 3064 (1.00x) | - | >10x | >10x | - | - | - | - | - | - | - | - |
| local linear score float64 | **1121** | fill=striped: 1191 | 1128 (0.99x) | 1226 (0.91x) | - | 140 (7.99x) | 222 (5.04x) | - | - | - | - | 338 (3.32x) | - | - | - |
| local linear path int16 | **2494** | - | 2874 (0.87x) | 2474 (1.01x) | 2134 (1.17x) | 1201 (2.08x) | >10x | 557 (4.48x) sat | - | - | - | - | - | 330 (7.57x) sat | - |
| local linear path int32 | **1298** | - | 1448 (0.90x) | 1414 (0.92x) | 1137 (1.14x) | 604 (2.15x) | 172 (7.55x) | - | - | - | - | - | - | - | - |
| local linear path int64 | **241** | - | 371 (0.65x) | 243 (0.99x) | 83 (2.92x) | 89 (2.71x) | 173 (1.39x) | - | - | - | - | - | - | - | - |
| local linear path float32 | **1200** | traceback=scores: 1477 | 1113 (1.08x) | 1199 (1.00x) | - | >10x | 146 (8.22x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **552** | traceback=scores: 738 | 555 (1.00x) | 540 (1.02x) | - | 85 (6.48x) | 146 (3.78x) | - | - | - | - | 69 (8.02x) | 194 (2.85x) | - | - |
| local linear hard int16 | **2638** | - | 2921 (0.90x) | 2728 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **1409** | - | 1478 (0.95x) | 1436 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **253** | - | 351 (0.72x) | 256 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1228** | traceback=scores: 1538 | 1135 (1.08x) | 1214 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **559** | traceback=scores: 778 | 580 (0.96x) | 545 (1.03x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **143** | api=functions: 170 | 150 (0.95x) | 151 (0.95x) | - | - | - | - | - | - | - | - | - | - | - |

#### oldsolace · dna_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global affine score int32 | **1889** | - | 1918 (0.98x) | 1741 (1.08x) | 2631 (0.72x) | 658 (2.87x) | 547 (3.45x) | 778 (2.43x) | 3408 (0.55x) | - | - | - | - |
| global affine score int64 | **737** | - | 756 (0.97x) | 741 (0.99x) | 102 (7.21x) | 497 (1.48x) | 509 (1.45x) | - | - | - | - | - | - |
| global affine score float32 | **1638** | fill=striped: 1646 | 1572 (1.04x) | 1545 (1.06x) | - | 232 (7.05x) | 221 (7.41x) | - | - | - | - | - | - |
| global affine score float64 | **845** | fill=striped: 816 | 834 (1.01x) | 849 (0.99x) | - | 230 (3.68x) | 233 (3.63x) | - | - | - | 262 (3.23x) | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global affine path int32 | **726** | - | 749 (0.97x) | 740 (0.98x) | 951 (0.76x) | 181 (4.02x) | 215 (3.37x) | 656 (1.11x) | 1460 (0.50x) | - | - | - | - |
| global affine path int64 | **303** | - | 310 (0.98x) | 300 (1.01x) | 86 (3.53x) | 141 (2.15x) | 168 (1.80x) | - | - | - | - | - | - |
| global affine path float32 | **1571** | fill=striped: 1644 | 1596 (0.98x) | 1606 (0.98x) | - | >10x | 163 (9.62x) | - | - | - | - | - | - |
| global affine path float64 | **825** | traceback=grid: 833 | 848 (0.97x) | 815 (1.01x) | - | 147 (5.63x) | 184 (4.48x) | - | - | - | >10x | 130 (6.32x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global affine hard int32 | **715** | - | 787 (0.91x) | 746 (0.96x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **298** | - | 305 (0.98x) | 302 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1598** | api=functions: 1601 | 1472 (1.09x) | 1579 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **809** | api=functions: 823 | 803 (1.01x) | 838 (0.97x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **88** | fill=rowwise: 91 | 96 (0.92x) | 92 (0.97x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global linear score int32 | **1659** | - | 1745 (0.95x) | 1860 (0.89x) | 3179 (0.52x) | 642 (2.58x) | 714 (2.32x) | 729 (2.28x) | 55949 (0.03x) | 93908 (0.02x) | - | - | - |
| global linear score int64 | **721** | - | 688 (1.05x) | 709 (1.02x) | 102 (7.05x) | 486 (1.48x) | 678 (1.06x) | - | - | - | - | - | - |
| global linear score float32 | **2143** | fill=striped: 2151 | 1967 (1.09x) | 2146 (1.00x) | - | 233 (9.21x) | 309 (6.93x) | - | - | - | - | - | - |
| global linear score float64 | **951** | api=functions: 947 | 881 (1.08x) | 941 (1.01x) | - | 225 (4.22x) | 286 (3.32x) | - | - | - | 472 (2.01x) | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global linear path int32 | **1073** | - | 1069 (1.00x) | 1044 (1.03x) | 935 (1.15x) | 185 (5.80x) | 241 (4.44x) | 681 (1.57x) | 31870 (0.03x) | 23489 (0.05x) | - | - | - |
| global linear path int64 | **467** | - | 482 (0.97x) | 439 (1.06x) | 83 (5.59x) | 128 (3.64x) | 245 (1.90x) | - | - | - | - | - | - |
| global linear path float32 | **1326** | traceback=scores: 1613 | 1243 (1.07x) | 1383 (0.96x) | - | >10x | 287 (4.62x) | - | - | - | - | - | - |
| global linear path float64 | **641** | traceback=scores: 802 | 594 (1.08x) | 599 (1.07x) | - | 122 (5.27x) | 276 (2.33x) | - | - | - | 120 (5.35x) | 218 (2.94x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global linear hard int32 | **1115** | - | 1097 (1.02x) | 1093 (1.02x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **465** | - | 532 (0.87x) | 445 (1.04x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1350** | traceback=scores: 1630 | 1262 (1.07x) | 1368 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **595** | traceback=scores: 761 | 625 (0.95x) | 634 (0.94x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **130** | api=functions: 147 | 133 (0.98x) | 136 (0.96x) | - | - | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local affine score int32 | **1629** | - | 1652 (0.99x) | 1686 (0.97x) | 3015 (0.54x) | 385 (4.23x) | 476 (3.42x) | - | - | - | - | - | - |
| local affine score int64 | **632** | - | 638 (0.99x) | 631 (1.00x) | 104 (6.09x) | 282 (2.24x) | 427 (1.48x) | - | - | - | - | - | - |
| local affine score float32 | **1912** | api=functions: 1893 | 1925 (0.99x) | 1835 (1.04x) | - | >10x | >10x | - | - | - | - | - | - |
| local affine score float64 | **923** | fill=striped: 920 | 858 (1.08x) | 916 (1.01x) | - | 135 (6.84x) | 167 (5.53x) | - | - | - | 163 (5.67x) | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local affine path int32 | **322** | - | 306 (1.05x) | 317 (1.01x) | 809 (0.40x) | 159 (2.02x) | 128 (2.51x) | - | - | - | - | - | - |
| local affine path int64 | **120** | - | 133 (0.90x) | 129 (0.93x) | 79 (1.52x) | 101 (1.19x) | 134 (0.89x) | - | - | - | - | - | - |
| local affine path float32 | **311** | traceback=pointers: 778 | 298 (1.04x) | 309 (1.01x) | - | 102 (3.05x) | 114 (2.74x) | - | - | - | - | - | - |
| local affine path float64 | **389** | traceback=pointers: 388 | 338 (1.15x) | 372 (1.05x) | - | 95 (4.11x) | 108 (3.59x) | - | - | - | 46 (8.43x) | 118 (3.29x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local affine hard int32 | **322** | - | 315 (1.02x) | 311 (1.04x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **123** | - | 132 (0.94x) | 124 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **290** | traceback=scores: 780 | 321 (0.90x) | 322 (0.90x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **347** | traceback=grid: 385 | 378 (0.92x) | 385 (0.90x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **56** | api=functions: 58 | 66 (0.85x) | 56 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local linear score int32 | **1582** | - | 1590 (0.99x) | 1704 (0.93x) | 3705 (0.43x) | 372 (4.25x) | 575 (2.75x) | - | - | - | - | - | - |
| local linear score int64 | **686** | - | 666 (1.03x) | 688 (1.00x) | 103 (6.64x) | 280 (2.45x) | 602 (1.14x) | - | - | - | - | - | - |
| local linear score float32 | **2004** | api=functions: 2082 | 2104 (0.95x) | 1938 (1.03x) | - | >10x | 219 (9.13x) | - | - | - | - | - | - |
| local linear score float64 | **924** | api=functions: 929 | 934 (0.99x) | 855 (1.08x) | - | 140 (6.59x) | 221 (4.18x) | - | - | - | 344 (2.69x) | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local linear path int32 | **910** | - | 890 (1.02x) | 920 (0.99x) | 940 (0.97x) | 160 (5.68x) | 168 (5.41x) | - | - | - | - | - | - |
| local linear path int64 | **258** | - | 365 (0.71x) | 232 (1.11x) | 80 (3.22x) | 93 (2.78x) | 171 (1.51x) | - | - | - | - | - | - |
| local linear path float32 | **1052** | traceback=scores: 1231 | 974 (1.08x) | 1050 (1.00x) | - | >10x | 137 (7.66x) | - | - | - | - | - | - |
| local linear path float64 | **502** | traceback=scores: 602 | 476 (1.06x) | 472 (1.06x) | - | 91 (5.50x) | 136 (3.68x) | - | - | - | 69 (7.30x) | 185 (2.72x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local linear hard int32 | **922** | - | 896 (1.03x) | 926 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **263** | - | 376 (0.70x) | 257 (1.02x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1077** | traceback=scores: 1244 | 1002 (1.07x) | 1069 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **515** | traceback=scores: 641 | 484 (1.06x) | 485 (1.06x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **131** | api=functions: 156 | 130 (1.00x) | 124 (1.06x) | - | - | - | - | - | - | - | - | - |

#### oldsolace · mirna · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **2889** | - | 2487 (1.16x) | 2795 (1.03x) | 599 (4.82x) | 4058 (0.71x) | 400 (7.22x) | - | - | - | - | - | - | >10x | - |
| global affine score int32 | **1813** | - | 1424 (1.27x) | 1886 (0.96x) | 738 (2.46x) | 2286 (0.79x) | 463 (3.92x) | - | 281 (6.46x) | >10x | - | - | - | >10x | - |
| global affine score int64 | **733** | - | 662 (1.11x) | 696 (1.05x) | 88 (8.28x) | 446 (1.64x) | 409 (1.79x) | - | - | - | - | - | - | - | - |
| global affine score float32 | **1733** | kernel=sse4: 957 | 1420 (1.22x) | 1771 (0.98x) | - | 244 (7.09x) | 223 (7.78x) | - | - | - | - | - | - | - | - |
| global affine score float64 | **845** | fill=striped: 556 | 785 (1.08x) | 926 (0.91x) | - | 234 (3.62x) | 218 (3.88x) | - | - | - | - | 90 (9.38x) | - | - | - |
| global affine path int16 | **597** | - | 550 (1.09x) | 607 (0.98x) | 321 (1.86x) | 1613 (0.37x) | 88 (6.82x) | - | - | - | - | - | - | >10x | - |
| global affine path int32 | **495** | - | 488 (1.02x) | 499 (0.99x) | 363 (1.36x) | 802 (0.62x) | 106 (4.66x) | - | 222 (2.23x) | >10x | - | - | - | >10x | - |
| global affine path int64 | **323** | - | 310 (1.04x) | 332 (0.97x) | 73 (4.46x) | 88 (3.67x) | 92 (3.53x) | - | - | - | - | - | - | - | - |
| global affine path float32 | **496** | traceback=scores: 522 | 470 (1.05x) | 513 (0.97x) | - | 103 (4.83x) | 75 (6.57x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **379** | traceback=pointers: 400 | 340 (1.12x) | 381 (0.99x) | - | 101 (3.75x) | 86 (4.43x) | - | - | - | - | >10x | >10x | - | - |
| global affine hard int16 | **1044** | - | 1081 (0.97x) | 1121 (0.93x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **840** | - | 792 (1.06x) | 830 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **453** | - | 405 (1.12x) | 466 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **828** | traceback=hirschberg_pmax: 829 | 716 (1.16x) | 784 (1.06x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **507** | traceback=hirschberg_pmax: 539 | 494 (1.03x) | 565 (0.90x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **148** | api=functions: 94 | 121 (1.22x) | 140 (1.06x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **3163** | - | 2578 (1.23x) | 3560 (0.89x) | 681 (4.65x) | 4363 (0.72x) | 611 (5.18x) | - | - | - | - | - | - | >10x | - |
| global linear score int32 | **2098** | - | 1769 (1.19x) | 2277 (0.92x) | 761 (2.76x) | 2386 (0.88x) | 588 (3.57x) | - | 320 (6.55x) | 395 (5.31x) | 355 (5.90x) | - | - | >10x | - |
| global linear score int64 | **940** | - | 924 (1.02x) | 1022 (0.92x) | >10x | 463 (2.03x) | 597 (1.58x) | - | - | - | - | - | - | - | - |
| global linear score float32 | **2244** | kernel=sse4: 1278 | 1622 (1.38x) | 2382 (0.94x) | - | 246 (9.14x) | 311 (7.21x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **1259** | fill=striped: 681 | 879 (1.43x) | 1274 (0.99x) | - | 241 (5.23x) | 280 (4.50x) | - | - | - | - | >10x | - | - | - |
| global linear path int16 | **489** | - | 473 (1.03x) | 484 (1.01x) | 322 (1.52x) | 1526 (0.32x) | 182 (2.69x) | - | - | - | - | - | - | >10x | - |
| global linear path int32 | **466** | - | 425 (1.10x) | 459 (1.01x) | 377 (1.23x) | 858 (0.54x) | 174 (2.67x) | - | 253 (1.84x) | 370 (1.26x) | 128 (3.63x) | - | - | - | - |
| global linear path int64 | **375** | - | 362 (1.04x) | 391 (0.96x) | 76 (4.91x) | 103 (3.63x) | 184 (2.04x) | - | - | - | - | - | - | - | - |
| global linear path float32 | **412** | traceback=pointers: 436 | 446 (0.92x) | 457 (0.90x) | - | 111 (3.70x) | 172 (2.40x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **379** | traceback=pointers: 379 | 346 (1.10x) | 371 (1.02x) | - | 110 (3.45x) | 180 (2.10x) | - | - | - | - | 43 (8.76x) | >10x | - | - |
| global linear hard int16 | **1113** | - | 997 (1.12x) | 1197 (0.93x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **968** | - | 825 (1.17x) | 997 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **636** | - | 591 (1.07x) | 638 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **851** | traceback=pointers: 909 | 802 (1.06x) | 851 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **641** | traceback=hirschberg: 641 | 607 (1.06x) | 657 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **216** | api=functions: 142 | 183 (1.18x) | 217 (0.99x) | - | - | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 | **2408** | - | 2192 (1.10x) | 2557 (0.94x) | 1312 (1.84x) sat | 2073 (1.16x) | 372 (6.47x) | 950 (2.54x) sat | - | - | - | - | - | >10x | - |
| local affine score int32 | **1578** | - | 1330 (1.19x) | 1596 (0.99x) | 852 (1.85x) | 1085 (1.45x) | 446 (3.54x) | - | - | - | - | - | - | >10x | - |
| local affine score int64 | **603** | - | 628 (0.96x) | 650 (0.93x) | 77 (7.86x) | 230 (2.63x) | 393 (1.54x) | - | - | - | - | - | - | - | - |
| local affine score float32 | **1489** | fill=striped: 904 | 1333 (1.12x) | 1525 (0.98x) | - | >10x | 177 (8.42x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **707** | fill=rowwise: 796 | 716 (0.99x) | 766 (0.92x) | - | 127 (5.59x) | 176 (4.02x) | - | - | - | - | >10x | - | - | - |
| local affine path int16 | **1203** | - | 1110 (1.08x) | 1214 (0.99x) | 422 (2.85x) | 1493 (0.81x) | >10x | 620 (1.94x) sat | - | - | - | - | - | >10x | - |
| local affine path int32 | **885** | - | 848 (1.04x) | 891 (0.99x) | 498 (1.78x) | 772 (1.15x) | 90 (9.80x) | - | - | - | - | - | - | >10x | - |
| local affine path int64 | **391** | - | 382 (1.02x) | 400 (0.98x) | 64 (6.12x) | 86 (4.54x) | 96 (4.07x) | - | - | - | - | - | - | - | - |
| local affine path float32 | **780** | traceback=hirschberg_pmax: 838 | 736 (1.06x) | 787 (0.99x) | - | 85 (9.22x) | 94 (8.34x) | - | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **494** | traceback=scores: 497 | 483 (1.02x) | 491 (1.01x) | - | 79 (6.27x) | 93 (5.29x) | - | - | - | - | >10x | >10x | - | - |
| local affine hard int16 | **1358** | - | 1353 (1.00x) | 1489 (0.91x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int32 | **957** | - | 915 (1.05x) | 972 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **403** | - | 377 (1.07x) | 381 (1.06x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **879** | traceback=pointers: 909 | 792 (1.11x) | 940 (0.94x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **521** | traceback=hirschberg: 519 | 516 (1.01x) | 509 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **138** | api=functions: 89 | 130 (1.06x) | 145 (0.95x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **2738** | - | 2750 (1.00x) | 2781 (0.98x) | 564 (4.85x) | 2178 (1.26x) | 374 (7.33x) | wrong | - | - | - | - | - | >10x | - |
| local linear score int32 | **1977** | - | 1799 (1.10x) | 1689 (1.17x) | 688 (2.87x) | 1136 (1.74x) | 477 (4.14x) | - | - | - | - | - | - | >10x | - |
| local linear score int64 | **785** | - | 741 (1.06x) | 859 (0.91x) | >10x | 240 (3.27x) | 516 (1.52x) | - | - | - | - | - | - | - | - |
| local linear score float32 | **1711** | fill=striped: 928 | 1562 (1.10x) | 1809 (0.95x) | - | >10x | 223 (7.67x) | - | - | - | - | - | - | - | - |
| local linear score float64 | **963** | fill=striped: 816 | 807 (1.19x) | 981 (0.98x) | - | 132 (7.27x) | 230 (4.18x) | - | - | - | - | >10x | - | - | - |
| local linear path int16 | **1078** | - | 1038 (1.04x) | 1097 (0.98x) | 385 (2.80x) | 1446 (0.75x) | 116 (9.29x) | wrong | - | - | - | - | - | >10x | - |
| local linear path int32 | **835** | - | 744 (1.12x) | 886 (0.94x) | 430 (1.94x) | 741 (1.13x) | 123 (6.77x) | - | - | - | - | - | - | >10x | - |
| local linear path int64 | **480** | - | 452 (1.06x) | 492 (0.98x) | 63 (7.60x) | 76 (6.32x) | 127 (3.79x) | - | - | - | - | - | - | - | - |
| local linear path float32 | **780** | traceback=hirschberg: 781 | 723 (1.08x) | 776 (1.01x) | - | 79 (9.83x) | 111 (7.04x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **532** | traceback=scores: 551 | 490 (1.08x) | 563 (0.94x) | - | 72 (7.37x) | 113 (4.71x) | - | - | - | - | >10x | >10x | - | - |
| local linear hard int16 | **1453** | - | 1372 (1.06x) | 1520 (0.96x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **1126** | - | 958 (1.18x) | 1202 (0.94x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **566** | - | 554 (1.02x) | 598 (0.95x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1035** | traceback=pointers: 1080 | 938 (1.10x) | 1077 (0.96x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **627** | traceback=hirschberg: 667 | 608 (1.03x) | 653 (0.96x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **195** | api=functions: 133 | 171 (1.14x) | 196 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |

#### oldsolace · protein_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **2368** | - | 2541 (0.93x) | 2270 (1.04x) | 1307 (1.81x) | 1338 (1.77x) | 545 (4.34x) | - | - | - | - | >10x | - |
| global affine score int32 | **1521** | - | 1904 (0.80x) | 1524 (1.00x) | 1197 (1.27x) | 1352 (1.12x) | 566 (2.69x) | - | 500 (3.04x) | - | - | - | - |
| global affine score int64 | **765** | - | 970 (0.79x) | 768 (1.00x) | 99 (7.72x) | 755 (1.01x) | 517 (1.48x) | - | - | - | - | - | - |
| global affine score float32 | **765** | fill=striped: 760 | 778 (0.98x) | 758 (1.01x) | - | 242 (3.16x) | 248 (3.09x) | - | - | - | - | - | - |
| global affine score float64 | **750** | fill=striped: 742 | 705 (1.06x) | 752 (1.00x) | - | 242 (3.10x) | 248 (3.02x) | - | - | 242 (3.10x) | - | - | - |
| global affine path int16 | **914** | - | 777 (1.18x) | 926 (0.99x) | 886 (1.03x) | 972 (0.94x) | 187 (4.89x) | - | - | - | - | 117 (7.80x) sat | - |
| global affine path int32 | **739** | - | 593 (1.25x) | 705 (1.05x) | 726 (1.02x) | 700 (1.06x) | 195 (3.78x) | - | 450 (1.64x) | - | - | - | - |
| global affine path int64 | **532** | - | 456 (1.17x) | 538 (0.99x) | 89 (5.99x) | 383 (1.39x) | 164 (3.25x) | - | - | - | - | - | - |
| global affine path float32 | **711** | traceback=scores: 713 | 605 (1.17x) | 719 (0.99x) | - | 144 (4.92x) | 167 (4.25x) | - | - | - | - | - | - |
| global affine path float64 | **542** | traceback=hirschberg_pmax: 542 | 501 (1.08x) | 542 (1.00x) | - | 135 (4.00x) | 149 (3.65x) | - | - | 61 (8.94x) | >10x | - | - |
| global affine hard int16 | **944** | - | 810 (1.17x) | 895 (1.05x) | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **747** | - | 673 (1.11x) | 752 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **539** | - | 466 (1.16x) | 536 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **694** | traceback=scores: 746 | 666 (1.04x) | 706 (0.98x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **554** | traceback=hirschberg: 551 | 504 (1.10x) | 548 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **113** | api=functions: 146 | 121 (0.94x) | 113 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **2240** | - | 2357 (0.95x) | 2168 (1.03x) | 1242 (1.80x) | 1293 (1.73x) | 673 (3.33x) | - | - | - | - | >10x | - |
| global linear score int32 | **1349** | - | 1667 (0.81x) | 1371 (0.98x) | 1139 (1.18x) | 1336 (1.01x) | 676 (1.99x) | - | 491 (2.74x) | - | - | - | - |
| global linear score int64 | **713** | - | 899 (0.79x) | 717 (0.99x) | 86 (8.25x) | 728 (0.98x) | 734 (0.97x) | - | - | - | - | - | - |
| global linear score float32 | **982** | fill=striped: 1054 | 1082 (0.91x) | 1024 (0.96x) | - | 242 (4.06x) | 346 (2.84x) | - | - | - | - | - | - |
| global linear score float64 | **943** | fill=striped: 921 | 908 (1.04x) | 960 (0.98x) | - | 242 (3.90x) | 342 (2.76x) | - | - | 376 (2.51x) | - | - | - |
| global linear path int16 | **991** | - | 1205 (0.82x) | 1110 (0.89x) | 871 (1.14x) | 986 (1.00x) | 344 (2.88x) | - | - | - | - | 128 (7.75x) sat | - |
| global linear path int32 | **916** | - | 1020 (0.90x) | 926 (0.99x) | 752 (1.22x) | 630 (1.45x) | 323 (2.84x) | - | 449 (2.04x) | - | - | - | - |
| global linear path int64 | **676** | - | 508 (1.33x) | 687 (0.98x) | >10x | 376 (1.80x) | 299 (2.26x) | - | - | - | - | - | - |
| global linear path float32 | **799** | traceback=scores: 798 | 732 (1.09x) | 764 (1.04x) | - | 160 (5.00x) | 449 (1.78x) | - | - | - | - | - | - |
| global linear path float64 | **676** | traceback=scores: 677 | 564 (1.20x) | 637 (1.06x) | - | 147 (4.60x) | 415 (1.63x) | - | - | 137 (4.95x) | >10x | - | - |
| global linear hard int16 | **1355** | - | 1513 (0.90x) | 1322 (1.02x) | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **1077** | - | 1220 (0.88x) | 1036 (1.04x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **726** | - | 530 (1.37x) | 730 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **885** | traceback=pointers: 883 | 745 (1.19x) | 887 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **685** | traceback=hirschberg: 740 | 667 (1.03x) | 745 (0.92x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **153** | api=functions: 193 | 148 (1.03x) | 158 (0.97x) | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 | **1899** | - | 1986 (0.96x) | 2147 (0.88x) | 1064 (1.79x) | 1101 (1.72x) | 436 (4.36x) | 520 (3.65x) sat | - | - | - | >10x | - |
| local affine score int32 | **1275** | - | 1535 (0.83x) | 1291 (0.99x) | 1049 (1.22x) | 936 (1.36x) | 506 (2.52x) | - | - | - | - | - | - |
| local affine score int64 | **670** | - | 770 (0.87x) | 669 (1.00x) | 80 (8.36x) | 448 (1.49x) | 350 (1.92x) | - | - | - | - | - | - |
| local affine score float32 | **794** | fill=striped: 833 | 790 (1.00x) | 807 (0.98x) | - | 138 (5.75x) | 186 (4.27x) | - | - | - | - | - | - |
| local affine score float64 | **770** | fill=striped: 754 | 739 (1.04x) | 758 (1.02x) | - | 137 (5.61x) | 186 (4.14x) | - | - | 148 (5.19x) | - | - | - |
| local affine path int16 | **767** | - | 662 (1.16x) | 786 (0.98x) | 743 (1.03x) | 941 (0.82x) | 115 (6.68x) | 217 (3.53x) sat | - | - | - | 90 (8.51x) sat | - |
| local affine path int32 | **712** | - | 599 (1.19x) | 709 (1.00x) | 629 (1.13x) | 599 (1.19x) | 122 (5.82x) | - | - | - | - | - | - |
| local affine path int64 | **368** | - | 402 (0.91x) | 376 (0.98x) | 71 (5.21x) | 303 (1.21x) | 116 (3.16x) | - | - | - | - | - | - |
| local affine path float32 | **584** | traceback=hirschberg_pmax: 599 | 550 (1.06x) | 612 (0.95x) | - | 103 (5.68x) | 143 (4.08x) | - | - | - | - | - | - |
| local affine path float64 | **458** | traceback=hirschberg_pmax: 484 | 431 (1.06x) | 464 (0.99x) | - | 102 (4.49x) | 144 (3.17x) | - | - | >10x | 53 (8.63x) | - | - |
| local affine hard int16 | **851** | - | 736 (1.16x) | 861 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard int32 | **689** | - | 655 (1.05x) | 714 (0.96x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **390** | - | 378 (1.03x) | 391 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **638** | traceback=scores: 687 | 581 (1.10x) | 634 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **503** | traceback=grid: 505 | 446 (1.13x) | 478 (1.05x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **98** | api=functions: 131 | 104 (0.95x) | 103 (0.95x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **2011** | - | 2063 (0.98x) | 2052 (0.98x) | 1124 (1.79x) | 1101 (1.83x) | 590 (3.41x) | wrong | - | - | - | >10x | - |
| local linear score int32 | **1279** | - | 1561 (0.82x) | 1298 (0.98x) | 968 (1.32x) | 906 (1.41x) | 766 (1.67x) | - | - | - | - | - | - |
| local linear score int64 | **633** | - | 728 (0.87x) | 627 (1.01x) | 68 (9.37x) | 425 (1.49x) | 383 (1.65x) | - | - | - | - | - | - |
| local linear score float32 | **1082** | fill=striped: 1084 | 1094 (0.99x) | 1224 (0.88x) | - | 131 (8.28x) | 230 (4.71x) | - | - | - | - | - | - |
| local linear score float64 | **887** | fill=striped: 870 | 850 (1.04x) | 1025 (0.87x) | - | 132 (6.74x) | 232 (3.82x) | - | - | 199 (4.46x) | - | - | - |
| local linear path int16 | **871** | - | 733 (1.19x) | 885 (0.98x) | 735 (1.18x) | 856 (1.02x) | 144 (6.06x) | wrong | - | - | - | 117 (7.44x) sat | - |
| local linear path int32 | **788** | - | 630 (1.25x) | 795 (0.99x) | 612 (1.29x) | 562 (1.40x) | 149 (5.28x) | - | - | - | - | - | - |
| local linear path int64 | **515** | - | 469 (1.10x) | 525 (0.98x) | 65 (7.95x) | 303 (1.70x) | 172 (3.00x) | - | - | - | - | - | - |
| local linear path float32 | **647** | traceback=scores: 647 | 575 (1.13x) | 673 (0.96x) | - | 105 (6.17x) | 143 (4.54x) | - | - | - | - | - | - |
| local linear path float64 | **506** | traceback=scores: 554 | 454 (1.12x) | 504 (1.00x) | - | 103 (4.90x) | 145 (3.49x) | - | - | 66 (7.63x) | 63 (8.02x) | - | - |
| local linear hard int16 | **982** | - | 792 (1.24x) | 994 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **890** | - | 694 (1.28x) | 985 (0.90x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **597** | - | 533 (1.12x) | 581 (1.03x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **723** | traceback=pointers: 763 | 668 (1.08x) | 766 (0.94x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **547** | fill=striped: 555 | 473 (1.16x) | 557 (0.98x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **166** | api=functions: 209 | 154 (1.08x) | 160 (1.04x) | - | - | - | - | - | - | - | - | - |

#### oldsolace · protein_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global affine score int32 | **2870** | - | 3041 (0.94x) | 2764 (1.04x) | 2684 (1.07x) | 1265 (2.27x) | 556 (5.16x) | 769 (3.73x) | - | - | - | - |
| global affine score int64 | **920** | - | 987 (0.93x) | 919 (1.00x) | 103 (8.96x) | 693 (1.33x) | 560 (1.64x) | - | - | - | - | - |
| global affine score float32 | **1116** | fill=striped: 1128 | 1149 (0.97x) | 1078 (1.04x) | - | 236 (4.73x) | 233 (4.78x) | - | - | - | - | - |
| global affine score float64 | **769** | fill=striped: 794 | 782 (0.98x) | 737 (1.04x) | - | 235 (3.27x) | 236 (3.26x) | - | 261 (2.95x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global affine path int32 | **926** | - | 1080 (0.86x) | 980 (0.94x) | 1272 (0.73x) | 513 (1.80x) | 226 (4.10x) | 686 (1.35x) | - | - | - | - |
| global affine path int64 | **371** | - | 402 (0.92x) | 399 (0.93x) | 91 (4.08x) | 257 (1.45x) | 184 (2.02x) | - | - | - | - | - |
| global affine path float32 | **859** | fill=striped: 856 | 865 (0.99x) | 861 (1.00x) | - | 155 (5.56x) | 185 (4.65x) | - | - | - | - | - |
| global affine path float64 | **695** | fill=rowwise: 688 | 652 (1.07x) | 708 (0.98x) | - | 134 (5.19x) | 153 (4.54x) | - | >10x | 117 (5.91x) | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global affine hard int32 | **963** | - | 1115 (0.86x) | 1046 (0.92x) | - | - | - | - | - | - | - | - |
| global affine hard int64 | **397** | - | 399 (1.00x) | 398 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **881** | traceback=grid: 875 | 879 (1.00x) | 885 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **705** | traceback=grid: 709 | 702 (1.00x) | 704 (1.00x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **65** | fill=striped: 65 | 76 (0.86x) | 65 (1.00x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global linear score int32 | **2813** | - | 3257 (0.86x) | 2813 (1.00x) | 4229 (0.67x) | 1243 (2.26x) | 723 (3.89x) | 767 (3.67x) | - | - | - | - |
| global linear score int64 | **940** | - | 996 (0.94x) | 926 (1.02x) | >10x | 672 (1.40x) | 700 (1.34x) | - | - | - | - | - |
| global linear score float32 | **2945** | api=functions: 2847 | 2935 (1.00x) | 2898 (1.02x) | - | >10x | 318 (9.26x) | - | - | - | - | - |
| global linear score float64 | **1128** | fill=striped: 1207 | 1180 (0.96x) | 1204 (0.94x) | - | 236 (4.79x) | 318 (3.55x) | - | 452 (2.50x) | - | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global linear path int32 | **1529** | - | 1668 (0.92x) | 1699 (0.90x) | 1203 (1.27x) | 504 (3.03x) | 415 (3.68x) | 695 (2.20x) | - | - | >10x | - |
| global linear path int64 | **430** | - | 500 (0.86x) | 485 (0.89x) | 67 (6.40x) | 263 (1.64x) | 327 (1.31x) | - | - | - | - | - |
| global linear path float32 | **1595** | traceback=scores: 1953 | 1580 (1.01x) | 1549 (1.03x) | - | 175 (9.14x) | 493 (3.23x) | - | - | - | - | - |
| global linear path float64 | **770** | traceback=scores: 980 | 761 (1.01x) | 772 (1.00x) | - | 161 (4.79x) | 486 (1.59x) | - | 184 (4.19x) | 229 (3.37x) | - | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global linear hard int32 | **1562** | - | 1556 (1.00x) | 1780 (0.88x) | - | - | - | - | - | - | - | - |
| global linear hard int64 | **468** | - | 530 (0.88x) | 468 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1452** | traceback=scores: 2009 | 1612 (0.90x) | 1569 (0.93x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **757** | traceback=scores: 980 | 766 (0.99x) | 787 (0.96x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **131** | api=functions: 145 | 126 (1.04x) | 132 (0.99x) | - | - | - | - | - | - | - | >10x |
| local affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local affine score int32 | **2308** | - | 2843 (0.81x) | 2546 (0.91x) | 2315 (1.00x) | 885 (2.61x) | 520 (4.44x) | - | - | - | - | - |
| local affine score int64 | **861** | - | 912 (0.94x) | 816 (1.05x) | >10x | 422 (2.04x) | 382 (2.26x) | - | - | - | - | - |
| local affine score float32 | **1284** | fill=striped: 1312 | 1276 (1.01x) | 1225 (1.05x) | - | 138 (9.32x) | 174 (7.37x) | - | - | - | - | - |
| local affine score float64 | **868** | fill=striped: 835 | 799 (1.09x) | 861 (1.01x) | - | 136 (6.39x) | 179 (4.85x) | - | 206 (4.21x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local affine path int32 | **677** | - | 748 (0.91x) | 698 (0.97x) | 1012 (0.67x) | 459 (1.48x) | 147 (4.60x) | - | - | - | - | - |
| local affine path int64 | **302** | - | 340 (0.89x) | 302 (1.00x) | 77 (3.94x) | 245 (1.23x) | 129 (2.34x) | - | - | - | - | - |
| local affine path float32 | **584** | traceback=scores: 646 | 636 (0.92x) | 606 (0.96x) | - | 109 (5.36x) | 162 (3.61x) | - | - | - | - | - |
| local affine path float64 | **408** | fill=striped: 406 | 361 (1.13x) | 412 (0.99x) | - | 108 (3.76x) | 150 (2.72x) | - | 42 (9.69x) | 108 (3.78x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local affine hard int32 | **743** | - | 766 (0.97x) | 719 (1.03x) | - | - | - | - | - | - | - | - |
| local affine hard int64 | **320** | - | 312 (1.03x) | 325 (0.99x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **629** | traceback=scores: 696 | 646 (0.97x) | 615 (1.02x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **387** | traceback=pointers: 411 | 373 (1.04x) | 387 (1.00x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **54** | fill=striped: 54 | 67 (0.81x) | 57 (0.95x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local linear score int32 | **2487** | - | 2645 (0.94x) | 2361 (1.05x) | 4555 (0.55x) | 812 (3.06x) | 809 (3.08x) | - | - | - | - | - |
| local linear score int64 | **809** | - | 827 (0.98x) | 827 (0.98x) | 105 (7.71x) | 395 (2.05x) | 387 (2.09x) | - | - | - | - | - |
| local linear score float32 | **2647** | fill=striped: 2827 | 2627 (1.01x) | 2662 (0.99x) | - | >10x | >10x | - | - | - | - | - |
| local linear score float64 | **1101** | fill=striped: 1091 | 1100 (1.00x) | 1199 (0.92x) | - | 137 (8.01x) | 220 (5.00x) | - | 302 (3.65x) | - | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local linear path int32 | **1226** | - | 1324 (0.93x) | 1395 (0.88x) | 1026 (1.20x) | 464 (2.64x) | 182 (6.73x) | - | - | - | - | - |
| local linear path int64 | **234** | - | 360 (0.65x) | 250 (0.94x) | 82 (2.84x) | 236 (0.99x) | 187 (1.25x) | - | - | - | - | - |
| local linear path float32 | **1118** | traceback=scores: 1337 | 1090 (1.03x) | 1177 (0.95x) | - | >10x | 156 (7.18x) | - | - | - | - | - |
| local linear path float64 | **525** | traceback=scores: 724 | 539 (0.97x) | 564 (0.93x) | - | 104 (5.02x) | 150 (3.49x) | - | 78 (6.71x) | 207 (2.53x) | - | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local linear hard int32 | **1293** | - | 1468 (0.88x) | 1342 (0.96x) | - | - | - | - | - | - | - | - |
| local linear hard int64 | **235** | - | 381 (0.62x) | 236 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1141** | traceback=scores: 1388 | 1113 (1.03x) | 1182 (0.97x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **558** | traceback=scores: 734 | 603 (0.92x) | 566 (0.99x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **110** | api=functions: 130 | 110 (0.99x) | 117 (0.94x) | - | - | - | - | - | - | - | - |

#### oldsolace · protein_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global affine score int32 | **1724** | - | 1805 (0.96x) | 1734 (0.99x) | 2238 (0.77x) | 560 (3.08x) | 581 (2.97x) | 752 (2.29x) | - | - | - |
| global affine score int64 | **714** | - | 751 (0.95x) | 714 (1.00x) | 101 (7.09x) | 493 (1.45x) | 529 (1.35x) | - | - | - | - |
| global affine score float32 | **967** | kernel=sse4: 1037 | 992 (0.97x) | 984 (0.98x) | - | 235 (4.11x) | 236 (4.10x) | - | - | - | - |
| global affine score float64 | **648** | kernel=sse2: 704 | 651 (0.99x) | 691 (0.94x) | - | 235 (2.76x) | 234 (2.77x) | - | 285 (2.28x) | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global affine path int32 | **757** | - | 821 (0.92x) | 788 (0.96x) | 965 (0.78x) | 150 (5.03x) | 218 (3.48x) | 711 (1.07x) | - | - | - |
| global affine path int64 | **306** | - | 310 (0.99x) | 322 (0.95x) | 91 (3.37x) | 139 (2.21x) | 183 (1.67x) | - | - | - | - |
| global affine path float32 | **974** | traceback=grid: 975 | 975 (1.00x) | 1037 (0.94x) | - | 147 (6.64x) | 184 (5.30x) | - | - | - | - |
| global affine path float64 | **647** | fill=rowwise: 667 | 641 (1.01x) | 644 (1.00x) | - | 142 (4.56x) | 164 (3.96x) | - | >10x | 140 (4.64x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global affine hard int32 | **739** | - | 752 (0.98x) | 788 (0.94x) | - | - | - | - | - | - | - |
| global affine hard int64 | **300** | - | 314 (0.95x) | 320 (0.94x) | - | - | - | - | - | - | - |
| global affine hard float32 | **1033** | fill=striped: 1034 | 1020 (1.01x) | 963 (1.07x) | - | - | - | - | - | - | - |
| global affine hard float64 | **688** | fill=striped: 686 | 693 (0.99x) | 667 (1.03x) | - | - | - | - | - | - | - |
| global affine soft float64 | **82** | fill=striped: 85 | 90 (0.91x) | 82 (1.00x) | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global linear score int32 | **1820** | - | 1928 (0.94x) | 1696 (1.07x) | 3422 (0.53x) | 596 (3.06x) | 702 (2.59x) | 792 (2.30x) | - | - | 397 (4.59x) sat |
| global linear score int64 | **740** | - | 711 (1.04x) | 766 (0.97x) | 93 (7.92x) | 513 (1.44x) | 727 (1.02x) | - | - | - | - |
| global linear score float32 | **1975** | api=functions: 2107 | 1963 (1.01x) | 2037 (0.97x) | - | 234 (8.43x) | 315 (6.27x) | - | - | - | - |
| global linear score float64 | **955** | fill=striped: 954 | 927 (1.03x) | 943 (1.01x) | - | 234 (4.08x) | 315 (3.04x) | - | 472 (2.02x) | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global linear path int32 | **1078** | - | 1101 (0.98x) | 1054 (1.02x) | 959 (1.12x) | 159 (6.77x) | 363 (2.97x) | 697 (1.55x) | - | - | 147 (7.34x) sat |
| global linear path int64 | **476** | - | 509 (0.94x) | 449 (1.06x) | 66 (7.24x) | 141 (3.38x) | 304 (1.57x) | - | - | - | - |
| global linear path float32 | **1340** | traceback=scores: 1593 | 1213 (1.10x) | 1335 (1.00x) | - | 160 (8.38x) | 516 (2.60x) | - | - | - | - |
| global linear path float64 | **597** | traceback=scores: 748 | 620 (0.96x) | 597 (1.00x) | - | 145 (4.11x) | 490 (1.22x) | - | 155 (3.85x) | 227 (2.64x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global linear hard int32 | **1017** | - | 1030 (0.99x) | 1088 (0.94x) | - | - | - | - | - | - | - |
| global linear hard int64 | **472** | - | 511 (0.92x) | 450 (1.05x) | - | - | - | - | - | - | - |
| global linear hard float32 | **1341** | traceback=scores: 1563 | 1199 (1.12x) | 1303 (1.03x) | - | - | - | - | - | - | - |
| global linear hard float64 | **600** | traceback=scores: 753 | 598 (1.00x) | 632 (0.95x) | - | - | - | - | - | - | - |
| global linear soft float64 | **170** | api=functions: 192 | 159 (1.07x) | 165 (1.03x) | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local affine score int32 | **1699** | - | 1644 (1.03x) | 1549 (1.10x) | 1671 (1.02x) | 292 (5.82x) | 516 (3.30x) | - | - | - | - |
| local affine score int64 | **655** | - | 705 (0.93x) | 656 (1.00x) | 83 (7.85x) | 288 (2.27x) | 382 (1.71x) | - | - | - | - |
| local affine score float32 | **1132** | kernel=sse2: 1203 | 1266 (0.89x) | 1210 (0.94x) | - | 137 (8.28x) | 177 (6.41x) | - | - | - | - |
| local affine score float64 | **716** | api=functions: 734 | 745 (0.96x) | 762 (0.94x) | - | 137 (5.23x) | 181 (3.95x) | - | 237 (3.02x) | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local affine path int32 | **325** | - | 331 (0.98x) | 324 (1.00x) | 805 (0.40x) | 133 (2.44x) | 153 (2.13x) | - | - | - | - |
| local affine path int64 | **121** | - | 126 (0.96x) | 124 (0.98x) | 77 (1.58x) | 131 (0.93x) | 128 (0.95x) | - | - | - | - |
| local affine path float32 | **328** | traceback=scores: 638 | 304 (1.08x) | 330 (0.99x) | - | 112 (2.94x) | 174 (1.88x) | - | - | - | - |
| local affine path float64 | **332** | traceback=pointers: 363 | 320 (1.04x) | 348 (0.95x) | - | 111 (3.00x) | 171 (1.94x) | - | 40 (8.22x) | 130 (2.55x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local affine hard int32 | **322** | - | 316 (1.02x) | 324 (1.00x) | - | - | - | - | - | - | - |
| local affine hard int64 | **123** | - | 126 (0.98x) | 123 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float32 | **311** | traceback=scores: 600 | 319 (0.98x) | 306 (1.02x) | - | - | - | - | - | - | - |
| local affine hard float64 | **330** | traceback=pointers: 348 | 337 (0.98x) | 346 (0.96x) | - | - | - | - | - | - | - |
| local affine soft float64 | **70** | api=functions: 72 | 71 (0.98x) | 68 (1.03x) | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local linear score int32 | **1527** | - | 1673 (0.91x) | 1582 (0.97x) | 3362 (0.45x) | 273 (5.59x) | 780 (1.96x) | - | - | - | 283 (5.40x) sat |
| local linear score int64 | **678** | - | 660 (1.03x) | 655 (1.03x) | 105 (6.45x) | 267 (2.54x) | 422 (1.61x) | - | - | - | - |
| local linear score float32 | **2051** | api=functions: 1963 | 2085 (0.98x) | 2001 (1.02x) | - | >10x | 226 (9.07x) | - | - | - | - |
| local linear score float64 | **928** | api=functions: 930 | 916 (1.01x) | 893 (1.04x) | - | 136 (6.85x) | 219 (4.23x) | - | 338 (2.75x) | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local linear path int32 | **893** | - | 900 (0.99x) | 889 (1.00x) | 888 (1.00x) | 130 (6.89x) | 173 (5.17x) | - | - | - | 219 (4.08x) sat |
| local linear path int64 | **263** | - | 362 (0.73x) | 252 (1.05x) | 82 (3.21x) | 124 (2.12x) | 182 (1.44x) | - | - | - | - |
| local linear path float32 | **1027** | traceback=scores: 1216 | 996 (1.03x) | 1031 (1.00x) | - | >10x | 148 (6.93x) | - | - | - | - |
| local linear path float64 | **516** | traceback=scores: 627 | 478 (1.08x) | 525 (0.98x) | - | 104 (4.96x) | 158 (3.27x) | - | 79 (6.56x) | 202 (2.56x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local linear hard int32 | **925** | - | 836 (1.11x) | 883 (1.05x) | - | - | - | - | - | - | - |
| local linear hard int64 | **262** | - | 346 (0.76x) | 264 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float32 | **1085** | traceback=scores: 1188 | 993 (1.09x) | 999 (1.09x) | - | - | - | - | - | - | - |
| local linear hard float64 | **513** | traceback=scores: 623 | 453 (1.13x) | 511 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **152** | api=functions: 185 | 148 (1.03x) | 141 (1.08x) | - | - | - | - | - | - | - |

#### oldsolace · protein_L300 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| global affine score int32 | **2303** | - | 2562 (0.90x) | 2373 (0.97x) | 1926 (1.20x) | 1418 (1.62x) | 577 (3.99x) | 645 (3.57x) | - | - | - | - |
| global affine score int64 | **827** | - | 1139 (0.73x) | 889 (0.93x) | 102 (8.14x) | 791 (1.05x) | 516 (1.60x) | - | - | - | - | - |
| global affine score float32 | **997** | fill=striped: 978 | 1030 (0.97x) | 990 (1.01x) | - | 238 (4.20x) | 240 (4.15x) | - | - | - | - | - |
| global affine score float64 | **827** | fill=striped: 883 | 826 (1.00x) | 895 (0.92x) | - | 237 (3.48x) | 241 (3.44x) | - | 277 (2.99x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| global affine path int32 | **987** | - | 990 (1.00x) | 997 (0.99x) | 1156 (0.85x) | 561 (1.76x) | 205 (4.81x) | 619 (1.60x) | - | - | - | - |
| global affine path int64 | **389** | - | 445 (0.87x) | 421 (0.92x) | 91 (4.25x) | 313 (1.24x) | 164 (2.38x) | - | - | - | - | - |
| global affine path float32 | **609** | traceback=scores: 724 | 569 (1.07x) | 621 (0.98x) | - | 151 (4.03x) | 184 (3.30x) | - | - | - | - | - |
| global affine path float64 | **433** | traceback=scores: 504 | 441 (0.98x) | 425 (1.02x) | - | 138 (3.13x) | 155 (2.80x) | - | 62 (6.95x) | 124 (3.50x) | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| global affine hard int32 | **949** | - | 1054 (0.90x) | 1050 (0.90x) | - | - | - | - | - | - | - | - |
| global affine hard int64 | **417** | - | 441 (0.95x) | 419 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **608** | traceback=scores: 680 | 617 (0.98x) | 615 (0.99x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **440** | traceback=scores: 552 | 438 (1.00x) | 474 (0.93x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **71** | fill=striped: 71 | 82 (0.87x) | 71 (1.00x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| global linear score int32 | **2310** | - | 2783 (0.83x) | 2336 (0.99x) | 2935 (0.79x) | 1409 (1.64x) | 650 (3.55x) | 667 (3.46x) | - | - | - | - |
| global linear score int64 | **874** | - | 1090 (0.80x) | 858 (1.02x) | 90 (9.72x) | 772 (1.13x) | 693 (1.26x) | - | - | - | - | - |
| global linear score float32 | **2015** | fill=striped: 1998 | 1854 (1.09x) | 2059 (0.98x) | - | 238 (8.47x) | 323 (6.23x) | - | - | - | - | - |
| global linear score float64 | **1404** | fill=striped: 1384 | 1245 (1.13x) | 1444 (0.97x) | - | 238 (5.91x) | 325 (4.32x) | - | 464 (3.02x) | - | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| global linear path int32 | **1465** | - | 1615 (0.91x) | 1492 (0.98x) | 1011 (1.45x) | 570 (2.57x) | 359 (4.08x) | 619 (2.37x) | - | - | - | - |
| global linear path int64 | **786** | - | 514 (1.53x) | 813 (0.97x) | >10x | 292 (2.69x) | 332 (2.37x) | - | - | - | - | - |
| global linear path float32 | **1319** | fill=striped: 1307 | 1278 (1.03x) | 1357 (0.97x) | - | 166 (7.96x) | 460 (2.87x) | - | - | - | - | - |
| global linear path float64 | **883** | traceback=scores: 883 | 928 (0.95x) | 894 (0.99x) | - | 152 (5.82x) | 495 (1.78x) | - | 175 (5.06x) | 179 (4.94x) | - | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| global linear hard int32 | **1596** | - | 1719 (0.93x) | 1577 (1.01x) | - | - | - | - | - | - | - | - |
| global linear hard int64 | **814** | - | 570 (1.43x) | 837 (0.97x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1446** | fill=striped: 1485 | 1425 (1.01x) | 1530 (0.94x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **877** | fill=striped: 950 | 987 (0.89x) | 949 (0.92x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **100** | api=functions: 105 | 99 (1.02x) | 103 (0.98x) | - | - | - | - | - | - | - | >10x |
| local affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| local affine score int32 | **2099** | - | 2235 (0.94x) | 2185 (0.96x) | 1772 (1.18x) | 874 (2.40x) | 478 (4.39x) | - | - | - | - | - |
| local affine score int64 | **822** | - | 939 (0.88x) | 806 (1.02x) | >10x | 461 (1.78x) | 349 (2.35x) | - | - | - | - | - |
| local affine score float32 | **1036** | fill=striped: 1117 | 1169 (0.89x) | 1093 (0.95x) | - | 138 (7.51x) | 179 (5.80x) | - | - | - | - | - |
| local affine score float64 | **935** | fill=striped: 937 | 860 (1.09x) | 969 (0.96x) | - | 131 (7.15x) | 177 (5.29x) | - | 190 (4.92x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| local affine path int32 | **780** | - | 871 (0.89x) | 789 (0.99x) | 949 (0.82x) | 529 (1.47x) | 131 (5.95x) | - | - | - | - | - |
| local affine path int64 | **296** | - | 349 (0.85x) | 321 (0.92x) | 76 (3.87x) | 276 (1.07x) | 115 (2.56x) | - | - | - | - | - |
| local affine path float32 | **549** | traceback=scores: 633 | 523 (1.05x) | 572 (0.96x) | - | 107 (5.12x) | 153 (3.58x) | - | - | - | - | - |
| local affine path float64 | **375** | traceback=scores: 457 | 373 (1.00x) | 368 (1.02x) | - | 102 (3.67x) | 153 (2.45x) | - | 41 (9.09x) | 118 (3.16x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| local affine hard int32 | **763** | - | 907 (0.84x) | 798 (0.96x) | - | - | - | - | - | - | - | - |
| local affine hard int64 | **310** | - | 329 (0.94x) | 310 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **538** | traceback=scores: 681 | 541 (0.99x) | 572 (0.94x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **401** | traceback=scores: 460 | 399 (1.00x) | 401 (1.00x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **30** | fill=striped: 30 | 43 (0.71x) | 30 (1.00x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| local linear score int32 | **2109** | - | 2345 (0.90x) | 2184 (0.97x) | 3042 (0.69x) | 936 (2.25x) | 822 (2.57x) | - | - | - | - | - |
| local linear score int64 | **770** | - | 891 (0.86x) | 821 (0.94x) | >10x | 436 (1.76x) | 406 (1.90x) | - | - | - | - | - |
| local linear score float32 | **1944** | fill=striped: 2080 | 2080 (0.93x) | 2143 (0.91x) | - | >10x | 224 (8.67x) | - | - | - | - | - |
| local linear score float64 | **1343** | fill=striped: 1337 | 1321 (1.02x) | 1298 (1.03x) | - | >10x | 219 (6.12x) | - | 271 (4.95x) | - | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| local linear path int32 | **1103** | - | 1313 (0.84x) | 1281 (0.86x) | 967 (1.14x) | 536 (2.06x) | 173 (6.38x) | - | - | - | - | - |
| local linear path int64 | **594** | - | 372 (1.60x) | 582 (1.02x) | 76 (7.77x) | 277 (2.14x) | 171 (3.48x) | - | - | - | - | - |
| local linear path float32 | **1059** | fill=striped: 1051 | 1013 (1.05x) | 1093 (0.97x) | - | 113 (9.41x) | 146 (7.27x) | - | - | - | - | - |
| local linear path float64 | **726** | traceback=scores: 740 | 715 (1.02x) | 737 (0.99x) | - | 110 (6.57x) | 146 (4.99x) | - | 73 (9.92x) | 169 (4.30x) | - | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| local linear hard int32 | **1371** | - | 1406 (0.98x) | 1333 (1.03x) | - | - | - | - | - | - | - | - |
| local linear hard int64 | **600** | - | 354 (1.69x) | 561 (1.07x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1101** | traceback=scores: 1141 | 1108 (0.99x) | 1216 (0.91x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **779** | fill=striped: 724 | 695 (1.12x) | 802 (0.97x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **67** | api=functions: 74 | 69 (0.97x) | 70 (0.94x) | - | - | - | - | - | - | - | - |

#### oldsolace · protein_L3000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global affine score int32 | **2131** | - | 2261 (0.94x) | 2120 (1.01x) | 2506 (0.85x) | 564 (3.78x) | 553 (3.86x) | 724 (2.94x) | - | - | - | - |
| global affine score int64 | **787** | - | 856 (0.92x) | 878 (0.90x) | 103 (7.65x) | 550 (1.43x) | 546 (1.44x) | - | - | - | - | - |
| global affine score float32 | **979** | fill=striped: 1076 | 989 (0.99x) | 1007 (0.97x) | - | 236 (4.15x) | 237 (4.14x) | - | - | - | - | - |
| global affine score float64 | **731** | fill=striped: 774 | 703 (1.04x) | 764 (0.96x) | - | 235 (3.11x) | 237 (3.09x) | - | 262 (2.79x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global affine path int32 | **825** | - | 897 (0.92x) | 877 (0.94x) | 942 (0.88x) | 197 (4.19x) | 222 (3.72x) | 689 (1.20x) | - | - | - | - |
| global affine path int64 | **336** | - | 383 (0.88x) | 350 (0.96x) | 93 (3.62x) | 220 (1.53x) | 177 (1.89x) | - | - | - | - | - |
| global affine path float32 | **1029** | traceback=grid: 1011 | 919 (1.12x) | 930 (1.11x) | - | 145 (7.11x) | 203 (5.07x) | - | - | - | - | - |
| global affine path float64 | **699** | fill=rowwise: 774 | 696 (1.00x) | 742 (0.94x) | - | 131 (5.34x) | 158 (4.43x) | - | >10x | 138 (5.05x) | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global affine hard int32 | **841** | - | 869 (0.97x) | 943 (0.89x) | - | - | - | - | - | - | - | - |
| global affine hard int64 | **340** | - | 346 (0.98x) | 378 (0.90x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1017** | traceback=grid: 991 | 946 (1.07x) | 927 (1.10x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **715** | traceback=grid: 759 | 714 (1.00x) | 794 (0.90x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **69** | fill=rowwise: 72 | 83 (0.83x) | 69 (1.00x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global linear score int32 | **1944** | - | 2007 (0.97x) | 1948 (1.00x) | 3835 (0.51x) | 577 (3.37x) | 733 (2.65x) | 761 (2.55x) | - | - | - | - |
| global linear score int64 | **823** | - | 932 (0.88x) | 866 (0.95x) | 93 (8.90x) | 538 (1.53x) | 707 (1.17x) | - | - | - | - | - |
| global linear score float32 | **2145** | fill=striped: 2425 | 2259 (0.95x) | 2252 (0.95x) | - | 236 (9.11x) | 316 (6.78x) | - | - | - | - | - |
| global linear score float64 | **1140** | fill=striped: 1108 | 1079 (1.06x) | 1151 (0.99x) | - | 235 (4.84x) | 316 (3.61x) | - | 468 (2.44x) | - | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global linear path int32 | **1204** | - | 1103 (1.09x) | 1153 (1.04x) | 1009 (1.19x) | 212 (5.68x) | 395 (3.05x) | 733 (1.64x) | - | - | - | - |
| global linear path int64 | **476** | - | 539 (0.88x) | 480 (0.99x) | 66 (7.19x) | 219 (2.17x) | 331 (1.44x) | - | - | - | - | - |
| global linear path float32 | **1379** | traceback=scores: 1745 | 1248 (1.10x) | 1474 (0.94x) | - | 162 (8.50x) | 503 (2.74x) | - | - | - | - | - |
| global linear path float64 | **728** | traceback=scores: 789 | 642 (1.13x) | 723 (1.01x) | - | 161 (4.52x) | 527 (1.38x) | - | 186 (3.92x) | 227 (3.20x) | - | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global linear hard int32 | **1182** | - | 1202 (0.98x) | 1164 (1.02x) | - | - | - | - | - | - | - | - |
| global linear hard int64 | **477** | - | 538 (0.89x) | 478 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1407** | traceback=scores: 1685 | 1346 (1.05x) | 1411 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **653** | traceback=scores: 792 | 657 (1.00x) | 731 (0.89x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **154** | api=functions: 184 | 161 (0.96x) | 164 (0.94x) | - | - | - | - | - | - | - | >10x |
| local affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local affine score int32 | **1818** | - | 2062 (0.88x) | 1949 (0.93x) | 1818 (1.00x) | 356 (5.11x) | 527 (3.45x) | - | - | - | - | - |
| local affine score int64 | **836** | - | 765 (1.09x) | 875 (0.95x) | >10x | 308 (2.71x) | 366 (2.28x) | - | - | - | - | - |
| local affine score float32 | **1232** | api=functions: 1181 | 1223 (1.01x) | 1235 (1.00x) | - | 138 (8.95x) | 178 (6.92x) | - | - | - | - | - |
| local affine score float64 | **834** | fill=rowwise: 840 | 871 (0.96x) | 830 (1.00x) | - | 135 (6.19x) | 178 (4.69x) | - | 206 (4.04x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local affine path int32 | **687** | - | 712 (0.96x) | 653 (1.05x) | 871 (0.79x) | 199 (3.45x) | 150 (4.59x) | - | - | - | - | - |
| local affine path int64 | **282** | - | 280 (1.01x) | 295 (0.96x) | 74 (3.80x) | 198 (1.43x) | 131 (2.15x) | - | - | - | - | - |
| local affine path float32 | **623** | traceback=pointers: 639 | 577 (1.08x) | 648 (0.96x) | - | 110 (5.65x) | 163 (3.82x) | - | - | - | - | - |
| local affine path float64 | **361** | traceback=scores: 370 | 347 (1.04x) | 360 (1.00x) | - | 109 (3.31x) | 152 (2.37x) | - | 42 (8.53x) | 129 (2.80x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local affine hard int32 | **639** | - | 665 (0.96x) | 664 (0.96x) | - | - | - | - | - | - | - | - |
| local affine hard int64 | **290** | - | 282 (1.03x) | 279 (1.04x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **637** | traceback=grid: 642 | 630 (1.01x) | 637 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **384** | traceback=pointers: 390 | 364 (1.05x) | 375 (1.02x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **52** | api=functions: 54 | 61 (0.86x) | 52 (1.00x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local linear score int32 | **1952** | - | 2001 (0.98x) | 1976 (0.99x) | 3502 (0.56x) | 354 (5.52x) | 837 (2.33x) | - | - | - | - | - |
| local linear score int64 | **767** | - | 848 (0.90x) | 832 (0.92x) | 106 (7.25x) | 322 (2.38x) | 412 (1.86x) | - | - | - | - | - |
| local linear score float32 | **2181** | fill=striped: 2315 | 2383 (0.92x) | 2200 (0.99x) | - | >10x | >10x | - | - | - | - | - |
| local linear score float64 | **1094** | fill=striped: 1140 | 1162 (0.94x) | 1132 (0.97x) | - | 138 (7.94x) | 229 (4.78x) | - | 324 (3.38x) | - | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local linear path int32 | **1000** | - | 1026 (0.97x) | 1084 (0.92x) | 1032 (0.97x) | 193 (5.17x) | 188 (5.32x) | - | - | - | - | - |
| local linear path int64 | **244** | - | 372 (0.66x) | 246 (0.99x) | 83 (2.92x) | 186 (1.31x) | 200 (1.22x) | - | - | - | - | - |
| local linear path float32 | **1074** | traceback=scores: 1330 | 1037 (1.04x) | 1082 (0.99x) | - | 109 (9.82x) | 152 (7.06x) | - | - | - | - | - |
| local linear path float64 | **523** | traceback=scores: 635 | 515 (1.02x) | 552 (0.95x) | - | 110 (4.74x) | 162 (3.23x) | - | 85 (6.17x) | 202 (2.59x) | - | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local linear hard int32 | **1079** | - | 1036 (1.04x) | 1076 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard int64 | **260** | - | 383 (0.68x) | 260 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1155** | traceback=scores: 1267 | 1054 (1.10x) | 1090 (1.06x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **582** | traceback=scores: 643 | 492 (1.18x) | 581 (1.00x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **146** | api=functions: 174 | 137 (1.06x) | 147 (1.00x) | - | - | - | - | - | - | - | - |

#### oldsolace · dna_L100 · 4 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **6742** | schedule=sorted: 6531 | 5098 (1.32x) | 6641 (1.02x) | 4945 (1.36x) | 8400 (0.80x) | - | 1317 (5.12x) | 1186 (5.68x) | - | - | - | - |
| global affine score float32 | **6006** | schedule=sorted: 6011 | 5116 (1.17x) | 6016 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **3075** | schedule=sorted: 3016 | 2622 (1.17x) | 3053 (1.01x) | - | - | - | - | - | - | 636 (4.84x) | - | - |
| global affine path int32 | **3181** | schedule=sorted: 3183 | 2950 (1.08x) | 3208 (0.99x) | 2571 (1.24x) | 2668 (1.19x) | - | 1197 (2.66x) | 1319 (2.41x) | - | - | - | - |
| global affine path float32 | **3114** | schedule=sorted: 3061 | 2698 (1.15x) | 3134 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 | **1848** | schedule=sorted: 1847 | 1661 (1.11x) | 1858 (0.99x) | - | - | - | - | - | - | 190 (9.71x) | >10x | - |
| global affine hard int32 | **3712** | schedule=sorted: 3652 | 3476 (1.07x) | 3657 (1.02x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **3561** | schedule=sorted: 3655 | 3215 (1.11x) | 3588 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **2010** | schedule=sorted: 1975 | 1863 (1.08x) | 2010 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **324** | schedule=sorted: 326 | 340 (0.95x) | 326 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 | **7532** | schedule=sorted: 7629 | 6415 (1.17x) | 8223 (0.92x) | 4930 (1.53x) | 8356 (0.90x) | - | 1362 (5.53x) | 20837 (0.36x) | 6011 (1.25x) | - | - | - |
| global linear score float32 | **8253** | schedule=sorted: 8354 | 5958 (1.39x) | 8290 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **4392** | schedule=sorted: 4403 | 2907 (1.51x) | 4366 (1.01x) | - | - | - | - | - | - | 1045 (4.20x) | - | - |
| global linear path int32 | **4544** | schedule=sorted: 4541 | 3797 (1.20x) | 4495 (1.01x) | 2604 (1.74x) | 2681 (1.69x) | - | 1227 (3.70x) | 15963 (0.28x) | 2019 (2.25x) | - | - | - |
| global linear path float32 | **4387** | schedule=sorted: 4262 | 3917 (1.12x) | 4374 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 | **2844** | schedule=sorted: 2895 | 2476 (1.15x) | 2898 (0.98x) | - | - | - | - | - | - | 314 (9.07x) | >10x | - |
| global linear hard int32 | **6444** | schedule=sorted: 6458 | 4958 (1.30x) | 6246 (1.03x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **6201** | schedule=sorted: 6330 | 5221 (1.19x) | 6139 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **3645** | schedule=sorted: 3654 | 2930 (1.24x) | 3555 (1.03x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **582** | fill=striped: 582 | 554 (1.05x) | 577 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **5867** | schedule=sorted: 5930 | 4953 (1.18x) | 5648 (1.04x) | 4867 (1.21x) | 3889 (1.51x) | - | - | - | - | - | - | - |
| local affine score float32 | **5475** | schedule=sorted: 5510 | 4893 (1.12x) | 5336 (1.03x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **2391** | fill=striped: 3075 | 2479 (0.96x) | 2389 (1.00x) | - | - | - | - | - | - | 379 (6.31x) | - | - |
| local affine path int32 | **2680** | schedule=sorted: 2685 | 2614 (1.03x) | 2731 (0.98x) | 2339 (1.15x) | 2426 (1.10x) | - | - | - | - | - | - | - |
| local affine path float32 | **2478** | schedule=sorted: 2504 | 2428 (1.02x) | 2539 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **1417** | schedule=sorted: 1422 | 1443 (0.98x) | 1444 (0.98x) | - | - | - | - | - | - | >10x | 159 (8.94x) | - |
| local affine hard int32 | **3119** | schedule=sorted: 3061 | 3067 (1.02x) | 3125 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **2888** | schedule=sorted: 2857 | 2756 (1.05x) | 2852 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **1544** | schedule=sorted: 1546 | 1592 (0.97x) | 1532 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **309** | fill=striped: 308 | 320 (0.96x) | 306 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 | **6829** | schedule=sorted: 6991 | 6513 (1.05x) | 5695 (1.20x) | 4157 (1.64x) | 3862 (1.77x) | - | - | - | - | - | - | - |
| local linear score float32 | **5770** | schedule=sorted: 5875 | 5372 (1.07x) | 5914 (0.98x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **3118** | fill=striped: 3188 | 2714 (1.15x) | 3092 (1.01x) | - | - | - | - | - | - | 698 (4.47x) | - | - |
| local linear path int32 | **3678** | schedule=sorted: 3674 | 3078 (1.19x) | 3730 (0.99x) | 2238 (1.64x) | 2478 (1.48x) | - | - | - | - | - | - | - |
| local linear path float32 | **3403** | schedule=sorted: 3372 | 3206 (1.06x) | 3308 (1.03x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **2034** | schedule=sorted: 2037 | 1917 (1.06x) | 2033 (1.00x) | - | - | - | - | - | - | >10x | >10x | - |
| local linear hard int32 | **4895** | schedule=sorted: 4797 | 3825 (1.28x) | 4799 (1.02x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **4332** | schedule=sorted: 4335 | 4050 (1.07x) | 4225 (1.03x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **2340** | schedule=sorted: 2340 | 2187 (1.07x) | 2343 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **508** | fill=striped: 509 | 478 (1.06x) | 507 (1.00x) | - | - | - | - | - | - | - | - | - |

#### oldsolace · dna_L1000 · 4 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **8672** | schedule=sorted: 8686 | 9798 (0.89x) | 8352 (1.04x) | 8334 (1.04x) | 9900 (0.88x) | - | 2095 (4.14x) | 5475 (1.58x) | - | - | - | - |
| global affine score float32 | **4821** | schedule=sorted: 4854 | 4946 (0.97x) | 4760 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **2785** | schedule=sorted: 2781 | 2769 (1.01x) | 2768 (1.01x) | - | - | - | - | - | - | 766 (3.64x) | - | - |
| global affine path int32 | **3140** | schedule=sorted: 3161 | 3426 (0.92x) | 3125 (1.00x) | 3422 (0.92x) | 1375 (2.28x) | - | 1906 (1.65x) | 2180 (1.44x) | - | - | - | - |
| global affine path float32 | **3256** | fill=striped: 3314 | 3225 (1.01x) | 3223 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 | **2265** | fill=striped: 2342 | 2340 (0.97x) | 2288 (0.99x) | - | - | - | - | - | - | >10x | 391 (5.80x) | - |
| global affine hard int32 | **3193** | fill=striped: 3172 | 3460 (0.92x) | 3131 (1.02x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **3296** | schedule=sorted: 3334 | 3252 (1.01x) | 3258 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **2348** | schedule=sorted: 2342 | 2296 (1.02x) | 2279 (1.03x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **166** | fill=striped: 164 | 191 (0.87x) | 164 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 | **8300** | schedule=sorted: 8267 | 9525 (0.87x) | 8113 (1.02x) | 12528 (0.66x) | 9194 (0.90x) | - | 2058 (4.03x) | 82019 (0.10x) | 52257 (0.16x) | - | - | - |
| global linear score float32 | **9451** | schedule=sorted: 9642 | 9703 (0.97x) | 9340 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **3535** | fill=striped: 3541 | 3483 (1.02x) | 3469 (1.02x) | - | - | - | - | - | - | 1366 (2.59x) | - | - |
| global linear path int32 | **4931** | schedule=sorted: 4960 | 4693 (1.05x) | 4958 (0.99x) | 3350 (1.47x) | 1377 (3.58x) | - | 1900 (2.59x) | 68874 (0.07x) | 20939 (0.24x) | - | - | >10x |
| global linear path float32 | **4655** | schedule=sorted: 4690 | 4844 (0.96x) | 4638 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 | **2118** | fill=striped: 2153 | 2143 (0.99x) | 2143 (0.99x) | - | - | - | - | - | - | 367 (5.78x) | 688 (3.08x) | - |
| global linear hard int32 | **5115** | schedule=sorted: 5154 | 5021 (1.02x) | 5097 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **4923** | fill=striped: 4891 | 5112 (0.96x) | 4773 (1.03x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **2228** | schedule=sorted: 2230 | 2201 (1.01x) | 2193 (1.02x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **410** | fill=striped: 407 | 393 (1.04x) | 406 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **7503** | schedule=sorted: 7532 | 8155 (0.92x) | 7358 (1.02x) | 9391 (0.80x) | 4118 (1.82x) | - | - | - | - | - | - | - |
| local affine score float32 | **6892** | schedule=sorted: 6936 | 6980 (0.99x) | 6697 (1.03x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **3219** | schedule=sorted: 3212 | 3267 (0.99x) | 3163 (1.02x) | - | - | - | - | - | - | 445 (7.24x) | - | - |
| local affine path int32 | **2027** | fill=striped: 2031 | 2125 (0.95x) | 2042 (0.99x) | 2926 (0.69x) | 1328 (1.53x) | - | - | - | - | - | - | - |
| local affine path float32 | **2264** | fill=striped: 2268 | 2304 (0.98x) | 2286 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **1235** | schedule=sorted: 1254 | 1242 (0.99x) | 1257 (0.98x) | - | - | - | - | - | - | 130 (9.50x) | 377 (3.27x) | - |
| local affine hard int32 | **2081** | fill=striped: 2109 | 2139 (0.97x) | 2085 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **2340** | schedule=sorted: 2359 | 2380 (0.98x) | 2306 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **1289** | schedule=sorted: 1277 | 1274 (1.01x) | 1280 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **65** | schedule=sorted: 65 | 88 (0.74x) | 65 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 | **7656** | fill=striped: 7481 | 8487 (0.90x) | 7447 (1.03x) | 13808 (0.55x) | 4188 (1.83x) | - | - | - | - | - | - | - |
| local linear score float32 | **8881** | fill=striped: 8888 | 8750 (1.01x) | 8504 (1.04x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **3382** | fill=striped: 3414 | 3439 (0.98x) | 3351 (1.01x) | - | - | - | - | - | - | 993 (3.41x) | - | - |
| local linear path int32 | **3796** | schedule=sorted: 3845 | 4095 (0.93x) | 3861 (0.98x) | 3041 (1.25x) | 1323 (2.87x) | - | - | - | - | - | - | - |
| local linear path float32 | **3413** | fill=striped: 3432 | 3240 (1.05x) | 3420 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **1541** | fill=striped: 1540 | 1654 (0.93x) | 1533 (1.00x) | - | - | - | - | - | - | 193 (8.00x) | 565 (2.73x) | - |
| local linear hard int32 | **4008** | schedule=sorted: 4041 | 4344 (0.92x) | 3958 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **3526** | schedule=sorted: 3516 | 3316 (1.06x) | 3511 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **1555** | schedule=sorted: 1575 | 1692 (0.92x) | 1553 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **415** | fill=striped: 413 | 416 (1.00x) | 410 (1.01x) | - | - | - | - | - | - | - | - | - |

#### oldsolace · mirna · 4 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **4925** | schedule=sorted: 4991 | 3957 (1.24x) | 5196 (0.95x) | 2188 (2.25x) | 4691 (1.05x) | - | 834 (5.91x) | >10x | - | - | - | >10x |
| global affine score float32 | **4708** | schedule=sorted: 4713 | 4299 (1.10x) | 5151 (0.91x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **2479** | schedule=sorted: 2439 | 2275 (1.09x) | 2564 (0.97x) | - | - | - | - | - | - | 271 (9.13x) | - | - |
| global affine path int32 | **1512** | schedule=sorted: 1513 | 1416 (1.07x) | 1526 (0.99x) | 1140 (1.33x) | 1829 (0.83x) | - | 691 (2.19x) | >10x | - | - | - | >10x |
| global affine path float32 | **1470** | schedule=sorted: 1496 | 1356 (1.08x) | 1488 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 | **1142** | schedule=sorted: 1143 | 1028 (1.11x) | 1130 (1.01x) | - | - | - | - | - | - | >10x | >10x | - |
| global affine hard int32 | **2317** | schedule=sorted: 2303 | 2233 (1.04x) | 2358 (0.98x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **2299** | schedule=sorted: 2318 | 2175 (1.06x) | 2326 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **1551** | schedule=sorted: 1541 | 1420 (1.09x) | 1564 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **419** | schedule=sorted: 417 | 363 (1.16x) | 419 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 | **6096** | schedule=sorted: 6089 | 5197 (1.17x) | 6512 (0.94x) | 2208 (2.76x) | 4733 (1.29x) | - | 940 (6.49x) | 1159 (5.26x) | 926 (6.59x) | - | - | >10x |
| global linear score float32 | **6202** | schedule=sorted: 6503 | 4891 (1.27x) | 6752 (0.92x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **3718** | schedule=sorted: 3709 | 2541 (1.46x) | 3740 (0.99x) | - | - | - | - | - | - | >10x | - | - |
| global linear path int32 | **1370** | schedule=sorted: 1377 | 1345 (1.02x) | 1426 (0.96x) | 1103 (1.24x) | 1869 (0.73x) | - | 743 (1.84x) | 1115 (1.23x) | 346 (3.96x) | - | - | - |
| global linear path float32 | **1250** | schedule=sorted: 1276 | 1293 (0.97x) | 1283 (0.97x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 | **1106** | schedule=sorted: 1117 | 1079 (1.03x) | 1149 (0.96x) | - | - | - | - | - | - | 126 (8.81x) | >10x | - |
| global linear hard int32 | **2752** | schedule=sorted: 2767 | 2481 (1.11x) | 2761 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **2584** | schedule=sorted: 2573 | 2432 (1.06x) | 2610 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **1945** | schedule=sorted: 1946 | 1771 (1.10x) | 1964 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **667** | schedule=sorted: 662 | 564 (1.18x) | 660 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **4891** | schedule=sorted: 4592 | 3958 (1.24x) | 4738 (1.03x) | 2758 (1.77x) | 2773 (1.76x) | - | - | - | - | - | - | >10x |
| local affine score float32 | **4643** | schedule=sorted: 4479 | 4123 (1.13x) | 4474 (1.04x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **2172** | fill=striped: 2395 | 1991 (1.09x) | 2181 (1.00x) | - | - | - | - | - | - | >10x | - | - |
| local affine path int32 | **2551** | schedule=sorted: 2575 | 2418 (1.05x) | 2593 (0.98x) | 1517 (1.68x) | 1836 (1.39x) | - | - | - | - | - | - | >10x |
| local affine path float32 | **2396** | schedule=sorted: 2381 | 2263 (1.06x) | 2360 (1.02x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **1427** | schedule=sorted: 1421 | 1412 (1.01x) | 1426 (1.00x) | - | - | - | - | - | - | >10x | >10x | - |
| local affine hard int32 | **2734** | schedule=sorted: 2749 | 2621 (1.04x) | 2799 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **2584** | schedule=sorted: 2607 | 2410 (1.07x) | 2600 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **1502** | schedule=sorted: 1500 | 1496 (1.00x) | 1500 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **425** | schedule=sorted: 421 | 373 (1.14x) | 419 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 | **5686** | schedule=sorted: 5765 | 5158 (1.10x) | 4891 (1.16x) | 2069 (2.75x) | 2770 (2.05x) | - | - | - | - | - | - | >10x |
| local linear score float32 | **5003** | schedule=sorted: 4895 | 4549 (1.10x) | 5203 (0.96x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **2720** | schedule=sorted: 2760 | 2348 (1.16x) | 2774 (0.98x) | - | - | - | - | - | - | >10x | - | - |
| local linear path int32 | **2578** | schedule=sorted: 2580 | 2248 (1.15x) | 2627 (0.98x) | 1379 (1.87x) | 1755 (1.47x) | - | - | - | - | - | - | >10x |
| local linear path float32 | **2395** | schedule=sorted: 2433 | 2260 (1.06x) | 2429 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **1661** | schedule=sorted: 1657 | 1532 (1.08x) | 1646 (1.01x) | - | - | - | - | - | - | >10x | >10x | - |
| local linear hard int32 | **3447** | schedule=sorted: 3491 | 2841 (1.21x) | 3538 (0.97x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **3108** | schedule=sorted: 3183 | 2874 (1.08x) | 3186 (0.98x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **1954** | schedule=sorted: 1959 | 1787 (1.09x) | 1945 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **607** | schedule=sorted: 603 | 526 (1.15x) | 607 (1.00x) | - | - | - | - | - | - | - | - | - |

#### oldsolace · protein_L100 · 4 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **4653** | fill=striped: 4698 | 5719 (0.81x) | 4564 (1.02x) | 3699 (1.26x) | 4078 (1.14x) | - | 1505 (3.09x) | - | - | - |
| global affine score float32 | **2317** | schedule=sorted: 2317 | 2335 (0.99x) | 2277 (1.02x) | - | - | - | - | - | - | - |
| global affine score float64 | **2222** | schedule=sorted: 2221 | 2162 (1.03x) | 2200 (1.01x) | - | - | - | - | 704 (3.15x) | - | - |
| global affine path int32 | **2211** | schedule=sorted: 2222 | 1936 (1.14x) | 2230 (0.99x) | 2173 (1.02x) | 1969 (1.12x) | - | 1292 (1.71x) | - | - | - |
| global affine path float32 | **2051** | schedule=sorted: 2046 | 1839 (1.12x) | 2072 (0.99x) | - | - | - | - | - | - | - |
| global affine path float64 | **1584** | schedule=sorted: 1608 | 1423 (1.11x) | 1614 (0.98x) | - | - | - | - | 175 (9.03x) | 169 (9.39x) | - |
| global affine hard int32 | **2198** | schedule=sorted: 2199 | 1972 (1.11x) | 2206 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float32 | **2097** | schedule=sorted: 2084 | 1889 (1.11x) | 2058 (1.02x) | - | - | - | - | - | - | - |
| global affine hard float64 | **1592** | schedule=sorted: 1586 | 1488 (1.07x) | 1612 (0.99x) | - | - | - | - | - | - | - |
| global affine soft float64 | **317** | schedule=sorted: 317 | 333 (0.95x) | 319 (0.99x) | - | - | - | - | - | - | - |
| global linear score int32 | **4094** | fill=striped: 4125 | 5185 (0.79x) | 4129 (0.99x) | 3299 (1.24x) | 3709 (1.10x) | - | 1394 (2.94x) | - | - | - |
| global linear score float32 | **3071** | schedule=sorted: 3074 | 3103 (0.99x) | 3023 (1.02x) | - | - | - | - | - | - | - |
| global linear score float64 | **2695** | schedule=sorted: 2724 | 2610 (1.03x) | 2728 (0.99x) | - | - | - | - | 1093 (2.46x) | - | - |
| global linear path int32 | **2623** | schedule=sorted: 2618 | 2931 (0.89x) | 2620 (1.00x) | 2058 (1.27x) | 1878 (1.40x) | - | 1257 (2.09x) | - | - | - |
| global linear path float32 | **2291** | schedule=sorted: 2269 | 2043 (1.12x) | 2277 (1.01x) | - | - | - | - | - | - | - |
| global linear path float64 | **1939** | schedule=sorted: 1926 | 1703 (1.14x) | 1962 (0.99x) | - | - | - | - | 384 (5.05x) | 197 (9.83x) | - |
| global linear hard int32 | **3006** | fill=striped: 3028 | 3390 (0.89x) | 3016 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float32 | **2503** | schedule=sorted: 2504 | 2212 (1.13x) | 2516 (0.99x) | - | - | - | - | - | - | - |
| global linear hard float64 | **2094** | schedule=sorted: 2114 | 1863 (1.12x) | 2095 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 | **443** | fill=striped: 443 | 416 (1.06x) | 439 (1.01x) | - | - | - | - | - | - | - |
| local affine score int32 | **3887** | schedule=sorted: 3953 | 4756 (0.82x) | 3910 (0.99x) | 2944 (1.32x) | 2500 (1.55x) | - | - | - | - | - |
| local affine score float32 | **2400** | schedule=sorted: 2428 | 2488 (0.96x) | 2413 (0.99x) | - | - | - | - | - | - | - |
| local affine score float64 | **2200** | schedule=sorted: 2207 | 2115 (1.04x) | 2221 (0.99x) | - | - | - | - | 460 (4.79x) | - | - |
| local affine path int32 | **2006** | schedule=sorted: 1988 | 1736 (1.16x) | 1979 (1.01x) | 1841 (1.09x) | 1767 (1.14x) | - | - | - | - | - |
| local affine path float32 | **1739** | schedule=sorted: 1752 | 1628 (1.07x) | 1770 (0.98x) | - | - | - | - | - | - | - |
| local affine path float64 | **1343** | schedule=sorted: 1340 | 1264 (1.06x) | 1350 (0.99x) | - | - | - | - | >10x | 163 (8.26x) | - |
| local affine hard int32 | **2089** | schedule=sorted: 2086 | 1871 (1.12x) | 2097 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float32 | **1858** | schedule=sorted: 1852 | 1692 (1.10x) | 1855 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 | **1374** | schedule=sorted: 1352 | 1301 (1.06x) | 1375 (1.00x) | - | - | - | - | - | - | - |
| local affine soft float64 | **288** | fill=striped: 288 | 296 (0.97x) | 285 (1.01x) | - | - | - | - | - | - | - |
| local linear score int32 | **3868** | schedule=sorted: 3893 | 4696 (0.82x) | 3886 (1.00x) | 2896 (1.34x) | 2409 (1.61x) | - | - | - | - | - |
| local linear score float32 | **3355** | schedule=sorted: 3371 | 3339 (1.00x) | 3363 (1.00x) | - | - | - | - | - | - | - |
| local linear score float64 | **2759** | schedule=sorted: 2753 | 2636 (1.05x) | 2783 (0.99x) | - | - | - | - | 612 (4.51x) | - | - |
| local linear path int32 | **2428** | schedule=sorted: 2463 | 1909 (1.27x) | 2466 (0.98x) | 1817 (1.34x) | 1772 (1.37x) | - | - | - | - | - |
| local linear path float32 | **1970** | schedule=sorted: 1985 | 1772 (1.11x) | 1980 (1.00x) | - | - | - | - | - | - | - |
| local linear path float64 | **1528** | schedule=sorted: 1552 | 1381 (1.11x) | 1543 (0.99x) | - | - | - | - | 198 (7.73x) | 194 (7.85x) | - |
| local linear hard int32 | **2663** | schedule=sorted: 2689 | 2054 (1.30x) | 2688 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float32 | **2134** | schedule=sorted: 2144 | 1917 (1.11x) | 2124 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float64 | **1639** | schedule=sorted: 1639 | 1454 (1.13x) | 1635 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **468** | schedule=sorted: 467 | 444 (1.05x) | 466 (1.00x) | - | - | - | - | - | - | - |

#### oldsolace · protein_L1000 · 4 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **7917** | fill=striped: 7829 | 9201 (0.86x) | 7795 (1.02x) | 7471 (1.06x) | 3639 (2.18x) | - | 2065 (3.83x) | - | - | - |
| global affine score float32 | **3150** | fill=striped: 3160 | 3282 (0.96x) | 3159 (1.00x) | - | - | - | - | - | - | - |
| global affine score float64 | **2168** | fill=striped: 2184 | 2171 (1.00x) | 2173 (1.00x) | - | - | - | - | 772 (2.81x) | - | - |
| global affine path int32 | **2838** | fill=striped: 2854 | 3013 (0.94x) | 2829 (1.00x) | 3259 (0.87x) | 1132 (2.51x) | - | 1902 (1.49x) | - | - | - |
| global affine path float32 | **2414** | fill=striped: 2411 | 2434 (0.99x) | 2424 (1.00x) | - | - | - | - | - | - | - |
| global affine path float64 | **1884** | fill=striped: 1890 | 1896 (0.99x) | 1874 (1.01x) | - | - | - | - | 210 (8.97x) | 451 (4.17x) | - |
| global affine hard int32 | **2922** | fill=striped: 2908 | 3029 (0.96x) | 2871 (1.02x) | - | - | - | - | - | - | - |
| global affine hard float32 | **2492** | schedule=sorted: 2464 | 2461 (1.01x) | 2454 (1.02x) | - | - | - | - | - | - | - |
| global affine hard float64 | **1912** | schedule=sorted: 1915 | 1912 (1.00x) | 1906 (1.00x) | - | - | - | - | - | - | - |
| global affine soft float64 | **175** | fill=striped: 175 | 198 (0.88x) | 173 (1.01x) | - | - | - | - | - | - | - |
| global linear score int32 | **7727** | schedule=sorted: 7866 | 9053 (0.85x) | 7548 (1.02x) | 7328 (1.05x) | 3645 (2.12x) | - | 2033 (3.80x) | - | - | - |
| global linear score float32 | **8334** | fill=striped: 8347 | 8330 (1.00x) | 8088 (1.03x) | - | - | - | - | - | - | - |
| global linear score float64 | **3378** | fill=striped: 3390 | 3240 (1.04x) | 3264 (1.03x) | - | - | - | - | 1363 (2.48x) | - | - |
| global linear path int32 | **4693** | fill=striped: 4592 | 4617 (1.02x) | 4691 (1.00x) | 3194 (1.47x) | 1136 (4.13x) | - | 1884 (2.49x) | - | - | >10x |
| global linear path float32 | **4293** | fill=striped: 4311 | 4415 (0.97x) | 4241 (1.01x) | - | - | - | - | - | - | - |
| global linear path float64 | **2090** | fill=striped: 2078 | 2052 (1.02x) | 2072 (1.01x) | - | - | - | - | 484 (4.32x) | 697 (3.00x) | - |
| global linear hard int32 | **4995** | fill=striped: 4866 | 4739 (1.05x) | 4855 (1.03x) | - | - | - | - | - | - | - |
| global linear hard float32 | **4478** | fill=striped: 4551 | 4694 (0.95x) | 4420 (1.01x) | - | - | - | - | - | - | - |
| global linear hard float64 | **2135** | fill=striped: 2171 | 2128 (1.00x) | 2154 (0.99x) | - | - | - | - | - | - | - |
| global linear soft float64 | **363** | schedule=sorted: 361 | 350 (1.04x) | 362 (1.01x) | - | - | - | - | - | - | - |
| local affine score int32 | **6981** | fill=striped: 7162 | 7868 (0.89x) | 6943 (1.01x) | 6329 (1.10x) | 2428 (2.88x) | - | - | - | - | - |
| local affine score float32 | **3674** | fill=striped: 3688 | 3849 (0.95x) | 3669 (1.00x) | - | - | - | - | - | - | - |
| local affine score float64 | **2377** | schedule=sorted: 2398 | 2458 (0.97x) | 2403 (0.99x) | - | - | - | - | 557 (4.27x) | - | - |
| local affine path int32 | **2044** | schedule=sorted: 2052 | 2087 (0.98x) | 2008 (1.02x) | 2766 (0.74x) | 1108 (1.85x) | - | - | - | - | - |
| local affine path float32 | **1686** | schedule=sorted: 1686 | 1753 (0.96x) | 1682 (1.00x) | - | - | - | - | - | - | - |
| local affine path float64 | **1105** | schedule=sorted: 1086 | 1079 (1.02x) | 1095 (1.01x) | - | - | - | - | 121 (9.14x) | 416 (2.66x) | - |
| local affine hard int32 | **2033** | fill=striped: 2071 | 2110 (0.96x) | 2047 (0.99x) | - | - | - | - | - | - | - |
| local affine hard float32 | **1723** | fill=striped: 1728 | 1799 (0.96x) | 1711 (1.01x) | - | - | - | - | - | - | - |
| local affine hard float64 | **1118** | fill=striped: 1118 | 1090 (1.03x) | 1111 (1.01x) | - | - | - | - | - | - | - |
| local affine soft float64 | **153** | fill=striped: 153 | 183 (0.84x) | 151 (1.01x) | - | - | - | - | - | - | - |
| local linear score int32 | **6872** | fill=striped: 6949 | 7912 (0.87x) | 6683 (1.03x) | 6187 (1.11x) | 2406 (2.86x) | - | - | - | - | - |
| local linear score float32 | **7745** | fill=striped: 7909 | 7892 (0.98x) | 7430 (1.04x) | - | - | - | - | - | - | - |
| local linear score float64 | **3215** | fill=striped: 3249 | 3347 (0.96x) | 3245 (0.99x) | - | - | - | - | 881 (3.65x) | - | - |
| local linear path int32 | **3581** | schedule=sorted: 3661 | 3868 (0.93x) | 3574 (1.00x) | 2778 (1.29x) | 1095 (3.27x) | - | - | - | - | - |
| local linear path float32 | **3282** | schedule=sorted: 3282 | 3040 (1.08x) | 3271 (1.00x) | - | - | - | - | - | - | - |
| local linear path float64 | **1510** | schedule=sorted: 1515 | 1602 (0.94x) | 1507 (1.00x) | - | - | - | - | 223 (6.76x) | 623 (2.42x) | - |
| local linear hard int32 | **3762** | fill=striped: 3786 | 4042 (0.93x) | 3733 (1.01x) | - | - | - | - | - | - | - |
| local linear hard float32 | **3364** | schedule=sorted: 3392 | 3225 (1.04x) | 3344 (1.01x) | - | - | - | - | - | - | - |
| local linear hard float64 | **1535** | fill=striped: 1521 | 1660 (0.92x) | 1521 (1.01x) | - | - | - | - | - | - | - |
| local linear soft float64 | **318** | fill=striped: 318 | 321 (0.99x) | 318 (1.00x) | - | - | - | - | - | - | - |

#### oldsolace · protein_L300 · 4 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **6680** | fill=striped: 6743 | 8007 (0.83x) | 6543 (1.02x) | 5784 (1.15x) | 3909 (1.71x) | - | 1816 (3.68x) | - | - | - |
| global affine score float32 | **2813** | schedule=sorted: 2828 | 2908 (0.97x) | 2779 (1.01x) | - | - | - | - | - | - | - |
| global affine score float64 | **2476** | fill=striped: 2490 | 2520 (0.98x) | 2481 (1.00x) | - | - | - | - | 769 (3.22x) | - | - |
| global affine path int32 | **2761** | schedule=sorted: 2762 | 3066 (0.90x) | 2770 (1.00x) | 3174 (0.87x) | 1414 (1.95x) | - | 1688 (1.64x) | - | - | - |
| global affine path float32 | **1685** | fill=striped: 1691 | 1695 (0.99x) | 1696 (0.99x) | - | - | - | - | - | - | - |
| global affine path float64 | **1225** | fill=striped: 1226 | 1320 (0.93x) | 1255 (0.98x) | - | - | - | - | 191 (6.42x) | 378 (3.24x) | - |
| global affine hard int32 | **2906** | fill=striped: 2865 | 3232 (0.90x) | 2856 (1.02x) | - | - | - | - | - | - | - |
| global affine hard float32 | **1726** | fill=striped: 1726 | 1737 (0.99x) | 1745 (0.99x) | - | - | - | - | - | - | - |
| global affine hard float64 | **1256** | fill=striped: 1260 | 1358 (0.93x) | 1289 (0.97x) | - | - | - | - | - | - | - |
| global affine soft float64 | **194** | fill=striped: 193 | 216 (0.90x) | 192 (1.01x) | - | - | - | - | - | - | - |
| global linear score int32 | **6469** | fill=striped: 6531 | 7926 (0.82x) | 6387 (1.01x) | 5791 (1.12x) | 3851 (1.68x) | - | 1797 (3.60x) | - | - | - |
| global linear score float32 | **5746** | fill=striped: 5763 | 5918 (0.97x) | 5701 (1.01x) | - | - | - | - | - | - | - |
| global linear score float64 | **3977** | schedule=sorted: 4022 | 3875 (1.03x) | 3891 (1.02x) | - | - | - | - | 1359 (2.93x) | - | - |
| global linear path int32 | **4259** | fill=striped: 4148 | 4457 (0.96x) | 4093 (1.04x) | 3047 (1.40x) | 1408 (3.03x) | - | 1659 (2.57x) | - | - | - |
| global linear path float32 | **3750** | fill=striped: 3751 | 3999 (0.94x) | 3756 (1.00x) | - | - | - | - | - | - | - |
| global linear path float64 | **2503** | schedule=sorted: 2529 | 2630 (0.95x) | 2548 (0.98x) | - | - | - | - | 478 (5.24x) | 548 (4.57x) | - |
| global linear hard int32 | **4767** | fill=striped: 4823 | 5071 (0.94x) | 4605 (1.04x) | - | - | - | - | - | - | - |
| global linear hard float32 | **4169** | schedule=sorted: 4180 | 4452 (0.94x) | 4109 (1.01x) | - | - | - | - | - | - | - |
| global linear hard float64 | **2698** | schedule=sorted: 2705 | 2812 (0.96x) | 2690 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 | **287** | schedule=sorted: 287 | 277 (1.04x) | 285 (1.01x) | - | - | - | - | - | - | - |
| local affine score int32 | **6060** | fill=striped: 6102 | 7181 (0.84x) | 5901 (1.03x) | 4859 (1.25x) | 2543 (2.38x) | - | - | - | - | - |
| local affine score float32 | **3237** | schedule=sorted: 3222 | 3352 (0.97x) | 3244 (1.00x) | - | - | - | - | - | - | - |
| local affine score float64 | **2699** | schedule=sorted: 2708 | 2695 (1.00x) | 2705 (1.00x) | - | - | - | - | 531 (5.09x) | - | - |
| local affine path int32 | **2206** | schedule=sorted: 2227 | 2408 (0.92x) | 2196 (1.00x) | 2671 (0.83x) | 1357 (1.63x) | - | - | - | - | - |
| local affine path float32 | **1530** | fill=striped: 1537 | 1593 (0.96x) | 1537 (1.00x) | - | - | - | - | - | - | - |
| local affine path float64 | **1101** | fill=striped: 1093 | 1102 (1.00x) | 1099 (1.00x) | - | - | - | - | 123 (8.93x) | 358 (3.08x) | - |
| local affine hard int32 | **2310** | schedule=sorted: 2304 | 2573 (0.90x) | 2296 (1.01x) | - | - | - | - | - | - | - |
| local affine hard float32 | **1571** | schedule=sorted: 1588 | 1641 (0.96x) | 1571 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 | **1124** | fill=striped: 1125 | 1114 (1.01x) | 1122 (1.00x) | - | - | - | - | - | - | - |
| local affine soft float64 | **89** | schedule=sorted: 89 | 121 (0.74x) | 88 (1.01x) | - | - | - | - | - | - | - |
| local linear score int32 | **6071** | fill=striped: 6073 | 7159 (0.85x) | 5843 (1.04x) | 4875 (1.25x) | 2570 (2.36x) | - | - | - | - | - |
| local linear score float32 | **5921** | fill=striped: 5924 | 5978 (0.99x) | 5800 (1.02x) | - | - | - | - | - | - | - |
| local linear score float64 | **3764** | fill=striped: 3815 | 3676 (1.02x) | 3758 (1.00x) | - | - | - | - | 774 (4.86x) | - | - |
| local linear path int32 | **3433** | fill=striped: 3523 | 3602 (0.95x) | 3546 (0.97x) | 2647 (1.30x) | 1358 (2.53x) | - | - | - | - | - |
| local linear path float32 | **3024** | schedule=sorted: 3022 | 3117 (0.97x) | 2955 (1.02x) | - | - | - | - | - | - | - |
| local linear path float64 | **2022** | fill=striped: 2022 | 1971 (1.03x) | 2014 (1.00x) | - | - | - | - | 222 (9.09x) | 510 (3.97x) | - |
| local linear hard int32 | **3877** | schedule=sorted: 3915 | 4150 (0.93x) | 3792 (1.02x) | - | - | - | - | - | - | - |
| local linear hard float32 | **3225** | fill=striped: 3280 | 3405 (0.95x) | 3220 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float64 | **2142** | schedule=sorted: 2115 | 2097 (1.02x) | 2148 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **203** | schedule=sorted: 204 | 197 (1.03x) | 202 (1.01x) | - | - | - | - | - | - | - |

#### oldsolace · protein_L3000 · 4 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **5901** | fill=striped: 6114 | 5886 (1.00x) | 5943 (0.99x) | 6669 (0.88x) | 2375 (2.48x) | - | 2146 (2.75x) | - | - | - |
| global affine score float32 | **2898** | fill=striped: 2884 | 2970 (0.98x) | 2865 (1.01x) | - | - | - | - | - | - | - |
| global affine score float64 | **2069** | schedule=sorted: 2095 | 2061 (1.00x) | 2077 (1.00x) | - | - | - | - | 714 (2.90x) | - | - |
| global affine path int32 | **2462** | fill=striped: 2465 | 2534 (0.97x) | 2429 (1.01x) | 2228 (1.11x) | 807 (3.05x) | - | 1955 (1.26x) | - | - | - |
| global affine path float32 | **2694** | fill=striped: 2781 | 2655 (1.01x) | 2689 (1.00x) | - | - | - | - | - | - | - |
| global affine path float64 | **2073** | fill=striped: 1997 | 1999 (1.04x) | 2000 (1.04x) | - | - | - | - | >10x | 375 (5.53x) | - |
| global affine hard int32 | **2537** | schedule=sorted: 2531 | 2599 (0.98x) | 2519 (1.01x) | - | - | - | - | - | - | - |
| global affine hard float32 | **2732** | fill=striped: 2734 | 2720 (1.00x) | 2710 (1.01x) | - | - | - | - | - | - | - |
| global affine hard float64 | **2031** | schedule=sorted: 2034 | 2037 (1.00x) | 1985 (1.02x) | - | - | - | - | - | - | - |
| global affine soft float64 | **181** | fill=striped: 181 | 200 (0.91x) | 180 (1.01x) | - | - | - | - | - | - | - |
| global linear score int32 | **5821** | schedule=sorted: 5854 | 6088 (0.96x) | 5650 (1.03x) | 9601 (0.61x) | 2394 (2.43x) | - | 2092 (2.78x) | - | - | - |
| global linear score float32 | **6356** | fill=striped: 6663 | 6562 (0.97x) | 6355 (1.00x) | - | - | - | - | - | - | - |
| global linear score float64 | **2988** | fill=striped: 3153 | 3158 (0.95x) | 3139 (0.95x) | - | - | - | - | 1252 (2.39x) | - | - |
| global linear path int32 | **3272** | fill=striped: 3244 | 3195 (1.02x) | 3269 (1.00x) | 2256 (1.45x) | 791 (4.14x) | - | 1927 (1.70x) | - | - | - |
| global linear path float32 | **3787** | schedule=sorted: 3716 | 3615 (1.05x) | 3780 (1.00x) | - | - | - | - | - | - | - |
| global linear path float64 | **1869** | fill=striped: 1879 | 1835 (1.02x) | 1885 (0.99x) | - | - | - | - | 455 (4.11x) | 634 (2.95x) | - |
| global linear hard int32 | **3259** | fill=striped: 3290 | 3255 (1.00x) | 3227 (1.01x) | - | - | - | - | - | - | - |
| global linear hard float32 | **3763** | schedule=sorted: 3737 | 3661 (1.03x) | 3770 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float64 | **1920** | fill=striped: 1893 | 1880 (1.02x) | 1897 (1.01x) | - | - | - | - | - | - | - |
| global linear soft float64 | **433** | schedule=sorted: 431 | 419 (1.03x) | 428 (1.01x) | - | - | - | - | - | - | - |
| local affine score int32 | **5256** | schedule=sorted: 5426 | 5585 (0.94x) | 5198 (1.01x) | 5169 (1.02x) | 1676 (3.14x) | - | - | - | - | - |
| local affine score float32 | **3314** | fill=striped: 3333 | 3486 (0.95x) | 3324 (1.00x) | - | - | - | - | - | - | - |
| local affine score float64 | **2304** | schedule=sorted: 2257 | 2293 (1.00x) | 2251 (1.02x) | - | - | - | - | 505 (4.56x) | - | - |
| local affine path int32 | **1842** | schedule=sorted: 1872 | 1885 (0.98x) | 1837 (1.00x) | 2034 (0.91x) | 766 (2.41x) | - | - | - | - | - |
| local affine path float32 | **1653** | schedule=sorted: 1674 | 1630 (1.01x) | 1649 (1.00x) | - | - | - | - | - | - | - |
| local affine path float64 | **1000** | fill=striped: 995 | 964 (1.04x) | 1000 (1.00x) | - | - | - | - | 111 (8.98x) | 348 (2.87x) | - |
| local affine hard int32 | **1866** | schedule=sorted: 1857 | 1928 (0.97x) | 1792 (1.04x) | - | - | - | - | - | - | - |
| local affine hard float32 | **1677** | schedule=sorted: 1691 | 1659 (1.01x) | 1685 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 | **998** | fill=striped: 1013 | 980 (1.02x) | 1006 (0.99x) | - | - | - | - | - | - | - |
| local affine soft float64 | **135** | fill=striped: 133 | 157 (0.86x) | 132 (1.02x) | - | - | - | - | - | - | - |
| local linear score int32 | **5558** | fill=striped: 5290 | 5481 (1.01x) | 5219 (1.07x) | 9658 (0.58x) | 1633 (3.40x) | - | - | - | - | - |
| local linear score float32 | **6513** | fill=striped: 6450 | 6564 (0.99x) | 6234 (1.04x) | - | - | - | - | - | - | - |
| local linear score float64 | **3020** | schedule=sorted: 3090 | 3118 (0.97x) | 3084 (0.98x) | - | - | - | - | 882 (3.42x) | - | - |
| local linear path int32 | **2846** | fill=striped: 2837 | 2902 (0.98x) | 2875 (0.99x) | 2273 (1.25x) | 751 (3.79x) | - | - | - | - | - |
| local linear path float32 | **3033** | schedule=sorted: 3073 | 2829 (1.07x) | 3065 (0.99x) | - | - | - | - | - | - | - |
| local linear path float64 | **1518** | schedule=sorted: 1513 | 1406 (1.08x) | 1508 (1.01x) | - | - | - | - | 209 (7.26x) | 566 (2.68x) | - |
| local linear hard int32 | **2982** | schedule=sorted: 2948 | 3010 (0.99x) | 2910 (1.02x) | - | - | - | - | - | - | - |
| local linear hard float32 | **3097** | schedule=sorted: 3089 | 2863 (1.08x) | 3072 (1.01x) | - | - | - | - | - | - | - |
| local linear hard float64 | **1519** | fill=striped: 1530 | 1429 (1.06x) | 1525 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **390** | fill=striped: 390 | 386 (1.01x) | 389 (1.00x) | - | - | - | - | - | - | - |

</details>

<details><summary><b>solace</b> — i7-11850H (Tiger Lake), AVX-512 — under WSL2: every cell</summary>


#### solace · dna_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **4642** | - | 5226 (0.89x) | 4857 (0.96x) | 2591 (1.79x) | 15967 (0.29x) | 755 (6.14x) | - | - | - | - | - | 491 (9.45x) sat |
| global affine score int32 | **4027** | - | 3974 (1.01x) | 4122 (0.98x) | 2753 (1.46x) | 8190 (0.49x) | 845 (4.76x) | - | 761 (5.29x) | 827 (4.87x) | - | - | - |
| global affine score int64 | **2793** | - | 3212 (0.87x) | 2885 (0.97x) | >10x | 476 (5.87x) | 685 (4.08x) | - | - | - | - | - | - |
| global affine score float32 | **1093** | kernel=avx2: 3188 | 1118 (0.98x) | 1113 (0.98x) | - | 322 (3.39x) | 325 (3.36x) | - | - | - | - | - | - |
| global affine score float64 (load!) | **1119** | kernel=avx2: 1657 | 1123 (1.00x) | 1126 (0.99x) | - | 321 (3.48x) | 323 (3.46x) | - | - | - | - | 339 (3.30x) | - |
| global affine path int16 (load!) | **2078** | - | 2080 (1.00x) | 2074 (1.00x) | 1754 (1.18x) | 1800 (1.15x) | 282 (7.37x) | - | - | - | - | - | 342 (6.08x) sat |
| global affine path int32 (load!) | **1746** | - | 1776 (0.98x) | 1794 (0.97x) | 1544 (1.13x) | 991 (1.76x) | 305 (5.72x) | - | 677 (2.58x) | 808 (2.16x) | - | - | - |
| global affine path int64 | **1107** | - | 1148 (0.96x) | 1102 (1.00x) | >10x | 237 (4.67x) | 293 (3.77x) | - | - | - | - | - | - |
| global affine path float32 | **796** | kernel=avx2: 1845 | 773 (1.03x) | 792 (1.00x) | - | 203 (3.91x) | 259 (3.07x) | - | - | - | - | - | - |
| global affine path float64 | **696** | kernel=avx2: 1074 | 680 (1.02x) | 696 (1.00x) | - | 206 (3.38x) | 265 (2.63x) | - | - | - | - | 100 (6.95x) | - |
| global affine hard int16 | **2637** | - | 2637 (1.00x) | 2548 (1.03x) | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **2018** | - | 2182 (0.92x) | 2130 (0.95x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **1183** | - | 1337 (0.89x) | 1201 (0.98x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **852** | kernel=avx2: 2159 | 822 (1.04x) | 869 (0.98x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **740** | kernel=avx2: 1177 | 741 (1.00x) | 756 (0.98x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **185** | api=functions: 231 | 189 (0.98x) | 189 (0.98x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **11495** | - | 11331 (1.01x) | 12364 (0.93x) | 2572 (4.47x) | 15384 (0.75x) | >10x | - | - | - | - | - | >10x |
| global linear score int32 | **8873** | - | 7973 (1.11x) | 9317 (0.95x) | 2668 (3.33x) | 8201 (1.08x) | 1242 (7.15x) | - | >10x | 9370 (0.95x) | 3200 (2.77x) | - | >10x |
| global linear score int64 | **4323** | - | 4090 (1.06x) | 4556 (0.95x) | >10x | 458 (9.43x) | 753 (5.74x) | - | - | - | - | - | - |
| global linear score float32 | **6963** | kernel=avx2: 4094 | 6820 (1.02x) | 7086 (0.98x) | - | >10x | >10x | - | - | - | - | - | - |
| global linear score float64 (load!) | **4124** | kernel=avx2: 2104 | 3678 (1.12x) | 4026 (1.02x) | - | >10x | 457 (9.02x) | - | - | - | - | 601 (6.86x) | - |
| global linear path int16 (load!) | **3947** | - | 3935 (1.00x) | 3909 (1.01x) | 1738 (2.27x) | 1815 (2.17x) | >10x | - | - | - | - | - | >10x |
| global linear path int32 (load!) | **3465** | - | 3124 (1.11x) | 3330 (1.04x) | 1556 (2.23x) | 1002 (3.46x) | 373 (9.28x) | - | 716 (4.84x) | 7925 (0.44x) | 1094 (3.17x) | - | >10x |
| global linear path int64 (load!) | **2599** | - | 2319 (1.12x) | 2794 (0.93x) | >10x | >10x | 365 (7.13x) | - | - | - | - | - | - |
| global linear path float32 (load!) | **2830** | traceback=pointers: 2968 | 3011 (0.94x) | 3044 (0.93x) | - | >10x | 334 (8.47x) | - | - | - | - | - | - |
| global linear path float64 (load!) | **2276** | traceback=pointers: 2250 | 2146 (1.06x) | 2367 (0.96x) | - | >10x | 338 (6.73x) | - | - | - | - | >10x | - |
| global linear hard int16 (load!) | **7152** | - | 6479 (1.10x) | 7142 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard int32 (load!) | **5886** | - | 4787 (1.23x) | 5846 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **3955** | - | 3158 (1.25x) | 4116 (0.96x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **4470** | traceback=pointers: 4744 | 4401 (1.02x) | 4420 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **3047** | traceback=hirschberg: 3118 | 2879 (1.06x) | 3224 (0.94x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **340** | api=functions: 394 | 293 (1.16x) | 340 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score int16 (load!) | **4445** | - | 4845 (0.92x) | 4477 (0.99x) | 2920 (1.52x) | 6778 (0.66x) | 645 (6.90x) | 2906 (1.53x) sat | - | - | - | - | >10x |
| local affine score int32 (load!) | **3573** | - | 3620 (0.99x) | 3756 (0.95x) | 2785 (1.28x) | 3327 (1.07x) | 733 (4.87x) | - | - | - | - | - | - |
| local affine score int64 (load!) | **2441** | - | 2771 (0.88x) | 2515 (0.97x) | >10x | 250 (9.77x) | 633 (3.86x) | - | - | - | - | - | - |
| local affine score float32 (load!) | **1780** | kernel=avx2: 2883 | 1825 (0.98x) | 1844 (0.97x) | - | >10x | 251 (7.08x) | - | - | - | - | - | - |
| local affine score float64 (load!) | **1520** | kernel=avx2: 1575 | 1530 (0.99x) | 1520 (1.00x) | - | 175 (8.69x) | 251 (6.06x) | - | - | - | - | 212 (7.18x) | - |
| local affine path int16 (load!) | **1746** | - | 1765 (0.99x) | 1694 (1.03x) | 1615 (1.08x) | 1774 (0.98x) | >10x | 593 (2.94x) sat | - | - | - | - | 240 (7.27x) sat |
| local affine path int32 (load!) | **1443** | - | 1481 (0.97x) | 1390 (1.04x) | 1321 (1.09x) | 918 (1.57x) | 164 (8.80x) | - | - | - | - | - | - |
| local affine path int64 (load!) | **885** | - | 926 (0.96x) | 846 (1.05x) | 92 (9.62x) | 286 (3.09x) | 156 (5.68x) | - | - | - | - | - | - |
| local affine path float32 (load!) | **885** | kernel=avx2: 1448 | 877 (1.01x) | 918 (0.96x) | - | 123 (7.22x) | 154 (5.73x) | - | - | - | - | - | - |
| local affine path float64 | **716** | kernel=avx2: 845 | 708 (1.01x) | 712 (1.01x) | - | 123 (5.85x) | 154 (4.64x) | - | - | - | - | >10x | - |
| local affine hard int16 | **2022** | - | 2128 (0.95x) | 2050 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard int32 | **1698** | - | 1819 (0.93x) | 1620 (1.05x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **921** | - | 1017 (0.91x) | 923 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **975** | kernel=avx2: 1648 | 971 (1.00x) | 1009 (0.97x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **780** | kernel=avx2: 896 | 760 (1.03x) | 768 (1.02x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **182** | api=functions: 220 | 181 (1.01x) | 184 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **6745** | - | 9546 (0.71x) | 11620 (0.58x) | 2283 (2.95x) | 6827 (0.99x) | 786 (8.58x) | wrong | - | - | - | - | >10x |
| local linear score int32 | **8124** | - | 6462 (1.26x) | 8094 (1.00x) | 2128 (3.82x) | 3487 (2.33x) | 1046 (7.76x) | - | - | - | - | - | >10x |
| local linear score int64 | **4002** | - | 2862 (1.40x) | 4126 (0.97x) | >10x | >10x | 927 (4.32x) | - | - | - | - | - | - |
| local linear score float32 (load!) | **4284** | kernel=avx2: 2877 | 4883 (0.88x) | 5310 (0.81x) | - | >10x | >10x | - | - | - | - | - | - |
| local linear score float64 (load!) | **2759** | fill=striped: 1629 | 2615 (1.06x) | 2886 (0.96x) | - | >10x | 326 (8.46x) | - | - | - | - | 404 (6.82x) | - |
| local linear path int16 (load!) | **3731** | - | 3462 (1.08x) | 4106 (0.91x) | 1611 (2.32x) | 1747 (2.14x) | >10x | wrong | - | - | - | - | >10x |
| local linear path int32 (load!) | **3224** | - | 2725 (1.18x) | 3241 (0.99x) | 1339 (2.41x) | 927 (3.48x) | >10x | - | - | - | - | - | - |
| local linear path int64 (load!) | **2285** | - | 1882 (1.21x) | 2293 (1.00x) | >10x | 284 (8.04x) | >10x | - | - | - | - | - | - |
| local linear path float32 (load!) | **2653** | traceback=scores: 2641 | 2516 (1.05x) | 2750 (0.96x) | - | >10x | >10x | - | - | - | - | - | - |
| local linear path float64 (load!) | **1881** | traceback=hirschberg: 1880 | 1674 (1.12x) | 1917 (0.98x) | - | >10x | >10x | - | - | - | - | >10x | - |
| local linear hard int16 (load!) | **6459** | - | 5396 (1.20x) | 6655 (0.97x) | - | - | - | - | - | - | - | - | - |
| local linear hard int32 (load!) | **5122** | - | 3869 (1.32x) | 5105 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **3022** | - | 2276 (1.33x) | 3032 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **3816** | traceback=scores: 3862 | 3416 (1.12x) | 3771 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **2298** | traceback=pointers: 2342 | 2104 (1.09x) | 2368 (0.97x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **291** | api=functions: 367 | 264 (1.10x) | 298 (0.98x) | - | - | - | - | - | - | - | - | - |

#### solace · dna_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| global affine score int32 (load!) | **7756** | - | 8443 (0.92x) | 7711 (1.01x) | 4240 (1.83x) | 8589 (0.90x) | 902 (8.60x) | - | 1230 (6.31x) | 3172 (2.44x) | - | - | >10x |
| global affine score int64 (load!) | **2583** | - | 2743 (0.94x) | 2665 (0.97x) | >10x | 449 (5.76x) | 655 (3.95x) | - | - | - | - | - | - |
| global affine score float32 (load!) | **2206** | kernel=avx2: 2445 | 2309 (0.96x) | 2274 (0.97x) | - | 299 (7.39x) | 299 (7.37x) | - | - | - | - | - | - |
| global affine score float64 (load!) | **1699** | fill=rowwise: 1682 | 1530 (1.11x) | 1546 (1.10x) | - | 298 (5.70x) | 298 (5.70x) | - | - | - | - | 364 (4.67x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| global affine path int32 (load!) | **2831** | - | 3139 (0.90x) | 2952 (0.96x) | 1961 (1.44x) | 462 (6.13x) | 396 (7.16x) | - | 1163 (2.43x) | 1464 (1.93x) | - | - | >10x |
| global affine path int64 (load!) | **857** | - | 863 (0.99x) | 846 (1.01x) | 112 (7.67x) | 270 (3.17x) | 398 (2.16x) | - | - | - | - | - | - |
| global affine path float32 (load!) | **1636** | kernel=avx2: 1690 | 1568 (1.04x) | 1594 (1.03x) | - | 247 (6.62x) | 339 (4.83x) | - | - | - | - | - | - |
| global affine path float64 (load!) | **1245** | fill=rowwise: 1258 | 1257 (0.99x) | 1276 (0.98x) | - | 250 (4.98x) | 336 (3.71x) | - | - | - | - | >10x | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| global affine hard int32 (load!) | **2841** | - | 3254 (0.87x) | 2966 (0.96x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **874** | - | 830 (1.05x) | 881 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **1645** | kernel=avx2: 1711 | 1583 (1.04x) | 1650 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1259** | fill=rowwise: 1277 | 1261 (1.00x) | 1280 (0.98x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **89** | fill=rowwise: 90 | 113 (0.79x) | 89 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 (load!) | **9668** | - | 9975 (0.97x) | 14134 (0.68x) | 10914 (0.89x) | 17153 (0.56x) | >10x | - | - | - | - | - | >10x |
| global linear score int32 (load!) | **8136** | - | 9248 (0.88x) | 8080 (1.01x) | 6751 (1.21x) | 8523 (0.95x) | 1392 (5.84x) | - | 1280 (6.35x) | 40047 (0.20x) | 27129 (0.30x) | - | >10x |
| global linear score int64 (load!) | **2575** | - | 2694 (0.96x) | 2506 (1.03x) | >10x | 445 (5.78x) | 685 (3.76x) | - | - | - | - | - | - |
| global linear score float32 (load!) | **4633** | fill=striped: 5210 | 4911 (0.94x) | 5199 (0.89x) | - | >10x | >10x | - | - | - | - | - | - |
| global linear score float64 (load!) | **2559** | api=functions: 2568 | 2535 (1.01x) | 2632 (0.97x) | - | 298 (8.60x) | 401 (6.38x) | - | - | - | - | 598 (4.28x) | - |
| global linear path int16 (load!) | **5102** | - | 5360 (0.95x) | 5246 (0.97x) | 3677 (1.39x) | 923 (5.53x) | >10x | - | - | - | - | - | >10x |
| global linear path int32 (load!) | **5791** | - | 5156 (1.12x) | 5612 (1.03x) | 1882 (3.08x) | >10x | >10x | - | 1202 (4.82x) | 24973 (0.23x) | 7231 (0.80x) | - | >10x |
| global linear path int64 (load!) | **1738** | - | 1793 (0.97x) | 1761 (0.99x) | >10x | 233 (7.47x) | 403 (4.31x) | - | - | - | - | - | - |
| global linear path float32 (load!) | **3835** | traceback=scores: 4706 | 3780 (1.01x) | 3980 (0.96x) | - | >10x | >10x | - | - | - | - | - | - |
| global linear path float64 (load!) | **2121** | traceback=scores: 2477 | 2149 (0.99x) | 2190 (0.97x) | - | >10x | 384 (5.52x) | - | - | - | - | >10x | - |
| global linear hard int16 (load!) | **5538** | - | 5315 (1.04x) | 5288 (1.05x) | - | - | - | - | - | - | - | - | - |
| global linear hard int32 (load!) | **5641** | - | 5511 (1.02x) | 6076 (0.93x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **1786** | - | 1783 (1.00x) | 1799 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **4072** | traceback=scores: 4879 | 3992 (1.02x) | 4152 (0.98x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **2141** | traceback=scores: 2573 | 2153 (0.99x) | 2231 (0.96x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **261** | fill=striped: 259 | 253 (1.03x) | 261 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| local affine score int32 (load!) | **6906** | - | 7208 (0.96x) | 7057 (0.98x) | 5940 (1.16x) | 3525 (1.96x) | 783 (8.82x) | - | - | - | - | - | - |
| local affine score int64 (load!) | **2041** | - | 2109 (0.97x) | 1993 (1.02x) | >10x | 255 (8.02x) | 658 (3.10x) | - | - | - | - | - | - |
| local affine score float32 (load!) | **3272** | kernel=avx2: 3590 | 3283 (1.00x) | 3294 (0.99x) | - | >10x | >10x | - | - | - | - | - | - |
| local affine score float64 (load!) | **2023** | fill=rowwise: 2068 | 2049 (0.99x) | 2090 (0.97x) | - | >10x | 233 (8.67x) | - | - | - | - | 234 (8.65x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| local affine path int32 (load!) | **1669** | - | 1664 (1.00x) | 1816 (0.92x) | 1806 (0.92x) | 438 (3.81x) | 192 (8.70x) | - | - | - | - | - | - |
| local affine path int64 (load!) | **751** | - | 773 (0.97x) | 793 (0.95x) | 94 (7.97x) | 299 (2.51x) | 178 (4.21x) | - | - | - | - | - | - |
| local affine path float32 (load!) | **1756** | traceback=scores: 1792 | 1695 (1.04x) | 1773 (0.99x) | - | >10x | 180 (9.78x) | - | - | - | - | - | - |
| local affine path float64 (load!) | **976** | traceback=grid: 968 | 928 (1.05x) | 1002 (0.97x) | - | 130 (7.53x) | 180 (5.41x) | - | - | - | - | >10x | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| local affine hard int32 (load!) | **1767** | - | 1697 (1.04x) | 1743 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **766** | - | 763 (1.00x) | 771 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1738** | traceback=scores: 1861 | 1631 (1.07x) | 1697 (1.02x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **967** | api=functions: 999 | 925 (1.04x) | 1003 (0.96x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **36** | fill=striped: 36 | 57 (0.63x) | 36 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 (load!) | **8609** | - | 8274 (1.04x) | 12299 (0.70x) | 14203 (0.61x) | 7353 (1.17x) | >10x | 5919 (1.45x) sat | - | - | - | - | >10x |
| local linear score int32 (load!) | **6973** | - | 7676 (0.91x) | 7058 (0.99x) | 8310 (0.84x) | 3780 (1.84x) | 1146 (6.08x) | - | - | - | - | - | >10x |
| local linear score int64 (load!) | **2074** | - | 2206 (0.94x) | 2102 (0.99x) | >10x | 256 (8.09x) | 932 (2.23x) | - | - | - | - | - | - |
| local linear score float32 (load!) | **4063** | kernel=avx2: 4461 | 4616 (0.88x) | 4630 (0.88x) | - | >10x | >10x | - | - | - | - | - | - |
| local linear score float64 (load!) | **2425** | fill=striped: 2420 | 2369 (1.02x) | 2422 (1.00x) | - | >10x | 291 (8.34x) | - | - | - | - | 534 (4.54x) | - |
| local linear path int16 (load!) | **5687** | - | 5655 (1.01x) | 5176 (1.10x) | 3287 (1.73x) | 889 (6.40x) | >10x | 805 (7.06x) sat | - | - | - | - | >10x |
| local linear path int32 (load!) | **4038** | - | 4116 (0.98x) | 4249 (0.95x) | 1840 (2.19x) | 435 (9.29x) | >10x | - | - | - | - | - | - |
| local linear path int64 (load!) | **1261** | - | 1377 (0.92x) | 1307 (0.96x) | >10x | 290 (4.35x) | 245 (5.16x) | - | - | - | - | - | - |
| local linear path float32 (load!) | **2911** | traceback=scores: 3293 | 2954 (0.99x) | 2918 (1.00x) | - | >10x | >10x | - | - | - | - | - | - |
| local linear path float64 (load!) | **1514** | traceback=scores: 1844 | 1497 (1.01x) | 1527 (0.99x) | - | >10x | 200 (7.58x) | - | - | - | - | >10x | - |
| local linear hard int16 (load!) | **6092** | - | 6272 (0.97x) | 5899 (1.03x) | - | - | - | - | - | - | - | - | - |
| local linear hard int32 (load!) | **4108** | - | 4236 (0.97x) | 4155 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **1270** | - | 1353 (0.94x) | 1277 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **2950** | traceback=scores: 3463 | 2960 (1.00x) | 2972 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **1507** | traceback=scores: 1856 | 1552 (0.97x) | 1552 (0.97x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **287** | fill=striped: 286 | 274 (1.05x) | 280 (1.02x) | - | - | - | - | - | - | - | - | - |

#### solace · dna_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global affine score int32 (load!) | **3994** | - | 3914 (1.02x) | 4024 (0.99x) | 3585 (1.11x) | 1014 (3.94x) | 919 (4.34x) | 1332 (3.00x) | 4368 (0.91x) | - | - | - |
| global affine score int64 (load!) | **1994** | - | 2046 (0.97x) | 1994 (1.00x) | >10x | 445 (4.48x) | 669 (2.98x) | - | - | - | - | - |
| global affine score float32 (load!) | **3022** | api=functions: 3084 | 3037 (1.00x) | 3040 (0.99x) | - | >10x | >10x | - | - | - | - | - |
| global affine score float64 (load!) | **1992** | fill=striped: 1995 | 2006 (0.99x) | 1971 (1.01x) | - | 296 (6.74x) | 295 (6.74x) | - | - | - | 364 (5.47x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global affine path int32 (load!) | **1686** | - | 1761 (0.96x) | 1687 (1.00x) | 631 (2.67x) | 237 (7.12x) | 298 (5.66x) | 782 (2.16x) | 1896 (0.89x) | - | - | >10x |
| global affine path int64 | **815** | - | 837 (0.97x) | 812 (1.00x) | 109 (7.50x) | 237 (3.44x) | 297 (2.74x) | - | - | - | - | - |
| global affine path float32 | **2741** | kernel=avx2: 2873 | 2708 (1.01x) | 2758 (0.99x) | - | >10x | >10x | - | - | - | - | - |
| global affine path float64 | **1835** | kernel=avx2: 1880 | 1810 (1.01x) | 1794 (1.02x) | - | 223 (8.24x) | 272 (6.74x) | - | - | - | >10x | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global affine hard int32 | **1716** | - | 1745 (0.98x) | 1713 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard int64 | **820** | - | 833 (0.98x) | 830 (0.99x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **2717** | kernel=avx2: 2863 | 2663 (1.02x) | 2738 (0.99x) | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1802** | kernel=avx2: 1885 | 1838 (0.98x) | 1819 (0.99x) | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **146** | fill=rowwise: 148 | 158 (0.92x) | 147 (0.99x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global linear score int32 (load!) | **4000** | - | 3912 (1.02x) | 4049 (0.99x) | 4283 (0.93x) | 1014 (3.95x) | 1377 (2.90x) | 1295 (3.09x) | 58339 (0.07x) | 119832 (0.03x) | - | - |
| global linear score int64 | **2017** | - | 2022 (1.00x) | 2027 (1.00x) | >10x | 445 (4.54x) | 676 (2.99x) | - | - | - | - | - |
| global linear score float32 (load!) | **4700** | fill=striped: 4608 | 4649 (1.01x) | 4544 (1.03x) | - | >10x | >10x | - | - | - | - | - |
| global linear score float64 (load!) | **2434** | api=functions: 2333 | 2407 (1.01x) | 2420 (1.01x) | - | 293 (8.30x) | 386 (6.30x) | - | - | - | 587 (4.15x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global linear path int32 (load!) | **2595** | - | 2587 (1.00x) | 2560 (1.01x) | 606 (4.28x) | >10x | 300 (8.64x) | 755 (3.43x) | 29688 (0.09x) | 24520 (0.11x) | - | >10x |
| global linear path int64 | **1195** | - | 1198 (1.00x) | 1235 (0.97x) | >10x | 207 (5.77x) | 299 (4.00x) | - | - | - | - | - |
| global linear path float32 | **3004** | fill=striped: 3076 | 3013 (1.00x) | 3071 (0.98x) | - | >10x | >10x | - | - | - | - | - |
| global linear path float64 | **1572** | fill=striped: 1566 | 1571 (1.00x) | 1576 (1.00x) | - | 173 (9.07x) | 293 (5.37x) | - | - | - | 162 (9.71x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global linear hard int32 | **2637** | - | 2619 (1.01x) | 2642 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard int64 | **1241** | - | 1216 (1.02x) | 1208 (1.03x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **2965** | traceback=pointers: 3022 | 3037 (0.98x) | 2989 (0.99x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **1600** | traceback=pointers: 1577 | 1619 (0.99x) | 1575 (1.02x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **225** | fill=striped: 230 | 228 (0.99x) | 229 (0.98x) | - | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local affine score int32 | **3670** | - | 3728 (0.98x) | 3740 (0.98x) | 4000 (0.92x) | 463 (7.93x) | 786 (4.67x) | - | - | - | - | - |
| local affine score int64 | **1817** | - | 1835 (0.99x) | 1862 (0.98x) | >10x | 249 (7.30x) | 644 (2.82x) | - | - | - | - | - |
| local affine score float32 | **3858** | fill=striped: 3813 | 3766 (1.02x) | 3772 (1.02x) | - | >10x | >10x | - | - | - | - | - |
| local affine score float64 | **2141** | fill=striped: 2144 | 2197 (0.97x) | 2181 (0.98x) | - | >10x | 229 (9.34x) | - | - | - | 229 (9.35x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local affine path int32 | **707** | - | 712 (0.99x) | 708 (1.00x) | 569 (1.24x) | 268 (2.64x) | 179 (3.95x) | - | - | - | - | - |
| local affine path int64 | **345** | - | 345 (1.00x) | 346 (1.00x) | 90 (3.85x) | 265 (1.30x) | 168 (2.06x) | - | - | - | - | - |
| local affine path float32 | **680** | traceback=grid: 1739 | 670 (1.01x) | 690 (0.99x) | - | 126 (5.39x) | 166 (4.10x) | - | - | - | - | - |
| local affine path float64 | **964** | fill=striped: 974 | 882 (1.09x) | 951 (1.01x) | - | 125 (7.73x) | 166 (5.82x) | - | - | - | >10x | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local affine hard int32 | **707** | - | 719 (0.98x) | 713 (0.99x) | - | - | - | - | - | - | - | - |
| local affine hard int64 | **347** | - | 348 (1.00x) | 347 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **677** | traceback=pointers: 1726 | 685 (0.99x) | 681 (0.99x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **972** | traceback=pointers: 974 | 887 (1.10x) | 943 (1.03x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **90** | fill=striped: 92 | 112 (0.80x) | 93 (0.96x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local linear score int32 | **3714** | - | 3712 (1.00x) | 3664 (1.01x) | 4338 (0.86x) | 450 (8.26x) | 1108 (3.35x) | - | - | - | - | - |
| local linear score int64 | **1861** | - | 1888 (0.99x) | 1877 (0.99x) | >10x | 249 (7.46x) | 911 (2.04x) | - | - | - | - | - |
| local linear score float32 | **4692** | fill=striped: 4754 | 4578 (1.02x) | 4591 (1.02x) | - | >10x | >10x | - | - | - | - | - |
| local linear score float64 | **2323** | api=functions: 2406 | 2332 (1.00x) | 2413 (0.96x) | - | >10x | 281 (8.26x) | - | - | - | 540 (4.30x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local linear path int32 | **2269** | - | 2393 (0.95x) | 2349 (0.97x) | 576 (3.94x) | 264 (8.58x) | >10x | - | - | - | - | - |
| local linear path int64 | **1096** | - | 1080 (1.01x) | 1112 (0.99x) | >10x | 263 (4.17x) | 226 (4.84x) | - | - | - | - | - |
| local linear path float32 | **2477** | fill=striped: 2516 | 2552 (0.97x) | 2594 (0.96x) | - | >10x | >10x | - | - | - | - | - |
| local linear path float64 | **1288** | fill=striped: 1302 | 1324 (0.97x) | 1303 (0.99x) | - | >10x | 185 (6.96x) | - | - | - | >10x | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local linear hard int32 | **2363** | - | 2328 (1.02x) | 2362 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard int64 | **1110** | - | 1123 (0.99x) | 1102 (1.01x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **2519** | traceback=pointers: 2639 | 2507 (1.01x) | 2540 (0.99x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **1306** | fill=striped: 1304 | 1328 (0.98x) | 1301 (1.00x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **224** | fill=striped: 225 | 224 (1.00x) | 223 (1.00x) | - | - | - | - | - | - | - | - |

#### solace · mirna · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **4512** | - | 6551 (0.69x) | 6191 (0.73x) | 960 (4.70x) | 8163 (0.55x) | 657 (6.86x) | - | - | - | - | - | >10x |
| global affine score int32 | **4455** | - | 4490 (0.99x) | 4627 (0.96x) | 1188 (3.75x) | 4638 (0.96x) | 737 (6.05x) | - | 454 (9.82x) | >10x | - | - | >10x |
| global affine score int64 | **3005** | - | 2666 (1.13x) | 3032 (0.99x) | >10x | 485 (6.20x) | 619 (4.85x) | - | - | - | - | - | - |
| global affine score float32 | **3871** | kernel=avx2: 2617 | 3703 (1.05x) | 3847 (1.01x) | - | >10x | >10x | - | - | - | - | - | - |
| global affine score float64 | **2343** | kernel=avx2: 1401 | 2108 (1.11x) | 2308 (1.02x) | - | 343 (6.83x) | 328 (7.13x) | - | - | - | - | >10x | - |
| global affine path int16 | **1006** | - | 936 (1.08x) | 990 (1.02x) | 526 (1.91x) | 2505 (0.40x) | 158 (6.36x) | - | - | - | - | - | >10x |
| global affine path int32 | **881** | - | 897 (0.98x) | 903 (0.98x) | 620 (1.42x) | 1413 (0.62x) | 161 (5.46x) | - | 392 (2.25x) | >10x | - | - | - |
| global affine path int64 | **771** | - | 769 (1.00x) | 782 (0.99x) | 94 (8.18x) | 212 (3.64x) | 161 (4.80x) | - | - | - | - | - | - |
| global affine path float32 | **844** | traceback=grid: 833 | 810 (1.04x) | 858 (0.98x) | - | 133 (6.35x) | 147 (5.73x) | - | - | - | - | - | - |
| global affine path float64 | **706** | traceback=hirschberg: 712 | 662 (1.07x) | 687 (1.03x) | - | 130 (5.42x) | 142 (4.98x) | - | - | - | - | >10x | - |
| global affine hard int16 | **1923** | - | 1898 (1.01x) | 1916 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **1615** | - | 1629 (0.99x) | 1647 (0.98x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **1235** | - | 1248 (0.99x) | 1253 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1372** | traceback=pointers: 1429 | 1442 (0.95x) | 1364 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **1087** | traceback=pointers: 1091 | 1057 (1.03x) | 1092 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **282** | api=functions: 143 | 250 (1.13x) | 286 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **6855** | - | 6863 (1.00x) | 7059 (0.97x) | 820 (8.36x) | 7127 (0.96x) | 856 (8.01x) | - | - | - | - | - | >10x |
| global linear score int32 | **5518** | - | 5225 (1.06x) | 5402 (1.02x) | 1122 (4.92x) | 4378 (1.26x) | 1002 (5.51x) | - | >10x | >10x | 679 (8.13x) | - | >10x |
| global linear score int64 | **3508** | - | 3042 (1.15x) | 3412 (1.03x) | >10x | 491 (7.14x) | 782 (4.49x) | - | - | - | - | - | - |
| global linear score float32 | **4686** | kernel=avx2: 3239 | 4831 (0.97x) | 4900 (0.96x) | - | >10x | >10x | - | - | - | - | - | - |
| global linear score float64 | **3214** | kernel=avx2: 1765 | 2896 (1.11x) | 3303 (0.97x) | - | 345 (9.32x) | 462 (6.95x) | - | - | - | - | >10x | - |
| global linear path int16 | **731** | - | 775 (0.94x) | 767 (0.95x) | 512 (1.43x) | 2457 (0.30x) | 234 (3.13x) | - | - | - | - | - | 100 (7.31x) sat |
| global linear path int32 | **699** | - | 747 (0.94x) | 739 (0.95x) | 590 (1.18x) | 1394 (0.50x) | 240 (2.91x) | - | 401 (1.74x) | 518 (1.35x) | 213 (3.27x) | - | - |
| global linear path int64 | **682** | - | 667 (1.02x) | 701 (0.97x) | 95 (7.18x) | 213 (3.21x) | 237 (2.88x) | - | - | - | - | - | - |
| global linear path float32 (load!) | **646** | traceback=hirschberg: 651 | 706 (0.91x) | 683 (0.95x) | - | 155 (4.17x) | 214 (3.02x) | - | - | - | - | - | - |
| global linear path float64 (load!) | **615** | traceback=hirschberg: 615 | 645 (0.95x) | 633 (0.97x) | - | 157 (3.93x) | 217 (2.84x) | - | - | - | - | 86 (7.14x) | - |
| global linear hard int16 | **1756** | - | 1822 (0.96x) | 1744 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **1600** | - | 1632 (0.98x) | 1593 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **1407** | - | 1356 (1.04x) | 1388 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **1460** | traceback=pointers: 1472 | 1540 (0.95x) | 1391 (1.05x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1244** | traceback=scores: 1288 | 1296 (0.96x) | 1293 (0.96x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **454** | api=functions: 234 | 409 (1.11x) | 468 (0.97x) | - | - | - | - | - | - | - | - | - |
| local affine score int16 (load!) | **5631** | - | 5712 (0.99x) | 5999 (0.94x) | 1931 (2.92x) | 4404 (1.28x) | 569 (9.90x) | 1614 (3.49x) sat | - | - | - | - | >10x |
| local affine score int32 (load!) | **4405** | - | 4201 (1.05x) | 4458 (0.99x) | 1283 (3.43x) | 2335 (1.89x) | 647 (6.81x) | - | - | - | - | - | >10x |
| local affine score int64 (load!) | **2627** | - | 2613 (1.01x) | 2702 (0.97x) | >10x | >10x | 591 (4.44x) | - | - | - | - | - | - |
| local affine score float32 (load!) | **3719** | kernel=avx2: 2459 | 3304 (1.13x) | 3757 (0.99x) | - | >10x | >10x | - | - | - | - | - | - |
| local affine score float64 (load!) | **2116** | kernel=avx2: 1377 | 1948 (1.09x) | 2097 (1.01x) | - | >10x | 258 (8.21x) | - | - | - | - | >10x | - |
| local affine path int16 (load!) | **2794** | - | 2622 (1.07x) | 2698 (1.04x) | 608 (4.59x) | 2464 (1.13x) | >10x | 967 (2.89x) sat | - | - | - | - | >10x |
| local affine path int32 | **2199** | - | 2055 (1.07x) | 2165 (1.02x) | 760 (2.89x) | 1304 (1.69x) | >10x | - | - | - | - | - | >10x |
| local affine path int64 | **1416** | - | 1323 (1.07x) | 1341 (1.06x) | >10x | 252 (5.62x) | >10x | - | - | - | - | - | - |
| local affine path float32 | **1814** | traceback=grid: 1843 | 1758 (1.03x) | 1749 (1.04x) | - | >10x | >10x | - | - | - | - | - | - |
| local affine path float64 | **1123** | traceback=grid: 1125 | 1098 (1.02x) | 1118 (1.00x) | - | >10x | 137 (8.20x) | - | - | - | - | >10x | - |
| local affine hard int16 | **2921** | - | 3007 (0.97x) | 2980 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine hard int32 | **2281** | - | 2236 (1.02x) | 2465 (0.93x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **1466** | - | 1386 (1.06x) | 1459 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1882** | traceback=grid: 1992 | 1865 (1.01x) | 1944 (0.97x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **1158** | traceback=scores: 1212 | 1165 (0.99x) | 1216 (0.95x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **298** | api=functions: 144 | 272 (1.10x) | 297 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 (load!) | **6780** | - | 6549 (1.04x) | 6623 (1.02x) | 784 (8.65x) | 4674 (1.45x) | 702 (9.66x) | wrong | - | - | - | - | >10x |
| local linear score int32 (load!) | **5308** | - | 4659 (1.14x) | 5560 (0.95x) | 932 (5.70x) | 2411 (2.20x) | 850 (6.25x) | - | - | - | - | - | >10x |
| local linear score int64 (load!) | **3404** | - | 2370 (1.44x) | 3405 (1.00x) | >10x | >10x | 768 (4.43x) | - | - | - | - | - | - |
| local linear score float32 (load!) | **4043** | kernel=avx2: 2458 | 3716 (1.09x) | 3978 (1.02x) | - | >10x | >10x | - | - | - | - | - | - |
| local linear score float64 (load!) | **2496** | kernel=avx2: 1300 | 2164 (1.15x) | 2511 (0.99x) | - | >10x | 350 (7.14x) | - | - | - | - | >10x | - |
| local linear path int16 (load!) | **1870** | - | 1777 (1.05x) | 1994 (0.94x) | 583 (3.21x) | 2367 (0.79x) | >10x | wrong | - | - | - | - | >10x |
| local linear path int32 | **1771** | - | 1641 (1.08x) | 1920 (0.92x) | 673 (2.63x) | 1304 (1.36x) | 181 (9.76x) | - | - | - | - | - | >10x |
| local linear path int64 | **1422** | - | 1280 (1.11x) | 1468 (0.97x) | >10x | 242 (5.87x) | 180 (7.92x) | - | - | - | - | - | - |
| local linear path float32 | **1568** | traceback=pointers: 1547 | 1523 (1.03x) | 1702 (0.92x) | - | >10x | >10x | - | - | - | - | - | - |
| local linear path float64 | **1218** | traceback=scores: 1254 | 1148 (1.06x) | 1293 (0.94x) | - | >10x | 151 (8.08x) | - | - | - | - | >10x | - |
| local linear hard int16 | **3074** | - | 2631 (1.17x) | 3077 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **2661** | - | 2217 (1.20x) | 2718 (0.98x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **1977** | - | 1545 (1.28x) | 2004 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **2276** | traceback=hirschberg: 2328 | 2083 (1.09x) | 2429 (0.94x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **1658** | traceback=hirschberg: 1700 | 1469 (1.13x) | 1651 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **440** | api=functions: 215 | 394 (1.12x) | 437 (1.01x) | - | - | - | - | - | - | - | - | - |

#### solace · protein_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **2718** | - | 3081 (0.88x) | 2897 (0.94x) | 1818 (1.50x) | 1559 (1.74x) | 751 (3.62x) | - | - | - | >10x |
| global affine score int32 | **2830** | - | 2867 (0.99x) | 2831 (1.00x) | 1742 (1.62x) | 1597 (1.77x) | 817 (3.46x) | - | 737 (3.84x) | - | >10x |
| global affine score int64 | **2167** | - | 2593 (0.84x) | 2226 (0.97x) | >10x | 1525 (1.42x) | 658 (3.29x) | - | - | - | - |
| global affine score float32 | **866** | kernel=sse2: 1003 | 851 (1.02x) | 842 (1.03x) | - | 315 (2.75x) | 326 (2.65x) | - | - | - | - |
| global affine score float64 | **885** | kernel=avx2: 969 | 921 (0.96x) | 907 (0.98x) | - | 315 (2.81x) | 329 (2.69x) | - | - | 338 (2.62x) | - |
| global affine path int16 | **1638** | - | 1626 (1.01x) | 1685 (0.97x) | 1359 (1.21x) | 1049 (1.56x) | 262 (6.26x) | - | - | - | 183 (8.94x) sat |
| global affine path int32 | **1450** | - | 1458 (0.99x) | 1471 (0.99x) | 1114 (1.30x) | 965 (1.50x) | 284 (5.11x) | - | 667 (2.17x) | - | - |
| global affine path int64 | **995** | - | 1050 (0.95x) | 989 (1.01x) | 107 (9.27x) | 646 (1.54x) | 277 (3.59x) | - | - | - | - |
| global affine path float32 | **654** | kernel=avx2: 1025 | 627 (1.04x) | 643 (1.02x) | - | 189 (3.47x) | 204 (3.20x) | - | - | - | - |
| global affine path float64 | **606** | kernel=avx2: 799 | 609 (1.00x) | 601 (1.01x) | - | 187 (3.24x) | 207 (2.92x) | - | - | 93 (6.49x) | - |
| global affine hard int16 | **1746** | - | 1823 (0.96x) | 1787 (0.98x) | - | - | - | - | - | - | - |
| global affine hard int32 | **1526** | - | 1600 (0.95x) | 1532 (1.00x) | - | - | - | - | - | - | - |
| global affine hard int64 | **1011** | - | 1118 (0.90x) | 1002 (1.01x) | - | - | - | - | - | - | - |
| global affine hard float32 | **671** | kernel=avx2: 1099 | 658 (1.02x) | 679 (0.99x) | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **615** | kernel=avx2: 811 | 616 (1.00x) | 617 (1.00x) | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **170** | api=functions: 211 | 170 (1.00x) | 166 (1.02x) | - | - | - | - | - | - | - |
| global linear score int16 | **2768** | - | 3042 (0.91x) | 2862 (0.97x) | 1761 (1.57x) | 1565 (1.77x) | 924 (3.00x) | - | - | - | 331 (8.37x) sat |
| global linear score int32 | **2841** | - | 2851 (1.00x) | 2724 (1.04x) | 1821 (1.56x) | 1506 (1.89x) | 1359 (2.09x) | - | 752 (3.78x) | - | - |
| global linear score int64 | **2151** | - | 2448 (0.88x) | 2180 (0.99x) | >10x | 1568 (1.37x) | 763 (2.82x) | - | - | - | - |
| global linear score float32 | **1265** | kernel=avx2: 1427 | 1282 (0.99x) | 1274 (0.99x) | - | 312 (4.05x) | 454 (2.78x) | - | - | - | - |
| global linear score float64 | **1234** | kernel=avx2: 1249 | 1230 (1.00x) | 1192 (1.04x) | - | 315 (3.91x) | 466 (2.65x) | - | - | 586 (2.11x) | - |
| global linear path int16 | **1783** | - | 1815 (0.98x) | 1780 (1.00x) | 1316 (1.36x) | 1036 (1.72x) | 444 (4.01x) | - | - | - | 232 (7.70x) sat |
| global linear path int32 | **1894** | - | 1853 (1.02x) | 1886 (1.00x) | 1127 (1.68x) | 941 (2.01x) | 444 (4.26x) | - | 683 (2.77x) | - | - |
| global linear path int64 | **1458** | - | 1526 (0.96x) | 1485 (0.98x) | >10x | 651 (2.24x) | 451 (3.23x) | - | - | - | - |
| global linear path float32 | **1295** | traceback=hirschberg: 1298 | 1106 (1.17x) | 1338 (0.97x) | - | 211 (6.13x) | 389 (3.33x) | - | - | - | - |
| global linear path float64 | **1158** | traceback=scores: 1159 | 997 (1.16x) | 1177 (0.98x) | - | 207 (5.59x) | 379 (3.06x) | - | - | 209 (5.53x) | - |
| global linear hard int16 | **2011** | - | 2182 (0.92x) | 2083 (0.97x) | - | - | - | - | - | - | - |
| global linear hard int32 | **2280** | - | 2209 (1.03x) | 2256 (1.01x) | - | - | - | - | - | - | - |
| global linear hard int64 | **1627** | - | 1676 (0.97x) | 1659 (0.98x) | - | - | - | - | - | - | - |
| global linear hard float32 | **1456** | traceback=hirschberg: 1468 | 1097 (1.33x) | 1495 (0.97x) | - | - | - | - | - | - | - |
| global linear hard float64 | **1236** | traceback=pointers: 1288 | 1048 (1.18x) | 1341 (0.92x) | - | - | - | - | - | - | - |
| global linear soft float64 | **231** | api=functions: 279 | 204 (1.13x) | 231 (1.00x) | - | - | - | - | - | - | - |
| local affine score int16 | **2816** | - | 2921 (0.96x) | 2786 (1.01x) | 1624 (1.73x) | 1390 (2.03x) | 614 (4.58x) | 898 (3.14x) sat | - | - | >10x |
| local affine score int32 | **2559** | - | 2625 (0.97x) | 2557 (1.00x) | 1482 (1.73x) | 1371 (1.87x) | 722 (3.55x) | - | - | - | >10x |
| local affine score int64 | **1963** | - | 2268 (0.87x) | 1945 (1.01x) | >10x | 1230 (1.60x) | 633 (3.10x) | - | - | - | - |
| local affine score float32 | **1003** | kernel=avx2: 1144 | 1013 (0.99x) | 1011 (0.99x) | - | 169 (5.93x) | 238 (4.21x) | - | - | - | - |
| local affine score float64 (load!) | **987** | kernel=avx2: 1060 | 1003 (0.98x) | 1007 (0.98x) | - | 171 (5.76x) | 251 (3.93x) | - | - | 230 (4.28x) | - |
| local affine path int16 (load!) | **1357** | - | 1361 (1.00x) | 1299 (1.04x) | 1225 (1.11x) | 1050 (1.29x) | 185 (7.32x) | 342 (3.96x) sat | - | - | 145 (9.34x) sat |
| local affine path int32 | **1185** | - | 1251 (0.95x) | 1158 (1.02x) | 1023 (1.16x) | 913 (1.30x) | 192 (6.18x) | - | - | - | - |
| local affine path int64 | **774** | - | 815 (0.95x) | 789 (0.98x) | 89 (8.67x) | 557 (1.39x) | 194 (4.00x) | - | - | - | - |
| local affine path float32 | **626** | kernel=avx2: 1049 | 615 (1.02x) | 624 (1.00x) | - | 141 (4.44x) | 169 (3.70x) | - | - | - | - |
| local affine path float64 (load!) | **565** | kernel=avx2: 730 | 541 (1.05x) | 562 (1.01x) | - | 142 (3.99x) | 168 (3.38x) | - | - | >10x | - |
| local affine hard int16 (load!) | **1362** | - | 1447 (0.94x) | 1403 (0.97x) | - | - | - | - | - | - | - |
| local affine hard int32 (load!) | **1232** | - | 1360 (0.91x) | 1207 (1.02x) | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **795** | - | 838 (0.95x) | 791 (1.01x) | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **625** | kernel=avx2: 1029 | 621 (1.01x) | 605 (1.03x) | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **543** | kernel=avx2: 757 | 500 (1.09x) | 547 (0.99x) | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **155** | api=functions: 203 | 156 (1.00x) | 155 (1.00x) | - | - | - | - | - | - | - |
| local linear score int16 (load!) | **2781** | - | 3014 (0.92x) | 2831 (0.98x) | 1674 (1.66x) | 1411 (1.97x) | 843 (3.30x) | wrong | - | - | >10x |
| local linear score int32 (load!) | **2483** | - | 2561 (0.97x) | 2518 (0.99x) | 1372 (1.81x) | 1393 (1.78x) | 1140 (2.18x) | - | - | - | >10x |
| local linear score int64 (load!) | **2039** | - | 2330 (0.88x) | 1965 (1.04x) | >10x | 1196 (1.71x) | 1098 (1.86x) | - | - | - | - |
| local linear score float32 (load!) | **1512** | kernel=avx2: 1682 | 1534 (0.99x) | 1518 (1.00x) | - | 172 (8.80x) | 325 (4.65x) | - | - | - | - |
| local linear score float64 (load!) | **1311** | kernel=avx2: 1339 | 1334 (0.98x) | 1311 (1.00x) | - | 168 (7.80x) | 326 (4.01x) | - | - | 317 (4.13x) | - |
| local linear path int16 (load!) | **1493** | - | 1178 (1.27x) | 1528 (0.98x) | 1218 (1.23x) | 1047 (1.43x) | 217 (6.87x) | wrong | - | - | 179 (8.36x) sat |
| local linear path int32 (load!) | **1407** | - | 1094 (1.29x) | 1456 (0.97x) | 1014 (1.39x) | 920 (1.53x) | 219 (6.43x) | - | - | - | - |
| local linear path int64 (load!) | **1168** | - | 863 (1.35x) | 1198 (0.98x) | >10x | 570 (2.05x) | 216 (5.41x) | - | - | - | - |
| local linear path float32 (load!) | **1197** | traceback=hirschberg: 1198 | 1001 (1.20x) | 1224 (0.98x) | - | 145 (8.23x) | 178 (6.74x) | - | - | - | - |
| local linear path float64 | **1043** | traceback=pointers: 1043 | 871 (1.20x) | 1081 (0.97x) | - | 144 (7.25x) | 179 (5.84x) | - | - | >10x | - |
| local linear hard int16 | **1667** | - | 1229 (1.36x) | 1707 (0.98x) | - | - | - | - | - | - | - |
| local linear hard int32 | **1627** | - | 1204 (1.35x) | 1601 (1.02x) | - | - | - | - | - | - | - |
| local linear hard int64 | **1287** | - | 959 (1.34x) | 1277 (1.01x) | - | - | - | - | - | - | - |
| local linear hard float32 | **1399** | traceback=pointers: 1369 | 1096 (1.28x) | 1393 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **1109** | traceback=scores: 1138 | 924 (1.20x) | 1114 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **248** | api=functions: 329 | 229 (1.09x) | 253 (0.98x) | - | - | - | - | - | - | - |

#### solace · protein_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global affine score int32 (load!) | **7037** | - | 4437 (1.59x) | 7049 (1.00x) | 3821 (1.84x) | 1492 (4.72x) | 864 (8.14x) | 1216 (5.79x) | - | >10x |
| global affine score int64 (load!) | **2562** | - | 2752 (0.93x) | 2555 (1.00x) | >10x | 1599 (1.60x) | 647 (3.96x) | - | - | - |
| global affine score float32 (load!) | **1459** | kernel=avx2: 1535 | 1422 (1.03x) | 1462 (1.00x) | - | 290 (5.03x) | 297 (4.90x) | - | - | - |
| global affine score float64 (load!) | **1214** | fill=striped: 1193 | 1224 (0.99x) | 1212 (1.00x) | - | 290 (4.18x) | 297 (4.09x) | - | 359 (3.39x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global affine path int32 (load!) | **2300** | - | 2542 (0.90x) | 2262 (1.02x) | 1804 (1.27x) | 366 (6.28x) | 323 (7.12x) | 1143 (2.01x) | - | - |
| global affine path int64 (load!) | **840** | - | 822 (1.02x) | 845 (0.99x) | 111 (7.58x) | 249 (3.37x) | 306 (2.74x) | - | - | - |
| global affine path float32 (load!) | **1142** | traceback=hirschberg_pmax: 1210 | 1115 (1.02x) | 1148 (1.00x) | - | 203 (5.64x) | 224 (5.11x) | - | - | - |
| global affine path float64 (load!) | **964** | kernel=avx2: 960 | 929 (1.04x) | 964 (1.00x) | - | 204 (4.73x) | 220 (4.38x) | - | >10x | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global affine hard int32 (load!) | **2288** | - | 2511 (0.91x) | 2394 (0.96x) | - | - | - | - | - | - |
| global affine hard int64 (load!) | **856** | - | 825 (1.04x) | 870 (0.98x) | - | - | - | - | - | - |
| global affine hard float32 (load!) | **1098** | traceback=hirschberg_pmax: 1231 | 1094 (1.00x) | 1154 (0.95x) | - | - | - | - | - | - |
| global affine hard float64 (load!) | **950** | kernel=avx2: 992 | 990 (0.96x) | 980 (0.97x) | - | - | - | - | - | - |
| global affine soft float64 (load!) | **97** | fill=rowwise: 97 | 120 (0.81x) | 97 (1.00x) | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global linear score int32 (load!) | **7047** | - | 7823 (0.90x) | 7495 (0.94x) | 6144 (1.15x) | 1450 (4.86x) | 1522 (4.63x) | 1263 (5.58x) | - | >10x |
| global linear score int64 (load!) | **2564** | - | 2711 (0.95x) | 2636 (0.97x) | >10x | 1578 (1.62x) | 688 (3.73x) | - | - | - |
| global linear score float32 (load!) | **4283** | kernel=avx2: 4477 | 4267 (1.00x) | 4318 (0.99x) | - | >10x | >10x | - | - | - |
| global linear score float64 (load!) | **2394** | api=functions: 2375 | 2375 (1.01x) | 2391 (1.00x) | - | 292 (8.20x) | 402 (5.96x) | - | 583 (4.11x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global linear path int32 (load!) | **5032** | - | 2994 (1.68x) | 4640 (1.08x) | 1657 (3.04x) | >10x | 545 (9.24x) | 1155 (4.36x) | - | >10x |
| global linear path int64 (load!) | **1685** | - | 1713 (0.98x) | 1661 (1.01x) | >10x | 252 (6.68x) | 529 (3.18x) | - | - | - |
| global linear path float32 (load!) | **3144** | traceback=scores: 3497 | 3302 (0.95x) | 3413 (0.92x) | - | >10x | 410 (7.67x) | - | - | - |
| global linear path float64 (load!) | **1726** | traceback=scores: 1932 | 1708 (1.01x) | 1689 (1.02x) | - | 212 (8.14x) | 458 (3.77x) | - | 250 (6.90x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global linear hard int32 (load!) | **5160** | - | 5031 (1.03x) | 5275 (0.98x) | - | - | - | - | - | - |
| global linear hard int64 (load!) | **1677** | - | 1744 (0.96x) | 1666 (1.01x) | - | - | - | - | - | - |
| global linear hard float32 (load!) | **3241** | traceback=scores: 3714 | 3297 (0.98x) | 3553 (0.91x) | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1978** | traceback=scores: 2306 | 1950 (1.01x) | 2013 (0.98x) | - | - | - | - | - | - |
| global linear soft float64 (load!) | **225** | fill=striped: 217 | 218 (1.03x) | 222 (1.01x) | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local affine score int32 (load!) | **5945** | - | 6982 (0.85x) | 6473 (0.92x) | 3197 (1.86x) | 1376 (4.32x) | 759 (7.84x) | - | - | - |
| local affine score int64 (load!) | **1951** | - | 2124 (0.92x) | 2042 (0.96x) | >10x | 1201 (1.62x) | 641 (3.04x) | - | - | - |
| local affine score float32 (load!) | **1700** | kernel=avx2: 1915 | 1710 (0.99x) | 1718 (0.99x) | - | 172 (9.89x) | 234 (7.26x) | - | - | - |
| local affine score float64 (load!) | **1363** | fill=rowwise: 1349 | 1379 (0.99x) | 1344 (1.01x) | - | 170 (8.00x) | 233 (5.85x) | - | 271 (5.03x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local affine path int32 (load!) | **1698** | - | 1712 (0.99x) | 1745 (0.97x) | 1708 (0.99x) | 366 (4.64x) | 227 (7.47x) | - | - | - |
| local affine path int64 (load!) | **762** | - | 753 (1.01x) | 775 (0.98x) | 94 (8.12x) | 297 (2.57x) | 233 (3.27x) | - | - | - |
| local affine path float32 (load!) | **1150** | traceback=scores: 1221 | 1079 (1.07x) | 1142 (1.01x) | - | 160 (7.20x) | 189 (6.09x) | - | - | - |
| local affine path float64 (load!) | **766** | traceback=pointers: 807 | 737 (1.04x) | 769 (1.00x) | - | 159 (4.81x) | 185 (4.14x) | - | >10x | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local affine hard int32 (load!) | **1693** | - | 1714 (0.99x) | 1798 (0.94x) | - | - | - | - | - | - |
| local affine hard int64 | **758** | - | 768 (0.99x) | 777 (0.98x) | - | - | - | - | - | - |
| local affine hard float32 | **1157** | traceback=scores: 1228 | 1089 (1.06x) | 1212 (0.95x) | - | - | - | - | - | - |
| local affine hard float64 | **775** | fill=striped: 787 | 736 (1.05x) | 785 (0.99x) | - | - | - | - | - | - |
| local affine soft float64 | **89** | fill=striped: 90 | 114 (0.79x) | 88 (1.01x) | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local linear score int32 | **6096** | - | 6713 (0.91x) | 6568 (0.93x) | 6828 (0.89x) | 1374 (4.44x) | 1294 (4.71x) | - | - | - |
| local linear score int64 | **1987** | - | 2162 (0.92x) | 2034 (0.98x) | >10x | 1209 (1.64x) | 1161 (1.71x) | - | - | - |
| local linear score float32 | **4092** | kernel=avx2: 4082 | 4162 (0.98x) | 4095 (1.00x) | - | >10x | >10x | - | - | - |
| local linear score float64 | **2285** | fill=striped: 2282 | 2247 (1.02x) | 2302 (0.99x) | - | >10x | 291 (7.84x) | - | 463 (4.93x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local linear path int32 | **3713** | - | 3826 (0.97x) | 3956 (0.94x) | 1699 (2.19x) | >10x | >10x | - | - | - |
| local linear path int64 | **1237** | - | 1342 (0.92x) | 1260 (0.98x) | >10x | 300 (4.12x) | 256 (4.83x) | - | - | - |
| local linear path float32 | **2687** | traceback=scores: 2863 | 2645 (1.02x) | 2679 (1.00x) | - | >10x | >10x | - | - | - |
| local linear path float64 (load!) | **1472** | traceback=scores: 1672 | 1465 (1.01x) | 1487 (0.99x) | - | >10x | 206 (7.14x) | - | >10x | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local linear hard int32 (load!) | **3826** | - | 3979 (0.96x) | 3999 (0.96x) | - | - | - | - | - | - |
| local linear hard int64 (load!) | **1228** | - | 1339 (0.92x) | 1296 (0.95x) | - | - | - | - | - | - |
| local linear hard float32 (load!) | **2682** | traceback=scores: 3150 | 2803 (0.96x) | 2703 (0.99x) | - | - | - | - | - | - |
| local linear hard float64 (load!) | **1458** | traceback=scores: 1813 | 1454 (1.00x) | 1513 (0.96x) | - | - | - | - | - | - |
| local linear soft float64 (load!) | **207** | fill=striped: 206 | 197 (1.05x) | 207 (1.00x) | - | - | - | - | - | - |

#### solace · protein_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global affine score int32 (load!) | **3803** | - | 3890 (0.98x) | 3942 (0.96x) | 2870 (1.32x) | 1048 (3.63x) | 906 (4.20x) | 1305 (2.92x) | - | - |
| global affine score int64 (load!) | **1970** | - | 1938 (1.02x) | 1942 (1.01x) | >10x | 443 (4.45x) | 663 (2.97x) | - | - | - |
| global affine score float32 (load!) | **1589** | api=functions: 1599 | 1575 (1.01x) | 1601 (0.99x) | - | 289 (5.49x) | 294 (5.41x) | - | - | - |
| global affine score float64 (load!) | **1254** | fill=striped: 1257 | 1275 (0.98x) | 1248 (1.00x) | - | 289 (4.33x) | 297 (4.23x) | - | 362 (3.46x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global affine path int32 (load!) | **1682** | - | 1688 (1.00x) | 1666 (1.01x) | 641 (2.63x) | 231 (7.29x) | 296 (5.67x) | 794 (2.12x) | - | >10x |
| global affine path int64 | **811** | - | 816 (0.99x) | 813 (1.00x) | 109 (7.46x) | 232 (3.49x) | 278 (2.91x) | - | - | - |
| global affine path float32 (load!) | **1482** | kernel=avx2: 1520 | 1491 (0.99x) | 1490 (0.99x) | - | 200 (7.40x) | 210 (7.05x) | - | - | - |
| global affine path float64 (load!) | **1177** | kernel=avx2: 1226 | 1097 (1.07x) | 1144 (1.03x) | - | 198 (5.94x) | 208 (5.66x) | - | >10x | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global affine hard int32 (load!) | **1672** | - | 1637 (1.02x) | 1661 (1.01x) | - | - | - | - | - | - |
| global affine hard int64 (load!) | **793** | - | 766 (1.04x) | 762 (1.04x) | - | - | - | - | - | - |
| global affine hard float32 (load!) | **1462** | kernel=avx2: 1549 | 1470 (0.99x) | 1408 (1.04x) | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1131** | kernel=avx2: 1260 | 1171 (0.97x) | 1167 (0.97x) | - | - | - | - | - | - |
| global affine soft float64 (load!) | **138** | fill=striped: 137 | 148 (0.93x) | 139 (0.99x) | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global linear score int32 | **3855** | - | 4050 (0.95x) | 3993 (0.97x) | 4247 (0.91x) | 1025 (3.76x) | 1502 (2.57x) | 1304 (2.96x) | - | 800 (4.82x) sat |
| global linear score int64 | **1899** | - | 1923 (0.99x) | 1935 (0.98x) | >10x | 441 (4.30x) | 662 (2.87x) | - | - | - |
| global linear score float32 | **4767** | fill=striped: 4672 | 4647 (1.03x) | 4724 (1.01x) | - | >10x | >10x | - | - | - |
| global linear score float64 | **2363** | fill=striped: 2356 | 2337 (1.01x) | 2298 (1.03x) | - | 288 (8.19x) | 395 (5.98x) | - | 592 (3.99x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global linear path int32 | **2658** | - | 2689 (0.99x) | 2753 (0.97x) | 588 (4.52x) | >10x | 437 (6.08x) | 786 (3.38x) | - | >10x |
| global linear path int64 | **1277** | - | 1266 (1.01x) | 1260 (1.01x) | >10x | 245 (5.21x) | 437 (2.92x) | - | - | - |
| global linear path float32 | **2989** | traceback=pointers: 3078 | 3085 (0.97x) | 3047 (0.98x) | - | >10x | 373 (8.01x) | - | - | - |
| global linear path float64 | **1589** | traceback=pointers: 1571 | 1562 (1.02x) | 1580 (1.01x) | - | 213 (7.46x) | 374 (4.25x) | - | 218 (7.30x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global linear hard int32 | **2695** | - | 2656 (1.01x) | 2588 (1.04x) | - | - | - | - | - | - |
| global linear hard int64 | **1259** | - | 1269 (0.99x) | 1244 (1.01x) | - | - | - | - | - | - |
| global linear hard float32 | **2963** | traceback=pointers: 3075 | 3071 (0.96x) | 3073 (0.96x) | - | - | - | - | - | - |
| global linear hard float64 | **1572** | fill=striped: 1565 | 1605 (0.98x) | 1590 (0.99x) | - | - | - | - | - | - |
| global linear soft float64 | **278** | fill=striped: 282 | 274 (1.01x) | 281 (0.99x) | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local affine score int32 | **3727** | - | 3760 (0.99x) | 3758 (0.99x) | 2532 (1.47x) | >10x | 769 (4.85x) | - | - | - |
| local affine score int64 | **1873** | - | 1859 (1.01x) | 1867 (1.00x) | >10x | 318 (5.88x) | 648 (2.89x) | - | - | - |
| local affine score float32 | **1919** | fill=striped: 1919 | 1965 (0.98x) | 1926 (1.00x) | - | >10x | 232 (8.28x) | - | - | - |
| local affine score float64 | **1433** | fill=striped: 1437 | 1459 (0.98x) | 1410 (1.02x) | - | 170 (8.43x) | 232 (6.18x) | - | 315 (4.55x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local affine path int32 | **704** | - | 709 (0.99x) | 714 (0.99x) | 572 (1.23x) | 280 (2.51x) | 215 (3.27x) | - | - | - |
| local affine path int64 | **346** | - | 348 (0.99x) | 352 (0.98x) | 90 (3.83x) | 269 (1.28x) | 220 (1.57x) | - | - | - |
| local affine path float32 | **689** | traceback=grid: 1089 | 681 (1.01x) | 644 (1.07x) | - | 156 (4.42x) | 177 (3.90x) | - | - | - |
| local affine path float64 | **753** | fill=striped: 758 | 683 (1.10x) | 759 (0.99x) | - | 153 (4.91x) | 177 (4.27x) | - | >10x | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local affine hard int32 | **723** | - | 732 (0.99x) | 723 (1.00x) | - | - | - | - | - | - |
| local affine hard int64 | **347** | - | 354 (0.98x) | 351 (0.99x) | - | - | - | - | - | - |
| local affine hard float32 | **698** | traceback=grid: 1095 | 699 (1.00x) | 693 (1.01x) | - | - | - | - | - | - |
| local affine hard float64 | **762** | fill=striped: 767 | 695 (1.10x) | 739 (1.03x) | - | - | - | - | - | - |
| local affine soft float64 | **118** | fill=striped: 118 | 130 (0.91x) | 118 (1.00x) | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local linear score int32 | **3697** | - | 3740 (0.99x) | 3690 (1.00x) | 4341 (0.85x) | >10x | 1264 (2.92x) | - | - | 539 (6.85x) sat |
| local linear score int64 | **1824** | - | 1877 (0.97x) | 1819 (1.00x) | >10x | 318 (5.74x) | 1158 (1.58x) | - | - | - |
| local linear score float32 | **4726** | api=functions: 4688 | 4626 (1.02x) | 4504 (1.05x) | - | >10x | >10x | - | - | - |
| local linear score float64 | **2383** | api=functions: 2386 | 2360 (1.01x) | 2333 (1.02x) | - | >10x | 288 (8.26x) | - | 545 (4.37x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local linear path int32 | **2365** | - | 2466 (0.96x) | 2459 (0.96x) | 606 (3.90x) | 282 (8.39x) | >10x | - | - | 396 (5.98x) sat |
| local linear path int64 | **1123** | - | 1144 (0.98x) | 1124 (1.00x) | >10x | 277 (4.06x) | 238 (4.72x) | - | - | - |
| local linear path float32 (load!) | **2588** | fill=striped: 2636 | 2608 (0.99x) | 2547 (1.02x) | - | >10x | >10x | - | - | - |
| local linear path float64 | **1302** | fill=striped: 1315 | 1336 (0.97x) | 1302 (1.00x) | - | 135 (9.68x) | 195 (6.68x) | - | >10x | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local linear hard int32 | **2376** | - | 2361 (1.01x) | 2370 (1.00x) | - | - | - | - | - | - |
| local linear hard int64 | **1098** | - | 1113 (0.99x) | 1103 (1.00x) | - | - | - | - | - | - |
| local linear hard float32 | **2478** | fill=striped: 2523 | 2639 (0.94x) | 2505 (0.99x) | - | - | - | - | - | - |
| local linear hard float64 | **1281** | fill=striped: 1293 | 1329 (0.96x) | 1329 (0.96x) | - | - | - | - | - | - |
| local linear soft float64 | **262** | fill=striped: 262 | 264 (0.99x) | 261 (1.00x) | - | - | - | - | - | - |

#### solace · protein_L300 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| global affine score int32 (load!) | **5220** | - | 5590 (0.93x) | 5405 (0.97x) | 2867 (1.82x) | 1602 (3.26x) | 869 (6.00x) | 1059 (4.93x) | - | >10x |
| global affine score int64 (load!) | **2994** | - | 3922 (0.76x) | 3207 (0.93x) | >10x | 1613 (1.86x) | 643 (4.65x) | - | - | - |
| global affine score float32 (load!) | **1207** | kernel=avx2: 1259 | 1216 (0.99x) | 1203 (1.00x) | - | 297 (4.07x) | 305 (3.95x) | - | - | - |
| global affine score float64 (load!) | **1079** | kernel=avx2: 1201 | 1133 (0.95x) | 1076 (1.00x) | - | 294 (3.67x) | 303 (3.56x) | - | 357 (3.02x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| global affine path int32 (load!) | **2393** | - | 2588 (0.92x) | 2510 (0.95x) | 1628 (1.47x) | 791 (3.03x) | 299 (8.01x) | 992 (2.41x) | - | - |
| global affine path int64 (load!) | **1239** | - | 1451 (0.85x) | 1256 (0.99x) | >10x | 483 (2.57x) | 288 (4.30x) | - | - | - |
| global affine path float32 (load!) | **917** | traceback=scores: 981 | 884 (1.04x) | 946 (0.97x) | - | 194 (4.72x) | 218 (4.20x) | - | - | - |
| global affine path float64 (load!) | **728** | traceback=scores: 754 | 742 (0.98x) | 737 (0.99x) | - | 194 (3.76x) | 216 (3.36x) | - | 99 (7.33x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| global affine hard int32 (load!) | **2483** | - | 2806 (0.88x) | 2597 (0.96x) | - | - | - | - | - | - |
| global affine hard int64 (load!) | **1293** | - | 1546 (0.84x) | 1274 (1.01x) | - | - | - | - | - | - |
| global affine hard float32 (load!) | **937** | traceback=scores: 1016 | 918 (1.02x) | 952 (0.98x) | - | - | - | - | - | - |
| global affine hard float64 (load!) | **757** | traceback=scores: 770 | 750 (1.01x) | 736 (1.03x) | - | - | - | - | - | - |
| global affine soft float64 (load!) | **110** | fill=striped: 110 | 126 (0.87x) | 111 (0.98x) | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| global linear score int32 (load!) | **5207** | - | 5566 (0.94x) | 5232 (1.00x) | 3987 (1.31x) | 1579 (3.30x) | 1442 (3.61x) | 1043 (4.99x) | - | >10x |
| global linear score int64 (load!) | **3155** | - | 3926 (0.80x) | 3117 (1.01x) | >10x | 1731 (1.82x) | 698 (4.52x) | - | - | - |
| global linear score float32 (load!) | **2704** | kernel=avx2: 2742 | 2575 (1.05x) | 2610 (1.04x) | - | 294 (9.21x) | 411 (6.57x) | - | - | - |
| global linear score float64 (load!) | **1927** | kernel=avx2: 1996 | 2013 (0.96x) | 1959 (0.98x) | - | 296 (6.50x) | 416 (4.63x) | - | 584 (3.30x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| global linear path int32 (load!) | **2526** | - | 2771 (0.91x) | 2847 (0.89x) | 1676 (1.51x) | 805 (3.14x) | 498 (5.07x) | 1005 (2.51x) | - | - |
| global linear path int64 (load!) | **1523** | - | 1514 (1.01x) | 1461 (1.04x) | >10x | 485 (3.14x) | 495 (3.08x) | - | - | - |
| global linear path float32 (load!) | **2330** | traceback=scores: 2325 | 2368 (0.98x) | 2331 (1.00x) | - | >10x | 418 (5.58x) | - | - | - |
| global linear path float64 (load!) | **1663** | traceback=scores: 1747 | 1847 (0.90x) | 1695 (0.98x) | - | 219 (7.60x) | 426 (3.91x) | - | 242 (6.88x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| global linear hard int32 (load!) | **3114** | - | 3216 (0.97x) | 3145 (0.99x) | - | - | - | - | - | - |
| global linear hard int64 (load!) | **1603** | - | 1677 (0.96x) | 1609 (1.00x) | - | - | - | - | - | - |
| global linear hard float32 (load!) | **2498** | traceback=scores: 2508 | 2655 (0.94x) | 2575 (0.97x) | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1805** | fill=striped: 1808 | 1958 (0.92x) | 1817 (0.99x) | - | - | - | - | - | - |
| global linear soft float64 (load!) | **162** | api=functions: 176 | 155 (1.05x) | 163 (1.00x) | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| local affine score int32 (load!) | **4720** | - | 4999 (0.94x) | 4896 (0.96x) | 2427 (1.94x) | 1482 (3.18x) | 727 (6.49x) | - | - | >10x |
| local affine score int64 (load!) | **2771** | - | 3272 (0.85x) | 2793 (0.99x) | >10x | 1210 (2.29x) | 643 (4.31x) | - | - | - |
| local affine score float32 (load!) | **1462** | kernel=avx2: 1545 | 1444 (1.01x) | 1449 (1.01x) | - | 170 (8.59x) | 236 (6.19x) | - | - | - |
| local affine score float64 (load!) | **1189** | kernel=avx2: 1358 | 1243 (0.96x) | 1202 (0.99x) | - | 165 (7.19x) | 238 (5.00x) | - | 258 (4.61x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| local affine path int32 (load!) | **1900** | - | 2138 (0.89x) | 1959 (0.97x) | 1460 (1.30x) | 757 (2.51x) | 206 (9.23x) | - | - | >10x |
| local affine path int64 (load!) | **1008** | - | 1189 (0.85x) | 1006 (1.00x) | >10x | 458 (2.20x) | 209 (4.82x) | - | - | - |
| local affine path float32 (load!) | **902** | traceback=scores: 983 | 923 (0.98x) | 953 (0.95x) | - | 149 (6.05x) | 178 (5.06x) | - | - | - |
| local affine path float64 (load!) | **692** | traceback=scores: 733 | 684 (1.01x) | 712 (0.97x) | - | 148 (4.68x) | 179 (3.85x) | - | >10x | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| local affine hard int32 (load!) | **1989** | - | 2224 (0.89x) | 1947 (1.02x) | - | - | - | - | - | - |
| local affine hard int64 (load!) | **1014** | - | 1180 (0.86x) | 1039 (0.98x) | - | - | - | - | - | - |
| local affine hard float32 (load!) | **943** | traceback=scores: 991 | 909 (1.04x) | 967 (0.98x) | - | - | - | - | - | - |
| local affine hard float64 (load!) | **698** | traceback=scores: 764 | 704 (0.99x) | 698 (1.00x) | - | - | - | - | - | - |
| local affine soft float64 (load!) | **49** | fill=striped: 49 | 76 (0.65x) | 49 (1.01x) | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| local linear score int32 | **4703** | - | 5009 (0.94x) | 5022 (0.94x) | 4442 (1.06x) | 1410 (3.33x) | 1236 (3.80x) | - | - | >10x |
| local linear score int64 | **2763** | - | 3362 (0.82x) | 2798 (0.99x) | >10x | 1236 (2.24x) | 1106 (2.50x) | - | - | - |
| local linear score float32 | **2824** | kernel=avx2: 2863 | 2850 (0.99x) | 2868 (0.98x) | - | >10x | 299 (9.44x) | - | - | - |
| local linear score float64 | **1869** | kernel=avx2: 1938 | 1911 (0.98x) | 1863 (1.00x) | - | >10x | 298 (6.28x) | - | 393 (4.75x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| local linear path int32 | **2849** | - | 2855 (1.00x) | 3023 (0.94x) | 1434 (1.99x) | 752 (3.79x) | >10x | - | - | - |
| local linear path int64 | **1396** | - | 1485 (0.94x) | 1430 (0.98x) | >10x | 450 (3.10x) | 229 (6.09x) | - | - | - |
| local linear path float32 (load!) | **1870** | fill=striped: 1912 | 1870 (1.00x) | 1955 (0.96x) | - | >10x | 193 (9.68x) | - | - | - |
| local linear path float64 (load!) | **1355** | traceback=scores: 1361 | 1424 (0.95x) | 1375 (0.99x) | - | 144 (9.44x) | 194 (6.97x) | - | >10x | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| local linear hard int32 (load!) | **3245** | - | 3053 (1.06x) | 3239 (1.00x) | - | - | - | - | - | - |
| local linear hard int64 (load!) | **1530** | - | 1543 (0.99x) | 1438 (1.06x) | - | - | - | - | - | - |
| local linear hard float32 (load!) | **2048** | fill=striped: 2095 | 2007 (1.02x) | 2157 (0.95x) | - | - | - | - | - | - |
| local linear hard float64 (load!) | **1454** | fill=striped: 1431 | 1533 (0.95x) | 1477 (0.98x) | - | - | - | - | - | - |
| local linear soft float64 (load!) | **119** | api=functions: 128 | 116 (1.03x) | 120 (0.99x) | - | - | - | - | - | - |

#### solace · protein_L3000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global affine score int32 (load!) | **3796** | - | 3370 (1.13x) | 4123 (0.92x) | 3732 (1.02x) | 1104 (3.44x) | 891 (4.26x) | 1281 (2.96x) | - | - |
| global affine score int64 (load!) | **2030** | - | 1952 (1.04x) | 1958 (1.04x) | >10x | 653 (3.11x) | 669 (3.03x) | - | - | - |
| global affine score float32 (load!) | **1504** | fill=striped: 1508 | 1513 (0.99x) | 1510 (1.00x) | - | 291 (5.17x) | 298 (5.05x) | - | - | - |
| global affine score float64 (load!) | **1181** | fill=rowwise: 1200 | 1216 (0.97x) | 1198 (0.99x) | - | 290 (4.08x) | 298 (3.97x) | - | 365 (3.23x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global affine path int32 (load!) | **1990** | - | 2033 (0.98x) | 1995 (1.00x) | 770 (2.58x) | 239 (8.31x) | 329 (6.04x) | 1224 (1.63x) | - | >10x |
| global affine path int64 (load!) | **816** | - | 854 (0.96x) | 834 (0.98x) | 112 (7.29x) | 239 (3.42x) | 316 (2.58x) | - | - | - |
| global affine path float32 (load!) | **1341** | traceback=hirschberg_pmax: 1532 | 1333 (1.01x) | 1308 (1.03x) | - | 204 (6.58x) | 231 (5.81x) | - | - | - |
| global affine path float64 (load!) | **1069** | kernel=avx2: 1127 | 1077 (0.99x) | 1080 (0.99x) | - | 202 (5.28x) | 227 (4.70x) | - | >10x | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global affine hard int32 (load!) | **1942** | - | 2010 (0.97x) | 2003 (0.97x) | - | - | - | - | - | - |
| global affine hard int64 (load!) | **838** | - | 831 (1.01x) | 818 (1.02x) | - | - | - | - | - | - |
| global affine hard float32 (load!) | **1299** | traceback=hirschberg_pmax: 1499 | 1317 (0.99x) | 1330 (0.98x) | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1064** | kernel=avx2: 1097 | 1070 (0.99x) | 1078 (0.99x) | - | - | - | - | - | - |
| global affine soft float64 (load!) | **105** | fill=rowwise: 106 | 122 (0.86x) | 102 (1.02x) | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global linear score int32 (load!) | **3789** | - | 3862 (0.98x) | 4075 (0.93x) | 6012 (0.63x) | 1091 (3.47x) | 1358 (2.79x) | 1276 (2.97x) | - | - |
| global linear score int64 (load!) | **1812** | - | 2050 (0.88x) | 1976 (0.92x) | >10x | 632 (2.87x) | 678 (2.67x) | - | - | - |
| global linear score float32 (load!) | **4542** | api=functions: 4523 | 4266 (1.06x) | 4392 (1.03x) | - | >10x | >10x | - | - | - |
| global linear score float64 (load!) | **2261** | fill=striped: 2348 | 2202 (1.03x) | 2198 (1.03x) | - | 288 (7.85x) | 372 (6.08x) | - | 583 (3.88x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global linear path int32 (load!) | **3035** | - | 3005 (1.01x) | 3154 (0.96x) | 711 (4.27x) | >10x | 536 (5.67x) | 1192 (2.55x) | - | >10x |
| global linear path int64 (load!) | **1393** | - | 1382 (1.01x) | 1447 (0.96x) | >10x | 237 (5.88x) | 544 (2.56x) | - | - | - |
| global linear path float32 (load!) | **3167** | traceback=pointers: 3181 | 3161 (1.00x) | 2989 (1.06x) | - | >10x | 446 (7.10x) | - | - | - |
| global linear path float64 (load!) | **1561** | fill=striped: 1560 | 1559 (1.00x) | 1531 (1.02x) | - | 220 (7.10x) | 462 (3.38x) | - | 254 (6.15x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global linear hard int32 (load!) | **2932** | - | 3036 (0.97x) | 3168 (0.93x) | - | - | - | - | - | - |
| global linear hard int64 (load!) | **1445** | - | 1387 (1.04x) | 1443 (1.00x) | - | - | - | - | - | - |
| global linear hard float32 (load!) | **3256** | traceback=pointers: 3178 | 3126 (1.04x) | 3182 (1.02x) | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1478** | fill=striped: 1539 | 1544 (0.96x) | 1470 (1.01x) | - | - | - | - | - | - |
| global linear soft float64 (load!) | **244** | fill=striped: 246 | 243 (1.01x) | 248 (0.99x) | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local affine score int32 (load!) | **3578** | - | 3732 (0.96x) | 3736 (0.96x) | 2959 (1.21x) | 367 (9.74x) | 769 (4.65x) | - | - | - |
| local affine score int64 (load!) | **1857** | - | 1842 (1.01x) | 1921 (0.97x) | >10x | 507 (3.66x) | 660 (2.81x) | - | - | - |
| local affine score float32 (load!) | **1798** | fill=striped: 1811 | 1826 (0.98x) | 1786 (1.01x) | - | >10x | 233 (7.73x) | - | - | - |
| local affine score float64 (load!) | **1353** | fill=striped: 1351 | 1363 (0.99x) | 1338 (1.01x) | - | 171 (7.92x) | 225 (6.00x) | - | 274 (4.94x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local affine path int32 (load!) | **1462** | - | 1425 (1.03x) | 1455 (1.00x) | 670 (2.18x) | 278 (5.27x) | 222 (6.57x) | - | - | - |
| local affine path int64 (load!) | **672** | - | 693 (0.97x) | 698 (0.96x) | 94 (7.14x) | 257 (2.61x) | 222 (3.03x) | - | - | - |
| local affine path float32 (load!) | **1106** | fill=striped: 1142 | 1061 (1.04x) | 1183 (0.93x) | - | 151 (7.34x) | 184 (6.01x) | - | - | - |
| local affine path float64 (load!) | **711** | fill=striped: 713 | 641 (1.11x) | 722 (0.98x) | - | 152 (4.66x) | 187 (3.81x) | - | >10x | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local affine hard int32 (load!) | **1430** | - | 1441 (0.99x) | 1464 (0.98x) | - | - | - | - | - | - |
| local affine hard int64 (load!) | **675** | - | 665 (1.02x) | 686 (0.98x) | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1126** | fill=striped: 1136 | 1033 (1.09x) | 1158 (0.97x) | - | - | - | - | - | - |
| local affine hard float64 (load!) | **736** | fill=striped: 722 | 678 (1.09x) | 715 (1.03x) | - | - | - | - | - | - |
| local affine soft float64 (load!) | **79** | fill=striped: 80 | 102 (0.78x) | 81 (0.99x) | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local linear score int32 (load!) | **3422** | - | 3497 (0.98x) | 3697 (0.93x) | 5900 (0.58x) | 371 (9.21x) | 1286 (2.66x) | - | - | - |
| local linear score int64 (load!) | **1807** | - | 1898 (0.95x) | 1872 (0.97x) | >10x | 476 (3.80x) | 1155 (1.56x) | - | - | - |
| local linear score float32 (load!) | **4005** | api=functions: 4357 | 4488 (0.89x) | 4603 (0.87x) | - | >10x | >10x | - | - | - |
| local linear score float64 (load!) | **2352** | fill=striped: 2286 | 2327 (1.01x) | 2389 (0.98x) | - | >10x | 289 (8.12x) | - | 507 (4.64x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local linear path int32 (load!) | **2752** | - | 2781 (0.99x) | 2903 (0.95x) | 672 (4.09x) | >10x | >10x | - | - | - |
| local linear path int64 (load!) | **1211** | - | 1219 (0.99x) | 1183 (1.02x) | >10x | 270 (4.49x) | 259 (4.67x) | - | - | - |
| local linear path float32 (load!) | **2609** | fill=striped: 2690 | 2466 (1.06x) | 2700 (0.97x) | - | >10x | >10x | - | - | - |
| local linear path float64 (load!) | **1315** | fill=striped: 1271 | 1270 (1.03x) | 1291 (1.02x) | - | 138 (9.55x) | 214 (6.15x) | - | >10x | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local linear hard int32 (load!) | **2773** | - | 2848 (0.97x) | 2975 (0.93x) | - | - | - | - | - | - |
| local linear hard int64 (load!) | **1204** | - | 1172 (1.03x) | 1184 (1.02x) | - | - | - | - | - | - |
| local linear hard float32 (load!) | **2803** | traceback=pointers: 2670 | 2592 (1.08x) | 2763 (1.01x) | - | - | - | - | - | - |
| local linear hard float64 (load!) | **1299** | traceback=pointers: 1299 | 1308 (0.99x) | 1321 (0.98x) | - | - | - | - | - | - |
| local linear soft float64 (load!) | **229** | fill=striped: 229 | 229 (1.00x) | 227 (1.01x) | - | - | - | - | - | - |

#### solace · dna_L100 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **20567** | fill=striped: 21791 | 21398 (0.96x) | 21605 (0.95x) | 14409 (1.43x) | 34111 (0.60x) | - | 4175 (4.93x) | 4047 (5.08x) | - | - | - |
| global affine score float32 | **6109** | schedule=sorted: 6148 | 6422 (0.95x) | 6354 (0.96x) | - | - | - | - | - | - | - | - |
| global affine score float64 | **5928** | schedule=sorted: 6444 | 6664 (0.89x) | 6457 (0.92x) | - | - | - | - | - | - | 1969 (3.01x) | - |
| global affine path int32 | **9314** | fill=striped: 9747 | 9507 (0.98x) | 10101 (0.92x) | 8791 (1.06x) | 8925 (1.04x) | - | 3764 (2.47x) | 4193 (2.22x) | - | - | - |
| global affine path float32 | **4416** | schedule=sorted: 4526 | 4323 (1.02x) | 4438 (0.99x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **3660** | fill=striped: 3793 | 3743 (0.98x) | 3718 (0.98x) | - | - | - | - | - | - | 540 (6.78x) | - |
| global affine hard int32 | **11230** | schedule=sorted: 9937 | 11849 (0.95x) | 11383 (0.99x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **4858** | fill=striped: 4955 | 4766 (1.02x) | 4812 (1.01x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **3855** | fill=striped: 4035 | 3939 (0.98x) | 4025 (0.96x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **966** | fill=striped: 998 | 967 (1.00x) | 956 (1.01x) | - | - | - | - | - | - | - | - |
| global linear score int32 | **47550** | schedule=sorted: 42462 | 41758 (1.14x) | 45062 (1.06x) | 15304 (3.11x) | 36017 (1.32x) | - | >10x | 49498 (0.96x) | 12405 (3.83x) | - | >10x |
| global linear score float32 | **36536** | schedule=sorted: 39134 | 34897 (1.05x) | 36590 (1.00x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **21634** | schedule=sorted: 21257 | 18410 (1.18x) | 21640 (1.00x) | - | - | - | - | - | - | 3262 (6.63x) | - |
| global linear path int32 | **17693** | schedule=sorted: 18438 | 17015 (1.04x) | 19136 (0.92x) | 8178 (2.16x) | 8871 (1.99x) | - | 4013 (4.41x) | 40465 (0.44x) | 5125 (3.45x) | - | >10x |
| global linear path float32 | **14817** | schedule=sorted: 15219 | 14331 (1.03x) | 14693 (1.01x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **12199** | schedule=sorted: 12134 | 11449 (1.07x) | 12186 (1.00x) | - | - | - | - | - | - | >10x | - |
| global linear hard int32 | **27194** | schedule=sorted: 28264 | 22619 (1.20x) | 25731 (1.06x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **23168** | schedule=sorted: 24672 | 21481 (1.08x) | 22159 (1.05x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **15539** | schedule=sorted: 15548 | 14459 (1.07x) | 15136 (1.03x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **1863** | fill=striped: 1804 | 1594 (1.17x) | 1787 (1.04x) | - | - | - | - | - | - | - | - |
| local affine score int32 | **19979** | fill=striped: 20305 | 19228 (1.04x) | 20933 (0.95x) | 15258 (1.31x) | 16746 (1.19x) | - | - | - | - | - | >10x |
| local affine score float32 | **10012** | fill=striped: 10152 | 10018 (1.00x) | 9971 (1.00x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **7589** | schedule=sorted: 8551 | 8445 (0.90x) | 7737 (0.98x) | - | - | - | - | - | - | 1107 (6.86x) | - |
| local affine path int32 | **7650** | fill=striped: 7769 | 7907 (0.97x) | 7836 (0.98x) | 7916 (0.97x) | 7513 (1.02x) | - | - | - | - | - | - |
| local affine path float32 | **4813** | schedule=sorted: 4915 | 4736 (1.02x) | 4877 (0.99x) | - | - | - | - | - | - | - | - |
| local affine path float64 | **3728** | schedule=sorted: 3841 | 3560 (1.05x) | 3720 (1.00x) | - | - | - | - | - | - | >10x | - |
| local affine hard int32 | **7990** | schedule=sorted: 9141 | 10038 (0.80x) | 8216 (0.97x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **5298** | schedule=sorted: 5143 | 5298 (1.00x) | 5276 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **3832** | schedule=sorted: 4288 | 4100 (0.93x) | 4023 (0.95x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **951** | fill=striped: 966 | 994 (0.96x) | 936 (1.02x) | - | - | - | - | - | - | - | - |
| local linear score int32 | **41181** | schedule=sorted: 45758 | 35913 (1.15x) | 42801 (0.96x) | 12441 (3.31x) | 16651 (2.47x) | - | - | - | - | - | >10x |
| local linear score float32 | **29158** | schedule=sorted: 29062 | 27560 (1.06x) | 29342 (0.99x) | - | - | - | - | - | - | - | - |
| local linear score float64 | **15090** | schedule=sorted: 16132 | 14701 (1.03x) | 16253 (0.93x) | - | - | - | - | - | - | 2344 (6.44x) | - |
| local linear path int32 | **17340** | schedule=sorted: 16495 | 15218 (1.14x) | 17283 (1.00x) | 7354 (2.36x) | 7886 (2.20x) | - | - | - | - | - | >10x |
| local linear path float32 | **14157** | schedule=sorted: 14322 | 14018 (1.01x) | 13664 (1.04x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **10187** | schedule=sorted: 10493 | 8830 (1.15x) | 10076 (1.01x) | - | - | - | - | - | - | >10x | - |
| local linear hard int32 (load!) | **25226** | schedule=sorted: 22627 | 20355 (1.24x) | 27854 (0.91x) | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **20391** | schedule=sorted: 19671 | 18294 (1.11x) | 19493 (1.05x) | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **12181** | schedule=sorted: 10890 | 10369 (1.17x) | 12967 (0.94x) | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **1572** | schedule=sorted: 1594 | 1458 (1.08x) | 1610 (0.98x) | - | - | - | - | - | - | - | - |

#### solace · dna_L1000 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 (load!) | **46390** | fill=striped: 41772 | 43084 (1.08x) | 41574 (1.12x) | 20253 (2.29x) | 36056 (1.29x) | - | 6369 (7.28x) | 17378 (2.67x) | - | - | >10x |
| global affine score float32 (load!) | **13123** | fill=striped: 12918 | 12376 (1.06x) | 12182 (1.08x) | - | - | - | - | - | - | - | - |
| global affine score float64 (load!) | **8895** | fill=striped: 8838 | 8891 (1.00x) | 8527 (1.04x) | - | - | - | - | - | - | 2201 (4.04x) | - |
| global affine path int32 (load!) | **14572** | fill=striped: 16528 | 17124 (0.85x) | 13148 (1.11x) | 9188 (1.59x) | >10x | - | 6555 (2.22x) | 5972 (2.44x) | - | - | >10x |
| global affine path float32 | **7539** | schedule=sorted: 8539 | 8317 (0.91x) | 8630 (0.87x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **6286** | fill=striped: 6256 | 6843 (0.92x) | 6582 (0.95x) | - | - | - | - | - | - | >10x | - |
| global affine hard int32 | **14819** | schedule=sorted: 16106 | 18132 (0.82x) | 13533 (1.10x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **8715** | schedule=sorted: 8719 | 8436 (1.03x) | 8352 (1.04x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **6751** | schedule=sorted: 6272 | 6573 (1.03x) | 6462 (1.04x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **431** | fill=striped: 432 | 458 (0.94x) | 427 (1.01x) | - | - | - | - | - | - | - | - |
| global linear score int32 | **41977** | fill=striped: 40171 | 37703 (1.11x) | 38417 (1.09x) | 35299 (1.19x) | 30616 (1.37x) | - | 6513 (6.45x) | 173481 (0.24x) | 66774 (0.63x) | - | >10x |
| global linear score float32 | **26599** | fill=striped: 29969 | 26559 (1.00x) | 27524 (0.97x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **11654** | schedule=sorted: 13033 | 13045 (0.89x) | 12567 (0.93x) | - | - | - | - | - | - | 3664 (3.18x) | - |
| global linear path int32 | **26633** | fill=striped: 28690 | 26124 (1.02x) | 25446 (1.05x) | 9729 (2.74x) | >10x | - | 6775 (3.93x) | 118451 (0.22x) | 48407 (0.55x) | - | >10x |
| global linear path float32 | **20944** | fill=striped: 20835 | 20216 (1.04x) | 21815 (0.96x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **10949** | fill=striped: 11539 | 10739 (1.02x) | 10136 (1.08x) | - | - | - | - | - | - | >10x | - |
| global linear hard int32 | **25865** | fill=striped: 29604 | 29275 (0.88x) | 28612 (0.90x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **21156** | schedule=sorted: 21468 | 21322 (0.99x) | 20657 (1.02x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **10323** | schedule=sorted: 10921 | 11020 (0.94x) | 10943 (0.94x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **1182** | fill=striped: 1189 | 1162 (1.02x) | 1228 (0.96x) | - | - | - | - | - | - | - | - |
| local affine score int32 | **35018** | fill=striped: 35275 | 37646 (0.93x) | 35634 (0.98x) | 33358 (1.05x) | 17762 (1.97x) | - | - | - | - | - | >10x |
| local affine score float32 | **18637** | fill=striped: 19272 | 16802 (1.11x) | 19293 (0.97x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **10747** | schedule=sorted: 10787 | 10790 (1.00x) | 10636 (1.01x) | - | - | - | - | - | - | 1277 (8.42x) | - |
| local affine path int32 (load!) | **9238** | fill=striped: 10617 | 8755 (1.06x) | 8665 (1.07x) | 9371 (0.99x) | 1558 (5.93x) | - | - | - | - | - | >10x |
| local affine path float32 (load!) | **8554** | fill=striped: 9074 | 8875 (0.96x) | 9502 (0.90x) | - | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **5175** | fill=striped: 5237 | 4918 (1.05x) | 4991 (1.04x) | - | - | - | - | - | - | >10x | - |
| local affine hard int32 (load!) | **9309** | schedule=sorted: 9514 | 9024 (1.03x) | 10331 (0.90x) | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **9122** | fill=striped: 9423 | 9362 (0.97x) | 9354 (0.98x) | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **5070** | schedule=sorted: 5024 | 4490 (1.13x) | 5172 (0.98x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **203** | schedule=sorted: 203 | 302 (0.67x) | 203 (1.00x) | - | - | - | - | - | - | - | - |
| local linear score int32 | **35515** | fill=striped: 35074 | 44290 (0.80x) | 34845 (1.02x) | 40646 (0.87x) | 18958 (1.87x) | - | - | - | - | - | >10x |
| local linear score float32 | **23690** | schedule=sorted: 26650 | 21338 (1.11x) | 21985 (1.08x) | - | - | - | - | - | - | - | - |
| local linear score float64 | **11115** | fill=striped: 13658 | 13117 (0.85x) | 13694 (0.81x) | - | - | - | - | - | - | 3289 (3.38x) | - |
| local linear path int32 | **20947** | fill=striped: 22026 | 21295 (0.98x) | 23353 (0.90x) | 9520 (2.20x) | >10x | - | - | - | - | - | - |
| local linear path float32 | **14153** | schedule=sorted: 16048 | 15194 (0.93x) | 15384 (0.92x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **7774** | schedule=sorted: 7824 | 8085 (0.96x) | 7999 (0.97x) | - | - | - | - | - | - | >10x | - |
| local linear hard int32 | **21168** | fill=striped: 21987 | 21226 (1.00x) | 19662 (1.08x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **14663** | schedule=sorted: 16022 | 16165 (0.91x) | 15832 (0.93x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **8104** | schedule=sorted: 8119 | 7910 (1.02x) | 8075 (1.00x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **1246** | schedule=sorted: 1257 | 1265 (0.99x) | 1244 (1.00x) | - | - | - | - | - | - | - | - |

#### solace · mirna · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **22028** | schedule=sorted: 23912 | 20923 (1.05x) | 22319 (0.99x) | 6276 (3.51x) | 8601 (2.56x) | - | 2731 (8.07x) | >10x | - | - | >10x |
| global affine score float32 | **17476** | schedule=sorted: 17921 | 18189 (0.96x) | 17959 (0.97x) | - | - | - | - | - | - | - | - |
| global affine score float64 | **11668** | schedule=sorted: 12711 | 10918 (1.07x) | 11783 (0.99x) | - | - | - | - | - | - | >10x | - |
| global affine path int32 | **4888** | schedule=sorted: 4736 | 4762 (1.03x) | 5317 (0.92x) | 3526 (1.39x) | 5683 (0.86x) | - | 2208 (2.21x) | >10x | - | - | >10x |
| global affine path float32 | **4475** | schedule=sorted: 4031 | 4219 (1.06x) | 4449 (1.01x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **3666** | schedule=sorted: 3502 | 3404 (1.08x) | 3740 (0.98x) | - | - | - | - | - | - | >10x | - |
| global affine hard int32 | **8453** | schedule=sorted: 8353 | 8477 (1.00x) | 8468 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **7035** | schedule=sorted: 7152 | 7292 (0.96x) | 7030 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **5529** | schedule=sorted: 5517 | 5528 (1.00x) | 5433 (1.02x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **1507** | schedule=sorted: 1451 | 1309 (1.15x) | 1446 (1.04x) | - | - | - | - | - | - | - | - |
| global linear score int32 | **26023** | schedule=sorted: 26115 | 25166 (1.03x) | 26275 (0.99x) | 6470 (4.02x) | 8641 (3.01x) | - | 3133 (8.31x) | 3023 (8.61x) | >10x | - | >10x |
| global linear score float32 | **21936** | schedule=sorted: 21527 | 22096 (0.99x) | 22205 (0.99x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **16161** | schedule=sorted: 15970 | 14361 (1.13x) | 15468 (1.04x) | - | - | - | - | - | - | >10x | - |
| global linear path int32 | **4018** | schedule=sorted: 4288 | 4460 (0.90x) | 4134 (0.97x) | 3634 (1.11x) | 5682 (0.71x) | - | 2291 (1.75x) | 2748 (1.46x) | 1074 (3.74x) | - | - |
| global linear path float32 | **3688** | schedule=sorted: 3341 | 3723 (0.99x) | 3639 (1.01x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **3295** | schedule=sorted: 3314 | 3491 (0.94x) | 3506 (0.94x) | - | - | - | - | - | - | 445 (7.40x) | - |
| global linear hard int32 | **7924** | schedule=sorted: 8186 | 8493 (0.93x) | 7704 (1.03x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **7441** | schedule=sorted: 7260 | 7517 (0.99x) | 7105 (1.05x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **6482** | schedule=sorted: 6310 | 6243 (1.04x) | 6322 (1.03x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **2438** | schedule=sorted: 2386 | 2239 (1.09x) | 2447 (1.00x) | - | - | - | - | - | - | - | - |
| local affine score int32 | **21688** | schedule=sorted: 21704 | 20708 (1.05x) | 22346 (0.97x) | 8040 (2.70x) | 8506 (2.55x) | - | - | - | - | - | >10x |
| local affine score float32 | **16447** | schedule=sorted: 18145 | 16841 (0.98x) | 18030 (0.91x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **11148** | schedule=sorted: 10914 | 10065 (1.11x) | 10812 (1.03x) | - | - | - | - | - | - | >10x | - |
| local affine path int32 | **10791** | schedule=sorted: 10372 | 10055 (1.07x) | 10696 (1.01x) | 4507 (2.39x) | 5127 (2.10x) | - | - | - | - | - | >10x |
| local affine path float32 | **7920** | schedule=sorted: 8595 | 8389 (0.94x) | 8734 (0.91x) | - | - | - | - | - | - | - | - |
| local affine path float64 | **5751** | schedule=sorted: 5323 | 5547 (1.04x) | 5724 (1.00x) | - | - | - | - | - | - | >10x | - |
| local affine hard int32 | **11213** | schedule=sorted: 11041 | 10635 (1.05x) | 11211 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **9786** | schedule=sorted: 9221 | 8893 (1.10x) | 8990 (1.09x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **5646** | schedule=sorted: 5697 | 5796 (0.97x) | 5683 (0.99x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **1612** | schedule=sorted: 1548 | 1375 (1.17x) | 1581 (1.02x) | - | - | - | - | - | - | - | - |
| local linear score int32 | **24915** | schedule=sorted: 27008 | 23359 (1.07x) | 26898 (0.93x) | 5710 (4.36x) | 8422 (2.96x) | - | - | - | - | - | >10x |
| local linear score float32 | **21554** | schedule=sorted: 20760 | 19319 (1.12x) | 21314 (1.01x) | - | - | - | - | - | - | - | - |
| local linear score float64 | **12920** | schedule=sorted: 13137 | 12690 (1.02x) | 13814 (0.94x) | - | - | - | - | - | - | >10x | - |
| local linear path int32 | **9244** | schedule=sorted: 8905 | 8951 (1.03x) | 9692 (0.95x) | 3783 (2.44x) | 5127 (1.80x) | - | - | - | - | - | >10x |
| local linear path float32 | **8178** | schedule=sorted: 7923 | 7623 (1.07x) | 8630 (0.95x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **6551** | schedule=sorted: 6887 | 6176 (1.06x) | 6718 (0.98x) | - | - | - | - | - | - | >10x | - |
| local linear hard int32 | **13384** | schedule=sorted: 11718 | 9816 (1.36x) | 12716 (1.05x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **10684** | schedule=sorted: 11318 | 10273 (1.04x) | 11247 (0.95x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **8402** | schedule=sorted: 8613 | 7432 (1.13x) | 7936 (1.06x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **2340** | schedule=sorted: 2309 | 2105 (1.11x) | 2548 (0.92x) | - | - | - | - | - | - | - | - |

#### solace · protein_L100 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **14715** | schedule=sorted: 16653 | 15532 (0.95x) | 15906 (0.93x) | 10825 (1.36x) | 7511 (1.96x) | - | 3997 (3.68x) | - | >10x |
| global affine score float32 | **4846** | fill=striped: 4882 | 4840 (1.00x) | 4990 (0.97x) | - | - | - | - | - | - |
| global affine score float64 | **5051** | fill=striped: 5140 | 5147 (0.98x) | 4892 (1.03x) | - | - | - | - | 1954 (2.58x) | - |
| global affine path int32 | **8267** | schedule=sorted: 8528 | 8508 (0.97x) | 8315 (0.99x) | 6364 (1.30x) | 5725 (1.44x) | - | 4046 (2.04x) | - | - |
| global affine path float32 | **3774** | fill=striped: 3696 | 3600 (1.05x) | 3757 (1.00x) | - | - | - | - | - | - |
| global affine path float64 | **3341** | schedule=sorted: 3426 | 3112 (1.07x) | 3545 (0.94x) | - | - | - | - | 516 (6.48x) | - |
| global affine hard int32 | **8130** | schedule=sorted: 8057 | 8565 (0.95x) | 8055 (1.01x) | - | - | - | - | - | - |
| global affine hard float32 | **3729** | schedule=sorted: 3700 | 3644 (1.02x) | 3751 (0.99x) | - | - | - | - | - | - |
| global affine hard float64 | **3211** | schedule=sorted: 3322 | 3301 (0.97x) | 3240 (0.99x) | - | - | - | - | - | - |
| global affine soft float64 | **857** | schedule=sorted: 907 | 907 (0.94x) | 885 (0.97x) | - | - | - | - | - | - |
| global linear score int32 | **14205** | schedule=sorted: 15706 | 15209 (0.93x) | 13834 (1.03x) | 10810 (1.31x) | 7656 (1.86x) | - | 4041 (3.52x) | - | - |
| global linear score float32 | **6773** | schedule=sorted: 7212 | 7299 (0.93x) | 7345 (0.92x) | - | - | - | - | - | - |
| global linear score float64 | **6802** | schedule=sorted: 6812 | 7448 (0.91x) | 6760 (1.01x) | - | - | - | - | 3347 (2.03x) | - |
| global linear path int32 | **10022** | fill=striped: 11462 | 10708 (0.94x) | 9519 (1.05x) | 6266 (1.60x) | 5661 (1.77x) | - | 3831 (2.62x) | - | - |
| global linear path float32 | **6917** | schedule=sorted: 6847 | 5572 (1.24x) | 6877 (1.01x) | - | - | - | - | - | - |
| global linear path float64 | **6012** | schedule=sorted: 6290 | 5364 (1.12x) | 6191 (0.97x) | - | - | - | - | 1165 (5.16x) | - |
| global linear hard int32 | **11498** | fill=striped: 11178 | 11219 (1.02x) | 11201 (1.03x) | - | - | - | - | - | - |
| global linear hard float32 | **7207** | schedule=sorted: 7417 | 5868 (1.23x) | 7104 (1.01x) | - | - | - | - | - | - |
| global linear hard float64 | **6186** | schedule=sorted: 6336 | 5060 (1.22x) | 6245 (0.99x) | - | - | - | - | - | - |
| global linear soft float64 | **1191** | schedule=sorted: 1256 | 1159 (1.03x) | 1249 (0.95x) | - | - | - | - | - | - |
| local affine score int32 | **14883** | schedule=sorted: 14357 | 14183 (1.05x) | 14137 (1.05x) | 8765 (1.70x) | 7556 (1.97x) | - | - | - | >10x |
| local affine score float32 | **5597** | schedule=sorted: 5683 | 5679 (0.99x) | 5575 (1.00x) | - | - | - | - | - | - |
| local affine score float64 | **5800** | fill=striped: 5431 | 5528 (1.05x) | 5475 (1.06x) | - | - | - | - | 1265 (4.59x) | - |
| local affine path int32 | **6359** | schedule=sorted: 6635 | 6708 (0.95x) | 6020 (1.06x) | 5573 (1.14x) | 5188 (1.23x) | - | - | - | - |
| local affine path float32 | **3436** | schedule=sorted: 3466 | 3412 (1.01x) | 3496 (0.98x) | - | - | - | - | - | - |
| local affine path float64 | **2877** | fill=striped: 3284 | 3123 (0.92x) | 2989 (0.96x) | - | - | - | - | 317 (9.08x) | - |
| local affine hard int32 | **6294** | fill=striped: 6985 | 7064 (0.89x) | 6595 (0.95x) | - | - | - | - | - | - |
| local affine hard float32 | **3266** | schedule=sorted: 3472 | 3377 (0.97x) | 3530 (0.93x) | - | - | - | - | - | - |
| local affine hard float64 | **2917** | schedule=sorted: 3117 | 2940 (0.99x) | 3031 (0.96x) | - | - | - | - | - | - |
| local affine soft float64 | **830** | fill=striped: 826 | 827 (1.00x) | 839 (0.99x) | - | - | - | - | - | - |
| local linear score int32 | **14642** | schedule=sorted: 14583 | 14560 (1.01x) | 14594 (1.00x) | 8705 (1.68x) | 7194 (2.04x) | - | - | - | >10x |
| local linear score float32 | **8505** | schedule=sorted: 8895 | 8644 (0.98x) | 8497 (1.00x) | - | - | - | - | - | - |
| local linear score float64 | **6960** | fill=striped: 7190 | 7221 (0.96x) | 7118 (0.98x) | - | - | - | - | 1921 (3.62x) | - |
| local linear path int32 | **7673** | fill=striped: 9028 | 6095 (1.26x) | 7682 (1.00x) | 6345 (1.21x) | 5098 (1.51x) | - | - | - | - |
| local linear path float32 | **6843** | schedule=sorted: 6394 | 5793 (1.18x) | 6852 (1.00x) | - | - | - | - | - | - |
| local linear path float64 | **5030** | schedule=sorted: 5817 | 4754 (1.06x) | 5726 (0.88x) | - | - | - | - | 570 (8.83x) | - |
| local linear hard int32 | **8257** | fill=striped: 9436 | 6018 (1.37x) | 7924 (1.04x) | - | - | - | - | - | - |
| local linear hard float32 | **7093** | schedule=sorted: 6290 | 5843 (1.21x) | 6827 (1.04x) | - | - | - | - | - | - |
| local linear hard float64 | **5941** | schedule=sorted: 6083 | 4760 (1.25x) | 5928 (1.00x) | - | - | - | - | - | - |
| local linear soft float64 | **1357** | fill=striped: 1407 | 1263 (1.07x) | 1348 (1.01x) | - | - | - | - | - | - |

#### solace · protein_L1000 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **33027** | fill=striped: 40531 | 40237 (0.82x) | 34034 (0.97x) | 21763 (1.52x) | 7471 (4.42x) | - | 6242 (5.29x) | - | >10x |
| global affine score float32 | **8011** | schedule=sorted: 7861 | 8681 (0.92x) | 8093 (0.99x) | - | - | - | - | - | - |
| global affine score float64 | **6784** | fill=striped: 6499 | 6698 (1.01x) | 6273 (1.08x) | - | - | - | - | 2238 (3.03x) | - |
| global affine path int32 | **12464** | schedule=sorted: 13127 | 14337 (0.87x) | 12241 (1.02x) | 9319 (1.34x) | >10x | - | 6438 (1.94x) | - | >10x |
| global affine path float32 | **5988** | fill=striped: 6930 | 5819 (1.03x) | 6134 (0.98x) | - | - | - | - | - | - |
| global affine path float64 | **5309** | schedule=sorted: 5137 | 5386 (0.99x) | 5013 (1.06x) | - | - | - | - | >10x | - |
| global affine hard int32 | **12416** | fill=striped: 12387 | 15912 (0.78x) | 12552 (0.99x) | - | - | - | - | - | - |
| global affine hard float32 | **6086** | fill=striped: 6698 | 6223 (0.98x) | 6052 (1.01x) | - | - | - | - | - | - |
| global affine hard float64 | **5076** | schedule=sorted: 5227 | 5150 (0.99x) | 5150 (0.99x) | - | - | - | - | - | - |
| global affine soft float64 | **395** | schedule=sorted: 434 | 443 (0.89x) | 431 (0.92x) | - | - | - | - | - | - |
| global linear score int32 | **34292** | fill=striped: 35582 | 36658 (0.94x) | 37126 (0.92x) | 20275 (1.69x) | 7258 (4.72x) | - | 7787 (4.40x) | - | >10x |
| global linear score float32 | **22524** | schedule=sorted: 23638 | 25228 (0.89x) | 21774 (1.03x) | - | - | - | - | - | - |
| global linear score float64 | **12283** | schedule=sorted: 13609 | 12383 (0.99x) | 13001 (0.94x) | - | - | - | - | 3704 (3.32x) | - |
| global linear path int32 | **26348** | fill=striped: 26390 | 26351 (1.00x) | 25760 (1.02x) | 9020 (2.92x) | >10x | - | 6476 (4.07x) | - | >10x |
| global linear path float32 | **17785** | fill=striped: 17826 | 18037 (0.99x) | 17544 (1.01x) | - | - | - | - | - | - |
| global linear path float64 | **9577** | schedule=sorted: 9894 | 9887 (0.97x) | 9982 (0.96x) | - | - | - | - | 1372 (6.98x) | - |
| global linear hard int32 | **24428** | schedule=sorted: 25191 | 26085 (0.94x) | 24830 (0.98x) | - | - | - | - | - | - |
| global linear hard float32 | **18533** | schedule=sorted: 18940 | 18236 (1.02x) | 17741 (1.04x) | - | - | - | - | - | - |
| global linear hard float64 | **9946** | fill=striped: 10068 | 9909 (1.00x) | 9869 (1.01x) | - | - | - | - | - | - |
| global linear soft float64 | **1100** | schedule=sorted: 1111 | 1085 (1.01x) | 1106 (0.99x) | - | - | - | - | - | - |
| local affine score int32 | **34510** | fill=striped: 38570 | 38098 (0.91x) | 36112 (0.96x) | 21041 (1.64x) | 7597 (4.54x) | - | - | - | >10x |
| local affine score float32 | **9608** | schedule=sorted: 9716 | 9672 (0.99x) | 10035 (0.96x) | - | - | - | - | - | - |
| local affine score float64 | **8016** | fill=striped: 7491 | 7692 (1.04x) | 7650 (1.05x) | - | - | - | - | 1565 (5.12x) | - |
| local affine path int32 | **9567** | schedule=sorted: 10004 | 9120 (1.05x) | 10481 (0.91x) | 8864 (1.08x) | 1256 (7.61x) | - | - | - | - |
| local affine path float32 | **7028** | schedule=sorted: 7156 | 6667 (1.05x) | 6492 (1.08x) | - | - | - | - | - | - |
| local affine path float64 | **4341** | schedule=sorted: 4278 | 4298 (1.01x) | 3927 (1.11x) | - | - | - | - | >10x | - |
| local affine hard int32 | **9611** | schedule=sorted: 10117 | 9901 (0.97x) | 10832 (0.89x) | - | - | - | - | - | - |
| local affine hard float32 | **6922** | fill=striped: 6447 | 6847 (1.01x) | 6912 (1.00x) | - | - | - | - | - | - |
| local affine hard float64 | **4574** | fill=striped: 4829 | 4321 (1.06x) | 4827 (0.95x) | - | - | - | - | - | - |
| local affine soft float64 | **414** | schedule=sorted: 418 | 450 (0.92x) | 415 (1.00x) | - | - | - | - | - | - |
| local linear score int32 | **34549** | fill=striped: 34425 | 42509 (0.81x) | 39716 (0.87x) | 18679 (1.85x) | 7872 (4.39x) | - | - | - | >10x |
| local linear score float32 | **20065** | schedule=sorted: 22087 | 22339 (0.90x) | 21856 (0.92x) | - | - | - | - | - | - |
| local linear score float64 | **12239** | fill=striped: 13479 | 12127 (1.01x) | 13350 (0.92x) | - | - | - | - | 2975 (4.11x) | - |
| local linear path int32 (load!) | **23899** | fill=striped: 24105 | 24747 (0.97x) | 21166 (1.13x) | 8835 (2.70x) | >10x | - | - | - | - |
| local linear path float32 (load!) | **15757** | schedule=sorted: 15165 | 14641 (1.08x) | 14549 (1.08x) | - | - | - | - | - | - |
| local linear path float64 (load!) | **8270** | schedule=sorted: 8672 | 8065 (1.03x) | 8150 (1.01x) | - | - | - | - | >10x | - |
| local linear hard int32 (load!) | **19516** | schedule=sorted: 23348 | 21971 (0.89x) | 24646 (0.79x) | - | - | - | - | - | - |
| local linear hard float32 (load!) | **16028** | fill=striped: 15622 | 16607 (0.97x) | 15977 (1.00x) | - | - | - | - | - | - |
| local linear hard float64 (load!) | **8318** | schedule=sorted: 8182 | 8471 (0.98x) | 8132 (1.02x) | - | - | - | - | - | - |
| local linear soft float64 | **1041** | schedule=sorted: 1070 | 1006 (1.04x) | 1066 (0.98x) | - | - | - | - | - | - |

#### solace · protein_L300 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **28468** | fill=striped: 29759 | 30052 (0.95x) | 27788 (1.02x) | 16585 (1.72x) | 8862 (3.21x) | - | 6112 (4.66x) | - | >10x |
| global affine score float32 | **6887** | fill=striped: 6893 | 7012 (0.98x) | 7359 (0.94x) | - | - | - | - | - | - |
| global affine score float64 | **6560** | fill=striped: 6057 | 6243 (1.05x) | 6263 (1.05x) | - | - | - | - | 2182 (3.01x) | - |
| global affine path int32 | **13747** | fill=striped: 13880 | 14665 (0.94x) | 13834 (0.99x) | 9836 (1.40x) | 1971 (6.97x) | - | 5622 (2.45x) | - | >10x |
| global affine path float32 | **5325** | fill=striped: 5484 | 5225 (1.02x) | 5468 (0.97x) | - | - | - | - | - | - |
| global affine path float64 | **3917** | fill=striped: 4024 | 4083 (0.96x) | 3874 (1.01x) | - | - | - | - | 551 (7.11x) | - |
| global affine hard int32 | **13880** | fill=striped: 14434 | 15538 (0.89x) | 13892 (1.00x) | - | - | - | - | - | - |
| global affine hard float32 | **5236** | fill=striped: 5365 | 5611 (0.93x) | 5306 (0.99x) | - | - | - | - | - | - |
| global affine hard float64 | **4084** | fill=striped: 4323 | 4266 (0.96x) | 4132 (0.99x) | - | - | - | - | - | - |
| global affine soft float64 | **593** | fill=striped: 605 | 666 (0.89x) | 591 (1.00x) | - | - | - | - | - | - |
| global linear score int32 | **29742** | fill=striped: 29971 | 31573 (0.94x) | 29153 (1.02x) | 16412 (1.81x) | 7963 (3.74x) | - | 5823 (5.11x) | - | >10x |
| global linear score float32 | **14829** | fill=striped: 14890 | 14731 (1.01x) | 14344 (1.03x) | - | - | - | - | - | - |
| global linear score float64 (load!) | **10838** | schedule=sorted: 10532 | 10559 (1.03x) | 10679 (1.01x) | - | - | - | - | 3783 (2.86x) | - |
| global linear path int32 (load!) | **15821** | fill=striped: 15774 | 16196 (0.98x) | 15633 (1.01x) | 9330 (1.70x) | 1967 (8.04x) | - | 5701 (2.77x) | - | - |
| global linear path float32 (load!) | **12276** | fill=striped: 12882 | 13531 (0.91x) | 12845 (0.96x) | - | - | - | - | - | - |
| global linear path float64 (load!) | **9203** | fill=striped: 9296 | 9997 (0.92x) | 9400 (0.98x) | - | - | - | - | 1382 (6.66x) | - |
| global linear hard int32 | **16792** | fill=striped: 16591 | 16463 (1.02x) | 16241 (1.03x) | - | - | - | - | - | - |
| global linear hard float32 | **13550** | schedule=sorted: 13598 | 14692 (0.92x) | 13994 (0.97x) | - | - | - | - | - | - |
| global linear hard float64 | **9352** | fill=striped: 9818 | 11165 (0.84x) | 9676 (0.97x) | - | - | - | - | - | - |
| global linear soft float64 | **893** | fill=striped: 909 | 870 (1.03x) | 906 (0.98x) | - | - | - | - | - | - |
| local affine score int32 | **27162** | fill=striped: 29175 | 27364 (0.99x) | 25096 (1.08x) | 14087 (1.93x) | 7575 (3.59x) | - | - | - | >10x |
| local affine score float32 | **7888** | schedule=sorted: 7984 | 8425 (0.94x) | 8109 (0.97x) | - | - | - | - | - | - |
| local affine score float64 | **6601** | fill=striped: 6596 | 6278 (1.05x) | 6596 (1.00x) | - | - | - | - | 1401 (4.71x) | - |
| local affine path int32 | **10253** | schedule=sorted: 11111 | 11846 (0.87x) | 11590 (0.88x) | 8580 (1.20x) | 1969 (5.21x) | - | - | - | >10x |
| local affine path float32 | **5052** | fill=striped: 5657 | 5235 (0.97x) | 5305 (0.95x) | - | - | - | - | - | - |
| local affine path float64 | **3571** | fill=striped: 3907 | 3880 (0.92x) | 3731 (0.96x) | - | - | - | - | >10x | - |
| local affine hard int32 | **11032** | schedule=sorted: 10368 | 12612 (0.87x) | 11078 (1.00x) | - | - | - | - | - | - |
| local affine hard float32 | **5336** | schedule=sorted: 5200 | 5321 (1.00x) | 5301 (1.01x) | - | - | - | - | - | - |
| local affine hard float64 | **3869** | fill=striped: 3764 | 3920 (0.99x) | 3735 (1.04x) | - | - | - | - | - | - |
| local affine soft float64 | **278** | schedule=sorted: 279 | 428 (0.65x) | 279 (1.00x) | - | - | - | - | - | - |
| local linear score int32 | **29250** | fill=striped: 27298 | 28525 (1.03x) | 27522 (1.06x) | 15219 (1.92x) | 8045 (3.64x) | - | - | - | >10x |
| local linear score float32 | **15401** | fill=striped: 15484 | 16401 (0.94x) | 14765 (1.04x) | - | - | - | - | - | - |
| local linear score float64 | **10557** | schedule=sorted: 9975 | 10458 (1.01x) | 10220 (1.03x) | - | - | - | - | 2464 (4.28x) | - |
| local linear path int32 | **16507** | schedule=sorted: 16423 | 14810 (1.11x) | 16514 (1.00x) | 8367 (1.97x) | 1831 (9.02x) | - | - | - | - |
| local linear path float32 | **10417** | fill=striped: 11029 | 10531 (0.99x) | 11103 (0.94x) | - | - | - | - | - | - |
| local linear path float64 | **7336** | schedule=sorted: 8061 | 8524 (0.86x) | 7727 (0.95x) | - | - | - | - | >10x | - |
| local linear hard int32 | **17206** | schedule=sorted: 16729 | 15563 (1.11x) | 16703 (1.03x) | - | - | - | - | - | - |
| local linear hard float32 | **11259** | fill=striped: 11298 | 11237 (1.00x) | 11891 (0.95x) | - | - | - | - | - | - |
| local linear hard float64 | **8117** | fill=striped: 7862 | 8076 (1.01x) | 7944 (1.02x) | - | - | - | - | - | - |
| local linear soft float64 | **665** | schedule=sorted: 666 | 653 (1.02x) | 666 (1.00x) | - | - | - | - | - | - |

#### solace · protein_L3000 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **20012** | fill=striped: 22510 | 20927 (0.96x) | 18594 (1.08x) | 20646 (0.97x) | 3011 (6.65x) | - | 8283 (2.42x) | - | - |
| global affine score float32 | **8332** | schedule=sorted: 8420 | 8474 (0.98x) | 8055 (1.03x) | - | - | - | - | - | - |
| global affine score float64 | **6195** | fill=striped: 6645 | 6655 (0.93x) | 6641 (0.93x) | - | - | - | - | 2217 (2.79x) | - |
| global affine path int32 | **10207** | fill=striped: 11045 | 11579 (0.88x) | 10957 (0.93x) | 3317 (3.08x) | >10x | - | 6350 (1.61x) | - | >10x |
| global affine path float32 | **6662** | schedule=sorted: 6516 | 6895 (0.97x) | 6777 (0.98x) | - | - | - | - | - | - |
| global affine path float64 | **4978** | schedule=sorted: 5206 | 5333 (0.93x) | 5035 (0.99x) | - | - | - | - | >10x | - |
| global affine hard int32 | **9777** | fill=striped: 10101 | 10998 (0.89x) | 9715 (1.01x) | - | - | - | - | - | - |
| global affine hard float32 | **7024** | fill=striped: 6747 | 7147 (0.98x) | 6622 (1.06x) | - | - | - | - | - | - |
| global affine hard float64 | **5050** | fill=striped: 5141 | 5679 (0.89x) | 5132 (0.98x) | - | - | - | - | - | - |
| global affine soft float64 | **418** | fill=striped: 433 | 443 (0.94x) | 438 (0.95x) | - | - | - | - | - | - |
| global linear score int32 | **20075** | schedule=sorted: 18850 | 18728 (1.07x) | 17875 (1.12x) | 29954 (0.67x) | 2889 (6.95x) | - | 6554 (3.06x) | - | - |
| global linear score float32 | **19166** | fill=striped: 21812 | 19800 (0.97x) | 22280 (0.86x) | - | - | - | - | - | - |
| global linear score float64 | **11541** | schedule=sorted: 10986 | 11516 (1.00x) | 11529 (1.00x) | - | - | - | - | 3645 (3.17x) | - |
| global linear path int32 | **13550** | fill=striped: 15681 | 13487 (1.00x) | 14987 (0.90x) | 3052 (4.44x) | >10x | - | 6555 (2.07x) | - | >10x |
| global linear path float32 | **14798** | fill=striped: 15686 | 16075 (0.92x) | 14854 (1.00x) | - | - | - | - | - | - |
| global linear path float64 | **7719** | fill=striped: 8041 | 8160 (0.95x) | 7951 (0.97x) | - | - | - | - | 1275 (6.06x) | - |
| global linear hard int32 (load!) | **14849** | fill=striped: 15618 | 15507 (0.96x) | 15786 (0.94x) | - | - | - | - | - | - |
| global linear hard float32 (load!) | **16873** | schedule=sorted: 14843 | 14854 (1.14x) | 14538 (1.16x) | - | - | - | - | - | - |
| global linear hard float64 (load!) | **8304** | schedule=sorted: 8402 | 7240 (1.15x) | 7231 (1.15x) | - | - | - | - | - | - |
| global linear soft float64 | **1144** | schedule=sorted: 1137 | 1099 (1.04x) | 1124 (1.02x) | - | - | - | - | - | - |
| local affine score int32 | **19172** | fill=striped: 17156 | 19371 (0.99x) | 17211 (1.11x) | 15707 (1.22x) | >10x | - | - | - | - |
| local affine score float32 | **8951** | schedule=sorted: 9341 | 9465 (0.95x) | 9094 (0.98x) | - | - | - | - | - | - |
| local affine score float64 | **6986** | fill=striped: 7562 | 6706 (1.04x) | 6790 (1.03x) | - | - | - | - | 1510 (4.63x) | - |
| local affine path int32 | **7356** | schedule=sorted: 7484 | 7474 (0.98x) | 7298 (1.01x) | 2960 (2.49x) | >10x | - | - | - | >10x |
| local affine path float32 | **5406** | schedule=sorted: 5680 | 5510 (0.98x) | 5571 (0.97x) | - | - | - | - | - | - |
| local affine path float64 | **3565** | fill=striped: 3511 | 3330 (1.07x) | 3523 (1.01x) | - | - | - | - | >10x | - |
| local affine hard int32 | **7208** | fill=striped: 7439 | 7177 (1.00x) | 6805 (1.06x) | - | - | - | - | - | - |
| local affine hard float32 | **5444** | fill=striped: 5890 | 5458 (1.00x) | 6478 (0.84x) | - | - | - | - | - | - |
| local affine hard float64 | **3301** | schedule=sorted: 3755 | 3439 (0.96x) | 3603 (0.92x) | - | - | - | - | - | - |
| local affine soft float64 | **359** | fill=striped: 361 | 403 (0.89x) | 344 (1.04x) | - | - | - | - | - | - |
| local linear score int32 | **16432** | fill=striped: 18915 | 19548 (0.84x) | 16776 (0.98x) | 29939 (0.55x) | 2635 (6.24x) | - | - | - | - |
| local linear score float32 | **23616** | schedule=sorted: 21709 | 22225 (1.06x) | 19633 (1.20x) | - | - | - | - | - | - |
| local linear score float64 | **12082** | schedule=sorted: 11483 | 11808 (1.02x) | 11490 (1.05x) | - | - | - | - | 3157 (3.83x) | - |
| local linear path int32 | **11933** | schedule=sorted: 13973 | 14499 (0.82x) | 12160 (0.98x) | 2934 (4.07x) | >10x | - | - | - | - |
| local linear path float32 | **12958** | fill=striped: 12740 | 12708 (1.02x) | 12446 (1.04x) | - | - | - | - | - | - |
| local linear path float64 | **6724** | fill=striped: 6571 | 6267 (1.07x) | 6358 (1.06x) | - | - | - | - | >10x | - |
| local linear hard int32 | **14556** | fill=striped: 14790 | 12321 (1.18x) | 12085 (1.20x) | - | - | - | - | - | - |
| local linear hard float32 | **12381** | schedule=sorted: 12655 | 12918 (0.96x) | 12325 (1.00x) | - | - | - | - | - | - |
| local linear hard float64 | **6312** | fill=striped: 6830 | 6548 (0.96x) | 6240 (1.01x) | - | - | - | - | - | - |
| local linear soft float64 | **1063** | schedule=sorted: 1068 | 1040 (1.02x) | 1068 (1.00x) | - | - | - | - | - | - |

</details>

<details><summary><b>spot</b> — Apple M1, NEON — Fedora Asahi: every cell</summary>


#### spot · dna_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **4678** | - | 3687 (1.27x) | 4053 (1.15x) | 1577 (2.97x) | >10x | >10x | - | - | - | - | - | >10x |
| global affine score int32 | **2453** | - | 1928 (1.27x) | 2455 (1.00x) | 1152 (2.13x) | 417 (5.89x) | 609 (4.03x) | - | 524 (4.68x) | 947 (2.59x) | - | - | - |
| global affine score int64 | **945** | - | 890 (1.06x) | 947 (1.00x) | 385 (2.45x) | 415 (2.28x) | 605 (1.56x) | - | - | - | - | - | - |
| global affine score float32 | **1629** | fill=striped: 1062 | 1561 (1.04x) | 1631 (1.00x) | - | 212 (7.68x) | 219 (7.45x) | - | - | - | - | - | - |
| global affine score float64 | **827** | fill=rowwise: 928 | 797 (1.04x) | 827 (1.00x) | - | 212 (3.90x) | 219 (3.78x) | - | - | - | - | 183 (4.52x) | - |
| global affine path int16 | **2396** | - | 2054 (1.17x) | 2358 (1.02x) | 1238 (1.94x) | >10x | 265 (9.04x) | - | - | - | - | - | >10x |
| global affine path int32 | **1433** | - | 1204 (1.19x) | 1400 (1.02x) | 954 (1.50x) | 162 (8.82x) | 284 (5.04x) | - | 501 (2.86x) | 984 (1.46x) | - | - | - |
| global affine path int64 | **721** | - | 623 (1.16x) | 711 (1.01x) | 362 (1.99x) | 229 (3.14x) | 291 (2.48x) | - | - | - | - | - | - |
| global affine path float32 | **1246** | traceback=hirschberg_pmax: 1254 | 1115 (1.12x) | 1239 (1.01x) | - | 145 (8.61x) | 201 (6.19x) | - | - | - | - | - | - |
| global affine path float64 | **693** | traceback=scores: 702 | 602 (1.15x) | 690 (1.00x) | - | 144 (4.83x) | 201 (3.44x) | - | - | - | - | 110 (6.27x) | - |
| global affine hard int16 | **2814** | - | 2399 (1.17x) | 2797 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **1632** | - | 1366 (1.19x) | 1587 (1.03x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **776** | - | 659 (1.18x) | 775 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1383** | traceback=grid: 1401 | 1225 (1.13x) | 1370 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **741** | traceback=hirschberg: 741 | 632 (1.17x) | 741 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **125** | api=functions: 162 | 116 (1.07x) | 125 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **4225** | - | 4531 (0.93x) | 3658 (1.16x) | 2179 (1.94x) sat | >10x | >10x | - | - | - | - | - | >10x |
| global linear score int32 | **2193** | - | 2403 (0.91x) | 2195 (1.00x) | 1152 (1.90x) | 417 (5.26x) | 629 (3.48x) | - | 550 (3.99x) | 10158 (0.22x) | 4892 (0.45x) | - | - |
| global linear score int64 | **745** | - | 770 (0.97x) | 745 (1.00x) | 385 (1.93x) | 416 (1.79x) | 626 (1.19x) | - | - | - | - | - | - |
| global linear score float32 | **1818** | fill=striped: 1502 | 1755 (1.04x) | 1817 (1.00x) | - | 212 (8.56x) | 319 (5.69x) | - | - | - | - | - | - |
| global linear score float64 | **1055** | fill=striped: 1056 | 879 (1.20x) | 1053 (1.00x) | - | 212 (4.97x) | 319 (3.31x) | - | - | - | - | 524 (2.01x) | - |
| global linear path int16 | **3038** | - | 2562 (1.19x) | 2968 (1.02x) | 1306 (2.33x) sat | >10x | 649 (4.68x) | - | - | - | - | - | >10x |
| global linear path int32 | **2040** | - | 1697 (1.20x) | 2003 (1.02x) | 954 (2.14x) | >10x | 742 (2.75x) | - | 535 (3.81x) | 7161 (0.28x) | 1492 (1.37x) | - | - |
| global linear path int64 | **915** | - | 874 (1.05x) | 922 (0.99x) | 362 (2.52x) | 212 (4.30x) | 742 (1.23x) | - | - | - | - | - | - |
| global linear path float32 | **1400** | traceback=scores: 1401 | 1330 (1.05x) | 1390 (1.01x) | - | >10x | 235 (5.97x) | - | - | - | - | - | - |
| global linear path float64 | **704** | traceback=scores: 704 | 607 (1.16x) | 701 (1.00x) | - | 129 (5.47x) | 233 (3.01x) | - | - | - | - | 154 (4.56x) | - |
| global linear hard int16 | **4195** | - | 3444 (1.22x) | 3809 (1.10x) | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **2513** | - | 2048 (1.23x) | 2501 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **998** | - | 959 (1.04x) | 1000 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1627** | traceback=pointers: 1626 | 1548 (1.05x) | 1628 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **761** | fill=striped: 762 | 652 (1.17x) | 758 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **249** | api=functions: 239 | 203 (1.23x) | 250 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score int16 | **3883** | - | 3646 (1.06x) | 3717 (1.04x) | 1499 (2.59x) sat | >10x | >10x | 992 (3.91x) sat | - | - | - | - | >10x |
| local affine score int32 | **2010** | - | 1920 (1.05x) | 2011 (1.00x) | 973 (2.07x) | 210 (9.55x) | 412 (4.88x) | - | - | - | - | - | - |
| local affine score int64 | **871** | - | 830 (1.05x) | 872 (1.00x) | 369 (2.36x) | 178 (4.89x) | 410 (2.12x) | - | - | - | - | - | - |
| local affine score float32 | **1591** | fill=striped: 1483 | 1533 (1.04x) | 1590 (1.00x) | - | >10x | 166 (9.59x) | - | - | - | - | - | - |
| local affine score float64 | **807** | fill=striped: 1096 | 784 (1.03x) | 807 (1.00x) | - | 95 (8.46x) | 166 (4.86x) | - | - | - | - | 154 (5.23x) | - |
| local affine path int16 | **2094** | - | 1892 (1.11x) | 2088 (1.00x) | 1055 (1.98x) | >10x | >10x | 344 (6.10x) sat | - | - | - | - | >10x |
| local affine path int32 | **1249** | - | 1115 (1.12x) | 1245 (1.00x) | 753 (1.66x) | 172 (7.26x) | 162 (7.70x) | - | - | - | - | - | - |
| local affine path int64 | **545** | - | 524 (1.04x) | 553 (0.99x) | 294 (1.85x) | 219 (2.49x) | 164 (3.33x) | - | - | - | - | - | - |
| local affine path float32 | **954** | traceback=grid: 965 | 919 (1.04x) | 952 (1.00x) | - | >10x | 109 (8.73x) | - | - | - | - | - | - |
| local affine path float64 | **513** | traceback=scores: 516 | 492 (1.04x) | 516 (0.99x) | - | 95 (5.38x) | 110 (4.65x) | - | - | - | - | 67 (7.67x) | - |
| local affine hard int16 | **2495** | - | 2210 (1.13x) | 2473 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard int32 | **1406** | - | 1250 (1.12x) | 1386 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **582** | - | 545 (1.07x) | 582 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **1047** | traceback=hirschberg_pmax: 1047 | 982 (1.07x) | 1033 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **541** | traceback=hirschberg_pmax: 541 | 518 (1.05x) | 539 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **110** | api=functions: 151 | 107 (1.03x) | 109 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **3906** | - | 3503 (1.12x) | 3589 (1.09x) | 1274 (3.07x) sat | >10x | >10x | wrong | - | - | - | - | >10x |
| local linear score int32 | **2009** | - | 1885 (1.07x) | 2012 (1.00x) | 882 (2.28x) | 208 (9.64x) | 569 (3.53x) | - | - | - | - | - | - |
| local linear score int64 | **634** | - | 605 (1.05x) | 634 (1.00x) | 296 (2.14x) | 175 (3.63x) | 567 (1.12x) | - | - | - | - | - | - |
| local linear score float32 | **1157** | fill=striped: 1533 | 1117 (1.04x) | 1157 (1.00x) | - | >10x | 214 (5.41x) | - | - | - | - | - | - |
| local linear score float64 | **583** | fill=striped: 1106 | 566 (1.03x) | 583 (1.00x) | - | 95 (6.13x) | 214 (2.73x) | - | - | - | - | 331 (1.76x) | - |
| local linear path int16 | **2304** | - | 2119 (1.09x) | 2324 (0.99x) | 1142 (2.02x) sat | >10x | >10x | wrong | - | - | - | - | 264 (8.73x) sat |
| local linear path int32 | **1465** | - | 1361 (1.08x) | 1468 (1.00x) | 735 (1.99x) | 153 (9.59x) | 186 (7.89x) | - | - | - | - | - | - |
| local linear path int64 | **563** | - | 527 (1.07x) | 565 (1.00x) | 292 (1.93x) | 191 (2.95x) | 185 (3.04x) | - | - | - | - | - | - |
| local linear path float32 | **952** | traceback=scores: 953 | 914 (1.04x) | 951 (1.00x) | - | >10x | 133 (7.14x) | - | - | - | - | - | - |
| local linear path float64 | **520** | fill=striped: 539 | 505 (1.03x) | 520 (1.00x) | - | 96 (5.43x) | 135 (3.86x) | - | - | - | - | 65 (8.05x) | - |
| local linear hard int16 | **2961** | - | 2761 (1.07x) | 2996 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **1683** | - | 1592 (1.06x) | 1699 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **594** | - | 555 (1.07x) | 596 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1044** | traceback=scores: 1047 | 1016 (1.03x) | 1047 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **546** | fill=striped: 571 | 536 (1.02x) | 548 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **218** | api=functions: 223 | 186 (1.17x) | 217 (1.00x) | - | - | - | - | - | - | - | - | - |

#### spot · dna_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| global affine score int32 | **2219** | - | 2589 (0.86x) | 2217 (1.00x) | 1378 (1.61x) | 415 (5.35x) | 596 (3.72x) | - | 744 (2.98x) | 3335 (0.67x) | - | - | - |
| global affine score int64 | **769** | - | 783 (0.98x) | 770 (1.00x) | 407 (1.89x) | 412 (1.87x) | 592 (1.30x) | - | - | - | - | - | - |
| global affine score float32 | **1816** | fill=striped: 1817 | 1558 (1.17x) | 1817 (1.00x) | - | 200 (9.07x) | 201 (9.05x) | - | - | - | - | - | - |
| global affine score float64 | **1234** | fill=rowwise: 1235 | 1016 (1.21x) | 1235 (1.00x) | - | 200 (6.17x) | 200 (6.16x) | - | - | - | - | 187 (6.61x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| global affine path int32 | **868** | - | 1054 (0.82x) | 868 (1.00x) | 1049 (0.83x) | 171 (5.07x) | 401 (2.17x) | - | 690 (1.26x) | 2713 (0.32x) | - | - | - |
| global affine path int64 | **387** | - | 394 (0.98x) | 387 (1.00x) | 388 (1.00x) | 276 (1.40x) | 412 (0.94x) | - | - | - | - | - | - |
| global affine path float32 | **1329** | traceback=grid: 1330 | 1177 (1.13x) | 1329 (1.00x) | - | 172 (7.71x) | 261 (5.09x) | - | - | - | - | - | - |
| global affine path float64 | **984** | traceback=grid: 985 | 837 (1.18x) | 985 (1.00x) | - | 171 (5.76x) | 255 (3.86x) | - | - | - | - | 133 (7.42x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| global affine hard int32 | **870** | - | 1057 (0.82x) | 870 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **388** | - | 395 (0.98x) | 388 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1332** | fill=striped: 1335 | 1182 (1.13x) | 1334 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **988** | fill=rowwise: 988 | 838 (1.18x) | 986 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **133** | fill=rowwise: 133 | 133 (1.00x) | 133 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **4391** | - | 5138 (0.85x) | 3630 (1.21x) | 3089 (1.42x) | >10x | >10x | - | - | - | - | - | >10x |
| global linear score int32 | **2218** | - | 2589 (0.86x) | 2217 (1.00x) | 1744 (1.27x) | 415 (5.35x) | 597 (3.72x) | - | 742 (2.99x) | 27703 (0.08x) | 27314 (0.08x) | - | - |
| global linear score int64 | **904** | - | 938 (0.96x) | 904 (1.00x) | 407 (2.22x) | 412 (2.19x) | 596 (1.52x) | - | - | - | - | - | - |
| global linear score float32 | **2711** | fill=striped: 2711 | 2209 (1.23x) | 2714 (1.00x) | - | >10x | 275 (9.84x) | - | - | - | - | - | - |
| global linear score float64 | **1421** | fill=striped: 1421 | 1148 (1.24x) | 1422 (1.00x) | - | 200 (7.10x) | 275 (5.16x) | - | - | - | - | 438 (3.24x) | - |
| global linear path int16 | **3404** | - | 3351 (1.02x) | 3287 (1.04x) | 1907 (1.78x) | >10x | 752 (4.53x) | - | - | - | - | - | >10x |
| global linear path int32 | **1455** | - | 1616 (0.90x) | 1458 (1.00x) | 1049 (1.39x) | 167 (8.71x) | 906 (1.61x) | - | 684 (2.13x) | 23339 (0.06x) | 11925 (0.12x) | - | - |
| global linear path int64 | **569** | - | 567 (1.00x) | 556 (1.02x) | 389 (1.46x) | 221 (2.57x) | 904 (0.63x) | - | - | - | - | - | - |
| global linear path float32 | **1283** | traceback=scores: 1712 | 1214 (1.06x) | 1284 (1.00x) | - | 131 (9.77x) | 265 (4.83x) | - | - | - | - | - | - |
| global linear path float64 | **726** | traceback=scores: 908 | 630 (1.15x) | 727 (1.00x) | - | 132 (5.48x) | 265 (2.74x) | - | - | - | - | 185 (3.92x) | - |
| global linear hard int16 | **3507** | - | 3454 (1.02x) | 3420 (1.03x) | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **1473** | - | 1640 (0.90x) | 1473 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **571** | - | 569 (1.00x) | 571 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1295** | traceback=scores: 1742 | 1227 (1.06x) | 1295 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **730** | traceback=scores: 916 | 633 (1.15x) | 729 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **175** | api=functions: 212 | 175 (1.00x) | 175 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| local affine score int32 | **2016** | - | 2146 (0.94x) | 2019 (1.00x) | 1313 (1.54x) | 216 (9.35x) | 408 (4.94x) | - | - | - | - | - | - |
| local affine score int64 | **864** | - | 892 (0.97x) | 865 (1.00x) | 401 (2.15x) | 180 (4.80x) | 404 (2.14x) | - | - | - | - | - | - |
| local affine score float32 | **2458** | fill=striped: 2458 | 1991 (1.23x) | 2457 (1.00x) | - | >10x | >10x | - | - | - | - | - | - |
| local affine score float64 | **1406** | fill=striped: 1406 | 1111 (1.27x) | 1406 (1.00x) | - | >10x | 158 (8.92x) | - | - | - | - | 164 (8.55x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| local affine path int32 | **731** | - | 725 (1.01x) | 722 (1.01x) | 1037 (0.70x) | 186 (3.94x) | 179 (4.08x) | - | - | - | - | - | - |
| local affine path int64 | **309** | - | 314 (0.98x) | 309 (1.00x) | 324 (0.95x) | 247 (1.25x) | 182 (1.69x) | - | - | - | - | - | - |
| local affine path float32 | **678** | traceback=scores: 982 | 647 (1.05x) | 678 (1.00x) | - | 106 (6.42x) | 126 (5.40x) | - | - | - | - | - | - |
| local affine path float64 | **385** | traceback=scores: 512 | 321 (1.20x) | 385 (1.00x) | - | 105 (3.67x) | 126 (3.05x) | - | - | - | - | 76 (5.06x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | |
| local affine hard int32 | **737** | - | 728 (1.01x) | 727 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **309** | - | 315 (0.98x) | 310 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **681** | traceback=scores: 995 | 651 (1.05x) | 681 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **386** | traceback=scores: 517 | 322 (1.20x) | 386 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **104** | api=functions: 121 | 104 (1.00x) | 104 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **3972** | - | 4297 (0.92x) | 3630 (1.09x) | 3143 (1.26x) | >10x | >10x | 2237 (1.78x) sat | - | - | - | - | >10x |
| local linear score int32 | **2016** | - | 2146 (0.94x) | 2020 (1.00x) | 1421 (1.42x) | 210 (9.60x) | 557 (3.62x) | - | - | - | - | - | - |
| local linear score int64 | **881** | - | 912 (0.97x) | 882 (1.00x) | 405 (2.17x) | 175 (5.03x) | 557 (1.58x) | - | - | - | - | - | - |
| local linear score float32 | **2710** | fill=striped: 2711 | 2163 (1.25x) | 2709 (1.00x) | - | >10x | >10x | - | - | - | - | - | - |
| local linear score float64 | **1430** | fill=striped: 1429 | 1127 (1.27x) | 1430 (1.00x) | - | >10x | 200 (7.13x) | - | - | - | - | 415 (3.45x) | - |
| local linear path int16 | **2513** | - | 2520 (1.00x) | 2500 (1.01x) | 1957 (1.28x) | >10x | >10x | 591 (4.25x) sat | - | - | - | - | >10x |
| local linear path int32 | **1094** | - | 1179 (0.93x) | 1099 (0.99x) | 1189 (0.92x) | 155 (7.03x) | 204 (5.37x) | - | - | - | - | - | - |
| local linear path int64 | **407** | - | 395 (1.03x) | 407 (1.00x) | 328 (1.24x) | 197 (2.07x) | 204 (2.00x) | - | - | - | - | - | - |
| local linear path float32 | **952** | traceback=scores: 1126 | 827 (1.15x) | 935 (1.02x) | - | 98 (9.72x) | 146 (6.53x) | - | - | - | - | - | - |
| local linear path float64 | **518** | traceback=scores: 622 | 425 (1.22x) | 518 (1.00x) | - | 98 (5.28x) | 148 (3.49x) | - | - | - | - | 68 (7.65x) | - |
| local linear hard int16 | **2566** | - | 2582 (0.99x) | 2551 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **1104** | - | 1193 (0.93x) | 1107 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **409** | - | 396 (1.03x) | 409 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **960** | traceback=scores: 1138 | 833 (1.15x) | 941 (1.02x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **520** | traceback=scores: 626 | 426 (1.22x) | 520 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **149** | api=functions: 193 | 149 (1.00x) | 149 (1.00x) | - | - | - | - | - | - | - | - | - |

#### spot · dna_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global affine score int32 | **1939** | - | 1573 (1.23x) | 1984 (0.98x) | 1583 (1.22x) | 415 (4.67x) | 593 (3.27x) | 778 (2.49x) | 5043 (0.38x) | - | - | - |
| global affine score int64 | **867** | - | 881 (0.98x) | 867 (1.00x) | 392 (2.21x) | 408 (2.12x) | 591 (1.47x) | - | - | - | - | - |
| global affine score float32 | **2037** | fill=striped: 2037 | 1710 (1.19x) | 2043 (1.00x) | - | >10x | >10x | - | - | - | - | - |
| global affine score float64 | **1003** | fill=striped: 1007 | 970 (1.03x) | 1000 (1.00x) | - | 199 (5.04x) | 199 (5.05x) | - | - | - | 186 (5.40x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global affine path int32 | **709** | - | 735 (0.96x) | 710 (1.00x) | 880 (0.81x) | 165 (4.29x) | 302 (2.35x) | 706 (1.00x) | 2263 (0.31x) | - | - | - |
| global affine path int64 | **273** | - | 274 (0.99x) | 273 (1.00x) | 352 (0.78x) | 250 (1.09x) | 309 (0.88x) | - | - | - | - | - |
| global affine path float32 | **2148** | fill=striped: 2146 | 1833 (1.17x) | 2141 (1.00x) | - | >10x | >10x | - | - | - | - | - |
| global affine path float64 | **1267** | traceback=grid: 1268 | 1055 (1.20x) | 1269 (1.00x) | - | 160 (7.94x) | 215 (5.89x) | - | - | - | 128 (9.90x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global affine hard int32 | **710** | - | 732 (0.97x) | 635 (1.12x) | - | - | - | - | - | - | - | - |
| global affine hard int64 | **257** | - | 276 (0.93x) | 271 (0.95x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **2148** | fill=striped: 2146 | 1814 (1.18x) | 2148 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **1272** | traceback=grid: 1272 | 1055 (1.21x) | 1266 (1.00x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **123** | api=functions: 124 | 115 (1.07x) | 123 (1.00x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global linear score int32 | **1979** | - | 1538 (1.29x) | 1981 (1.00x) | 1762 (1.12x) | 415 (4.77x) | 593 (3.34x) | 782 (2.53x) | 41498 (0.05x) | 108625 (0.02x) | - | - |
| global linear score int64 | **920** | - | 941 (0.98x) | 920 (1.00x) | 396 (2.32x) | 408 (2.25x) | 592 (1.56x) | - | - | - | - | - |
| global linear score float32 | **2336** | api=functions: 2339 | 2046 (1.14x) | 2334 (1.00x) | - | >10x | 272 (8.60x) | - | - | - | - | - |
| global linear score float64 | **1078** | fill=striped: 1094 | 1038 (1.04x) | 1075 (1.00x) | - | 199 (5.42x) | 271 (3.97x) | - | - | - | 427 (2.52x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global linear path int32 | **1015** | - | 1034 (0.98x) | 1020 (0.99x) | 870 (1.17x) | 165 (6.16x) | 911 (1.11x) | 721 (1.41x) | 27000 (0.04x) | 34389 (0.03x) | - | - |
| global linear path int64 | **546** | - | 548 (1.00x) | 543 (1.01x) | 350 (1.56x) | 208 (2.63x) | 904 (0.60x) | - | - | - | - | - |
| global linear path float32 | **992** | traceback=scores: 1807 | 934 (1.06x) | 1017 (0.98x) | - | 124 (7.97x) | 214 (4.63x) | - | - | - | - | - |
| global linear path float64 | **474** | traceback=scores: 838 | 444 (1.07x) | 475 (1.00x) | - | 123 (3.87x) | 211 (2.25x) | - | - | - | 144 (3.28x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| global linear hard int32 | **1012** | - | 1030 (0.98x) | 1020 (0.99x) | - | - | - | - | - | - | - | - |
| global linear hard int64 | **547** | - | 559 (0.98x) | 546 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1014** | traceback=scores: 1816 | 935 (1.09x) | 1009 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **475** | traceback=scores: 838 | 445 (1.07x) | 475 (1.00x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **147** | api=functions: 201 | 153 (0.96x) | 147 (1.00x) | - | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local affine score int32 | **1843** | - | 1852 (1.00x) | 1848 (1.00x) | 1400 (1.32x) | 217 (8.51x) | 406 (4.54x) | - | - | - | - | - |
| local affine score int64 | **886** | - | 914 (0.97x) | 886 (1.00x) | 406 (2.18x) | 177 (5.00x) | 404 (2.19x) | - | - | - | - | - |
| local affine score float32 | **2015** | fill=striped: 2015 | 1893 (1.06x) | 2018 (1.00x) | - | >10x | >10x | - | - | - | - | - |
| local affine score float64 | **1054** | api=functions: 1065 | 992 (1.06x) | 1050 (1.00x) | - | >10x | 157 (6.72x) | - | - | - | 159 (6.62x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local affine path int32 | **293** | - | 296 (0.99x) | 292 (1.00x) | 999 (0.29x) | 185 (1.58x) | 185 (1.58x) | - | - | - | - | - |
| local affine path int64 | **114** | - | 114 (1.00x) | 114 (1.00x) | 314 (0.36x) | 239 (0.48x) | 189 (0.60x) | - | - | - | - | - |
| local affine path float32 | **283** | traceback=scores: 854 | 231 (1.22x) | 279 (1.01x) | - | 106 (2.67x) | 128 (2.20x) | - | - | - | - | - |
| local affine path float64 | **348** | traceback=scores: 439 | 311 (1.12x) | 348 (1.00x) | - | 105 (3.32x) | 128 (2.71x) | - | - | - | 74 (4.68x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local affine hard int32 | **291** | - | 296 (0.98x) | 293 (0.99x) | - | - | - | - | - | - | - | - |
| local affine hard int64 | **113** | - | 114 (0.98x) | 114 (0.99x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **283** | traceback=scores: 848 | 232 (1.22x) | 279 (1.02x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **349** | traceback=scores: 443 | 311 (1.12x) | 348 (1.00x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **94** | api=functions: 117 | 90 (1.05x) | 94 (0.99x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local linear score int32 | **1843** | - | 1832 (1.01x) | 1849 (1.00x) | 1440 (1.28x) | 217 (8.51x) | 556 (3.32x) | - | - | - | - | - |
| local linear score int64 | **896** | - | 924 (0.97x) | 896 (1.00x) | 407 (2.20x) | 172 (5.20x) | 553 (1.62x) | - | - | - | - | - |
| local linear score float32 | **2182** | fill=striped: 2184 | 2000 (1.09x) | 2220 (0.98x) | - | >10x | >10x | - | - | - | - | - |
| local linear score float64 | **1075** | fill=striped: 1087 | 1000 (1.08x) | 1080 (1.00x) | - | >10x | 199 (5.40x) | - | - | - | 422 (2.55x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local linear path int32 | **833** | - | 846 (0.99x) | 833 (1.00x) | 1111 (0.75x) | 156 (5.35x) | 203 (4.11x) | - | - | - | - | - |
| local linear path int64 | **410** | - | 392 (1.05x) | 409 (1.00x) | 317 (1.29x) | 191 (2.15x) | 203 (2.02x) | - | - | - | - | - |
| local linear path float32 | **757** | traceback=scores: 1155 | 668 (1.13x) | 741 (1.02x) | - | 97 (7.79x) | 143 (5.28x) | - | - | - | - | - |
| local linear path float64 | **378** | traceback=scores: 588 | 334 (1.13x) | 379 (1.00x) | - | 96 (3.93x) | 145 (2.60x) | - | - | - | 66 (5.69x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | |
| local linear hard int32 | **832** | - | 842 (0.99x) | 835 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard int64 | **409** | - | 389 (1.05x) | 410 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **757** | traceback=scores: 1156 | 667 (1.13x) | 740 (1.02x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **379** | traceback=scores: 588 | 334 (1.14x) | 378 (1.00x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **128** | api=functions: 183 | 133 (0.97x) | 128 (1.00x) | - | - | - | - | - | - | - | - |

#### spot · mirna · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **3969** | - | 3006 (1.32x) | 3730 (1.06x) | 1278 (3.11x) | >10x | >10x | - | - | - | - | - | >10x |
| global affine score int32 | **2200** | - | 1689 (1.30x) | 2209 (1.00x) | 811 (2.71x) | 430 (5.12x) | 616 (3.57x) | - | 345 (6.38x) | >10x | - | - | >10x |
| global affine score int64 | **952** | - | 842 (1.13x) | 958 (0.99x) | 317 (3.01x) | 424 (2.24x) | 590 (1.61x) | - | - | - | - | - | - |
| global affine score float32 | **1619** | fill=striped: 704 | 1437 (1.13x) | 1621 (1.00x) | - | 231 (7.01x) | 240 (6.76x) | - | - | - | - | - | - |
| global affine score float64 | **816** | fill=striped: 669 | 737 (1.11x) | 815 (1.00x) | - | 231 (3.53x) | 238 (3.43x) | - | - | - | - | 137 (5.95x) | - |
| global affine path int16 | **937** | - | 872 (1.07x) | 912 (1.03x) | 638 (1.47x) | 160 (5.87x) | 146 (6.41x) | - | - | - | - | - | 113 (8.29x) sat |
| global affine path int32 | **723** | - | 683 (1.06x) | 745 (0.97x) | 547 (1.32x) | 147 (4.92x) | 156 (4.64x) | - | 290 (2.50x) | >10x | - | - | - |
| global affine path int64 | **489** | - | 434 (1.13x) | 482 (1.02x) | 275 (1.78x) | 175 (2.79x) | 154 (3.18x) | - | - | - | - | - | - |
| global affine path float32 | **681** | traceback=grid: 684 | 601 (1.13x) | 672 (1.01x) | - | 95 (7.16x) | 110 (6.17x) | - | - | - | - | - | - |
| global affine path float64 | **466** | traceback=pointers: 468 | 411 (1.14x) | 462 (1.01x) | - | 95 (4.91x) | 110 (4.25x) | - | - | - | - | 55 (8.47x) | - |
| global affine hard int16 | **1661** | - | 1500 (1.11x) | 1637 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **1151** | - | 1034 (1.11x) | 1133 (1.02x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **633** | - | 553 (1.14x) | 628 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1010** | traceback=hirschberg_pmax: 1012 | 847 (1.19x) | 998 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **604** | traceback=hirschberg: 605 | 510 (1.18x) | 600 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **177** | api=functions: 126 | 159 (1.11x) | 177 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | **4061** | - | 3589 (1.13x) | 3808 (1.07x) | 1532 (2.65x) sat | >10x | >10x | - | - | - | - | - | >10x |
| global linear score int32 | **2220** | - | 2065 (1.07x) | 2252 (0.99x) | 810 (2.74x) | 429 (5.17x) | 663 (3.35x) | - | 395 (5.61x) | 482 (4.61x) | 959 (2.31x) | - | >10x |
| global linear score int64 | **573** | - | 593 (0.97x) | 574 (1.00x) | 348 (1.65x) | 423 (1.35x) | 637 (0.90x) | - | - | - | - | - | - |
| global linear score float32 | **1901** | fill=striped: 827 | 1720 (1.10x) | 1919 (0.99x) | - | 231 (8.22x) | 384 (4.95x) | - | - | - | - | - | - |
| global linear score float64 | **740** | fill=striped: 740 | 642 (1.15x) | 741 (1.00x) | - | 231 (3.21x) | 384 (1.93x) | - | - | - | - | 296 (2.50x) | - |
| global linear path int16 | **753** | - | 647 (1.16x) | 730 (1.03x) | 613 (1.23x) | 160 (4.72x) | 367 (2.05x) | - | - | - | - | - | 116 (6.47x) sat |
| global linear path int32 | **692** | - | 587 (1.18x) | 661 (1.05x) | 529 (1.31x) | 145 (4.75x) | 401 (1.72x) | - | 323 (2.14x) | 485 (1.43x) | 262 (2.64x) | - | - |
| global linear path int64 | **520** | - | 479 (1.09x) | 519 (1.00x) | 271 (1.92x) | 175 (2.97x) | 398 (1.31x) | - | - | - | - | - | - |
| global linear path float32 | **586** | traceback=scores: 587 | 517 (1.13x) | 564 (1.04x) | - | 107 (5.48x) | 164 (3.57x) | - | - | - | - | - | - |
| global linear path float64 | **387** | fill=striped: 387 | 363 (1.07x) | 383 (1.01x) | - | 107 (3.61x) | 162 (2.38x) | - | - | - | - | 87 (4.44x) | - |
| global linear hard int16 | **1634** | - | 1499 (1.09x) | 1631 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **1336** | - | 1141 (1.17x) | 1317 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **799** | - | 806 (0.99x) | 796 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1048** | traceback=hirschberg: 1052 | 957 (1.10x) | 1041 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **539** | fill=striped: 540 | 543 (0.99x) | 537 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **206** | api=functions: 191 | 175 (1.18x) | 205 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine score int16 | **3451** | - | 3044 (1.13x) | 3427 (1.01x) | 1213 (2.84x) sat | >10x | >10x | 528 (6.54x) sat | - | - | - | - | >10x |
| local affine score int32 | **1856** | - | 1708 (1.09x) | 1861 (1.00x) | 563 (3.30x) | 207 (8.97x) | 415 (4.47x) | - | - | - | - | - | >10x |
| local affine score int64 | **805** | - | 744 (1.08x) | 807 (1.00x) | 299 (2.69x) | 178 (4.52x) | 401 (2.01x) | - | - | - | - | - | - |
| local affine score float32 | **1504** | fill=striped: 1003 | 1354 (1.11x) | 1506 (1.00x) | - | >10x | 175 (8.60x) | - | - | - | - | - | - |
| local affine score float64 | **772** | fill=striped: 889 | 714 (1.08x) | 774 (1.00x) | - | 94 (8.18x) | 174 (4.44x) | - | - | - | - | 114 (6.80x) | - |
| local affine path int16 | **1942** | - | 1755 (1.11x) | 1899 (1.02x) | 652 (2.98x) | >10x | >10x | 393 (4.94x) sat | - | - | - | - | >10x |
| local affine path int32 | **1170** | - | 1050 (1.11x) | 1155 (1.01x) | 516 (2.27x) | 165 (7.08x) | 132 (8.89x) | - | - | - | - | - | - |
| local affine path int64 | **533** | - | 506 (1.05x) | 530 (1.01x) | 237 (2.24x) | 230 (2.31x) | 131 (4.06x) | - | - | - | - | - | - |
| local affine path float32 | **924** | traceback=scores: 924 | 864 (1.07x) | 914 (1.01x) | - | >10x | 99 (9.35x) | - | - | - | - | - | - |
| local affine path float64 | **496** | traceback=grid: 497 | 476 (1.04x) | 494 (1.01x) | - | 86 (5.79x) | 99 (5.00x) | - | - | - | - | 62 (7.97x) | - |
| local affine hard int16 | **2209** | - | 1909 (1.16x) | 2134 (1.04x) | - | - | - | - | - | - | - | - | - |
| local affine hard int32 | **1262** | - | 1144 (1.10x) | 1239 (1.02x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **544** | - | 519 (1.05x) | 548 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **958** | traceback=hirschberg: 981 | 914 (1.05x) | 968 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **514** | traceback=grid: 514 | 492 (1.04x) | 511 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **163** | api=functions: 123 | 153 (1.07x) | 162 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **3757** | - | 3147 (1.19x) | 3609 (1.04x) | 885 (4.25x) | >10x | >10x | wrong | - | - | - | - | >10x |
| local linear score int32 | **2034** | - | 1793 (1.13x) | 2039 (1.00x) | 561 (3.62x) | >10x | 571 (3.56x) | - | - | - | - | - | >10x |
| local linear score int64 | **704** | - | 623 (1.13x) | 707 (1.00x) | 249 (2.83x) | 174 (4.05x) | 556 (1.27x) | - | - | - | - | - | - |
| local linear score float32 | **1244** | fill=striped: 974 | 1138 (1.09x) | 1246 (1.00x) | - | >10x | 229 (5.42x) | - | - | - | - | - | - |
| local linear score float64 | **641** | fill=striped: 880 | 589 (1.09x) | 642 (1.00x) | - | 94 (6.81x) | 229 (2.80x) | - | - | - | - | 181 (3.54x) | - |
| local linear path int16 | **1650** | - | 1397 (1.18x) | 1559 (1.06x) | 619 (2.66x) | >10x | >10x | wrong | - | - | - | - | >10x |
| local linear path int32 | **1176** | - | 1073 (1.10x) | 1142 (1.03x) | 497 (2.36x) | 147 (8.01x) | 163 (7.21x) | - | - | - | - | - | - |
| local linear path int64 | **557** | - | 483 (1.15x) | 551 (1.01x) | 228 (2.44x) | 191 (2.91x) | 163 (3.42x) | - | - | - | - | - | - |
| local linear path float32 | **845** | traceback=pointers: 846 | 777 (1.09x) | 819 (1.03x) | - | 87 (9.68x) | 119 (7.12x) | - | - | - | - | - | - |
| local linear path float64 | **498** | traceback=scores: 499 | 474 (1.05x) | 493 (1.01x) | - | 87 (5.72x) | 119 (4.17x) | - | - | - | - | 54 (9.26x) | - |
| local linear hard int16 | **2315** | - | 2037 (1.14x) | 2281 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **1475** | - | 1334 (1.11x) | 1467 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **618** | - | 528 (1.17x) | 617 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1000** | traceback=hirschberg: 1000 | 914 (1.09x) | 990 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **545** | traceback=scores: 547 | 520 (1.05x) | 547 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **198** | api=functions: 178 | 166 (1.19x) | 198 (1.00x) | - | - | - | - | - | - | - | - | - |

#### spot · protein_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **2368** | - | 2714 (0.87x) | 2368 (1.00x) | 1586 (1.49x) | 301 (7.88x) | 332 (7.14x) | - | - | - | >10x |
| global affine score int32 | **1840** | - | 2033 (0.91x) | 1840 (1.00x) | 977 (1.88x) | 417 (4.42x) | 603 (3.05x) | - | 507 (3.63x) | - | - |
| global affine score int64 | **546** | - | 555 (0.98x) | 548 (1.00x) | 362 (1.51x) | 416 (1.31x) | 600 (0.91x) | - | - | - | - |
| global affine score float32 | **880** | fill=striped: 875 | 841 (1.05x) | 881 (1.00x) | - | 212 (4.15x) | 218 (4.03x) | - | - | - | - |
| global affine score float64 | **819** | fill=rowwise: 812 | 707 (1.16x) | 818 (1.00x) | - | 212 (3.86x) | 218 (3.76x) | - | - | 183 (4.47x) | - |
| global affine path int16 | **1346** | - | 1033 (1.30x) | 1354 (0.99x) | 1066 (1.26x) | 195 (6.89x) | 257 (5.24x) | - | - | - | 136 (9.87x) sat |
| global affine path int32 | **1081** | - | 857 (1.26x) | 1025 (1.05x) | 821 (1.32x) | 166 (6.52x) | 272 (3.98x) | - | 494 (2.19x) | - | - |
| global affine path int64 | **623** | - | 549 (1.13x) | 625 (1.00x) | 339 (1.84x) | 233 (2.67x) | 272 (2.29x) | - | - | - | - |
| global affine path float32 | **875** | traceback=hirschberg_pmax: 874 | 765 (1.14x) | 870 (1.01x) | - | 133 (6.57x) | 190 (4.60x) | - | - | - | - |
| global affine path float64 | **575** | traceback=scores: 572 | 521 (1.10x) | 569 (1.01x) | - | 132 (4.34x) | 190 (3.02x) | - | - | 105 (5.46x) | - |
| global affine hard int16 | **1416** | - | 1099 (1.29x) | 1385 (1.02x) | - | - | - | - | - | - | - |
| global affine hard int32 | **1121** | - | 889 (1.26x) | 1118 (1.00x) | - | - | - | - | - | - | - |
| global affine hard int64 | **640** | - | 565 (1.13x) | 640 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float32 | **899** | traceback=grid: 900 | 792 (1.14x) | 898 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float64 | **584** | traceback=hirschberg: 584 | 524 (1.11x) | 586 (1.00x) | - | - | - | - | - | - | - |
| global affine soft float64 | **129** | api=functions: 178 | 124 (1.04x) | 131 (0.98x) | - | - | - | - | - | - | - |
| global linear score int16 | **2357** | - | 2694 (0.88x) | 2357 (1.00x) | 1586 (1.49x) | 301 (7.84x) | 362 (6.52x) | - | - | - | 240 (9.82x) sat |
| global linear score int32 | **1814** | - | 2025 (0.90x) | 1810 (1.00x) | 978 (1.86x) | 417 (4.35x) | 637 (2.85x) | - | 523 (3.47x) | - | - |
| global linear score int64 | **661** | - | 693 (0.95x) | 660 (1.00x) | 362 (1.83x) | 416 (1.59x) | 634 (1.04x) | - | - | - | - |
| global linear score float32 | **1180** | fill=striped: 1169 | 1093 (1.08x) | 1181 (1.00x) | - | 212 (5.55x) | 321 (3.68x) | - | - | - | - |
| global linear score float64 | **925** | fill=striped: 925 | 803 (1.15x) | 923 (1.00x) | - | 212 (4.36x) | 320 (2.89x) | - | - | 524 (1.77x) | - |
| global linear path int16 | **1475** | - | 1545 (0.95x) | 1470 (1.00x) | 1064 (1.39x) | 213 (6.91x) | 732 (2.01x) | - | - | - | 173 (8.51x) sat |
| global linear path int32 | **1203** | - | 1253 (0.96x) | 1206 (1.00x) | 820 (1.47x) | 167 (7.22x) | 858 (1.40x) | - | 500 (2.40x) | - | - |
| global linear path int64 | **911** | - | 872 (1.04x) | 918 (0.99x) | 339 (2.69x) | 254 (3.59x) | 852 (1.07x) | - | - | - | - |
| global linear path float32 | **881** | traceback=scores: 883 | 781 (1.13x) | 877 (1.01x) | - | 149 (5.91x) | 325 (2.71x) | - | - | - | - |
| global linear path float64 | **630** | traceback=scores: 630 | 564 (1.12x) | 627 (1.00x) | - | 149 (4.23x) | 324 (1.94x) | - | - | 167 (3.76x) | - |
| global linear hard int16 | **1673** | - | 1783 (0.94x) | 1672 (1.00x) | - | - | - | - | - | - | - |
| global linear hard int32 | **1328** | - | 1404 (0.95x) | 1337 (0.99x) | - | - | - | - | - | - | - |
| global linear hard int64 | **968** | - | 931 (1.04x) | 970 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float32 | **932** | traceback=pointers: 932 | 826 (1.13x) | 936 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float64 | **661** | traceback=scores: 662 | 592 (1.12x) | 660 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 | **213** | api=functions: 282 | 163 (1.30x) | 213 (1.00x) | - | - | - | - | - | - | - |
| local affine score int16 | **2254** | - | 2593 (0.87x) | 2254 (1.00x) | 1109 (2.03x) | >10x | 244 (9.22x) | 372 (6.06x) sat | - | - | >10x |
| local affine score int32 | **1669** | - | 1871 (0.89x) | 1669 (1.00x) | 764 (2.19x) | 212 (7.86x) | 408 (4.09x) | - | - | - | >10x |
| local affine score int64 | **586** | - | 603 (0.97x) | 585 (1.00x) | 298 (1.97x) | 355 (1.65x) | 404 (1.45x) | - | - | - | - |
| local affine score float32 | **1007** | fill=striped: 998 | 924 (1.09x) | 1007 (1.00x) | - | >10x | 166 (6.07x) | - | - | - | - |
| local affine score float64 | **869** | fill=rowwise: 859 | 742 (1.17x) | 870 (1.00x) | - | 95 (9.13x) | 166 (5.25x) | - | - | 157 (5.54x) | - |
| local affine path int16 | **1231** | - | 1010 (1.22x) | 1196 (1.03x) | 898 (1.37x) | 156 (7.89x) | 217 (5.67x) | 193 (6.39x) sat | - | - | >10x |
| local affine path int32 | **938** | - | 805 (1.17x) | 932 (1.01x) | 640 (1.47x) | 171 (5.47x) | 230 (4.08x) | - | - | - | - |
| local affine path int64 | **495** | - | 457 (1.08x) | 496 (1.00x) | 256 (1.94x) | 233 (2.13x) | 230 (2.15x) | - | - | - | - |
| local affine path float32 | **757** | traceback=pointers: 757 | 672 (1.13x) | 756 (1.00x) | - | 99 (7.65x) | 132 (5.75x) | - | - | - | - |
| local affine path float64 | **473** | traceback=scores: 477 | 440 (1.08x) | 475 (1.00x) | - | 99 (4.80x) | 131 (3.63x) | - | - | 61 (7.74x) | - |
| local affine hard int16 | **1291** | - | 1058 (1.22x) | 1268 (1.02x) | - | - | - | - | - | - | - |
| local affine hard int32 | **974** | - | 836 (1.17x) | 970 (1.00x) | - | - | - | - | - | - | - |
| local affine hard int64 | **509** | - | 475 (1.07x) | 515 (0.99x) | - | - | - | - | - | - | - |
| local affine hard float32 | **782** | traceback=hirschberg_pmax: 784 | 698 (1.12x) | 771 (1.01x) | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **482** | traceback=hirschberg_pmax: 484 | 449 (1.07x) | 488 (0.99x) | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **116** | api=functions: 167 | 110 (1.05x) | 114 (1.01x) | - | - | - | - | - | - | - |
| local linear score int16 | **2254** | - | 2593 (0.87x) | 2254 (1.00x) | 1109 (2.03x) | >10x | 337 (6.68x) | wrong | - | - | >10x |
| local linear score int32 | **1667** | - | 1871 (0.89x) | 1668 (1.00x) | 767 (2.17x) | 212 (7.88x) | 569 (2.93x) | - | - | - | >10x |
| local linear score int64 | **694** | - | 728 (0.95x) | 692 (1.00x) | 282 (2.46x) | 354 (1.96x) | 568 (1.22x) | - | - | - | - |
| local linear score float32 | **1301** | fill=striped: 1288 | 1165 (1.12x) | 1303 (1.00x) | - | >10x | 214 (6.09x) | - | - | - | - |
| local linear score float64 | **984** | fill=striped: 986 | 833 (1.18x) | 986 (1.00x) | - | >10x | 214 (4.61x) | - | - | 265 (3.71x) | - |
| local linear path int16 | **1220** | - | 999 (1.22x) | 1214 (1.01x) | 902 (1.35x) | 155 (7.85x) | 180 (6.77x) | wrong | - | - | 148 (8.23x) sat |
| local linear path int32 | **937** | - | 810 (1.16x) | 940 (1.00x) | 647 (1.45x) | 157 (5.95x) | 190 (4.94x) | - | - | - | - |
| local linear path int64 | **441** | - | 410 (1.08x) | 442 (1.00x) | 276 (1.60x) | 220 (2.00x) | 190 (2.33x) | - | - | - | - |
| local linear path float32 | **674** | fill=striped: 708 | 612 (1.10x) | 673 (1.00x) | - | 102 (6.62x) | 145 (4.66x) | - | - | - | - |
| local linear path float64 | **505** | fill=striped: 505 | 426 (1.18x) | 506 (1.00x) | - | 102 (4.96x) | 148 (3.42x) | - | - | 74 (6.83x) | - |
| local linear hard int16 | **1300** | - | 1068 (1.22x) | 1309 (0.99x) | - | - | - | - | - | - | - |
| local linear hard int32 | **981** | - | 854 (1.15x) | 991 (0.99x) | - | - | - | - | - | - | - |
| local linear hard int64 | **450** | - | 420 (1.07x) | 454 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float32 | **696** | fill=striped: 749 | 639 (1.09x) | 698 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float64 | **523** | fill=striped: 524 | 443 (1.18x) | 525 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **179** | api=functions: 258 | 143 (1.25x) | 178 (1.00x) | - | - | - | - | - | - | - |

#### spot · protein_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global affine score int32 | **2173** | - | 2533 (0.86x) | 2173 (1.00x) | 1350 (1.61x) | 412 (5.28x) | 587 (3.70x) | 747 (2.91x) | - | - |
| global affine score int64 | **643** | - | 643 (1.00x) | 643 (1.00x) | 404 (1.59x) | 412 (1.56x) | 590 (1.09x) | - | - | - |
| global affine score float32 | **1313** | fill=striped: 1313 | 1165 (1.13x) | 1313 (1.00x) | - | 200 (6.56x) | 201 (6.54x) | - | - | - |
| global affine score float64 | **1047** | fill=rowwise: 1047 | 880 (1.19x) | 1047 (1.00x) | - | 200 (5.23x) | 201 (5.22x) | - | 187 (5.61x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global affine path int32 | **842** | - | 1005 (0.84x) | 842 (1.00x) | 1032 (0.82x) | 174 (4.85x) | 308 (2.73x) | 686 (1.23x) | - | - |
| global affine path int64 | **351** | - | 355 (0.99x) | 351 (1.00x) | 386 (0.91x) | 263 (1.34x) | 309 (1.14x) | - | - | - |
| global affine path float32 | **1043** | traceback=grid: 1044 | 945 (1.10x) | 1043 (1.00x) | - | 148 (7.06x) | 214 (4.88x) | - | - | - |
| global affine path float64 | **864** | traceback=grid: 864 | 745 (1.16x) | 864 (1.00x) | - | 148 (5.83x) | 216 (4.00x) | - | 119 (7.25x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global affine hard int32 | **844** | - | 1009 (0.84x) | 844 (1.00x) | - | - | - | - | - | - |
| global affine hard int64 | **352** | - | 356 (0.99x) | 352 (1.00x) | - | - | - | - | - | - |
| global affine hard float32 | **1046** | fill=striped: 1047 | 946 (1.11x) | 1047 (1.00x) | - | - | - | - | - | - |
| global affine hard float64 | **866** | traceback=grid: 866 | 745 (1.16x) | 865 (1.00x) | - | - | - | - | - | - |
| global affine soft float64 | **142** | fill=striped: 147 | 151 (0.94x) | 148 (0.96x) | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global linear score int32 | **2172** | - | 2533 (0.86x) | 2173 (1.00x) | 1701 (1.28x) | 412 (5.27x) | 601 (3.62x) | 740 (2.93x) | - | - |
| global linear score int64 | **886** | - | 920 (0.96x) | 887 (1.00x) | 404 (2.19x) | 412 (2.15x) | 601 (1.48x) | - | - | - |
| global linear score float32 | **2562** | fill=striped: 2563 | 2111 (1.21x) | 2560 (1.00x) | - | >10x | 276 (9.29x) | - | - | - |
| global linear score float64 | **1389** | fill=striped: 1389 | 1129 (1.23x) | 1390 (1.00x) | - | 200 (6.94x) | 276 (5.04x) | - | 438 (3.17x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global linear path int32 | **1382** | - | 1598 (0.86x) | 1383 (1.00x) | 1055 (1.31x) | 174 (7.92x) | 1068 (1.29x) | 695 (1.99x) | - | - |
| global linear path int64 | **569** | - | 566 (1.01x) | 556 (1.02x) | 386 (1.47x) | 288 (1.98x) | 1066 (0.53x) | - | - | - |
| global linear path float32 | **1216** | traceback=scores: 1638 | 1192 (1.02x) | 1217 (1.00x) | - | 165 (7.39x) | 387 (3.14x) | - | - | - |
| global linear path float64 | **716** | traceback=scores: 890 | 624 (1.15x) | 717 (1.00x) | - | 165 (4.34x) | 386 (1.85x) | - | 197 (3.63x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| global linear hard int32 | **1395** | - | 1619 (0.86x) | 1395 (1.00x) | - | - | - | - | - | - |
| global linear hard int64 | **571** | - | 569 (1.00x) | 571 (1.00x) | - | - | - | - | - | - |
| global linear hard float32 | **1227** | traceback=scores: 1663 | 1203 (1.02x) | 1227 (1.00x) | - | - | - | - | - | - |
| global linear hard float64 | **720** | traceback=scores: 899 | 627 (1.15x) | 720 (1.00x) | - | - | - | - | - | - |
| global linear soft float64 | **213** | api=functions: 297 | 221 (0.96x) | 212 (1.00x) | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local affine score int32 | **1978** | - | 2108 (0.94x) | 1981 (1.00x) | 989 (2.00x) | 211 (9.38x) | 401 (4.94x) | - | - | - |
| local affine score int64 | **701** | - | 711 (0.99x) | 701 (1.00x) | 362 (1.94x) | 369 (1.90x) | 400 (1.75x) | - | - | - |
| local affine score float32 | **1557** | fill=striped: 1556 | 1340 (1.16x) | 1556 (1.00x) | - | >10x | 158 (9.86x) | - | - | - |
| local affine score float64 | **1155** | fill=striped: 1155 | 939 (1.23x) | 1156 (1.00x) | - | >10x | 158 (7.32x) | - | 165 (7.01x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local affine path int32 | **727** | - | 719 (1.01x) | 717 (1.01x) | 821 (0.88x) | 180 (4.04x) | 270 (2.69x) | - | - | - |
| local affine path int64 | **284** | - | 287 (0.99x) | 284 (1.00x) | 285 (1.00x) | 251 (1.13x) | 275 (1.03x) | - | - | - |
| local affine path float32 | **580** | traceback=scores: 793 | 556 (1.04x) | 581 (1.00x) | - | 112 (5.20x) | 151 (3.85x) | - | - | - |
| local affine path float64 | **362** | traceback=scores: 470 | 305 (1.19x) | 362 (1.00x) | - | 111 (3.26x) | 150 (2.42x) | - | 66 (5.52x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local affine hard int32 | **731** | - | 723 (1.01x) | 721 (1.01x) | - | - | - | - | - | - |
| local affine hard int64 | **284** | - | 288 (0.99x) | 285 (1.00x) | - | - | - | - | - | - |
| local affine hard float32 | **583** | traceback=scores: 799 | 559 (1.04x) | 583 (1.00x) | - | - | - | - | - | - |
| local affine hard float64 | **363** | traceback=scores: 472 | 306 (1.19x) | 363 (1.00x) | - | - | - | - | - | - |
| local affine soft float64 | **110** | api=functions: 134 | 110 (1.00x) | 113 (0.97x) | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local linear score int32 | **1978** | - | 2108 (0.94x) | 1980 (1.00x) | 1392 (1.42x) | 217 (9.13x) | 559 (3.54x) | - | - | - |
| local linear score int64 | **870** | - | 902 (0.96x) | 870 (1.00x) | 403 (2.16x) | 369 (2.36x) | 558 (1.56x) | - | - | - |
| local linear score float32 | **2603** | fill=striped: 2601 | 2091 (1.24x) | 2602 (1.00x) | - | >10x | >10x | - | - | - |
| local linear score float64 | **1404** | fill=striped: 1404 | 1108 (1.27x) | 1404 (1.00x) | - | >10x | 200 (7.00x) | - | 365 (3.84x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local linear path int32 | **1064** | - | 1181 (0.90x) | 1067 (1.00x) | 1139 (0.93x) | 167 (6.38x) | 213 (4.99x) | - | - | - |
| local linear path int64 | **384** | - | 395 (0.97x) | 382 (1.00x) | 326 (1.18x) | 234 (1.64x) | 214 (1.79x) | - | - | - |
| local linear path float32 | **924** | traceback=scores: 1104 | 823 (1.12x) | 910 (1.02x) | - | 108 (8.54x) | 163 (5.65x) | - | - | - |
| local linear path float64 | **515** | traceback=scores: 617 | 423 (1.22x) | 515 (1.00x) | - | 108 (4.78x) | 168 (3.06x) | - | 78 (6.60x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | |
| local linear hard int32 | **1072** | - | 1193 (0.90x) | 1074 (1.00x) | - | - | - | - | - | - |
| local linear hard int64 | **385** | - | 396 (0.97x) | 385 (1.00x) | - | - | - | - | - | - |
| local linear hard float32 | **931** | traceback=scores: 1116 | 828 (1.12x) | 915 (1.02x) | - | - | - | - | - | - |
| local linear hard float64 | **516** | traceback=scores: 620 | 424 (1.22x) | 517 (1.00x) | - | - | - | - | - | - |
| local linear soft float64 | **176** | api=functions: 263 | 182 (0.97x) | 177 (1.00x) | - | - | - | - | - | - |

#### spot · protein_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global affine score int32 | **1859** | - | 1465 (1.27x) | 1864 (1.00x) | 1169 (1.59x) | 412 (4.52x) | 584 (3.18x) | 770 (2.41x) | - | - |
| global affine score int64 | **691** | - | 682 (1.01x) | 691 (1.00x) | 367 (1.88x) | 411 (1.68x) | 585 (1.18x) | - | - | - |
| global affine score float32 | **1307** | api=functions: 1310 | 1177 (1.11x) | 1309 (1.00x) | - | 199 (6.57x) | 199 (6.57x) | - | - | - |
| global affine score float64 | **849** | fill=striped: 851 | 815 (1.04x) | 851 (1.00x) | - | 199 (4.27x) | 199 (4.27x) | - | 186 (4.57x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global affine path int32 | **695** | - | 726 (0.96x) | 700 (0.99x) | 867 (0.80x) | 171 (4.07x) | 295 (2.35x) | 697 (1.00x) | - | - |
| global affine path int64 | **269** | - | 275 (0.98x) | 271 (1.00x) | 351 (0.77x) | 254 (1.06x) | 295 (0.91x) | - | - | - |
| global affine path float32 | **1389** | fill=striped: 1393 | 1229 (1.13x) | 1390 (1.00x) | - | 147 (9.42x) | 211 (6.57x) | - | - | - |
| global affine path float64 | **1027** | fill=rowwise: 1034 | 886 (1.16x) | 1029 (1.00x) | - | 142 (7.23x) | 211 (4.86x) | - | 119 (8.63x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global affine hard int32 | **711** | - | 728 (0.98x) | 704 (1.01x) | - | - | - | - | - | - |
| global affine hard int64 | **269** | - | 273 (0.99x) | 271 (0.99x) | - | - | - | - | - | - |
| global affine hard float32 | **1392** | traceback=grid: 1392 | 1230 (1.13x) | 1392 (1.00x) | - | - | - | - | - | - |
| global affine hard float64 | **1028** | traceback=grid: 1031 | 885 (1.16x) | 1031 (1.00x) | - | - | - | - | - | - |
| global affine soft float64 | **136** | api=functions: 141 | 128 (1.06x) | 136 (1.00x) | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global linear score int32 | **1829** | - | 1475 (1.24x) | 1859 (0.98x) | 1758 (1.04x) | 412 (4.44x) | 597 (3.06x) | 778 (2.35x) | - | 292 (6.26x) sat |
| global linear score int64 | **917** | - | 942 (0.97x) | 918 (1.00x) | 396 (2.32x) | 411 (2.23x) | 596 (1.54x) | - | - | - |
| global linear score float32 | **2369** | api=functions: 2393 | 1974 (1.20x) | 2371 (1.00x) | - | >10x | 272 (8.72x) | - | - | - |
| global linear score float64 | **1050** | fill=striped: 1052 | 1004 (1.05x) | 1052 (1.00x) | - | 199 (5.28x) | 271 (3.87x) | - | 427 (2.46x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global linear path int32 | **1018** | - | 1027 (0.99x) | 1011 (1.01x) | 1055 (0.97x) | 171 (5.95x) | 1064 (0.96x) | 720 (1.41x) | - | 190 (5.37x) sat |
| global linear path int64 | **539** | - | 556 (0.97x) | 542 (0.99x) | 350 (1.54x) | 274 (1.97x) | 1047 (0.51x) | - | - | - |
| global linear path float32 | **1037** | traceback=scores: 1803 | 947 (1.10x) | 1037 (1.00x) | - | 160 (6.49x) | 352 (2.95x) | - | - | - |
| global linear path float64 | **473** | traceback=scores: 828 | 445 (1.06x) | 475 (1.00x) | - | 156 (3.04x) | 350 (1.35x) | - | 184 (2.57x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| global linear hard int32 | **1030** | - | 1029 (1.00x) | 1017 (1.01x) | - | - | - | - | - | - |
| global linear hard int64 (load!) | **544** | - | 555 (0.98x) | 543 (1.00x) | - | - | - | - | - | - |
| global linear hard float32 (load!) | **1036** | traceback=scores: 1807 | 951 (1.09x) | 1043 (0.99x) | - | - | - | - | - | - |
| global linear hard float64 (load!) | **473** | traceback=scores: 831 | 444 (1.06x) | 474 (1.00x) | - | - | - | - | - | - |
| global linear soft float64 (load!) | **175** | api=functions: 256 | 185 (0.95x) | 175 (1.00x) | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local affine score int32 (load!) | **1834** | - | 1778 (1.03x) | 1835 (1.00x) | 972 (1.89x) | 219 (8.37x) | 399 (4.60x) | - | - | - |
| local affine score int64 (load!) | **754** | - | 761 (0.99x) | 754 (1.00x) | 377 (2.00x) | 365 (2.07x) | 398 (1.90x) | - | - | - |
| local affine score float32 (load!) | **1437** | api=functions: 1435 | 1347 (1.07x) | 1441 (1.00x) | - | >10x | 157 (9.15x) | - | - | - |
| local affine score float64 (load!) | **921** | fill=striped: 923 | 876 (1.05x) | 926 (0.99x) | - | 96 (9.57x) | 157 (5.87x) | - | 175 (5.25x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local affine path int32 (load!) | **290** | - | 295 (0.98x) | 290 (1.00x) | 722 (0.40x) | 174 (1.67x) | 311 (0.93x) | - | - | - |
| local affine path int64 (load!) | **113** | - | 114 (1.00x) | 113 (1.00x) | 279 (0.41x) | 232 (0.49x) | 321 (0.35x) | - | - | - |
| local affine path float32 (load!) | **280** | traceback=scores: 734 | 231 (1.21x) | 280 (1.00x) | - | 118 (2.37x) | 157 (1.79x) | - | - | - |
| local affine path float64 (load!) | **336** | traceback=scores: 418 | 299 (1.12x) | 336 (1.00x) | - | 116 (2.90x) | 156 (2.16x) | - | 62 (5.45x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local affine hard int32 | **291** | - | 296 (0.98x) | 291 (1.00x) | - | - | - | - | - | - |
| local affine hard int64 | **114** | - | 114 (0.99x) | 112 (1.01x) | - | - | - | - | - | - |
| local affine hard float32 | **281** | traceback=scores: 733 | 231 (1.22x) | 280 (1.00x) | - | - | - | - | - | - |
| local affine hard float64 | **336** | traceback=scores: 414 | 300 (1.12x) | 335 (1.00x) | - | - | - | - | - | - |
| local affine soft float64 | **102** | api=functions: 133 | 98 (1.04x) | 102 (1.00x) | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local linear score int32 | **1828** | - | 1792 (1.02x) | 1826 (1.00x) | 1437 (1.27x) | 219 (8.34x) | 558 (3.28x) | - | - | >10x |
| local linear score int64 | **894** | - | 925 (0.97x) | 894 (1.00x) | 407 (2.20x) | 364 (2.45x) | 556 (1.61x) | - | - | - |
| local linear score float32 | **2115** | api=functions: 2131 | 1967 (1.08x) | 2142 (0.99x) | - | >10x | >10x | - | - | - |
| local linear score float64 | **1062** | fill=striped: 1063 | 1001 (1.06x) | 1064 (1.00x) | - | >10x | 199 (5.34x) | - | 414 (2.57x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local linear path int32 | **825** | - | 837 (0.99x) | 825 (1.00x) | 1085 (0.76x) | 161 (5.13x) | 210 (3.92x) | - | - | 174 (4.75x) sat |
| local linear path int64 | **403** | - | 390 (1.03x) | 403 (1.00x) | 317 (1.27x) | 218 (1.84x) | 210 (1.91x) | - | - | - |
| local linear path float32 | **757** | traceback=scores: 1153 | 671 (1.13x) | 738 (1.03x) | - | 105 (7.22x) | 163 (4.63x) | - | - | - |
| local linear path float64 | **379** | traceback=scores: 584 | 334 (1.14x) | 380 (1.00x) | - | 103 (3.69x) | 167 (2.27x) | - | 75 (5.03x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | |
| local linear hard int32 | **826** | - | 836 (0.99x) | 828 (1.00x) | - | - | - | - | - | - |
| local linear hard int64 | **405** | - | 390 (1.04x) | 407 (0.99x) | - | - | - | - | - | - |
| local linear hard float32 | **757** | traceback=scores: 1154 | 670 (1.13x) | 738 (1.03x) | - | - | - | - | - | - |
| local linear hard float64 | **379** | traceback=scores: 585 | 334 (1.13x) | 379 (1.00x) | - | - | - | - | - | - |
| local linear soft float64 | **148** | api=functions: 228 | 156 (0.95x) | 148 (1.00x) | - | - | - | - | - | - |

#### spot · protein_L300 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| global affine score int32 | **2095** | - | 2404 (0.87x) | 2095 (1.00x) | 1261 (1.66x) | 413 (5.07x) | 592 (3.54x) | 668 (3.13x) | - | - |
| global affine score int64 | **610** | - | 614 (0.99x) | 610 (1.00x) | 394 (1.55x) | 413 (1.48x) | 597 (1.02x) | - | - | - |
| global affine score float32 | **1149** | fill=striped: 1149 | 1049 (1.10x) | 1149 (1.00x) | - | 203 (5.66x) | 205 (5.60x) | - | - | - |
| global affine score float64 | **970** | fill=rowwise: 970 | 823 (1.18x) | 970 (1.00x) | - | 203 (4.77x) | 205 (4.73x) | - | 187 (5.19x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| global affine path int32 | **763** | - | 960 (0.80x) | 761 (1.00x) | 972 (0.79x) | 171 (4.47x) | 294 (2.60x) | 627 (1.22x) | - | - |
| global affine path int64 | **339** | - | 341 (0.99x) | 339 (1.00x) | 375 (0.90x) | 251 (1.35x) | 295 (1.15x) | - | - | - |
| global affine path float32 | **603** | traceback=scores: 822 | 587 (1.03x) | 603 (1.00x) | - | 141 (4.28x) | 203 (2.97x) | - | - | - |
| global affine path float64 | **441** | traceback=scores: 605 | 387 (1.14x) | 441 (1.00x) | - | 140 (3.14x) | 205 (2.16x) | - | 115 (3.84x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| global affine hard int32 | **775** | - | 986 (0.79x) | 773 (1.00x) | - | - | - | - | - | - |
| global affine hard int64 | **342** | - | 344 (0.99x) | 342 (1.00x) | - | - | - | - | - | - |
| global affine hard float32 | **609** | traceback=scores: 833 | 597 (1.02x) | 609 (1.00x) | - | - | - | - | - | - |
| global affine hard float64 | **445** | traceback=scores: 614 | 391 (1.14x) | 445 (1.00x) | - | - | - | - | - | - |
| global affine soft float64 | **138** | api=functions: 165 | 136 (1.02x) | 139 (1.00x) | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| global linear score int32 | **2093** | - | 2404 (0.87x) | 2090 (1.00x) | 1261 (1.66x) | 413 (5.06x) | 610 (3.43x) | 660 (3.17x) | - | - |
| global linear score int64 | **817** | - | 848 (0.96x) | 811 (1.01x) | 394 (2.07x) | 413 (1.98x) | 610 (1.34x) | - | - | - |
| global linear score float32 (load!) | **1953** | fill=striped: 1954 | 1688 (1.16x) | 1952 (1.00x) | - | 203 (9.61x) | 286 (6.82x) | - | - | - |
| global linear score float64 (load!) | **1234** | fill=striped: 1234 | 1022 (1.21x) | 1235 (1.00x) | - | 203 (6.08x) | 286 (4.31x) | - | 464 (2.66x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| global linear path int32 (load!) | **1692** | - | 1673 (1.01x) | 1668 (1.01x) | 971 (1.74x) | 170 (9.96x) | 999 (1.69x) | 625 (2.71x) | - | - |
| global linear path int64 | **920** | - | 920 (1.00x) | 930 (0.99x) | 371 (2.48x) | 276 (3.34x) | 998 (0.92x) | - | - | - |
| global linear path float32 | **1288** | traceback=scores: 1289 | 1162 (1.11x) | 1276 (1.01x) | - | 157 (8.20x) | 363 (3.55x) | - | - | - |
| global linear path float64 | **796** | fill=striped: 796 | 686 (1.16x) | 797 (1.00x) | - | 158 (5.02x) | 362 (2.20x) | - | 188 (4.23x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| global linear hard int32 | **1775** | - | 1771 (1.00x) | 1755 (1.01x) | - | - | - | - | - | - |
| global linear hard int64 | **944** | - | 947 (1.00x) | 947 (1.00x) | - | - | - | - | - | - |
| global linear hard float32 | **1341** | traceback=scores: 1342 | 1208 (1.11x) | 1335 (1.00x) | - | - | - | - | - | - |
| global linear hard float64 | **814** | fill=striped: 815 | 703 (1.16x) | 817 (1.00x) | - | - | - | - | - | - |
| global linear soft float64 | **224** | api=functions: 298 | 224 (1.00x) | 224 (1.00x) | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| local affine score int32 | **1891** | - | 2043 (0.93x) | 1892 (1.00x) | 936 (2.02x) | 216 (8.76x) | 404 (4.68x) | - | - | - |
| local affine score int64 | **664** | - | 677 (0.98x) | 664 (1.00x) | 343 (1.94x) | 366 (1.81x) | 402 (1.65x) | - | - | - |
| local affine score float32 | **1357** | fill=striped: 1356 | 1195 (1.14x) | 1353 (1.00x) | - | >10x | 160 (8.48x) | - | - | - |
| local affine score float64 (load!) | **1061** | fill=rowwise: 1062 | 882 (1.20x) | 1061 (1.00x) | - | >10x | 160 (6.63x) | - | 163 (6.51x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| local affine path int32 (load!) | **633** | - | 772 (0.82x) | 633 (1.00x) | 766 (0.83x) | 177 (3.57x) | 253 (2.50x) | - | - | - |
| local affine path int64 | **275** | - | 277 (0.99x) | 276 (1.00x) | 276 (1.00x) | 247 (1.12x) | 257 (1.07x) | - | - | - |
| local affine path float32 | **548** | traceback=scores: 728 | 512 (1.07x) | 549 (1.00x) | - | 105 (5.20x) | 142 (3.85x) | - | - | - |
| local affine path float64 | **340** | traceback=scores: 473 | 306 (1.11x) | 340 (1.00x) | - | 105 (3.23x) | 141 (2.41x) | - | 65 (5.27x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | |
| local affine hard int32 | **641** | - | 788 (0.81x) | 641 (1.00x) | - | - | - | - | - | - |
| local affine hard int64 | **277** | - | 280 (0.99x) | 277 (1.00x) | - | - | - | - | - | - |
| local affine hard float32 | **554** | traceback=scores: 743 | 521 (1.06x) | 555 (1.00x) | - | - | - | - | - | - |
| local affine hard float64 | **342** | traceback=scores: 477 | 309 (1.11x) | 343 (1.00x) | - | - | - | - | - | - |
| local affine soft float64 | **121** | api=functions: 175 | 122 (0.99x) | 118 (1.02x) | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| local linear score int32 | **1890** | - | 2043 (0.93x) | 1892 (1.00x) | 937 (2.02x) | 214 (8.84x) | 562 (3.36x) | - | - | - |
| local linear score int64 | **816** | - | 846 (0.96x) | 804 (1.01x) | 392 (2.08x) | 366 (2.23x) | 561 (1.45x) | - | - | - |
| local linear score float32 | **2035** | fill=striped: 2030 | 1707 (1.19x) | 2033 (1.00x) | - | >10x | 204 (9.99x) | - | - | - |
| local linear score float64 | **1268** | fill=striped: 1268 | 1030 (1.23x) | 1268 (1.00x) | - | >10x | 204 (6.22x) | - | 317 (4.00x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| local linear path int32 | **1276** | - | 1294 (0.99x) | 1278 (1.00x) | 924 (1.38x) | 162 (7.89x) | 202 (6.32x) | - | - | - |
| local linear path int64 | **479** | - | 476 (1.01x) | 479 (1.00x) | 309 (1.55x) | 231 (2.07x) | 204 (2.35x) | - | - | - |
| local linear path float32 | **997** | traceback=scores: 997 | 838 (1.19x) | 998 (1.00x) | - | 105 (9.47x) | 154 (6.46x) | - | - | - |
| local linear path float64 | **584** | fill=striped: 584 | 470 (1.24x) | 584 (1.00x) | - | 106 (5.52x) | 158 (3.69x) | - | 77 (7.62x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | |
| local linear hard int32 | **1320** | - | 1356 (0.97x) | 1322 (1.00x) | - | - | - | - | - | - |
| local linear hard int64 | **485** | - | 483 (1.00x) | 485 (1.00x) | - | - | - | - | - | - |
| local linear hard float32 | **1027** | traceback=scores: 1027 | 865 (1.19x) | 1027 (1.00x) | - | - | - | - | - | - |
| local linear hard float64 | **594** | traceback=scores: 594 | 477 (1.24x) | 595 (1.00x) | - | - | - | - | - | - |
| local linear soft float64 | **194** | api=functions: 280 | 189 (1.02x) | 194 (1.00x) | - | - | - | - | - | - |

#### spot · protein_L3000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global affine score int32 | **2201** | - | 2562 (0.86x) | 2199 (1.00x) | 1368 (1.61x) | 412 (5.35x) | 585 (3.76x) | 767 (2.87x) | - | - |
| global affine score int64 | **665** | - | 658 (1.01x) | 665 (1.00x) | 406 (1.64x) | 411 (1.62x) | 586 (1.14x) | - | - | - |
| global affine score float32 | **1404** | fill=striped: 1402 | 1227 (1.14x) | 1403 (1.00x) | - | 199 (7.05x) | 200 (7.04x) | - | - | - |
| global affine score float64 | **877** | fill=rowwise: 885 | 836 (1.05x) | 884 (0.99x) | - | 199 (4.40x) | 199 (4.40x) | - | 187 (4.70x) | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global affine path int32 | **902** | - | 1038 (0.87x) | 903 (1.00x) | 897 (1.01x) | 171 (5.28x) | 310 (2.91x) | 715 (1.26x) | - | - |
| global affine path int64 | **352** | - | 356 (0.99x) | 352 (1.00x) | 339 (1.04x) | 261 (1.35x) | 310 (1.14x) | - | - | - |
| global affine path float32 | **1256** | fill=striped: 1256 | 1108 (1.13x) | 1254 (1.00x) | - | 147 (8.53x) | 217 (5.78x) | - | - | - |
| global affine path float64 | **946** | fill=rowwise: 946 | 816 (1.16x) | 941 (1.01x) | - | 149 (6.35x) | 217 (4.36x) | - | 119 (7.93x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global affine hard int32 | **901** | - | 1039 (0.87x) | 901 (1.00x) | - | - | - | - | - | - |
| global affine hard int64 | **354** | - | 356 (0.99x) | 353 (1.00x) | - | - | - | - | - | - |
| global affine hard float32 | **1255** | fill=striped: 1256 | 1108 (1.13x) | 1255 (1.00x) | - | - | - | - | - | - |
| global affine hard float64 | **948** | fill=rowwise: 947 | 819 (1.16x) | 947 (1.00x) | - | - | - | - | - | - |
| global affine soft float64 | **132** | api=functions: 136 | 132 (1.01x) | 134 (0.99x) | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global linear score int32 | **2201** | - | 2558 (0.86x) | 2201 (1.00x) | 1751 (1.26x) | 412 (5.35x) | 598 (3.68x) | 744 (2.96x) | - | - |
| global linear score int64 | **911** | - | 934 (0.98x) | 911 (1.00x) | 406 (2.25x) | 411 (2.21x) | 598 (1.52x) | - | - | - |
| global linear score float32 | **2808** | fill=striped: 2804 | 2274 (1.23x) | 2804 (1.00x) | - | >10x | >10x | - | - | - |
| global linear score float64 | **1123** | fill=striped: 1122 | 1052 (1.07x) | 1119 (1.00x) | - | 199 (5.64x) | 273 (4.12x) | - | 430 (2.61x) | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global linear path int32 | **1475** | - | 1522 (0.97x) | 1474 (1.00x) | 995 (1.48x) | 176 (8.40x) | 1094 (1.35x) | 695 (2.12x) | - | - |
| global linear path int64 | **571** | - | 570 (1.00x) | 558 (1.02x) | 344 (1.66x) | 289 (1.98x) | 1093 (0.52x) | - | - | - |
| global linear path float32 | **1391** | traceback=scores: 1765 | 1235 (1.13x) | 1391 (1.00x) | - | 166 (8.40x) | 388 (3.59x) | - | - | - |
| global linear path float64 | **707** | traceback=scores: 909 | 610 (1.16x) | 706 (1.00x) | - | 165 (4.29x) | 387 (1.83x) | - | 196 (3.61x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| global linear hard int32 | **1481** | - | 1528 (0.97x) | 1478 (1.00x) | - | - | - | - | - | - |
| global linear hard int64 | **571** | - | 570 (1.00x) | 571 (1.00x) | - | - | - | - | - | - |
| global linear hard float32 | **1394** | traceback=scores: 1775 | 1239 (1.12x) | 1394 (1.00x) | - | - | - | - | - | - |
| global linear hard float64 | **707** | traceback=scores: 912 | 614 (1.15x) | 708 (1.00x) | - | - | - | - | - | - |
| global linear soft float64 | **194** | api=functions: 245 | 208 (0.93x) | 196 (0.99x) | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local affine score int32 | **2010** | - | 2130 (0.94x) | 2009 (1.00x) | 1010 (1.99x) | 219 (9.20x) | 400 (5.03x) | - | - | - |
| local affine score int64 | **726** | - | 732 (0.99x) | 726 (1.00x) | 369 (1.97x) | 370 (1.96x) | 398 (1.83x) | - | - | - |
| local affine score float32 | **1666** | fill=striped: 1667 | 1416 (1.18x) | 1667 (1.00x) | - | >10x | >10x | - | - | - |
| local affine score float64 | **966** | fill=rowwise: 967 | 890 (1.09x) | 957 (1.01x) | - | >10x | 157 (6.15x) | - | 164 (5.88x) | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local affine path int32 | **609** | - | 596 (1.02x) | 605 (1.01x) | 740 (0.82x) | 181 (3.36x) | 270 (2.25x) | - | - | - |
| local affine path int64 | **287** | - | 290 (0.99x) | 287 (1.00x) | 261 (1.10x) | 250 (1.15x) | 276 (1.04x) | - | - | - |
| local affine path float32 | **595** | traceback=scores: 808 | 550 (1.08x) | 601 (0.99x) | - | 112 (5.33x) | 152 (3.91x) | - | - | - |
| local affine path float64 | **337** | traceback=scores: 421 | 303 (1.11x) | 337 (1.00x) | - | 112 (3.02x) | 151 (2.23x) | - | 66 (5.11x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local affine hard int32 | **611** | - | 598 (1.02x) | 603 (1.01x) | - | - | - | - | - | - |
| local affine hard int64 | **287** | - | 290 (0.99x) | 288 (1.00x) | - | - | - | - | - | - |
| local affine hard float32 | **596** | traceback=scores: 811 | 551 (1.08x) | 602 (0.99x) | - | - | - | - | - | - |
| local affine hard float64 | **337** | traceback=scores: 425 | 304 (1.11x) | 337 (1.00x) | - | - | - | - | - | - |
| local affine soft float64 | **101** | api=functions: 127 | 100 (1.02x) | 102 (0.99x) | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local linear score int32 | **2010** | - | 2131 (0.94x) | 2011 (1.00x) | 1426 (1.41x) | 219 (9.20x) | 558 (3.60x) | - | - | - |
| local linear score int64 | **889** | - | 920 (0.97x) | 889 (1.00x) | 406 (2.19x) | 370 (2.40x) | 557 (1.60x) | - | - | - |
| local linear score float32 | **2814** | fill=striped: 2818 | 2231 (1.26x) | 2670 (1.05x) | - | >10x | >10x | - | - | - |
| local linear score float64 | **1141** | fill=striped: 1137 | 1041 (1.10x) | 1145 (1.00x) | - | >10x | 199 (5.72x) | - | 400 (2.85x) | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local linear path int32 | **1117** | - | 1148 (0.97x) | 1123 (0.99x) | 1069 (1.05x) | 167 (6.70x) | 219 (5.11x) | - | - | - |
| local linear path int64 | **408** | - | 397 (1.03x) | 407 (1.00x) | 305 (1.34x) | 233 (1.75x) | 221 (1.85x) | - | - | - |
| local linear path float32 | **1000** | traceback=scores: 1137 | 837 (1.19x) | 1003 (1.00x) | - | 108 (9.22x) | 168 (5.96x) | - | - | - |
| local linear path float64 | **444** | traceback=scores: 616 | 386 (1.15x) | 446 (1.00x) | - | 109 (4.09x) | 172 (2.57x) | - | 79 (5.65x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | |
| local linear hard int32 | **1120** | - | 1150 (0.97x) | 1127 (0.99x) | - | - | - | - | - | - |
| local linear hard int64 | **409** | - | 397 (1.03x) | 409 (1.00x) | - | - | - | - | - | - |
| local linear hard float32 | **1002** | traceback=scores: 1140 | 839 (1.19x) | 1004 (1.00x) | - | - | - | - | - | - |
| local linear hard float64 | **445** | traceback=scores: 616 | 386 (1.15x) | 446 (1.00x) | - | - | - | - | - | - |
| local linear soft float64 | **156** | api=functions: 222 | 166 (0.94x) | 156 (1.00x) | - | - | - | - | - | - |

#### spot · dna_L100 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **11931** | schedule=sorted: 11527 | 8367 (1.43x) | 12168 (0.98x) | 6492 (1.84x) | 2280 (5.23x) | - | 3072 (3.88x) | 4685 (2.55x) | - | - | - |
| global affine score float32 | **8292** | schedule=sorted: 8481 | 7984 (1.04x) | 8433 (0.98x) | - | - | - | - | - | - | - | - |
| global affine score float64 | **4323** | fill=striped: 5123 | 3668 (1.18x) | 4306 (1.00x) | - | - | - | - | - | - | 1169 (3.70x) | - |
| global affine path int32 | **7107** | schedule=sorted: 7142 | 5696 (1.25x) | 7019 (1.01x) | 4999 (1.42x) | 1045 (6.80x) | - | 2900 (2.45x) | 4746 (1.50x) | - | - | - |
| global affine path float32 | **5811** | schedule=sorted: 5531 | 4628 (1.26x) | 5579 (1.04x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **3133** | schedule=sorted: 2948 | 2420 (1.30x) | 2834 (1.11x) | - | - | - | - | - | - | 612 (5.12x) | - |
| global affine hard int32 | **6771** | schedule=sorted: 6897 | 5506 (1.23x) | 6325 (1.07x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **5733** | schedule=sorted: 5663 | 4893 (1.17x) | 5924 (0.97x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **2838** | schedule=sorted: 2818 | 2274 (1.25x) | 2720 (1.04x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **576** | fill=striped: 577 | 557 (1.03x) | 576 (1.00x) | - | - | - | - | - | - | - | - |
| global linear score int32 | **12955** | schedule=sorted: 12613 | 9428 (1.37x) | 12553 (1.03x) | 6249 (2.07x) | 2274 (5.70x) | - | 3158 (4.10x) | 49278 (0.26x) | 18153 (0.71x) | - | - |
| global linear score float32 | **10368** | schedule=sorted: 9634 | 9646 (1.07x) | 10119 (1.02x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **5743** | fill=striped: 5780 | 4742 (1.21x) | 5764 (1.00x) | - | - | - | - | - | - | 2815 (2.04x) | - |
| global linear path int32 | **10756** | schedule=sorted: 10784 | 8245 (1.30x) | 10323 (1.04x) | 5065 (2.12x) | >10x | - | 3037 (3.54x) | 41068 (0.26x) | 6808 (1.58x) | - | - |
| global linear path float32 | **7763** | schedule=sorted: 7725 | 7158 (1.08x) | 7716 (1.01x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **4051** | fill=striped: 4051 | 3516 (1.15x) | 3996 (1.01x) | - | - | - | - | - | - | 927 (4.37x) | - |
| global linear hard int32 | **12763** | schedule=sorted: 12941 | 10184 (1.25x) | 12344 (1.03x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **9074** | schedule=sorted: 9061 | 8440 (1.08x) | 8300 (1.09x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **4384** | schedule=sorted: 4357 | 3767 (1.16x) | 4359 (1.01x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **1338** | schedule=sorted: 1238 | 993 (1.35x) | 1203 (1.11x) | - | - | - | - | - | - | - | - |
| local affine score int32 | **9325** | schedule=sorted: 9548 | 7680 (1.21x) | 9244 (1.01x) | 5948 (1.57x) | >10x | - | - | - | - | - | - |
| local affine score float32 | **7898** | fill=striped: 8236 | 6844 (1.15x) | 7993 (0.99x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **4025** | fill=striped: 5823 | 3379 (1.19x) | 3789 (1.06x) | - | - | - | - | - | - | 891 (4.52x) | - |
| local affine path int32 | **5309** | schedule=sorted: 5179 | 4818 (1.10x) | 5183 (1.02x) | 4128 (1.29x) | 942 (5.63x) | - | - | - | - | - | - |
| local affine path float32 | **4340** | schedule=sorted: 4259 | 3978 (1.09x) | 4284 (1.01x) | - | - | - | - | - | - | - | - |
| local affine path float64 | **2422** | schedule=sorted: 2294 | 1994 (1.22x) | 2184 (1.11x) | - | - | - | - | - | - | 373 (6.50x) | - |
| local affine hard int32 | **5612** | schedule=sorted: 5311 | 4744 (1.18x) | 5612 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **4458** | schedule=sorted: 4406 | 3982 (1.12x) | 4153 (1.07x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **2145** | schedule=sorted: 2130 | 1965 (1.09x) | 2113 (1.01x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **527** | fill=striped: 520 | 504 (1.05x) | 513 (1.03x) | - | - | - | - | - | - | - | - |
| local linear score int32 | **11229** | schedule=sorted: 11017 | 8872 (1.27x) | 11046 (1.02x) | 4916 (2.28x) | >10x | - | - | - | - | - | - |
| local linear score float32 | **6819** | fill=striped: 8490 | 6609 (1.03x) | 6769 (1.01x) | - | - | - | - | - | - | - | - |
| local linear score float64 | **3435** | fill=striped: 5834 | 3348 (1.03x) | 3427 (1.00x) | - | - | - | - | - | - | 1967 (1.75x) | - |
| local linear path int32 | **7985** | schedule=sorted: 7870 | 7086 (1.13x) | 7824 (1.02x) | 3957 (2.02x) | >10x | - | - | - | - | - | - |
| local linear path float32 | **5353** | schedule=sorted: 5391 | 5113 (1.05x) | 5309 (1.01x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **2970** | fill=striped: 3057 | 2879 (1.03x) | 2959 (1.00x) | - | - | - | - | - | - | 406 (7.31x) | - |
| local linear hard int32 | **8961** | schedule=sorted: 9108 | 8440 (1.06x) | 9062 (0.99x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **5877** | schedule=sorted: 5851 | 5760 (1.02x) | 5873 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **3132** | fill=striped: 3252 | 3065 (1.02x) | 3128 (1.00x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **1146** | schedule=sorted: 1103 | 897 (1.28x) | 1069 (1.07x) | - | - | - | - | - | - | - | - |

#### spot · dna_L1000 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **10352** | schedule=sorted: 10531 | 10690 (0.97x) | 10885 (0.95x) | 7361 (1.41x) | 2541 (4.07x) | - | 4048 (2.56x) | 14440 (0.72x) | - | - | - |
| global affine score float32 | **10093** | schedule=sorted: 10018 | 8367 (1.21x) | 9745 (1.04x) | - | - | - | - | - | - | - | - |
| global affine score float64 | **6698** | fill=striped: 6685 | 5328 (1.26x) | 6518 (1.03x) | - | - | - | - | - | - | 1252 (5.35x) | - |
| global affine path int32 | **4547** | fill=striped: 4307 | 4774 (0.95x) | 4060 (1.12x) | 5313 (0.86x) | 1149 (3.96x) | - | 3903 (1.17x) | 5744 (0.79x) | - | - | - |
| global affine path float32 | **7397** | schedule=sorted: 7333 | 6364 (1.16x) | 7299 (1.01x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **5308** | fill=striped: 5263 | 4243 (1.25x) | 5157 (1.03x) | - | - | - | - | - | - | 718 (7.39x) | - |
| global affine hard int32 | **4034** | fill=striped: 4094 | 4344 (0.93x) | 3796 (1.06x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **7453** | schedule=sorted: 7379 | 6368 (1.17x) | 7357 (1.01x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **5192** | fill=striped: 5230 | 4034 (1.29x) | 4903 (1.06x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **580** | fill=striped: 587 | 575 (1.01x) | 568 (1.02x) | - | - | - | - | - | - | - | - |
| global linear score int32 | **9187** | fill=striped: 9631 | 10228 (0.90x) | 10188 (0.90x) | 9816 (0.94x) | 2535 (3.62x) | - | 4026 (2.28x) | 124117 (0.07x) | 124121 (0.07x) | - | - |
| global linear score float32 | **14323** | schedule=sorted: 14481 | 10828 (1.32x) | 13663 (1.05x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **7551** | schedule=sorted: 7589 | 5968 (1.27x) | 7241 (1.04x) | - | - | - | - | - | - | 2598 (2.91x) | - |
| global linear path int32 | **7856** | fill=striped: 7717 | 8226 (0.96x) | 7320 (1.07x) | 5287 (1.49x) | 1150 (6.83x) | 891 (8.82x) | 3883 (2.02x) | 110295 (0.07x) | 58471 (0.13x) | - | - |
| global linear path float32 | **7039** | fill=striped: 7039 | 6854 (1.03x) | 6898 (1.02x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **4075** | fill=striped: 4105 | 3631 (1.12x) | 4093 (1.00x) | - | - | - | - | - | - | 1085 (3.76x) | - |
| global linear hard int32 | **7987** | fill=striped: 7844 | 9038 (0.88x) | 7612 (1.05x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **7125** | fill=striped: 7121 | 7016 (1.02x) | 6955 (1.02x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **4097** | schedule=sorted: 4126 | 3640 (1.13x) | 4115 (1.00x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **1000** | schedule=sorted: 995 | 970 (1.03x) | 995 (1.01x) | - | - | - | - | - | - | - | - |
| local affine score int32 | **10924** | schedule=sorted: 10389 | 11113 (0.98x) | 9929 (1.10x) | 7461 (1.46x) | 1209 (9.04x) | - | - | - | - | - | - |
| local affine score float32 | **12834** | fill=striped: 13076 | 10320 (1.24x) | 12646 (1.01x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **7315** | schedule=sorted: 7370 | 5631 (1.30x) | 7078 (1.03x) | - | - | - | - | - | - | 973 (7.52x) | - |
| local affine path int32 | **3873** | fill=striped: 3876 | 3795 (1.02x) | 3762 (1.03x) | 4786 (0.81x) | 1019 (3.80x) | - | - | - | - | - | - |
| local affine path float32 | **3531** | fill=striped: 3528 | 3446 (1.02x) | 3538 (1.00x) | - | - | - | - | - | - | - | - |
| local affine path float64 | **2021** | schedule=sorted: 2032 | 1697 (1.19x) | 2019 (1.00x) | - | - | - | - | - | - | 421 (4.80x) | - |
| local affine hard int32 | **3698** | fill=striped: 3655 | 3584 (1.03x) | 3590 (1.03x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **3483** | schedule=sorted: 3451 | 3381 (1.03x) | 3458 (1.01x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **1886** | schedule=sorted: 1895 | 1585 (1.19x) | 1876 (1.01x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **489** | fill=striped: 477 | 453 (1.08x) | 460 (1.06x) | - | - | - | - | - | - | - | - |
| local linear score int32 | **9352** | schedule=sorted: 10014 | 9065 (1.03x) | 9844 (0.95x) | 8200 (1.14x) | 1203 (7.77x) | - | - | - | - | - | - |
| local linear score float32 | **13693** | schedule=sorted: 13700 | 10153 (1.35x) | 13359 (1.02x) | - | - | - | - | - | - | - | - |
| local linear score float64 | **7385** | schedule=sorted: 7436 | 5628 (1.31x) | 7245 (1.02x) | - | - | - | - | - | - | 2569 (2.87x) | - |
| local linear path int32 | **5997** | fill=striped: 6036 | 6487 (0.92x) | 5929 (1.01x) | 5499 (1.09x) | 1014 (5.91x) | - | - | - | - | - | - |
| local linear path float32 | **5265** | fill=striped: 5284 | 4755 (1.11x) | 5122 (1.03x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **2906** | fill=striped: 2906 | 2486 (1.17x) | 2917 (1.00x) | - | - | - | - | - | - | 426 (6.83x) | - |
| local linear hard int32 | **6089** | schedule=sorted: 6079 | 6626 (0.92x) | 6094 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **5285** | schedule=sorted: 5295 | 4766 (1.11x) | 5176 (1.02x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **2933** | schedule=sorted: 2930 | 2488 (1.18x) | 2920 (1.00x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **856** | fill=striped: 858 | 845 (1.01x) | 860 (1.00x) | - | - | - | - | - | - | - | - |

#### spot · mirna · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **10435** | schedule=sorted: 10313 | 7167 (1.46x) | 10448 (1.00x) | 4001 (2.61x) | >10x | - | 1991 (5.24x) | >10x | - | - | >10x |
| global affine score float32 | **7827** | schedule=sorted: 7589 | 6479 (1.21x) | 7385 (1.06x) | - | - | - | - | - | - | - | - |
| global affine score float64 | **3956** | schedule=sorted: 3996 | 3537 (1.12x) | 4016 (0.99x) | - | - | - | - | - | - | 791 (5.00x) | - |
| global affine path int32 | **3705** | schedule=sorted: 3812 | 3418 (1.08x) | 3790 (0.98x) | 2855 (1.30x) | 611 (6.06x) | - | 1737 (2.13x) | >10x | - | - | - |
| global affine path float32 | **3274** | schedule=sorted: 3219 | 2733 (1.20x) | 3216 (1.02x) | - | - | - | - | - | - | - | - |
| global affine path float64 | **2228** | schedule=sorted: 2069 | 1757 (1.27x) | 2037 (1.09x) | - | - | - | - | - | - | 310 (7.19x) | - |
| global affine hard int32 | **4686** | schedule=sorted: 4490 | 4158 (1.13x) | 4755 (0.99x) | - | - | - | - | - | - | - | - |
| global affine hard float32 | **4285** | schedule=sorted: 4237 | 3625 (1.18x) | 4024 (1.06x) | - | - | - | - | - | - | - | - |
| global affine hard float64 | **2419** | schedule=sorted: 2411 | 1995 (1.21x) | 2351 (1.03x) | - | - | - | - | - | - | - | - |
| global affine soft float64 | **642** | schedule=sorted: 637 | 529 (1.21x) | 625 (1.03x) | - | - | - | - | - | - | - | - |
| global linear score int32 | **10784** | schedule=sorted: 11029 | 8307 (1.30x) | 11279 (0.96x) | 4592 (2.35x) | >10x | - | 2220 (4.86x) | 2624 (4.11x) | 3565 (3.03x) | - | >10x |
| global linear score float32 | **9215** | schedule=sorted: 8859 | 8150 (1.13x) | 9004 (1.02x) | - | - | - | - | - | - | - | - |
| global linear score float64 | **3777** | schedule=sorted: 3810 | 3327 (1.14x) | 3829 (0.99x) | - | - | - | - | - | - | 1438 (2.63x) | - |
| global linear path int32 | **3685** | schedule=sorted: 3679 | 3179 (1.16x) | 3572 (1.03x) | 2745 (1.34x) | 602 (6.12x) | 393 (9.38x) | 1859 (1.98x) | 2610 (1.41x) | 1281 (2.88x) | - | - |
| global linear path float32 | **3099** | schedule=sorted: 3124 | 2849 (1.09x) | 3016 (1.03x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **2125** | fill=striped: 2135 | 2027 (1.05x) | 2111 (1.01x) | - | - | - | - | - | - | 474 (4.49x) | - |
| global linear hard int32 | **6329** | schedule=sorted: 6209 | 4778 (1.32x) | 6301 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float32 | **5133** | schedule=sorted: 4994 | 4601 (1.12x) | 5200 (0.99x) | - | - | - | - | - | - | - | - |
| global linear hard float64 | **2821** | schedule=sorted: 2838 | 2857 (0.99x) | 2834 (1.00x) | - | - | - | - | - | - | - | - |
| global linear soft float64 | **986** | schedule=sorted: 945 | 799 (1.23x) | 933 (1.06x) | - | - | - | - | - | - | - | - |
| local affine score int32 | **7437** | schedule=sorted: 7648 | 6106 (1.22x) | 7775 (0.96x) | 3228 (2.30x) | >10x | - | - | - | - | - | - |
| local affine score float32 | **6773** | schedule=sorted: 6990 | 5614 (1.21x) | 6815 (0.99x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **3655** | fill=striped: 4346 | 2900 (1.26x) | 3605 (1.01x) | - | - | - | - | - | - | 631 (5.80x) | - |
| local affine path int32 | **5017** | schedule=sorted: 4664 | 4185 (1.20x) | 4537 (1.11x) | 2749 (1.82x) | 570 (8.80x) | - | - | - | - | - | - |
| local affine path float32 | **3979** | schedule=sorted: 3850 | 3485 (1.14x) | 3807 (1.05x) | - | - | - | - | - | - | - | - |
| local affine path float64 | **2257** | schedule=sorted: 2147 | 1875 (1.20x) | 2058 (1.10x) | - | - | - | - | - | - | 333 (6.78x) | - |
| local affine hard int32 | **4601** | schedule=sorted: 4661 | 3974 (1.16x) | 4670 (0.99x) | - | - | - | - | - | - | - | - |
| local affine hard float32 | **3915** | schedule=sorted: 4103 | 3555 (1.10x) | 3984 (0.98x) | - | - | - | - | - | - | - | - |
| local affine hard float64 | **2098** | schedule=sorted: 2142 | 1890 (1.11x) | 2091 (1.00x) | - | - | - | - | - | - | - | - |
| local affine soft float64 | **596** | schedule=sorted: 593 | 501 (1.19x) | 586 (1.02x) | - | - | - | - | - | - | - | - |
| local linear score int32 | **9951** | schedule=sorted: 9808 | 7100 (1.40x) | 10118 (0.98x) | 3120 (3.19x) | >10x | - | - | - | - | - | >10x |
| local linear score float32 | **6378** | schedule=sorted: 6357 | 6117 (1.04x) | 6405 (1.00x) | - | - | - | - | - | - | - | - |
| local linear score float64 | **3410** | fill=striped: 4277 | 3168 (1.08x) | 3456 (0.99x) | - | - | - | - | - | - | 986 (3.46x) | - |
| local linear path int32 | **5876** | schedule=sorted: 5756 | 5386 (1.09x) | 5859 (1.00x) | 2622 (2.24x) | >10x | - | - | - | - | - | - |
| local linear path float32 | **4405** | schedule=sorted: 4332 | 4061 (1.08x) | 4365 (1.01x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **2674** | schedule=sorted: 2671 | 2528 (1.06x) | 2638 (1.01x) | - | - | - | - | - | - | 318 (8.41x) | - |
| local linear hard int32 | **6753** | schedule=sorted: 6454 | 5977 (1.13x) | 6524 (1.04x) | - | - | - | - | - | - | - | - |
| local linear hard float32 | **4755** | schedule=sorted: 4891 | 4581 (1.04x) | 4920 (0.97x) | - | - | - | - | - | - | - | - |
| local linear hard float64 | **2775** | schedule=sorted: 2793 | 2632 (1.05x) | 2833 (0.98x) | - | - | - | - | - | - | - | - |
| local linear soft float64 | **918** | schedule=sorted: 888 | 726 (1.26x) | 864 (1.06x) | - | - | - | - | - | - | - | - |

#### spot · protein_L100 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **9648** | schedule=sorted: 9539 | 10549 (0.91x) | 9504 (1.02x) | 5461 (1.77x) | 2187 (4.41x) | - | 2946 (3.27x) | - | - |
| global affine score float32 | **5153** | schedule=sorted: 5142 | 4743 (1.09x) | 5070 (1.02x) | - | - | - | - | - | - |
| global affine score float64 | **4559** | schedule=sorted: 4534 | 3831 (1.19x) | 4521 (1.01x) | - | - | - | - | 1191 (3.83x) | - |
| global affine path int32 | **5613** | schedule=sorted: 5693 | 4495 (1.25x) | 5623 (1.00x) | 4281 (1.31x) | 1064 (5.28x) | - | 2805 (2.00x) | - | - |
| global affine path float32 | **4622** | schedule=sorted: 4647 | 3999 (1.16x) | 4596 (1.01x) | - | - | - | - | - | - |
| global affine path float64 | **3053** | schedule=sorted: 3050 | 2659 (1.15x) | 3056 (1.00x) | - | - | - | - | 606 (5.04x) | - |
| global affine hard int32 | **5465** | schedule=sorted: 5367 | 4340 (1.26x) | 5468 (1.00x) | - | - | - | - | - | - |
| global affine hard float32 | **4472** | schedule=sorted: 4380 | 3783 (1.18x) | 4486 (1.00x) | - | - | - | - | - | - |
| global affine hard float64 | **2973** | schedule=sorted: 2953 | 2328 (1.28x) | 2963 (1.00x) | - | - | - | - | - | - |
| global affine soft float64 | **594** | fill=striped: 580 | 555 (1.07x) | 563 (1.05x) | - | - | - | - | - | - |
| global linear score int32 | **8383** | schedule=sorted: 8750 | 9761 (0.86x) | 8629 (0.97x) | 5381 (1.56x) | 2104 (3.98x) | - | 3014 (2.78x) | - | - |
| global linear score float32 | **6666** | schedule=sorted: 6690 | 6063 (1.10x) | 6627 (1.01x) | - | - | - | - | - | - |
| global linear score float64 | **5058** | fill=striped: 5050 | 4308 (1.17x) | 5047 (1.00x) | - | - | - | - | 2734 (1.85x) | - |
| global linear path int32 | **6430** | schedule=sorted: 6525 | 6776 (0.95x) | 6491 (0.99x) | 4244 (1.52x) | 1069 (6.01x) | 849 (7.57x) | 2855 (2.25x) | - | - |
| global linear path float32 | **4993** | schedule=sorted: 5033 | 4424 (1.13x) | 4964 (1.01x) | - | - | - | - | - | - |
| global linear path float64 | **3594** | schedule=sorted: 3578 | 3245 (1.11x) | 3572 (1.01x) | - | - | - | - | 1015 (3.54x) | - |
| global linear hard int32 | **6534** | fill=striped: 6604 | 6897 (0.95x) | 6551 (1.00x) | - | - | - | - | - | - |
| global linear hard float32 | **4975** | schedule=sorted: 4926 | 4389 (1.13x) | 4905 (1.01x) | - | - | - | - | - | - |
| global linear hard float64 | **3617** | schedule=sorted: 3607 | 3264 (1.11x) | 3618 (1.00x) | - | - | - | - | - | - |
| global linear soft float64 | **1126** | fill=striped: 1078 | 919 (1.22x) | 1019 (1.10x) | - | - | - | - | - | - |
| local affine score int32 | **7384** | schedule=sorted: 7469 | 8528 (0.87x) | 8034 (0.92x) | 4214 (1.75x) | 1101 (6.71x) | - | - | - | - |
| local affine score float32 | **5604** | schedule=sorted: 5700 | 5109 (1.10x) | 5684 (0.99x) | - | - | - | - | - | - |
| local affine score float64 | **4689** | schedule=sorted: 4716 | 3929 (1.19x) | 4702 (1.00x) | - | - | - | - | 943 (4.97x) | - |
| local affine path int32 | **4826** | schedule=sorted: 4688 | 3912 (1.23x) | 4603 (1.05x) | 3448 (1.40x) | 907 (5.32x) | - | - | - | - |
| local affine path float32 | **3754** | schedule=sorted: 3728 | 2970 (1.26x) | 3439 (1.09x) | - | - | - | - | - | - |
| local affine path float64 | **2301** | schedule=sorted: 2171 | 1879 (1.22x) | 2118 (1.09x) | - | - | - | - | 352 (6.54x) | - |
| local affine hard int32 | **4575** | schedule=sorted: 4705 | 4112 (1.11x) | 4676 (0.98x) | - | - | - | - | - | - |
| local affine hard float32 | **3800** | schedule=sorted: 3864 | 3476 (1.09x) | 3786 (1.00x) | - | - | - | - | - | - |
| local affine hard float64 | **2376** | schedule=sorted: 2416 | 2260 (1.05x) | 2424 (0.98x) | - | - | - | - | - | - |
| local affine soft float64 | **627** | fill=striped: 628 | 603 (1.04x) | 628 (1.00x) | - | - | - | - | - | - |
| local linear score int32 | **8803** | schedule=sorted: 8648 | 9731 (0.90x) | 8704 (1.01x) | 4244 (2.07x) | 1093 (8.05x) | - | - | - | - |
| local linear score float32 | **6863** | fill=striped: 6949 | 6175 (1.11x) | 7100 (0.97x) | - | - | - | - | - | - |
| local linear score float64 | **5268** | fill=striped: 5267 | 4300 (1.23x) | 5220 (1.01x) | - | - | - | - | 1573 (3.35x) | - |
| local linear path int32 | **5223** | fill=striped: 5239 | 4468 (1.17x) | 5309 (0.98x) | 3432 (1.52x) | 873 (5.98x) | - | - | - | - |
| local linear path float32 | **3857** | fill=striped: 3981 | 3396 (1.14x) | 3899 (0.99x) | - | - | - | - | - | - |
| local linear path float64 | **2838** | schedule=sorted: 2845 | 2433 (1.17x) | 2855 (0.99x) | - | - | - | - | 455 (6.23x) | - |
| local linear hard int32 | **5124** | fill=striped: 5271 | 4480 (1.14x) | 5237 (0.98x) | - | - | - | - | - | - |
| local linear hard float32 | **3852** | fill=striped: 4055 | 3486 (1.11x) | 3880 (0.99x) | - | - | - | - | - | - |
| local linear hard float64 | **2847** | fill=striped: 2854 | 2468 (1.15x) | 2870 (0.99x) | - | - | - | - | - | - |
| local linear soft float64 | **991** | schedule=sorted: 947 | 827 (1.20x) | 937 (1.06x) | - | - | - | - | - | - |

#### spot · protein_L1000 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **10802** | schedule=sorted: 11748 | 12546 (0.86x) | 11206 (0.96x) | 7128 (1.52x) | 2569 (4.21x) | - | 4074 (2.65x) | - | - |
| global affine score float32 | **7477** | schedule=sorted: 7501 | 6480 (1.15x) | 7343 (1.02x) | - | - | - | - | - | - |
| global affine score float64 | **5699** | schedule=sorted: 5748 | 4646 (1.23x) | 5655 (1.01x) | - | - | - | - | 1246 (4.57x) | - |
| global affine path int32 | **4468** | fill=striped: 4472 | 4747 (0.94x) | 4121 (1.08x) | 5309 (0.84x) | 1154 (3.87x) | - | 3893 (1.15x) | - | - |
| global affine path float32 | **5865** | fill=striped: 5896 | 5184 (1.13x) | 5886 (1.00x) | - | - | - | - | - | - |
| global affine path float64 | **4602** | schedule=sorted: 4631 | 3890 (1.18x) | 4590 (1.00x) | - | - | - | - | 676 (6.80x) | - |
| global affine hard int32 | **3932** | fill=striped: 3994 | 4338 (0.91x) | 3812 (1.03x) | - | - | - | - | - | - |
| global affine hard float32 | **5865** | schedule=sorted: 5914 | 5229 (1.12x) | 5900 (0.99x) | - | - | - | - | - | - |
| global affine hard float64 | **4578** | fill=striped: 4429 | 3575 (1.28x) | 4370 (1.05x) | - | - | - | - | - | - |
| global affine soft float64 | **596** | fill=striped: 579 | 565 (1.05x) | 557 (1.07x) | - | - | - | - | - | - |
| global linear score int32 | **9315** | schedule=sorted: 8691 | 9383 (0.99x) | 10404 (0.90x) | 10027 (0.93x) | 2562 (3.64x) | - | 4065 (2.29x) | - | - |
| global linear score float32 | **12727** | schedule=sorted: 13308 | 10943 (1.16x) | 13044 (0.98x) | - | - | - | - | - | - |
| global linear score float64 | **7349** | schedule=sorted: 7397 | 5440 (1.35x) | 6889 (1.07x) | - | - | - | - | 2620 (2.81x) | - |
| global linear path int32 | **7318** | fill=striped: 7243 | 7709 (0.95x) | 6707 (1.09x) | 5144 (1.42x) | 1146 (6.39x) | 1060 (6.90x) | 3822 (1.91x) | - | - |
| global linear path float32 | **6721** | fill=striped: 6671 | 6746 (1.00x) | 6596 (1.02x) | - | - | - | - | - | - |
| global linear path float64 | **4037** | schedule=sorted: 4045 | 3618 (1.12x) | 4044 (1.00x) | - | - | - | - | 1180 (3.42x) | - |
| global linear hard int32 | **7470** | fill=striped: 7567 | 8226 (0.91x) | 7185 (1.04x) | - | - | - | - | - | - |
| global linear hard float32 | **6727** | schedule=sorted: 6799 | 6825 (0.99x) | 6664 (1.01x) | - | - | - | - | - | - |
| global linear hard float64 | **4015** | schedule=sorted: 4060 | 3610 (1.11x) | 4033 (1.00x) | - | - | - | - | - | - |
| global linear soft float64 | **1156** | fill=striped: 1162 | 1113 (1.04x) | 1151 (1.00x) | - | - | - | - | - | - |
| local affine score int32 | **9833** | schedule=sorted: 9887 | 9655 (1.02x) | 9680 (1.02x) | 5489 (1.79x) | 1210 (8.13x) | - | - | - | - |
| local affine score float32 | **8643** | schedule=sorted: 8607 | 7167 (1.21x) | 8479 (1.02x) | - | - | - | - | - | - |
| local affine score float64 | **6088** | schedule=sorted: 6104 | 4812 (1.27x) | 5986 (1.02x) | - | - | - | - | 980 (6.21x) | - |
| local affine path int32 | **3862** | fill=striped: 3863 | 3813 (1.01x) | 3799 (1.02x) | 4373 (0.88x) | 991 (3.90x) | - | - | - | - |
| local affine path float32 | **3098** | schedule=sorted: 3090 | 2968 (1.04x) | 3072 (1.01x) | - | - | - | - | - | - |
| local affine path float64 | **1911** | fill=striped: 1912 | 1620 (1.18x) | 1895 (1.01x) | - | - | - | - | 368 (5.20x) | - |
| local affine hard int32 | **3836** | fill=striped: 3737 | 3658 (1.05x) | 3618 (1.06x) | - | - | - | - | - | - |
| local affine hard float32 | **3006** | fill=striped: 3025 | 2845 (1.06x) | 2991 (1.01x) | - | - | - | - | - | - |
| local affine hard float64 | **1886** | fill=striped: 1853 | 1534 (1.23x) | 1804 (1.05x) | - | - | - | - | - | - |
| local affine soft float64 | **511** | fill=striped: 502 | 484 (1.05x) | 488 (1.05x) | - | - | - | - | - | - |
| local linear score int32 | **8842** | fill=striped: 9459 | 10635 (0.83x) | 9232 (0.96x) | 8049 (1.10x) | 1205 (7.34x) | - | - | - | - |
| local linear score float32 | **13477** | schedule=sorted: 13428 | 10390 (1.30x) | 13477 (1.00x) | - | - | - | - | - | - |
| local linear score float64 | **7214** | fill=striped: 7240 | 5589 (1.29x) | 7091 (1.02x) | - | - | - | - | 2245 (3.21x) | - |
| local linear path int32 | **5929** | schedule=sorted: 5900 | 6569 (0.90x) | 5863 (1.01x) | 5425 (1.09x) | 1000 (5.93x) | - | - | - | - |
| local linear path float32 | **5089** | fill=striped: 5082 | 4719 (1.08x) | 4975 (1.02x) | - | - | - | - | - | - |
| local linear path float64 | **2905** | schedule=sorted: 2900 | 2478 (1.17x) | 2907 (1.00x) | - | - | - | - | 479 (6.07x) | - |
| local linear hard int32 | **5969** | schedule=sorted: 5949 | 6670 (0.89x) | 5845 (1.02x) | - | - | - | - | - | - |
| local linear hard float32 | **5114** | fill=striped: 5127 | 4733 (1.08x) | 5051 (1.01x) | - | - | - | - | - | - |
| local linear hard float64 | **2910** | fill=striped: 2918 | 2469 (1.18x) | 2899 (1.00x) | - | - | - | - | - | - |
| local linear soft float64 | **979** | schedule=sorted: 982 | 989 (0.99x) | 979 (1.00x) | - | - | - | - | - | - |

#### spot · protein_L300 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **9694** | schedule=sorted: 10569 | 10893 (0.89x) | 10358 (0.94x) | 6954 (1.39x) | 2541 (3.82x) | - | 3669 (2.64x) | - | - |
| global affine score float32 | **6515** | fill=striped: 6602 | 5764 (1.13x) | 6509 (1.00x) | - | - | - | - | - | - |
| global affine score float64 | **5230** | fill=striped: 5301 | 4336 (1.21x) | 5215 (1.00x) | - | - | - | - | 1237 (4.23x) | - |
| global affine path int32 | **4008** | schedule=sorted: 4089 | 4771 (0.84x) | 4019 (1.00x) | 5210 (0.77x) | 1147 (3.49x) | - | 3526 (1.14x) | - | - |
| global affine path float32 | **3198** | fill=striped: 3235 | 3082 (1.04x) | 3106 (1.03x) | - | - | - | - | - | - |
| global affine path float64 | **2350** | fill=striped: 2268 | 1930 (1.22x) | 2155 (1.09x) | - | - | - | - | 653 (3.60x) | - |
| global affine hard int32 | **3762** | schedule=sorted: 3927 | 4378 (0.86x) | 3800 (0.99x) | - | - | - | - | - | - |
| global affine hard float32 | **3142** | fill=striped: 3115 | 2862 (1.10x) | 2941 (1.07x) | - | - | - | - | - | - |
| global affine hard float64 | **2074** | schedule=sorted: 2114 | 1848 (1.12x) | 2075 (1.00x) | - | - | - | - | - | - |
| global affine soft float64 | **599** | fill=striped: 590 | 575 (1.04x) | 587 (1.02x) | - | - | - | - | - | - |
| global linear score int32 | **9465** | schedule=sorted: 10249 | 10304 (0.92x) | 9526 (0.99x) | 6941 (1.36x) | 2553 (3.71x) | - | 3664 (2.58x) | - | - |
| global linear score float32 | **10884** | fill=striped: 10881 | 8653 (1.26x) | 9878 (1.10x) | - | - | - | - | - | - |
| global linear score float64 | **6702** | fill=striped: 6647 | 5407 (1.24x) | 6590 (1.02x) | - | - | - | - | 2694 (2.49x) | - |
| global linear path int32 | **9238** | fill=striped: 9251 | 9246 (1.00x) | 9030 (1.02x) | 4999 (1.85x) | 1151 (8.03x) | 997 (9.26x) | 3465 (2.67x) | - | - |
| global linear path float32 | **6901** | fill=striped: 7517 | 6695 (1.03x) | 7383 (0.93x) | - | - | - | - | - | - |
| global linear path float64 | **4628** | schedule=sorted: 4621 | 4013 (1.15x) | 4632 (1.00x) | - | - | - | - | 1126 (4.11x) | - |
| global linear hard int32 | **9653** | schedule=sorted: 9653 | 9573 (1.01x) | 9461 (1.02x) | - | - | - | - | - | - |
| global linear hard float32 | **7751** | schedule=sorted: 7861 | 7031 (1.10x) | 7679 (1.01x) | - | - | - | - | - | - |
| global linear hard float64 | **4783** | schedule=sorted: 4775 | 4101 (1.17x) | 4703 (1.02x) | - | - | - | - | - | - |
| global linear soft float64 | **1301** | fill=striped: 1302 | 1269 (1.03x) | 1299 (1.00x) | - | - | - | - | - | - |
| local affine score int32 | **9931** | fill=striped: 10261 | 10116 (0.98x) | 9703 (1.02x) | 5250 (1.89x) | 1191 (8.34x) | - | - | - | - |
| local affine score float32 | **7469** | schedule=sorted: 7285 | 6414 (1.16x) | 7422 (1.01x) | - | - | - | - | - | - |
| local affine score float64 | **5748** | fill=striped: 5737 | 4631 (1.24x) | 5632 (1.02x) | - | - | - | - | 995 (5.78x) | - |
| local affine path int32 | **3371** | schedule=sorted: 3380 | 4087 (0.82x) | 3357 (1.00x) | 4130 (0.82x) | 979 (3.44x) | - | - | - | - |
| local affine path float32 | **2909** | schedule=sorted: 2890 | 2788 (1.04x) | 2910 (1.00x) | - | - | - | - | - | - |
| local affine path float64 | **1848** | schedule=sorted: 1847 | 1663 (1.11x) | 1802 (1.03x) | - | - | - | - | 371 (4.98x) | - |
| local affine hard int32 | **3214** | fill=striped: 3184 | 3624 (0.89x) | 2968 (1.08x) | - | - | - | - | - | - |
| local affine hard float32 | **2753** | fill=striped: 2753 | 2607 (1.06x) | 2638 (1.04x) | - | - | - | - | - | - |
| local affine hard float64 | **1699** | schedule=sorted: 1678 | 1504 (1.13x) | 1635 (1.04x) | - | - | - | - | - | - |
| local affine soft float64 | **533** | fill=striped: 529 | 515 (1.03x) | 524 (1.02x) | - | - | - | - | - | - |
| local linear score int32 | **9124** | schedule=sorted: 8946 | 10177 (0.90x) | 9221 (0.99x) | 5282 (1.73x) | 1194 (7.64x) | - | - | - | - |
| local linear score float32 | **10781** | fill=striped: 10984 | 9006 (1.20x) | 10884 (0.99x) | - | - | - | - | - | - |
| local linear score float64 | **6700** | fill=striped: 6643 | 5215 (1.28x) | 6583 (1.02x) | - | - | - | - | 1959 (3.42x) | - |
| local linear path int32 | **7023** | schedule=sorted: 7151 | 7198 (0.98x) | 6882 (1.02x) | 4406 (1.59x) | 972 (7.23x) | - | - | - | - |
| local linear path float32 | **5813** | fill=striped: 5864 | 4890 (1.19x) | 5738 (1.01x) | - | - | - | - | - | - |
| local linear path float64 | **3370** | schedule=sorted: 3363 | 2778 (1.21x) | 3409 (0.99x) | - | - | - | - | 478 (7.05x) | - |
| local linear hard int32 | **7305** | schedule=sorted: 7286 | 7423 (0.98x) | 7245 (1.01x) | - | - | - | - | - | - |
| local linear hard float32 | **5960** | fill=striped: 5994 | 5102 (1.17x) | 5881 (1.01x) | - | - | - | - | - | - |
| local linear hard float64 | **3419** | schedule=sorted: 3420 | 2827 (1.21x) | 3470 (0.99x) | - | - | - | - | - | - |
| local linear soft float64 | **1113** | schedule=sorted: 1113 | 1079 (1.03x) | 1118 (0.99x) | - | - | - | - | - | - |

#### spot · protein_L3000 · 8 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **10233** | fill=striped: 11281 | 10492 (0.98x) | 10919 (0.94x) | 6965 (1.47x) | 2461 (4.16x) | - | 4164 (2.46x) | - | - |
| global affine score float32 | **7329** | schedule=sorted: 7343 | 6322 (1.16x) | 7116 (1.03x) | - | - | - | - | - | - |
| global affine score float64 | **4694** | schedule=sorted: 4732 | 4142 (1.13x) | 4723 (0.99x) | - | - | - | - | 1212 (3.87x) | - |
| global affine path int32 | **4324** | schedule=sorted: 4338 | 4628 (0.93x) | 4198 (1.03x) | 4254 (1.02x) | 1054 (4.10x) | - | 3758 (1.15x) | - | - |
| global affine path float32 | **6522** | fill=striped: 6449 | 5567 (1.17x) | 6400 (1.02x) | - | - | - | - | - | - |
| global affine path float64 | **4547** | schedule=sorted: 4546 | 3772 (1.21x) | 4182 (1.09x) | - | - | - | - | 677 (6.72x) | - |
| global affine hard int32 | **3962** | fill=striped: 3876 | 4142 (0.96x) | 3935 (1.01x) | - | - | - | - | - | - |
| global affine hard float32 | **6671** | fill=striped: 6520 | 5691 (1.17x) | 6587 (1.01x) | - | - | - | - | - | - |
| global affine hard float64 | **4172** | fill=striped: 3908 | 3323 (1.26x) | 3921 (1.06x) | - | - | - | - | - | - |
| global affine soft float64 | **515** | fill=striped: 503 | 491 (1.05x) | 484 (1.06x) | - | - | - | - | - | - |
| global linear score int32 | **8008** | fill=striped: 8247 | 8476 (0.94x) | 9542 (0.84x) | 9748 (0.82x) | 2468 (3.24x) | - | 3963 (2.02x) | - | - |
| global linear score float32 | **13347** | schedule=sorted: 13344 | 10022 (1.33x) | 12044 (1.11x) | - | - | - | - | - | - |
| global linear score float64 | **5462** | fill=striped: 5265 | 4339 (1.26x) | 5145 (1.06x) | - | - | - | - | 2681 (2.04x) | - |
| global linear path int32 | **6784** | fill=striped: 7332 | 7539 (0.90x) | 6901 (0.98x) | 4223 (1.61x) | 1049 (6.47x) | 1087 (6.24x) | 3714 (1.83x) | - | - |
| global linear path float32 | **7104** | schedule=sorted: 6947 | 6575 (1.08x) | 6901 (1.03x) | - | - | - | - | - | - |
| global linear path float64 | **3689** | schedule=sorted: 3711 | 3343 (1.10x) | 3760 (0.98x) | - | - | - | - | 1127 (3.27x) | - |
| global linear hard int32 | **7302** | schedule=sorted: 7303 | 7484 (0.98x) | 7121 (1.03x) | - | - | - | - | - | - |
| global linear hard float32 | **7085** | schedule=sorted: 7271 | 6408 (1.11x) | 6767 (1.05x) | - | - | - | - | - | - |
| global linear hard float64 | **3759** | fill=striped: 3786 | 3333 (1.13x) | 3679 (1.02x) | - | - | - | - | - | - |
| global linear soft float64 | **1059** | fill=striped: 1064 | 1010 (1.05x) | 1056 (1.00x) | - | - | - | - | - | - |
| local affine score int32 | **8690** | schedule=sorted: 8429 | 8367 (1.04x) | 9117 (0.95x) | 5266 (1.65x) | 1145 (7.59x) | - | - | - | - |
| local affine score float32 | **8429** | fill=striped: 8514 | 5938 (1.42x) | 8381 (1.01x) | - | - | - | - | - | - |
| local affine score float64 | **4963** | fill=striped: 4985 | 4054 (1.22x) | 4820 (1.03x) | - | - | - | - | 952 (5.21x) | - |
| local affine path int32 | **3159** | fill=striped: 3180 | 3183 (0.99x) | 3138 (1.01x) | 3553 (0.89x) | 923 (3.42x) | - | - | - | - |
| local affine path float32 | **2935** | schedule=sorted: 2945 | 2716 (1.08x) | 2928 (1.00x) | - | - | - | - | - | - |
| local affine path float64 | **1687** | fill=striped: 1684 | 1444 (1.17x) | 1664 (1.01x) | - | - | - | - | 359 (4.70x) | - |
| local affine hard int32 | **3192** | fill=striped: 3183 | 3085 (1.03x) | 3075 (1.04x) | - | - | - | - | - | - |
| local affine hard float32 | **2921** | schedule=sorted: 2989 | 2670 (1.09x) | 2939 (0.99x) | - | - | - | - | - | - |
| local affine hard float64 | **1606** | fill=striped: 1623 | 1368 (1.17x) | 1573 (1.02x) | - | - | - | - | - | - |
| local affine soft float64 | **443** | fill=striped: 440 | 432 (1.03x) | 439 (1.01x) | - | - | - | - | - | - |
| local linear score int32 | **7633** | fill=striped: 8914 | 8329 (0.92x) | 8765 (0.87x) | 8087 (0.94x) | 1171 (6.52x) | - | - | - | - |
| local linear score float32 | **11507** | schedule=sorted: 12302 | 9306 (1.24x) | 12247 (0.94x) | - | - | - | - | - | - |
| local linear score float64 | **5674** | schedule=sorted: 5748 | 4695 (1.21x) | 5501 (1.03x) | - | - | - | - | 2523 (2.25x) | - |
| local linear path int32 | **5791** | fill=striped: 5723 | 6026 (0.96x) | 5938 (0.98x) | 4409 (1.31x) | 935 (6.19x) | - | - | - | - |
| local linear path float32 | **5067** | fill=striped: 5147 | 4460 (1.14x) | 5038 (1.01x) | - | - | - | - | - | - |
| local linear path float64 | **2522** | schedule=sorted: 2515 | 2233 (1.13x) | 2524 (1.00x) | - | - | - | - | 472 (5.35x) | - |
| local linear hard int32 | **5808** | schedule=sorted: 5777 | 5983 (0.97x) | 5733 (1.01x) | - | - | - | - | - | - |
| local linear hard float32 | **5073** | schedule=sorted: 5262 | 4529 (1.12x) | 5240 (0.97x) | - | - | - | - | - | - |
| local linear hard float64 | **2519** | schedule=sorted: 2523 | 2239 (1.13x) | 2505 (1.01x) | - | - | - | - | - | - |
| local linear soft float64 | **885** | schedule=sorted: 888 | 905 (0.98x) | 877 (1.01x) | - | - | - | - | - | - |

</details>

<details><summary><b>wloczykij</b> — 4× Opteron 6380 (Piledriver), SSE4 — 64 cores, 8 NUMA nodes: every cell</summary>


#### wloczykij · dna_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 (load!) | **1504** | - | 1246 (1.21x) | 1446 (1.04x) | 1022 (1.47x) | 2361 (0.64x) | 248 (6.05x) | - | - | - | - | - | - | 174 (8.63x) sat | - |
| global affine score int32 (load!) | **745** | - | 693 (1.08x) | 740 (1.01x) | 665 (1.12x) | 1221 (0.61x) | 268 (2.78x) | - | 296 (2.52x) | 251 (2.97x) | - | - | - | - | - |
| global affine score int64 (load!) | **354** | - | 309 (1.15x) | 351 (1.01x) | 38 (9.26x) | 258 (1.37x) | 262 (1.35x) | - | - | - | - | - | - | - | - |
| global affine score float32 (load!) | **684** | kernel=sse2: 699 | 616 (1.11x) | 695 (0.98x) | - | 252 (2.71x) | 146 (4.68x) | - | - | - | - | - | - | - | - |
| global affine score float64 (load!) | **330** | fill=rowwise: 395 | 292 (1.13x) | 330 (1.00x) | - | 253 (1.30x) | 144 (2.30x) | - | - | - | - | 128 (2.58x) | - | - | - |
| global affine path int16 (load!) | **667** | - | 535 (1.25x) | 676 (0.99x) | 674 (0.99x) | 553 (1.21x) | 108 (6.17x) | - | - | - | - | - | - | 120 (5.54x) sat | - |
| global affine path int32 (load!) | **375** | - | 370 (1.01x) | 397 (0.94x) | 411 (0.91x) | 268 (1.40x) | 124 (3.02x) | - | 278 (1.35x) | 264 (1.42x) | - | - | - | - | - |
| global affine path int64 (load!) | **189** | - | 172 (1.10x) | 195 (0.97x) | 26 (7.22x) | 81 (2.33x) | 124 (1.53x) | - | - | - | - | - | - | - | - |
| global affine path float32 (load!) | **394** | traceback=scores: 399 | 324 (1.21x) | 397 (0.99x) | - | 73 (5.42x) | 105 (3.75x) | - | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **203** | traceback=scores: 207 | 180 (1.13x) | 197 (1.03x) | - | 68 (2.98x) | 104 (1.95x) | - | - | - | - | 36 (5.65x) | 35 (5.85x) | - | - |
| global affine hard int16 (load!) | **805** | - | 698 (1.15x) | 790 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int32 (load!) | **447** | - | 429 (1.04x) | 446 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **203** | - | 188 (1.08x) | 206 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **431** | traceback=scores: 446 | 367 (1.17x) | 434 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **208** | traceback=hirschberg_pmax: 226 | 197 (1.06x) | 218 (0.95x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **42** | api=functions: 51 | 41 (1.02x) | 39 (1.06x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear score int16 (load!) | **1720** | - | 1550 (1.11x) | 1835 (0.94x) | 1109 (1.55x) sat | 2232 (0.77x) | 348 (4.94x) | - | - | - | - | - | - | >10x | - |
| global linear score int32 (load!) | **1176** | - | 759 (1.55x) | 1153 (1.02x) | 664 (1.77x) | 1223 (0.96x) | 388 (3.03x) | - | 300 (3.92x) | 3052 (0.39x) | 1226 (0.96x) | - | - | - | - |
| global linear score int64 (load!) | **476** | - | 477 (1.00x) | 471 (1.01x) | >10x | 263 (1.81x) | 439 (1.09x) | - | - | - | - | - | - | - | - |
| global linear score float32 (load!) | **1020** | kernel=sse2: 786 | 747 (1.36x) | 1015 (1.01x) | - | 250 (4.07x) | 171 (5.95x) | - | - | - | - | - | - | - | - |
| global linear score float64 (load!) | **446** | fill=striped: 444 | 425 (1.05x) | 451 (0.99x) | - | 252 (1.77x) | 168 (2.66x) | - | - | - | - | 255 (1.75x) | - | - | - |
| global linear path int16 (load!) | **903** | - | 868 (1.04x) | 984 (0.92x) | 773 (1.17x) sat | 542 (1.67x) | 291 (3.10x) | - | - | - | - | - | - | 114 (7.95x) sat | - |
| global linear path int32 (load!) | **659** | - | 574 (1.15x) | 705 (0.94x) | 412 (1.60x) | 265 (2.49x) | 307 (2.14x) | - | 279 (2.37x) | 2507 (0.26x) | 454 (1.45x) | - | - | - | - |
| global linear path int64 (load!) | **373** | - | 392 (0.95x) | 367 (1.02x) | >10x | 79 (4.71x) | 307 (1.22x) | - | - | - | - | - | - | - | - |
| global linear path float32 (load!) | **676** | traceback=pointers: 693 | 534 (1.27x) | 686 (0.99x) | - | 71 (9.52x) | 150 (4.51x) | - | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **304** | traceback=scores: 307 | 306 (0.99x) | 311 (0.98x) | - | 70 (4.31x) | 151 (2.01x) | - | - | - | - | 60 (5.04x) | 48 (6.35x) | - | - |
| global linear hard int16 (load!) | **1399** | - | 1141 (1.23x) | 1357 (1.03x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int32 (load!) | **867** | - | 646 (1.34x) | 845 (1.03x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **414** | - | 439 (0.94x) | 406 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **863** | traceback=hirschberg: 883 | 642 (1.34x) | 814 (1.06x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **346** | fill=striped: 348 | 348 (0.99x) | 346 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **80** | api=functions: 83 | 71 (1.13x) | 80 (1.00x) | - | - | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 (load!) | **1218** | - | 1197 (1.02x) | 1241 (0.98x) | 1065 (1.14x) sat | 771 (1.58x) | 219 (5.58x) | 1024 (1.19x) sat | - | - | - | - | - | 157 (7.77x) sat | - |
| local affine score int32 (load!) | **631** | - | 634 (0.99x) | 662 (0.95x) | 717 (0.88x) | 397 (1.59x) | 224 (2.81x) | - | - | - | - | - | - | - | - |
| local affine score int64 (load!) | **272** | - | 263 (1.04x) | 284 (0.96x) | 38 (7.11x) | 116 (2.35x) | 224 (1.22x) | - | - | - | - | - | - | - | - |
| local affine score float32 (load!) | **506** | fill=striped: 693 | 534 (0.95x) | 516 (0.98x) | - | 101 (5.03x) | 132 (3.84x) | - | - | - | - | - | - | - | - |
| local affine score float64 (load!) | **266** | fill=rowwise: 431 | 268 (0.99x) | 278 (0.95x) | - | 100 (2.65x) | 132 (2.01x) | - | - | - | - | 80 (3.31x) | - | - | - |
| local affine path int16 (load!) | **564** | - | 451 (1.25x) | 574 (0.98x) | 612 (0.92x) | 534 (1.06x) | 78 (7.24x) | 244 (2.31x) sat | - | - | - | - | - | 96 (5.85x) sat | - |
| local affine path int32 (load!) | **345** | - | 332 (1.04x) | 349 (0.99x) | 361 (0.96x) | 249 (1.38x) | 85 (4.03x) | - | - | - | - | - | - | - | - |
| local affine path int64 (load!) | **159** | - | 154 (1.03x) | 162 (0.98x) | 22 (7.29x) | 77 (2.06x) | 88 (1.81x) | - | - | - | - | - | - | - | - |
| local affine path float32 (load!) | **287** | traceback=hirschberg_pmax: 298 | 276 (1.04x) | 290 (0.99x) | - | 49 (5.84x) | 62 (4.66x) | - | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **154** | traceback=hirschberg: 154 | 160 (0.96x) | 163 (0.94x) | - | 51 (2.99x) | 67 (2.29x) | - | - | - | - | 23 (6.63x) | 37 (4.16x) | - | - |
| local affine hard int16 (load!) | **696** | - | 613 (1.14x) | 642 (1.08x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int32 (load!) | **371** | - | 387 (0.96x) | 373 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **157** | - | 166 (0.94x) | 167 (0.94x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **320** | traceback=pointers: 330 | 323 (0.99x) | 326 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **165** | traceback=scores: 166 | 175 (0.95x) | 171 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **38** | api=functions: 48 | 37 (1.01x) | 36 (1.04x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear score int16 (load!) | **1573** | - | 1379 (1.14x) | 1601 (0.98x) | 985 (1.60x) sat | 772 (2.04x) | 321 (4.90x) | wrong | - | - | - | - | - | 186 (8.47x) sat | - |
| local linear score int32 (load!) | **858** | - | 727 (1.18x) | 861 (1.00x) | 512 (1.68x) | 394 (2.18x) | 325 (2.64x) | - | - | - | - | - | - | - | - |
| local linear score int64 (load!) | **325** | - | 313 (1.04x) | 329 (0.99x) | >10x | 115 (2.82x) | 326 (1.00x) | - | - | - | - | - | - | - | - |
| local linear score float32 (load!) | **733** | fill=striped: 718 | 667 (1.10x) | 723 (1.01x) | - | 102 (7.22x) | 152 (4.82x) | - | - | - | - | - | - | - | - |
| local linear score float64 (load!) | **379** | fill=striped: 432 | 332 (1.14x) | 372 (1.02x) | - | 102 (3.73x) | 153 (2.47x) | - | - | - | - | 185 (2.05x) | - | - | - |
| local linear path int16 (load!) | **782** | - | 464 (1.69x) | 810 (0.97x) | 697 (1.12x) sat | 530 (1.48x) | 106 (7.35x) | wrong | - | - | - | - | - | 149 (5.25x) sat | - |
| local linear path int32 (load!) | **538** | - | 462 (1.16x) | 547 (0.98x) | 365 (1.47x) | 253 (2.13x) | 106 (5.08x) | - | - | - | - | - | - | - | - |
| local linear path int64 (load!) | **284** | - | 213 (1.33x) | 278 (1.02x) | >10x | 77 (3.68x) | 111 (2.55x) | - | - | - | - | - | - | - | - |
| local linear path float32 (load!) | **491** | traceback=scores: 494 | 431 (1.14x) | 489 (1.00x) | - | 52 (9.37x) | 77 (6.41x) | - | - | - | - | - | - | - | - |
| local linear path float64 (load!) | **259** | traceback=hirschberg: 282 | 241 (1.08x) | 273 (0.95x) | - | 53 (4.87x) | 78 (3.33x) | - | - | - | - | 36 (7.27x) | 45 (5.81x) | - | - |
| local linear hard int16 (load!) | **1033** | - | 542 (1.90x) | 1008 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int32 (load!) | **638** | - | 548 (1.17x) | 632 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **310** | - | 237 (1.31x) | 309 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **586** | traceback=pointers: 587 | 495 (1.18x) | 562 (1.04x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **307** | traceback=scores: 310 | 261 (1.18x) | 296 (1.04x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **76** | api=functions: 78 | 65 (1.17x) | 75 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |

#### wloczykij · dna_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| global affine score int32 (load!) | **796** | - | 835 (0.95x) | 792 (1.00x) | 889 (0.90x) | 1114 (0.71x) | 269 (2.96x) | - | 443 (1.80x) | 933 (0.85x) | - | - | - | - | - |
| global affine score int64 (load!) | **405** | - | 417 (0.97x) | 406 (1.00x) | >10x | 257 (1.58x) | 232 (1.74x) | - | - | - | - | - | - | - | - |
| global affine score float32 (load!) | **711** | kernel=sse2: 711 | 707 (1.01x) | 701 (1.01x) | - | 261 (2.72x) | 146 (4.88x) | - | - | - | - | - | - | - | - |
| global affine score float64 (load!) | **424** | fill=striped: 428 | 424 (1.00x) | 429 (0.99x) | - | 253 (1.68x) | 140 (3.03x) | - | - | - | - | 161 (2.64x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| global affine path int32 (load!) | **311** | - | 324 (0.96x) | 308 (1.01x) | 430 (0.72x) | 211 (1.47x) | 137 (2.27x) | - | 414 (0.75x) | 658 (0.47x) | - | - | - | - | - |
| global affine path int64 (load!) | **166** | - | 184 (0.90x) | 166 (1.00x) | 23 (7.38x) | 87 (1.91x) | 132 (1.26x) | - | - | - | - | - | - | - | - |
| global affine path float32 (load!) | **492** | traceback=grid: 496 | 479 (1.03x) | 497 (0.99x) | - | 82 (6.02x) | 120 (4.11x) | - | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **328** | fill=rowwise: 329 | 323 (1.02x) | 328 (1.00x) | - | 80 (4.09x) | 111 (2.97x) | - | - | - | - | 44 (7.46x) | 65 (5.06x) | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| global affine hard int32 (load!) | **312** | - | 333 (0.94x) | 319 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **168** | - | 186 (0.90x) | 166 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **499** | traceback=grid: 499 | 493 (1.01x) | 495 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **326** | fill=striped: 331 | 318 (1.03x) | 324 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **8** | fill=striped: 8 | 8 (1.01x) | 8 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear score int16 (load!) | **1777** | - | 1755 (1.01x) | 1775 (1.00x) | 2508 (0.71x) | 2328 (0.76x) | 390 (4.56x) | - | - | - | - | - | - | 271 (6.55x) sat | - |
| global linear score int32 (load!) | **796** | - | 835 (0.95x) | 799 (1.00x) | 1227 (0.65x) | 1153 (0.69x) | 431 (1.84x) | - | 439 (1.81x) | 8101 (0.10x) | 12084 (0.07x) | - | - | - | - |
| global linear score int64 (load!) | **429** | - | 465 (0.92x) | 457 (0.94x) | >10x | 256 (1.68x) | 455 (0.94x) | - | - | - | - | - | - | - | - |
| global linear score float32 (load!) | **896** | fill=striped: 897 | 895 (1.00x) | 889 (1.01x) | - | 255 (3.52x) | 172 (5.21x) | - | - | - | - | - | - | - | - |
| global linear score float64 (load!) | **439** | kernel=sse2: 468 | 457 (0.96x) | 462 (0.95x) | - | 255 (1.73x) | 163 (2.69x) | - | - | - | - | 400 (1.10x) | - | - | - |
| global linear path int16 (load!) | **1096** | - | 1160 (0.94x) | 1099 (1.00x) | 851 (1.29x) | 468 (2.34x) | 341 (3.21x) | - | - | - | - | - | - | >10x | - |
| global linear path int32 (load!) | **493** | - | 515 (0.96x) | 494 (1.00x) | 427 (1.16x) | 212 (2.33x) | 358 (1.38x) | - | 413 (1.19x) | 8110 (0.06x) | 3233 (0.15x) | - | - | - | - |
| global linear path int64 (load!) | **226** | - | 290 (0.78x) | 227 (1.00x) | 23 (9.88x) | 83 (2.73x) | 362 (0.62x) | - | - | - | - | - | - | - | - |
| global linear path float32 (load!) | **588** | traceback=scores: 661 | 584 (1.01x) | 588 (1.00x) | - | 75 (7.88x) | 164 (3.59x) | - | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **258** | traceback=scores: 301 | 277 (0.93x) | 275 (0.94x) | - | 74 (3.50x) | 163 (1.59x) | - | - | - | - | 76 (3.41x) | 234 (1.10x) | - | - |
| global linear hard int16 (load!) | **1121** | - | 1131 (0.99x) | 1154 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int32 (load!) | **500** | - | 527 (0.95x) | 496 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **215** | - | 287 (0.75x) | 230 (0.94x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **557** | traceback=scores: 689 | 597 (0.93x) | 604 (0.92x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **279** | traceback=scores: 311 | 279 (1.00x) | 278 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **27** | fill=striped: 27 | 26 (1.04x) | 27 (0.99x) | - | - | - | - | - | - | - | - | - | - | 4 (7.23x) |
| local affine score int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| local affine score int32 | **712** | - | 710 (1.00x) | 707 (1.01x) | 1114 (0.64x) | 399 (1.79x) | 225 (3.17x) | - | - | - | - | - | - | - | - |
| local affine score int64 | **407** | - | 424 (0.96x) | 397 (1.02x) | 41 (9.86x) | 117 (3.48x) | 201 (2.02x) | - | - | - | - | - | - | - | - |
| local affine score float32 | **818** | fill=striped: 822 | 844 (0.97x) | 827 (0.99x) | - | 105 (7.80x) | 141 (5.81x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **443** | kernel=sse2: 446 | 439 (1.01x) | 438 (1.01x) | - | 103 (4.31x) | 130 (3.40x) | - | - | - | - | 93 (4.76x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| local affine path int32 (load!) | **251** | - | 267 (0.94x) | 248 (1.01x) | 379 (0.66x) | 201 (1.25x) | 97 (2.59x) | - | - | - | - | - | - | - | - |
| local affine path int64 (load!) | **136** | - | 150 (0.90x) | 138 (0.99x) | 24 (5.73x) | 77 (1.75x) | 92 (1.47x) | - | - | - | - | - | - | - | - |
| local affine path float32 (load!) | **285** | traceback=scores: 311 | 289 (0.99x) | 282 (1.01x) | - | 57 (5.04x) | 77 (3.72x) | - | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **144** | fill=striped: 143 | 145 (0.99x) | 144 (1.00x) | - | 57 (2.54x) | 76 (1.88x) | - | - | - | - | 26 (5.47x) | 96 (1.49x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 1000 x 1089 problem with a largest  | | | | | | | | | | | | | | |
| local affine hard int32 (load!) | **252** | - | 257 (0.98x) | 247 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **136** | - | 149 (0.91x) | 138 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **285** | traceback=scores: 314 | 290 (0.98x) | 284 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **136** | fill=striped: 141 | 146 (0.93x) | 145 (0.93x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **3** | fill=striped: 3 | 3 (0.96x) | 3 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **1551** | - | 1586 (0.98x) | 1535 (1.01x) | 2832 (0.55x) | 801 (1.94x) | 328 (4.72x) | 1942 (0.80x) sat | - | - | - | - | - | 242 (6.42x) sat | - |
| local linear score int32 | **710** | - | 712 (1.00x) | 702 (1.01x) | 1187 (0.60x) | 399 (1.78x) | 340 (2.09x) | - | - | - | - | - | - | - | - |
| local linear score int64 | **411** | - | 430 (0.95x) | 409 (1.00x) | 42 (9.81x) | 117 (3.51x) | 332 (1.24x) | - | - | - | - | - | - | - | - |
| local linear score float32 | **863** | fill=striped: 872 | 878 (0.98x) | 851 (1.01x) | - | 105 (8.21x) | 153 (5.66x) | - | - | - | - | - | - | - | - |
| local linear score float64 | **445** | fill=striped: 441 | 434 (1.03x) | 440 (1.01x) | - | 104 (4.29x) | 153 (2.90x) | - | - | - | - | 324 (1.38x) | - | - | - |
| local linear path int16 | **931** | - | 962 (0.97x) | 943 (0.99x) | 768 (1.21x) | 456 (2.04x) | 117 (7.93x) | 342 (2.72x) sat | - | - | - | - | - | 216 (4.31x) sat | - |
| local linear path int32 | **415** | - | 443 (0.94x) | 408 (1.02x) | 410 (1.01x) | 199 (2.09x) | 120 (3.45x) | - | - | - | - | - | - | - | - |
| local linear path int64 | **173** | - | 190 (0.91x) | 173 (1.00x) | 24 (7.10x) | 74 (2.35x) | 126 (1.38x) | - | - | - | - | - | - | - | - |
| local linear path float32 | **430** | traceback=scores: 504 | 457 (0.94x) | 425 (1.01x) | - | 55 (7.80x) | 84 (5.14x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **206** | traceback=scores: 240 | 211 (0.98x) | 207 (1.00x) | - | 57 (3.63x) | 84 (2.47x) | - | - | - | - | 41 (4.99x) | 175 (1.18x) | - | - |
| local linear hard int16 | **981** | - | 919 (1.07x) | 955 (1.03x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **421** | - | 443 (0.95x) | 405 (1.04x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **171** | - | 189 (0.91x) | 173 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **431** | traceback=scores: 535 | 461 (0.94x) | 431 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **208** | traceback=scores: 241 | 212 (0.99x) | 205 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **23** | api=functions: 23 | 30 (0.76x) | 23 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |

#### wloczykij · dna_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global affine score int32 | **734** | - | 760 (0.97x) | 740 (0.99x) | 1193 (0.62x) | 274 (2.68x) | 213 (3.44x) | 480 (1.53x) | 1478 (0.50x) | - | - | - | - |
| global affine score int64 | **426** | - | 458 (0.93x) | 444 (0.96x) | >10x | 252 (1.69x) | 232 (1.83x) | - | - | - | - | - | - |
| global affine score float32 | **833** | kernel=sse2: 839 | 832 (1.00x) | 839 (0.99x) | - | 259 (3.21x) | 142 (5.85x) | - | - | - | - | - | - |
| global affine score float64 | **452** | kernel=sse2: 455 | 441 (1.03x) | 459 (0.98x) | - | 258 (1.75x) | 139 (3.26x) | - | - | - | 161 (2.81x) | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global affine path int32 (load!) | **293** | - | 304 (0.96x) | 293 (1.00x) | 345 (0.85x) | 106 (2.77x) | 131 (2.24x) | 387 (0.76x) | 663 (0.44x) | - | - | - | - |
| global affine path int64 | **126** | - | 132 (0.95x) | 125 (1.01x) | 22 (5.60x) | 86 (1.46x) | 130 (0.96x) | - | - | - | - | - | - |
| global affine path float32 | **776** | kernel=sse2: 770 | 745 (1.04x) | 772 (1.00x) | - | 83 (9.33x) | 118 (6.60x) | - | - | - | - | - | - |
| global affine path float64 | **419** | fill=striped: 416 | 412 (1.02x) | 421 (1.00x) | - | 81 (5.17x) | 111 (3.79x) | - | - | - | 44 (9.48x) | 63 (6.65x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global affine hard int32 | **293** | - | 304 (0.96x) | 290 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **123** | - | 130 (0.95x) | 124 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **774** | fill=striped: 774 | 762 (1.02x) | 774 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **417** | fill=striped: 423 | 408 (1.02x) | 422 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **24** | fill=striped: 24 | 23 (1.04x) | 24 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global linear score int32 | **739** | - | 794 (0.93x) | 744 (0.99x) | 1331 (0.56x) | 271 (2.73x) | 435 (1.70x) | 479 (1.54x) | 12583 (0.06x) | 58482 (0.01x) | - | - | - |
| global linear score int64 | **452** | - | 474 (0.95x) | 463 (0.98x) | >10x | 255 (1.77x) | 432 (1.04x) | - | - | - | - | - | - |
| global linear score float32 (load!) | **889** | kernel=sse2: 939 | 916 (0.97x) | 924 (0.96x) | - | 260 (3.43x) | 172 (5.16x) | - | - | - | - | - | - |
| global linear score float64 (load!) | **463** | api=functions: 467 | 466 (0.99x) | 472 (0.98x) | - | 254 (1.82x) | 163 (2.84x) | - | - | - | 395 (1.17x) | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global linear path int32 (load!) | **475** | - | 484 (0.98x) | 478 (0.99x) | 342 (1.39x) | 106 (4.46x) | 350 (1.36x) | 388 (1.22x) | 7714 (0.06x) | 13443 (0.04x) | - | - | - |
| global linear path int64 | **231** | - | 291 (0.79x) | 214 (1.08x) | >10x | 81 (2.84x) | 346 (0.67x) | - | - | - | - | - | - |
| global linear path float32 | **566** | traceback=scores: 582 | 570 (0.99x) | 581 (0.97x) | - | 75 (7.59x) | 162 (3.50x) | - | - | - | - | - | - |
| global linear path float64 (load!) | **281** | traceback=scores: 293 | 283 (0.99x) | 281 (1.00x) | - | 73 (3.84x) | 161 (1.75x) | - | - | - | 75 (3.73x) | 190 (1.48x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| global linear hard int32 (load!) | **470** | - | 488 (0.96x) | 472 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **230** | - | 292 (0.79x) | 214 (1.08x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **569** | traceback=scores: 590 | 563 (1.01x) | 582 (0.98x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **282** | traceback=scores: 297 | 280 (1.01x) | 282 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **34** | fill=striped: 35 | 33 (1.05x) | 34 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local affine score int32 | **637** | - | 673 (0.95x) | 666 (0.96x) | 1221 (0.52x) | 216 (2.95x) | 187 (3.41x) | - | - | - | - | - | - |
| local affine score int64 | **407** | - | 429 (0.95x) | 409 (0.99x) | 42 (9.79x) | 116 (3.52x) | 197 (2.07x) | - | - | - | - | - | - |
| local affine score float32 | **861** | api=functions: 869 | 865 (1.00x) | 864 (1.00x) | - | 104 (8.28x) | 138 (6.25x) | - | - | - | - | - | - |
| local affine score float64 | **448** | kernel=sse2: 448 | 448 (1.00x) | 443 (1.01x) | - | 103 (4.34x) | 129 (3.46x) | - | - | - | 97 (4.60x) | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local affine path int32 | **122** | - | 128 (0.96x) | 122 (1.00x) | 310 (0.39x) | 102 (1.20x) | 96 (1.27x) | - | - | - | - | - | - |
| local affine path int64 | **51** | - | 53 (0.96x) | 51 (1.01x) | 23 (2.18x) | 76 (0.67x) | 91 (0.56x) | - | - | - | - | - | - |
| local affine path float32 (load!) | **119** | traceback=pointers: 297 | 120 (0.99x) | 120 (0.99x) | - | 58 (2.07x) | 76 (1.56x) | - | - | - | - | - | - |
| local affine path float64 (load!) | **145** | traceback=grid: 145 | 147 (0.99x) | 145 (1.01x) | - | 58 (2.51x) | 76 (1.90x) | - | - | - | 27 (5.43x) | 58 (2.49x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local affine hard int32 | **120** | - | 127 (0.95x) | 121 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **51** | - | 53 (0.96x) | 51 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **121** | traceback=scores: 315 | 120 (1.00x) | 120 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **145** | fill=striped: 145 | 147 (0.99x) | 144 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **11** | fill=striped: 11 | 12 (0.97x) | 11 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local linear score int32 | **661** | - | 655 (1.01x) | 667 (0.99x) | 1241 (0.53x) | 214 (3.09x) | 336 (1.97x) | - | - | - | - | - | - |
| local linear score int64 (load!) | **407** | - | 425 (0.96x) | 411 (0.99x) | 42 (9.68x) | 117 (3.47x) | 331 (1.23x) | - | - | - | - | - | - |
| local linear score float32 (load!) | **880** | api=functions: 894 | 904 (0.97x) | 887 (0.99x) | - | 104 (8.46x) | 153 (5.76x) | - | - | - | - | - | - |
| local linear score float64 (load!) | **442** | api=functions: 450 | 448 (0.99x) | 450 (0.98x) | - | 104 (4.26x) | 152 (2.90x) | - | - | - | 341 (1.30x) | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local linear path int32 (load!) | **393** | - | 416 (0.95x) | 394 (1.00x) | 330 (1.19x) | 102 (3.86x) | 119 (3.31x) | - | - | - | - | - | - |
| local linear path int64 (load!) | **177** | - | 192 (0.92x) | 167 (1.06x) | 24 (7.40x) | 77 (2.29x) | 124 (1.43x) | - | - | - | - | - | - |
| local linear path float32 (load!) | **423** | traceback=scores: 482 | 447 (0.95x) | 426 (0.99x) | - | 56 (7.52x) | 84 (5.04x) | - | - | - | - | - | - |
| local linear path float64 (load!) | **212** | traceback=scores: 218 | 216 (0.99x) | 212 (1.00x) | - | 57 (3.72x) | 84 (2.53x) | - | - | - | 42 (5.02x) | 152 (1.39x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10000 x 10549 problem with a larges | | | | | | | | | | | | |
| local linear hard int32 (load!) | **389** | - | 419 (0.93x) | 396 (0.98x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **177** | - | 191 (0.93x) | 166 (1.07x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **424** | traceback=scores: 480 | 445 (0.95x) | 426 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **212** | traceback=scores: 217 | 216 (0.98x) | 213 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **36** | api=functions: 37 | 36 (0.99x) | 36 (1.00x) | - | - | - | - | - | - | - | - | - |

#### wloczykij · mirna · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 (load!) | **1096** | - | 985 (1.11x) | 1143 (0.96x) | 620 (1.77x) | 1774 (0.62x) | 231 (4.75x) | - | - | - | - | - | - | >10x | - |
| global affine score int32 (load!) | **659** | - | 572 (1.15x) | 664 (0.99x) | 517 (1.28x) | 930 (0.71x) | 243 (2.71x) | - | 175 (3.77x) | >10x | - | - | - | >10x | - |
| global affine score int64 (load!) | **313** | - | 274 (1.14x) | 315 (0.99x) | 37 (8.43x) | 249 (1.26x) | 235 (1.33x) | - | - | - | - | - | - | - | - |
| global affine score float32 (load!) | **618** | kernel=sse2: 580 | 532 (1.16x) | 610 (1.01x) | - | 236 (2.62x) | 140 (4.42x) | - | - | - | - | - | - | - | - |
| global affine score float64 (load!) | **307** | fill=striped: 309 | 272 (1.13x) | 305 (1.01x) | - | 237 (1.30x) | 136 (2.25x) | - | - | - | - | 47 (6.54x) | - | - | - |
| global affine path int16 (load!) | **345** | - | 292 (1.18x) | 345 (1.00x) | 305 (1.13x) | 504 (0.68x) | 76 (4.51x) | - | - | - | - | - | - | >10x | - |
| global affine path int32 (load!) | **260** | - | 234 (1.11x) | 264 (0.99x) | 254 (1.02x) | 264 (0.98x) | 84 (3.09x) | - | 152 (1.71x) | >10x | - | - | - | - | - |
| global affine path int64 (load!) | **143** | - | 133 (1.08x) | 150 (0.96x) | 26 (5.51x) | 70 (2.06x) | 85 (1.68x) | - | - | - | - | - | - | - | - |
| global affine path float32 (load!) | **253** | traceback=pointers: 254 | 205 (1.23x) | 255 (0.99x) | - | 61 (4.15x) | 63 (4.01x) | - | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **153** | traceback=scores: 158 | 138 (1.11x) | 157 (0.98x) | - | 61 (2.52x) | 63 (2.44x) | - | - | - | - | 17 (9.29x) | >10x | - | - |
| global affine hard int16 (load!) | **500** | - | 460 (1.09x) | 522 (0.96x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int32 (load!) | **351** | - | 329 (1.07x) | 348 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **177** | - | 164 (1.08x) | 173 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **341** | traceback=hirschberg_pmax: 348 | 292 (1.17x) | 340 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **187** | traceback=hirschberg_pmax: 189 | 170 (1.10x) | 182 (1.03x) | - | - | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **50** | fill=rowwise: 38 | 44 (1.15x) | 50 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear score int16 (load!) | **1329** | - | 1167 (1.14x) | 1329 (1.00x) | 631 (2.11x) | 1785 (0.74x) | 321 (4.13x) | - | - | - | - | - | - | >10x | - |
| global linear score int32 (load!) | **901** | - | 662 (1.36x) | 924 (0.97x) | 589 (1.53x) | 925 (0.97x) | 349 (2.58x) | - | 180 (5.01x) | 175 (5.14x) | 186 (4.85x) | - | - | >10x | - |
| global linear score int64 (load!) | **340** | - | 360 (0.95x) | 336 (1.01x) | 37 (9.17x) | 251 (1.36x) | 366 (0.93x) | - | - | - | - | - | - | - | - |
| global linear score float32 (load!) | **827** | kernel=sse2: 693 | 626 (1.32x) | 829 (1.00x) | - | 235 (3.52x) | 167 (4.94x) | - | - | - | - | - | - | - | - |
| global linear score float64 (load!) | **339** | kernel=scalar_fallback: 346 | 342 (0.99x) | 340 (1.00x) | - | 236 (1.44x) | 162 (2.10x) | - | - | - | - | 62 (5.48x) | - | - | - |
| global linear path int16 | **305** | - | 275 (1.11x) | 309 (0.99x) | 295 (1.03x) | 500 (0.61x) | 178 (1.72x) | - | - | - | - | - | - | 32 (9.40x) sat | - |
| global linear path int32 | **263** | - | 240 (1.10x) | 274 (0.96x) | 249 (1.05x) | 266 (0.99x) | 185 (1.42x) | - | 154 (1.70x) | 163 (1.61x) | 79 (3.33x) | - | - | - | - |
| global linear path int64 | **207** | - | 212 (0.98x) | 211 (0.98x) | 26 (7.99x) | 69 (2.99x) | 185 (1.12x) | - | - | - | - | - | - | - | - |
| global linear path float32 | **244** | traceback=pointers: 243 | 214 (1.14x) | 253 (0.97x) | - | 54 (4.53x) | 111 (2.19x) | - | - | - | - | - | - | - | - |
| global linear path float64 | **181** | kernel=sse2: 179 | 177 (1.03x) | 185 (0.98x) | - | 62 (2.93x) | 112 (1.61x) | - | - | - | - | 25 (7.33x) | >10x | - | - |
| global linear hard int16 | **563** | - | 535 (1.05x) | 571 (0.99x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **449** | - | 395 (1.14x) | 456 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **299** | - | 309 (0.97x) | 298 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **424** | traceback=hirschberg: 424 | 359 (1.18x) | 423 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **252** | fill=striped: 257 | 254 (1.00x) | 251 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **74** | fill=striped: 60 | 65 (1.14x) | 73 (1.02x) | - | - | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 (load!) | **1039** | - | 937 (1.11x) | 945 (1.10x) | 738 (1.41x) | 684 (1.52x) | 203 (5.12x) | 561 (1.85x) sat | - | - | - | - | - | >10x | - |
| local affine score int32 (load!) | **582** | - | 544 (1.07x) | 562 (1.04x) | 448 (1.30x) | 343 (1.70x) | 204 (2.86x) | - | - | - | - | - | - | >10x | - |
| local affine score int64 (load!) | **263** | - | 233 (1.13x) | 259 (1.02x) | 33 (7.92x) | 111 (2.37x) | 201 (1.31x) | - | - | - | - | - | - | - | - |
| local affine score float32 (load!) | **480** | kernel=sse2: 509 | 471 (1.02x) | 467 (1.03x) | - | 97 (4.95x) | 135 (3.56x) | - | - | - | - | - | - | - | - |
| local affine score float64 | **261** | fill=rowwise: 374 | 242 (1.08x) | 258 (1.01x) | - | 96 (2.72x) | 127 (2.06x) | - | - | - | - | 36 (7.35x) | - | - | - |
| local affine path int16 | **574** | - | 517 (1.11x) | 564 (1.02x) | 349 (1.64x) | 489 (1.17x) | 69 (8.34x) | 372 (1.54x) sat | - | - | - | - | - | >10x | - |
| local affine path int32 | **368** | - | 344 (1.07x) | 365 (1.01x) | 273 (1.35x) | 255 (1.44x) | 76 (4.85x) | - | - | - | - | - | - | - | - |
| local affine path int64 (load!) | **162** | - | 152 (1.06x) | 159 (1.02x) | 17 (9.62x) | 70 (2.32x) | 76 (2.12x) | - | - | - | - | - | - | - | - |
| local affine path float32 (load!) | **303** | traceback=grid: 309 | 297 (1.02x) | 309 (0.98x) | - | 50 (6.04x) | 64 (4.74x) | - | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **163** | traceback=scores: 164 | 158 (1.03x) | 161 (1.01x) | - | 50 (3.25x) | 64 (2.55x) | - | - | - | - | 17 (9.54x) | >10x | - | - |
| local affine hard int16 (load!) | **602** | - | 564 (1.07x) | 599 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int32 (load!) | **373** | - | 364 (1.03x) | 370 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **164** | - | 159 (1.03x) | 161 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **312** | traceback=hirschberg_pmax: 326 | 299 (1.04x) | 321 (0.97x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **170** | traceback=hirschberg: 170 | 166 (1.02x) | 170 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **49** | api=functions: 36 | 44 (1.12x) | 48 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear score int16 (load!) | **1233** | - | 1078 (1.14x) | 1216 (1.01x) | 473 (2.61x) | 672 (1.83x) | 288 (4.29x) | wrong | - | - | - | - | - | >10x | - |
| local linear score int32 (load!) | **723** | - | 610 (1.19x) | 709 (1.02x) | 381 (1.90x) | 343 (2.11x) | 288 (2.51x) | - | - | - | - | - | - | >10x | - |
| local linear score int64 (load!) | **290** | - | 275 (1.05x) | 286 (1.01x) | >10x | 109 (2.67x) | 287 (1.01x) | - | - | - | - | - | - | - | - |
| local linear score float32 (load!) | **625** | kernel=sse2: 555 | 576 (1.09x) | 629 (0.99x) | - | 97 (6.41x) | 147 (4.24x) | - | - | - | - | - | - | - | - |
| local linear score float64 (load!) | **326** | fill=striped: 366 | 282 (1.16x) | 318 (1.02x) | - | 95 (3.44x) | 149 (2.19x) | - | - | - | - | 53 (6.18x) | - | - | - |
| local linear path int16 (load!) | **521** | - | 326 (1.60x) | 541 (0.96x) | 318 (1.64x) | 491 (1.06x) | 88 (5.92x) | wrong | - | - | - | - | - | >10x | - |
| local linear path int32 (load!) | **401** | - | 349 (1.15x) | 410 (0.98x) | 261 (1.54x) | 253 (1.59x) | 92 (4.36x) | - | - | - | - | - | - | >10x | - |
| local linear path int64 | **221** | - | 179 (1.24x) | 227 (0.97x) | >10x | 71 (3.12x) | 97 (2.28x) | - | - | - | - | - | - | - | - |
| local linear path float32 | **375** | traceback=scores: 378 | 336 (1.12x) | 369 (1.02x) | - | 48 (7.74x) | 72 (5.22x) | - | - | - | - | - | - | - | - |
| local linear path float64 | **229** | traceback=pointers: 227 | 198 (1.15x) | 226 (1.01x) | - | 50 (4.61x) | 72 (3.18x) | - | - | - | - | >10x | >10x | - | - |
| local linear hard int16 | **687** | - | 374 (1.84x) | 688 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int32 | **479** | - | 405 (1.18x) | 481 (1.00x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard int64 | **250** | - | 199 (1.26x) | 255 (0.98x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **449** | traceback=scores: 450 | 361 (1.24x) | 436 (1.03x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **254** | traceback=hirschberg: 254 | 212 (1.20x) | 248 (1.02x) | - | - | - | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **64** | api=functions: 49 | 55 (1.16x) | 63 (1.01x) | - | - | - | - | - | - | - | - | - | - | - |

#### wloczykij · protein_L100 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ssw | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | **818** | - | 1187 (0.69x) | 818 (1.00x) | 821 (1.00x) | 865 (0.95x) | 254 (3.21x) | - | - | - | - | 112 (7.28x) sat | - |
| global affine score int32 | **578** | - | 688 (0.84x) | 592 (0.98x) | 628 (0.92x) | 773 (0.75x) | 258 (2.24x) | - | 300 (1.93x) | - | - | - | - |
| global affine score int64 (load!) | **330** | - | 338 (0.98x) | 326 (1.01x) | 39 (8.55x) | 424 (0.78x) | 262 (1.26x) | - | - | - | - | - | - |
| global affine score float32 | **402** | fill=striped: 410 | 438 (0.92x) | 401 (1.00x) | - | 264 (1.52x) | 233 (1.73x) | - | - | - | - | - | - |
| global affine score float64 (load!) | **336** | fill=striped: 341 | 336 (1.00x) | 335 (1.00x) | - | 235 (1.43x) | 209 (1.60x) | - | - | 132 (2.54x) | - | - | - |
| global affine path int16 (load!) | **472** | - | 333 (1.42x) | 431 (1.10x) | 463 (1.02x) | 388 (1.22x) | 114 (4.13x) | - | - | - | - | 77 (6.13x) sat | - |
| global affine path int32 (load!) | **314** | - | 320 (0.98x) | 317 (0.99x) | 315 (1.00x) | 257 (1.22x) | 132 (2.38x) | - | 276 (1.14x) | - | - | - | - |
| global affine path int64 (load!) | **211** | - | 181 (1.17x) | 216 (0.98x) | 24 (8.67x) | 137 (1.54x) | 132 (1.60x) | - | - | - | - | - | - |
| global affine path float32 | **319** | kernel=sse2: 329 | 268 (1.19x) | 323 (0.99x) | - | 73 (4.40x) | 106 (3.01x) | - | - | - | - | - | - |
| global affine path float64 | **212** | kernel=sse2: 220 | 181 (1.17x) | 217 (0.97x) | - | 72 (2.94x) | 103 (2.06x) | - | - | 37 (5.77x) | 36 (5.86x) | - | - |
| global affine hard int16 | **476** | - | 370 (1.29x) | 438 (1.09x) | - | - | - | - | - | - | - | - | - |
| global affine hard int32 | **375** | - | 333 (1.13x) | 326 (1.15x) | - | - | - | - | - | - | - | - | - |
| global affine hard int64 | **215** | - | 182 (1.18x) | 214 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **358** | kernel=sse2: 365 | 280 (1.28x) | 324 (1.11x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **217** | traceback=grid: 219 | 184 (1.17x) | 219 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **38** | api=functions: 48 | 38 (1.01x) | 38 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear score int16 (load!) | **829** | - | 1179 (0.70x) | 809 (1.02x) | 834 (0.99x) | 869 (0.95x) | 411 (2.02x) | - | - | - | - | 140 (5.92x) sat | - |
| global linear score int32 (load!) | **585** | - | 680 (0.86x) | 579 (1.01x) | 453 (1.29x) | 774 (0.76x) | 418 (1.40x) | - | 298 (1.96x) | - | - | - | - |
| global linear score int64 (load!) | **377** | - | 403 (0.94x) | 380 (0.99x) | 38 (9.91x) | 410 (0.92x) | 480 (0.79x) | - | - | - | - | - | - |
| global linear score float32 (load!) | **527** | kernel=sse2: 514 | 575 (0.92x) | 528 (1.00x) | - | 246 (2.14x) | 310 (1.70x) | - | - | - | - | - | - |
| global linear score float64 (load!) | **372** | kernel=sse2: 378 | 364 (1.02x) | 381 (0.98x) | - | 239 (1.56x) | 315 (1.18x) | - | - | 246 (1.52x) | - | - | - |
| global linear path int16 (load!) | **544** | - | 628 (0.87x) | 554 (0.98x) | 461 (1.18x) | 391 (1.39x) | 288 (1.89x) | - | - | - | - | 93 (5.82x) sat | - |
| global linear path int32 (load!) | **404** | - | 450 (0.90x) | 396 (1.02x) | 317 (1.27x) | 256 (1.58x) | 312 (1.29x) | - | 279 (1.45x) | - | - | - | - |
| global linear path int64 (load!) | **368** | - | 384 (0.96x) | 363 (1.01x) | >10x | 139 (2.65x) | 311 (1.18x) | - | - | - | - | - | - |
| global linear path float32 (load!) | **449** | traceback=scores: 448 | 352 (1.28x) | 445 (1.01x) | - | 81 (5.57x) | 279 (1.61x) | - | - | - | - | - | - |
| global linear path float64 (load!) | **261** | kernel=scalar_fallback: 311 | 278 (0.94x) | 260 (1.00x) | - | 78 (3.34x) | 274 (0.95x) | - | - | 76 (3.45x) | 45 (5.76x) | - | - |
| global linear hard int16 (load!) | **587** | - | 690 (0.85x) | 594 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard int32 | **427** | - | 476 (0.90x) | 425 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard int64 | **376** | - | 399 (0.94x) | 373 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **473** | traceback=pointers: 477 | 389 (1.22x) | 476 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **274** | kernel=scalar_fallback: 326 | 290 (0.94x) | 274 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **28** | api=functions: 42 | 27 (1.04x) | 28 (1.01x) | - | - | - | - | - | - | - | - | >10x |
| local affine score int16 | **779** | - | 1083 (0.72x) | 760 (1.02x) | 628 (1.24x) | 630 (1.24x) | 224 (3.47x) | 311 (2.51x) sat | - | - | - | 104 (7.52x) sat | - |
| local affine score int32 | **550** | - | 601 (0.92x) | 548 (1.00x) | 435 (1.26x) | 354 (1.55x) | 226 (2.44x) | - | - | - | - | - | - |
| local affine score int64 (load!) | **305** | - | 332 (0.92x) | 308 (0.99x) | 31 (9.81x) | 185 (1.64x) | 218 (1.40x) | - | - | - | - | - | - |
| local affine score float32 | **453** | fill=striped: 460 | 489 (0.93x) | 448 (1.01x) | - | 98 (4.60x) | 197 (2.30x) | - | - | - | - | - | - |
| local affine score float64 | **349** | fill=striped: 352 | 351 (0.99x) | 345 (1.01x) | - | 93 (3.76x) | 188 (1.86x) | - | - | 98 (3.55x) | - | - | - |
| local affine path int16 | **391** | - | 317 (1.23x) | 405 (0.96x) | 429 (0.91x) | 356 (1.10x) | 96 (4.08x) | 133 (2.94x) sat | - | - | - | 70 (5.55x) sat | - |
| local affine path int32 | **289** | - | 262 (1.10x) | 299 (0.97x) | 281 (1.03x) | 233 (1.24x) | 109 (2.66x) | - | - | - | - | - | - |
| local affine path int64 (load!) | **173** | - | 161 (1.08x) | 175 (0.99x) | 21 (8.21x) | 123 (1.41x) | 109 (1.59x) | - | - | - | - | - | - |
| local affine path float32 | **263** | traceback=grid: 272 | 248 (1.06x) | 272 (0.96x) | - | 57 (4.59x) | 103 (2.54x) | - | - | - | - | - | - |
| local affine path float64 | **175** | traceback=hirschberg: 177 | 165 (1.06x) | 177 (0.99x) | - | 57 (3.09x) | 103 (1.71x) | - | - | 23 (7.74x) | 37 (4.73x) | - | - |
| local affine hard int16 | **426** | - | 343 (1.24x) | 419 (1.02x) | - | - | - | - | - | - | - | - | - |
| local affine hard int32 | **324** | - | 304 (1.06x) | 299 (1.08x) | - | - | - | - | - | - | - | - | - |
| local affine hard int64 | **175** | - | 164 (1.06x) | 178 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **289** | traceback=scores: 290 | 266 (1.09x) | 287 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **179** | traceback=pointers: 181 | 168 (1.06x) | 181 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **36** | api=functions: 46 | 37 (0.97x) | 35 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score int16 | **782** | - | 1071 (0.73x) | 784 (1.00x) | 630 (1.24x) | 578 (1.35x) | 308 (2.54x) | wrong | - | - | - | 107 (7.32x) sat | - |
| local linear score int32 | **552** | - | 595 (0.93x) | 542 (1.02x) | 381 (1.45x) | 353 (1.56x) | 324 (1.70x) | - | - | - | - | - | - |
| local linear score int64 | **344** | - | 375 (0.92x) | 346 (0.99x) | >10x | 185 (1.85x) | 369 (0.93x) | - | - | - | - | - | - |
| local linear score float32 (load!) | **580** | kernel=sse2: 579 | 603 (0.96x) | 579 (1.00x) | - | 97 (5.99x) | 249 (2.33x) | - | - | - | - | - | - |
| local linear score float64 (load!) | **382** | kernel=sse2: 368 | 394 (0.97x) | 373 (1.02x) | - | 91 (4.21x) | 253 (1.51x) | - | - | 145 (2.63x) | - | - | - |
| local linear path int16 (load!) | **527** | - | 307 (1.71x) | 531 (0.99x) | 425 (1.24x) | 358 (1.47x) | 109 (4.82x) | wrong | - | - | - | 94 (5.61x) sat | - |
| local linear path int32 (load!) | **435** | - | 350 (1.24x) | 447 (0.97x) | 283 (1.54x) | 230 (1.89x) | 113 (3.85x) | - | - | - | - | - | - |
| local linear path int64 (load!) | **281** | - | 207 (1.35x) | 284 (0.99x) | >10x | 122 (2.30x) | 112 (2.50x) | - | - | - | - | - | - |
| local linear path float32 (load!) | **396** | traceback=hirschberg: 394 | 331 (1.20x) | 396 (1.00x) | - | 60 (6.55x) | 82 (4.84x) | - | - | - | - | - | - |
| local linear path float64 (load!) | **282** | traceback=hirschberg: 285 | 249 (1.13x) | 288 (0.98x) | - | 60 (4.69x) | 82 (3.44x) | - | - | 41 (6.85x) | 45 (6.22x) | - | - |
| local linear hard int16 (load!) | **548** | - | 318 (1.73x) | 544 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard int32 (load!) | **458** | - | 363 (1.26x) | 459 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **289** | - | 212 (1.37x) | 290 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **408** | traceback=hirschberg: 409 | 350 (1.17x) | 410 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **290** | traceback=pointers: 295 | 258 (1.12x) | 297 (0.98x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **67** | api=functions: 73 | 67 (1.00x) | 67 (1.01x) | - | - | - | - | - | - | - | - | - |

#### wloczykij · protein_L1000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global affine score int32 (load!) | **780** | - | 760 (1.03x) | 732 (1.07x) | 702 (1.11x) | 654 (1.19x) | 242 (3.23x) | 424 (1.84x) | - | - | - | - |
| global affine score int64 (load!) | **358** | - | 369 (0.97x) | 356 (1.01x) | 39 (9.26x) | 428 (0.84x) | 230 (1.56x) | - | - | - | - | - |
| global affine score float32 (load!) | **564** | kernel=sse2: 562 | 575 (0.98x) | 566 (1.00x) | - | 273 (2.06x) | 234 (2.41x) | - | - | - | - | - |
| global affine score float64 (load!) | **366** | fill=rowwise: 386 | 385 (0.95x) | 387 (0.94x) | - | 238 (1.54x) | 217 (1.69x) | - | 153 (2.39x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global affine path int32 (load!) | **284** | - | 295 (0.96x) | 280 (1.01x) | 398 (0.71x) | 187 (1.52x) | 134 (2.11x) | 391 (0.73x) | - | - | - | - |
| global affine path int64 (load!) | **148** | - | 163 (0.91x) | 147 (1.01x) | 23 (6.44x) | 106 (1.40x) | 126 (1.18x) | - | - | - | - | - |
| global affine path float32 (load!) | **403** | kernel=sse2: 426 | 424 (0.95x) | 426 (0.95x) | - | 79 (5.07x) | 118 (3.42x) | - | - | - | - | - |
| global affine path float64 (load!) | **299** | fill=rowwise: 302 | 295 (1.01x) | 296 (1.01x) | - | 75 (3.96x) | 110 (2.72x) | - | 39 (7.69x) | 76 (3.91x) | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global affine hard int32 (load!) | **300** | - | 315 (0.95x) | 300 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **158** | - | 173 (0.91x) | 155 (1.02x) | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **424** | traceback=grid: 428 | 427 (0.99x) | 420 (1.01x) | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **298** | fill=rowwise: 302 | 295 (1.01x) | 299 (1.00x) | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **11** | fill=striped: 11 | 11 (1.03x) | 11 (1.00x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global linear score int32 (load!) | **737** | - | 811 (0.91x) | 761 (0.97x) | 1091 (0.68x) | 660 (1.12x) | 399 (1.85x) | 424 (1.74x) | - | - | - | - |
| global linear score int64 (load!) | **425** | - | 445 (0.95x) | 425 (1.00x) | >10x | 404 (1.05x) | 473 (0.90x) | - | - | - | - | - |
| global linear score float32 (load!) | **857** | kernel=sse2: 859 | 836 (1.03x) | 803 (1.07x) | - | 257 (3.34x) | 304 (2.82x) | - | - | - | - | - |
| global linear score float64 (load!) | **444** | kernel=sse2: 457 | 444 (1.00x) | 452 (0.98x) | - | 238 (1.87x) | 306 (1.45x) | - | 403 (1.10x) | - | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global linear path int32 (load!) | **490** | - | 516 (0.95x) | 483 (1.01x) | 428 (1.15x) | 187 (2.62x) | 360 (1.36x) | 415 (1.18x) | - | - | - | - |
| global linear path int64 (load!) | **227** | - | 289 (0.79x) | 226 (1.01x) | 23 (9.79x) | 104 (2.19x) | 360 (0.63x) | - | - | - | - | - |
| global linear path float32 (load!) | **570** | traceback=scores: 649 | 567 (1.01x) | 564 (1.01x) | - | 90 (6.34x) | 323 (1.77x) | - | - | - | - | - |
| global linear path float64 (load!) | **270** | traceback=scores: 298 | 268 (1.01x) | 266 (1.02x) | - | 89 (3.04x) | 343 (0.79x) | - | 104 (2.61x) | 242 (1.12x) | - | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| global linear hard int32 (load!) | **496** | - | 514 (0.96x) | 485 (1.02x) | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **227** | - | 279 (0.81x) | 229 (0.99x) | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **569** | traceback=scores: 665 | 545 (1.04x) | 546 (1.04x) | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **255** | traceback=scores: 288 | 278 (0.92x) | 272 (0.93x) | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **27** | api=functions: 27 | 24 (1.11x) | 27 (1.01x) | - | - | - | - | - | - | - | 4 (7.39x) |
| local affine score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local affine score int32 (load!) | **685** | - | 699 (0.98x) | 686 (1.00x) | 763 (0.90x) | 326 (2.10x) | 210 (3.25x) | - | - | - | - | - |
| local affine score int64 (load!) | **336** | - | 370 (0.91x) | 340 (0.99x) | 36 (9.35x) | 196 (1.71x) | 198 (1.69x) | - | - | - | - | - |
| local affine score float32 (load!) | **630** | api=functions: 624 | 641 (0.98x) | 627 (1.00x) | - | 102 (6.17x) | 192 (3.28x) | - | - | - | - | - |
| local affine score float64 (load!) | **398** | fill=striped: 404 | 398 (1.00x) | 396 (1.01x) | - | 94 (4.23x) | 188 (2.11x) | - | 123 (3.24x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local affine path int32 (load!) | **238** | - | 256 (0.93x) | 238 (1.00x) | 346 (0.69x) | 165 (1.44x) | 113 (2.10x) | - | - | - | - | - |
| local affine path int64 (load!) | **130** | - | 143 (0.91x) | 129 (1.00x) | 22 (5.87x) | 94 (1.38x) | 107 (1.21x) | - | - | - | - | - |
| local affine path float32 (load!) | **243** | fill=striped: 243 | 245 (0.99x) | 241 (1.01x) | - | 55 (4.43x) | 102 (2.39x) | - | - | - | - | - |
| local affine path float64 (load!) | **135** | traceback=pointers: 138 | 133 (1.02x) | 129 (1.04x) | - | 60 (2.26x) | 106 (1.27x) | - | 23 (5.87x) | 70 (1.92x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local affine hard int32 (load!) | **251** | - | 273 (0.92x) | 250 (1.00x) | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **129** | - | 142 (0.91x) | 132 (0.98x) | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **258** | traceback=grid: 257 | 255 (1.02x) | 251 (1.03x) | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **137** | traceback=pointers: 137 | 140 (0.98x) | 137 (1.00x) | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **11** | fill=striped: 11 | 11 (0.99x) | 11 (1.00x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local linear score int32 (load!) | **669** | - | 695 (0.96x) | 674 (0.99x) | 1084 (0.62x) | 343 (1.95x) | 327 (2.05x) | - | - | - | - | - |
| local linear score int64 (load!) | **403** | - | 414 (0.97x) | 405 (1.00x) | 41 (9.76x) | 195 (2.07x) | 384 (1.05x) | - | - | - | - | - |
| local linear score float32 (load!) | **823** | kernel=sse2: 831 | 861 (0.96x) | 835 (0.99x) | - | 101 (8.12x) | 253 (3.26x) | - | - | - | - | - |
| local linear score float64 (load!) | **428** | api=functions: 436 | 443 (0.97x) | 433 (0.99x) | - | 95 (4.53x) | 248 (1.73x) | - | 271 (1.58x) | - | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local linear path int32 (load!) | **406** | - | 443 (0.92x) | 409 (0.99x) | 384 (1.06x) | 169 (2.41x) | 122 (3.33x) | - | - | - | - | - |
| local linear path int64 (load!) | **165** | - | 188 (0.88x) | 169 (0.97x) | 23 (7.03x) | 97 (1.70x) | 132 (1.24x) | - | - | - | - | - |
| local linear path float32 (load!) | **423** | traceback=scores: 513 | 453 (0.93x) | 419 (1.01x) | - | 63 (6.75x) | 88 (4.82x) | - | - | - | - | - |
| local linear path float64 (load!) | **203** | traceback=scores: 240 | 211 (0.96x) | 207 (0.98x) | - | 63 (3.23x) | 88 (2.32x) | - | 49 (4.11x) | 217 (0.94x) | - | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 969 x 1041 problem with a largest s | | | | | | | | | | | |
| local linear hard int32 (load!) | **409** | - | 448 (0.91x) | 410 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **169** | - | 190 (0.89x) | 168 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **429** | traceback=scores: 533 | 430 (1.00x) | 401 (1.07x) | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **195** | traceback=scores: 242 | 212 (0.92x) | 206 (0.95x) | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **20** | api=functions: 21 | 20 (1.00x) | 20 (0.99x) | - | - | - | - | - | - | - | - |

#### wloczykij · protein_L10000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global affine score int32 (load!) | **738** | - | 801 (0.92x) | 741 (1.00x) | 764 (0.97x) | 295 (2.50x) | 213 (3.46x) | 480 (1.54x) | - | - | - |
| global affine score int64 (load!) | **377** | - | 391 (0.96x) | 375 (1.01x) | 39 (9.70x) | 302 (1.25x) | 232 (1.63x) | - | - | - | - |
| global affine score float32 (load!) | **622** | kernel=sse2: 628 | 626 (0.99x) | 623 (1.00x) | - | 269 (2.31x) | 245 (2.54x) | - | - | - | - |
| global affine score float64 (load!) | **397** | api=functions: 402 | 399 (1.00x) | 407 (0.98x) | - | 238 (1.67x) | 234 (1.70x) | - | 158 (2.51x) | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global affine path int32 | **287** | - | 309 (0.93x) | 289 (0.99x) | 320 (0.90x) | 98 (2.93x) | 135 (2.12x) | 388 (0.74x) | - | - | - |
| global affine path int64 (load!) | **124** | - | 131 (0.95x) | 124 (1.00x) | 23 (5.50x) | 89 (1.39x) | 134 (0.92x) | - | - | - | - |
| global affine path float32 (load!) | **584** | traceback=grid: 564 | 575 (1.01x) | 580 (1.01x) | - | 82 (7.15x) | 114 (5.13x) | - | - | - | - |
| global affine path float64 | **367** | traceback=grid: 373 | 366 (1.00x) | 369 (0.99x) | - | 80 (4.57x) | 113 (3.24x) | - | 41 (8.98x) | 74 (4.95x) | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global affine hard int32 | **284** | - | 303 (0.94x) | 285 (0.99x) | - | - | - | - | - | - | - |
| global affine hard int64 | **122** | - | 131 (0.93x) | 125 (0.98x) | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **582** | kernel=sse2: 581 | 571 (1.02x) | 583 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **374** | fill=rowwise: 374 | 359 (1.04x) | 374 (1.00x) | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **23** | fill=striped: 23 | 22 (1.02x) | 23 (0.99x) | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global linear score int32 | **689** | - | 785 (0.88x) | 738 (0.93x) | 1287 (0.54x) | 294 (2.34x) | 416 (1.66x) | 474 (1.45x) | - | - | 271 (2.54x) sat |
| global linear score int64 (load!) | **455** | - | 476 (0.96x) | 448 (1.02x) | >10x | 303 (1.50x) | 495 (0.92x) | - | - | - | - |
| global linear score float32 (load!) | **915** | kernel=sse2: 938 | 917 (1.00x) | 931 (0.98x) | - | 271 (3.38x) | 329 (2.78x) | - | - | - | - |
| global linear score float64 (load!) | **457** | kernel=sse2: 463 | 448 (1.02x) | 470 (0.97x) | - | 238 (1.92x) | 305 (1.50x) | - | 403 (1.13x) | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global linear path int32 (load!) | **468** | - | 485 (0.97x) | 458 (1.02x) | 323 (1.45x) | 102 (4.60x) | 340 (1.38x) | 386 (1.21x) | - | - | 75 (6.22x) sat |
| global linear path int64 (load!) | **230** | - | 290 (0.79x) | 227 (1.01x) | >10x | 90 (2.55x) | 340 (0.68x) | - | - | - | - |
| global linear path float32 | **562** | traceback=scores: 568 | 559 (1.00x) | 577 (0.97x) | - | 88 (6.41x) | 332 (1.69x) | - | - | - | - |
| global linear path float64 (load!) | **274** | traceback=pointers: 276 | 273 (1.00x) | 269 (1.02x) | - | 86 (3.19x) | 330 (0.83x) | - | 100 (2.75x) | 220 (1.24x) | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| global linear hard int32 (load!) | **469** | - | 454 (1.03x) | 466 (1.01x) | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **229** | - | 289 (0.79x) | 227 (1.01x) | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **570** | traceback=scores: 572 | 564 (1.01x) | 570 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **276** | traceback=pointers: 275 | 275 (1.00x) | 277 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **62** | api=functions: 62 | 59 (1.04x) | 62 (1.00x) | - | - | - | - | - | - | - |
| local affine score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local affine score int32 | **673** | - | 670 (1.00x) | 669 (1.01x) | 839 (0.80x) | 182 (3.71x) | 184 (3.65x) | - | - | - | - |
| local affine score int64 | **364** | - | 377 (0.96x) | 364 (1.00x) | 37 (9.71x) | 179 (2.03x) | 198 (1.84x) | - | - | - | - |
| local affine score float32 | **681** | fill=striped: 686 | 695 (0.98x) | 689 (0.99x) | - | 101 (6.74x) | 190 (3.58x) | - | - | - | - |
| local affine score float64 | **415** | fill=rowwise: 410 | 410 (1.01x) | 416 (1.00x) | - | 94 (4.43x) | 188 (2.21x) | - | 138 (3.02x) | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local affine path int32 | **119** | - | 127 (0.94x) | 121 (0.99x) | 279 (0.43x) | 86 (1.39x) | 125 (0.95x) | - | - | - | - |
| local affine path int64 (load!) | **50** | - | 53 (0.96x) | 50 (1.00x) | 23 (2.23x) | 82 (0.61x) | 119 (0.42x) | - | - | - | - |
| local affine path float32 (load!) | **119** | traceback=grid: 264 | 117 (1.02x) | 117 (1.02x) | - | 65 (1.83x) | 121 (0.99x) | - | - | - | - |
| local affine path float64 (load!) | **138** | fill=striped: 138 | 142 (0.97x) | 139 (1.00x) | - | 64 (2.15x) | 116 (1.19x) | - | 23 (5.90x) | 71 (1.96x) | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local affine hard int32 (load!) | **121** | - | 125 (0.96x) | 120 (1.00x) | - | - | - | - | - | - | - |
| local affine hard int64 | **50** | - | 52 (0.96x) | 51 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **119** | traceback=grid: 264 | 118 (1.01x) | 119 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **138** | traceback=grid: 139 | 142 (0.97x) | 139 (0.99x) | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **17** | fill=striped: 17 | 17 (1.02x) | 17 (1.00x) | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local linear score int32 (load!) | **636** | - | 644 (0.99x) | 639 (1.00x) | 1160 (0.55x) | 174 (3.66x) | 313 (2.03x) | - | - | - | 173 (3.67x) sat |
| local linear score int64 (load!) | **395** | - | 421 (0.94x) | 404 (0.98x) | 42 (9.43x) | 177 (2.23x) | 378 (1.05x) | - | - | - | - |
| local linear score float32 (load!) | **846** | api=functions: 847 | 865 (0.98x) | 851 (0.99x) | - | 97 (8.76x) | 243 (3.49x) | - | - | - | - |
| local linear score float64 (load!) | **422** | api=functions: 447 | 435 (0.97x) | 423 (1.00x) | - | 88 (4.80x) | 231 (1.83x) | - | 332 (1.27x) | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local linear path int32 (load!) | **388** | - | 411 (0.94x) | 372 (1.04x) | 278 (1.39x) | 85 (4.58x) | 128 (3.04x) | - | - | - | 163 (2.38x) sat |
| local linear path int64 (load!) | **173** | - | 190 (0.91x) | 175 (0.99x) | 24 (7.37x) | 82 (2.11x) | 128 (1.35x) | - | - | - | - |
| local linear path float32 (load!) | **422** | traceback=scores: 438 | 444 (0.95x) | 426 (0.99x) | - | 59 (7.12x) | 85 (4.95x) | - | - | - | - |
| local linear path float64 (load!) | **197** | traceback=scores: 218 | 208 (0.94x) | 208 (0.95x) | - | 59 (3.35x) | 84 (2.33x) | - | 48 (4.09x) | 191 (1.03x) | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 10492 x 11143 problem with a larges | | | | | | | | | | |
| local linear hard int32 (load!) | **391** | - | 410 (0.95x) | 390 (1.00x) | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **175** | - | 190 (0.92x) | 177 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **420** | traceback=scores: 454 | 445 (0.94x) | 425 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **210** | traceback=scores: 219 | 210 (1.00x) | 210 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **50** | api=functions: 54 | 50 (1.00x) | 50 (1.00x) | - | - | - | - | - | - | - |

#### wloczykij · protein_L300 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| global affine score int32 (load!) | **828** | - | 888 (0.93x) | 818 (1.01x) | 696 (1.19x) | 799 (1.04x) | 271 (3.06x) | 399 (2.08x) | - | - | - | - |
| global affine score int64 (load!) | **365** | - | 376 (0.97x) | 362 (1.01x) | 39 (9.34x) | 437 (0.84x) | 265 (1.38x) | - | - | - | - | - |
| global affine score float32 (load!) | **537** | kernel=sse2: 553 | 559 (0.96x) | 533 (1.01x) | - | 272 (1.98x) | 239 (2.25x) | - | - | - | - | - |
| global affine score float64 (load!) | **379** | fill=rowwise: 379 | 378 (1.00x) | 380 (1.00x) | - | 239 (1.59x) | 215 (1.76x) | - | 158 (2.40x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| global affine path int32 (load!) | **305** | - | 329 (0.93x) | 316 (0.96x) | 433 (0.70x) | 222 (1.37x) | 141 (2.17x) | 374 (0.81x) | - | - | - | - |
| global affine path int64 (load!) | **155** | - | 171 (0.91x) | 153 (1.01x) | 24 (6.53x) | 124 (1.24x) | 138 (1.12x) | - | - | - | - | - |
| global affine path float32 (load!) | **275** | traceback=scores: 336 | 269 (1.02x) | 277 (1.00x) | - | 76 (3.63x) | 113 (2.44x) | - | - | - | - | - |
| global affine path float64 (load!) | **162** | traceback=scores: 185 | 170 (0.96x) | 162 (1.00x) | - | 76 (2.14x) | 111 (1.46x) | - | 41 (3.93x) | 94 (1.72x) | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| global affine hard int32 (load!) | **316** | - | 333 (0.95x) | 320 (0.99x) | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **154** | - | 172 (0.90x) | 153 (1.01x) | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **280** | traceback=scores: 352 | 277 (1.01x) | 278 (1.01x) | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **165** | traceback=scores: 189 | 172 (0.96x) | 163 (1.01x) | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **12** | fill=striped: 12 | 12 (1.03x) | 12 (0.99x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| global linear score int32 (load!) | **829** | - | 887 (0.93x) | 818 (1.01x) | 691 (1.20x) | 799 (1.04x) | 428 (1.94x) | 403 (2.05x) | - | - | - | - |
| global linear score int64 (load!) | **446** | - | 470 (0.95x) | 450 (0.99x) | >10x | 434 (1.03x) | 491 (0.91x) | - | - | - | - | - |
| global linear score float32 (load!) | **791** | fill=striped: 795 | 827 (0.96x) | 782 (1.01x) | - | 269 (2.94x) | 317 (2.50x) | - | - | - | - | - |
| global linear score float64 (load!) | **442** | kernel=sse2: 442 | 443 (1.00x) | 439 (1.01x) | - | 237 (1.87x) | 302 (1.46x) | - | 367 (1.21x) | - | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| global linear path int32 (load!) | **603** | - | 636 (0.95x) | 612 (0.99x) | 438 (1.38x) | 222 (2.72x) | 345 (1.75x) | 371 (1.63x) | - | - | - | - |
| global linear path int64 (load!) | **428** | - | 464 (0.92x) | 428 (1.00x) | >10x | 124 (3.45x) | 320 (1.34x) | - | - | - | - | - |
| global linear path float32 (load!) | **594** | traceback=scores: 593 | 629 (0.95x) | 594 (1.00x) | - | 86 (6.90x) | 318 (1.87x) | - | - | - | - | - |
| global linear path float64 (load!) | **346** | kernel=scalar_fallback: 362 | 357 (0.97x) | 345 (1.00x) | - | 85 (4.07x) | 317 (1.09x) | - | 98 (3.53x) | 171 (2.02x) | - | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| global linear hard int32 (load!) | **654** | - | 695 (0.94x) | 630 (1.04x) | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **442** | - | 476 (0.93x) | 435 (1.02x) | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **557** | traceback=scores: 645 | 690 (0.81x) | 637 (0.87x) | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **359** | kernel=scalar_fallback: 377 | 348 (1.03x) | 349 (1.03x) | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **16** | api=functions: 17 | 15 (1.06x) | 16 (0.99x) | - | - | - | - | - | - | - | 3 (4.70x) |
| local affine score int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| local affine score int32 | **739** | - | 742 (1.00x) | 721 (1.02x) | 619 (1.19x) | 360 (2.05x) | 230 (3.21x) | - | - | - | - | - |
| local affine score int64 | **341** | - | 352 (0.97x) | 344 (0.99x) | >10x | 191 (1.79x) | 218 (1.56x) | - | - | - | - | - |
| local affine score float32 | **607** | fill=striped: 609 | 627 (0.97x) | 604 (1.01x) | - | 102 (5.98x) | 194 (3.12x) | - | - | - | - | - |
| local affine score float64 | **398** | fill=striped: 399 | 390 (1.02x) | 396 (1.01x) | - | 87 (4.56x) | 175 (2.28x) | - | 117 (3.39x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| local affine path int32 | **255** | - | 279 (0.91x) | 252 (1.01x) | 370 (0.69x) | 199 (1.28x) | 115 (2.21x) | - | - | - | - | - |
| local affine path int64 | **129** | - | 143 (0.90x) | 133 (0.97x) | 23 (5.62x) | 112 (1.15x) | 115 (1.12x) | - | - | - | - | - |
| local affine path float32 | **251** | traceback=scores: 295 | 254 (0.99x) | 247 (1.02x) | - | 61 (4.15x) | 113 (2.22x) | - | - | - | - | - |
| local affine path float64 | **133** | traceback=scores: 158 | 139 (0.95x) | 138 (0.96x) | - | 60 (2.21x) | 111 (1.20x) | - | 25 (5.41x) | 92 (1.44x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 314 x 317 problem with a largest st | | | | | | | | | | | |
| local affine hard int32 (load!) | **256** | - | 280 (0.91x) | 258 (0.99x) | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **130** | - | 144 (0.90x) | 133 (0.98x) | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **252** | traceback=scores: 312 | 256 (0.98x) | 249 (1.01x) | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **138** | traceback=scores: 161 | 142 (0.97x) | 139 (1.00x) | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **6** | fill=striped: 6 | 6 (0.98x) | 5 (1.01x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| local linear score int32 (load!) | **744** | - | 744 (1.00x) | 694 (1.07x) | 527 (1.41x) | 362 (2.05x) | 333 (2.23x) | - | - | - | - | - |
| local linear score int64 (load!) | **390** | - | 406 (0.96x) | 390 (1.00x) | 41 (9.62x) | 192 (2.04x) | 376 (1.04x) | - | - | - | - | - |
| local linear score float32 | **828** | fill=striped: 827 | 840 (0.99x) | 815 (1.02x) | - | 101 (8.20x) | 253 (3.27x) | - | - | - | - | - |
| local linear score float64 | **439** | fill=striped: 422 | 435 (1.01x) | 435 (1.01x) | - | 94 (4.69x) | 248 (1.77x) | - | 221 (1.98x) | - | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| local linear path int32 | **481** | - | 527 (0.91x) | 497 (0.97x) | 360 (1.34x) | 199 (2.41x) | 122 (3.95x) | - | - | - | - | - |
| local linear path int64 | **298** | - | 241 (1.24x) | 297 (1.00x) | >10x | 112 (2.65x) | 121 (2.47x) | - | - | - | - | - |
| local linear path float32 | **479** | traceback=scores: 483 | 462 (1.04x) | 480 (1.00x) | - | 62 (7.67x) | 86 (5.56x) | - | - | - | - | - |
| local linear path float64 (load!) | **275** | fill=striped: 268 | 275 (1.00x) | 276 (1.00x) | - | 62 (4.44x) | 86 (3.21x) | - | 48 (5.73x) | 163 (1.69x) | - | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 328 x 352 problem with a largest st | | | | | | | | | | | |
| local linear hard int32 (load!) | **514** | - | 560 (0.92x) | 512 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **301** | - | 241 (1.25x) | 304 (0.99x) | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **505** | fill=striped: 510 | 494 (1.02x) | 500 (1.01x) | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **279** | traceback=scores: 281 | 281 (0.99x) | 281 (0.99x) | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **9** | api=functions: 10 | 9 (1.00x) | 9 (1.00x) | - | - | - | - | - | - | - | - |

#### wloczykij · protein_L3000 · 1 thread

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal | deepblast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global affine score int32 (load!) | **721** | - | 776 (0.93x) | 724 (1.00x) | 739 (0.98x) | 548 (1.32x) | 215 (3.35x) | 452 (1.59x) | - | - | - | - |
| global affine score int64 (load!) | **367** | - | 364 (1.01x) | 374 (0.98x) | 39 (9.39x) | 333 (1.10x) | 235 (1.56x) | - | - | - | - | - |
| global affine score float32 (load!) | **596** | api=functions: 597 | 594 (1.00x) | 579 (1.03x) | - | 272 (2.19x) | 241 (2.47x) | - | - | - | - | - |
| global affine score float64 (load!) | **390** | api=functions: 398 | 386 (1.01x) | 395 (0.99x) | - | 238 (1.64x) | 237 (1.65x) | - | 162 (2.41x) | - | - | - |
| global affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global affine path int32 (load!) | **283** | - | 288 (0.98x) | 272 (1.04x) | 340 (0.83x) | 123 (2.31x) | 134 (2.11x) | 394 (0.72x) | - | - | - | - |
| global affine path int64 (load!) | **152** | - | 168 (0.91x) | 150 (1.02x) | 21 (7.11x) | 84 (1.82x) | 130 (1.17x) | - | - | - | - | - |
| global affine path float32 (load!) | **504** | kernel=sse2: 496 | 496 (1.02x) | 494 (1.02x) | - | 79 (6.34x) | 116 (4.35x) | - | - | - | - | - |
| global affine path float64 (load!) | **340** | kernel=sse2: 342 | 330 (1.03x) | 331 (1.03x) | - | 80 (4.23x) | 118 (2.88x) | - | 42 (8.11x) | 47 (7.21x) | - | - |
| global affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global affine hard int32 (load!) | **289** | - | 303 (0.96x) | 286 (1.01x) | - | - | - | - | - | - | - | - |
| global affine hard int64 (load!) | **159** | - | 174 (0.91x) | 158 (1.00x) | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **493** | fill=striped: 499 | 497 (0.99x) | 499 (0.99x) | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **332** | fill=striped: 343 | 336 (0.99x) | 334 (1.00x) | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **12** | fill=striped: 13 | 13 (0.92x) | 13 (0.95x) | - | - | - | - | - | - | - | - |
| global linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global linear score int32 (load!) | **731** | - | 731 (1.00x) | 724 (1.01x) | 1251 (0.58x) | 547 (1.34x) | 416 (1.76x) | 460 (1.59x) | - | - | - | - |
| global linear score int64 (load!) | **457** | - | 480 (0.95x) | 458 (1.00x) | >10x | 332 (1.37x) | 488 (0.94x) | - | - | - | - | - |
| global linear score float32 (load!) | **855** | kernel=sse2: 926 | 927 (0.92x) | 906 (0.94x) | - | 268 (3.19x) | 320 (2.67x) | - | - | - | - | - |
| global linear score float64 (load!) | **463** | fill=striped: 464 | 462 (1.00x) | 473 (0.98x) | - | 234 (1.98x) | 292 (1.59x) | - | 401 (1.15x) | - | - | - |
| global linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global linear path int32 (load!) | **467** | - | 488 (0.96x) | 467 (1.00x) | 352 (1.33x) | 130 (3.60x) | 362 (1.29x) | 413 (1.13x) | - | - | - | - |
| global linear path int64 (load!) | **219** | - | 274 (0.80x) | 218 (1.00x) | >10x | 87 (2.52x) | 331 (0.66x) | - | - | - | - | - |
| global linear path float32 (load!) | **524** | traceback=scores: 528 | 523 (1.00x) | 525 (1.00x) | - | 81 (6.42x) | 324 (1.61x) | - | - | - | - | - |
| global linear path float64 (load!) | **278** | traceback=scores: 285 | 277 (1.01x) | 277 (1.00x) | - | 88 (3.16x) | 341 (0.82x) | - | 105 (2.66x) | 214 (1.30x) | - | - |
| global linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| global linear hard int32 (load!) | **465** | - | 476 (0.98x) | 464 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard int64 (load!) | **230** | - | 286 (0.81x) | 230 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **556** | fill=striped: 555 | 542 (1.03x) | 557 (1.00x) | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **278** | traceback=scores: 283 | 279 (1.00x) | 278 (1.00x) | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **43** | api=functions: 44 | 40 (1.08x) | 43 (1.01x) | - | - | - | - | - | - | - | >10x |
| local affine score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local affine score int32 (load!) | **639** | - | 679 (0.94x) | 663 (0.96x) | 792 (0.81x) | 270 (2.36x) | 186 (3.43x) | - | - | - | - | - |
| local affine score int64 (load!) | **359** | - | 384 (0.93x) | 364 (0.99x) | >10x | 147 (2.44x) | 203 (1.77x) | - | - | - | - | - |
| local affine score float32 (load!) | **614** | fill=striped: 658 | 664 (0.93x) | 651 (0.94x) | - | 102 (6.00x) | 192 (3.20x) | - | - | - | - | - |
| local affine score float64 (load!) | **414** | fill=striped: 413 | 375 (1.10x) | 412 (1.01x) | - | 95 (4.36x) | 186 (2.23x) | - | 117 (3.54x) | - | - | - |
| local affine path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local affine path int32 (load!) | **254** | - | 273 (0.93x) | 251 (1.01x) | 310 (0.82x) | 121 (2.11x) | 118 (2.15x) | - | - | - | - | - |
| local affine path int64 (load!) | **123** | - | 144 (0.86x) | 135 (0.91x) | 22 (5.48x) | 78 (1.59x) | 114 (1.08x) | - | - | - | - | - |
| local affine path float32 (load!) | **261** | traceback=grid: 261 | 265 (0.98x) | 259 (1.01x) | - | 64 (4.06x) | 118 (2.22x) | - | - | - | - | - |
| local affine path float64 (load!) | **141** | fill=striped: 141 | 145 (0.97x) | 141 (1.00x) | - | 64 (2.20x) | 114 (1.23x) | - | 25 (5.66x) | 46 (3.07x) | - | - |
| local affine hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local affine hard int32 (load!) | **248** | - | 273 (0.91x) | 253 (0.98x) | - | - | - | - | - | - | - | - |
| local affine hard int64 (load!) | **131** | - | 136 (0.96x) | 135 (0.97x) | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **258** | traceback=pointers: 261 | 264 (0.98x) | 254 (1.02x) | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **139** | traceback=grid: 142 | 145 (0.96x) | 141 (0.99x) | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **9** | api=functions: 9 | 9 (0.95x) | 9 (0.98x) | - | - | - | - | - | - | - | - |
| local linear score int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local linear score int32 (load!) | **634** | - | 665 (0.95x) | 664 (0.96x) | 1188 (0.53x) | 269 (2.36x) | 329 (1.93x) | - | - | - | - | - |
| local linear score int64 (load!) | **402** | - | 432 (0.93x) | 411 (0.98x) | 42 (9.66x) | 147 (2.73x) | 377 (1.07x) | - | - | - | - | - |
| local linear score float32 (load!) | **839** | api=functions: 889 | 907 (0.93x) | 891 (0.94x) | - | 102 (8.22x) | 261 (3.22x) | - | - | - | - | - |
| local linear score float64 (load!) | **440** | api=functions: 453 | 454 (0.97x) | 445 (0.99x) | - | 94 (4.69x) | 246 (1.79x) | - | 311 (1.42x) | - | - | - |
| local linear path int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local linear path int32 (load!) | **370** | - | 411 (0.90x) | 380 (0.97x) | 342 (1.08x) | 118 (3.13x) | 132 (2.80x) | - | - | - | - | - |
| local linear path int64 (load!) | **173** | - | 187 (0.92x) | 171 (1.01x) | 23 (7.37x) | 78 (2.22x) | 134 (1.29x) | - | - | - | - | - |
| local linear path float32 (load!) | **417** | traceback=scores: 460 | 428 (0.98x) | 412 (1.01x) | - | 62 (6.70x) | 89 (4.66x) | - | - | - | - | - |
| local linear path float64 (load!) | **196** | traceback=scores: 224 | 213 (0.92x) | 210 (0.93x) | - | 61 (3.20x) | 89 (2.21x) | - | 51 (3.87x) | 194 (1.01x) | - | - |
| local linear hard int16 | n/a: OverflowError: nwgrad: a 2958 x 3073 problem with a largest  | | | | | | | | | | | |
| local linear hard int32 (load!) | **383** | - | 413 (0.93x) | 387 (0.99x) | - | - | - | - | - | - | - | - |
| local linear hard int64 (load!) | **173** | - | 190 (0.91x) | 173 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **416** | traceback=scores: 462 | 426 (0.98x) | 416 (1.00x) | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **211** | traceback=scores: 224 | 210 (1.00x) | 211 (1.00x) | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **36** | api=functions: 38 | 36 (0.99x) | 36 (0.99x) | - | - | - | - | - | - | - | - |

#### wloczykij · dna_L100 · 32 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **11002** | fill=striped: 16138 | 15610 (0.70x) | 17849 (0.62x) | 13038 (0.84x) | 4468 (2.46x) | - | 7372 (1.49x) | 5777 (1.90x) | - | - | - | - |
| global affine score float32 | **16313** | fill=striped: 12135 | 9050 (1.80x) | 16381 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **8325** | fill=striped: 9447 | 4665 (1.78x) | 8369 (0.99x) | - | - | - | - | - | - | 2911 (2.86x) | - | - |
| global affine path int32 | **8888** | schedule=sorted: 9274 | 9003 (0.99x) | 8751 (1.02x) | 9953 (0.89x) | 5566 (1.60x) | - | 6699 (1.33x) | 5913 (1.50x) | - | - | - | - |
| global affine path float32 | **9877** | schedule=sorted: 9867 | 8167 (1.21x) | 9333 (1.06x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 | **3979** | schedule=sorted: 5236 | 3923 (1.01x) | 3925 (1.01x) | - | - | - | - | - | - | 828 (4.80x) | 851 (4.68x) | - |
| global affine hard int32 | **10139** | schedule=sorted: 7182 | 9549 (1.06x) | 10276 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **10413** | schedule=sorted: 10748 | 8534 (1.22x) | 10274 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **5539** | schedule=sorted: 5563 | 4183 (1.32x) | 5453 (1.02x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **1083** | schedule=sorted: 1086 | 1080 (1.00x) | 1083 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 | **25675** | schedule=sorted: 26443 | 19133 (1.34x) | 23598 (1.09x) | 15852 (1.62x) | 4537 (5.66x) | - | 6622 (3.88x) | 51535 (0.50x) | 6253 (4.11x) | - | - | - |
| global linear score float32 | **22911** | schedule=sorted: 20500 | 18253 (1.26x) | 23176 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **6971** | fill=striped: 10803 | 10480 (0.67x) | 10606 (0.66x) | - | - | - | - | - | - | 4613 (1.51x) | - | - |
| global linear path int32 | **15525** | schedule=sorted: 15341 | 10678 (1.45x) | 15787 (0.98x) | 9499 (1.63x) | 5873 (2.64x) | - | 6930 (2.24x) | 45573 (0.34x) | 5532 (2.81x) | - | - | - |
| global linear path float32 | **15201** | schedule=sorted: 15400 | 11956 (1.27x) | 15736 (0.97x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 | **7487** | fill=striped: 7388 | 7388 (1.01x) | 7461 (1.00x) | - | - | - | - | - | - | 1393 (5.38x) | 924 (8.10x) | - |
| global linear hard int32 | **11466** | schedule=sorted: 17497 | 15005 (0.76x) | 18655 (0.61x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **17495** | schedule=sorted: 17969 | 14396 (1.22x) | 18178 (0.96x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **7435** | fill=striped: 8299 | 8301 (0.90x) | 8266 (0.90x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **2028** | schedule=sorted: 2074 | 1837 (1.10x) | 1862 (1.09x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **15435** | fill=striped: 15520 | 14477 (1.07x) | 15639 (0.99x) | 16520 (0.93x) | 7385 (2.09x) | - | - | - | - | - | - | - |
| local affine score float32 | **12622** | fill=striped: 16148 | 12645 (1.00x) | 12755 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **7121** | fill=striped: 10373 | 6626 (1.07x) | 7246 (0.98x) | - | - | - | - | - | - | 1684 (4.23x) | - | - |
| local affine path int32 | **8112** | schedule=sorted: 8783 | 6821 (1.19x) | 8902 (0.91x) | 8246 (0.98x) | 5052 (1.61x) | - | - | - | - | - | - | - |
| local affine path float32 | **7539** | schedule=sorted: 7555 | 7400 (1.02x) | 7718 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **4277** | schedule=sorted: 4196 | 4190 (1.02x) | 4319 (0.99x) | - | - | - | - | - | - | 537 (7.97x) | 801 (5.34x) | - |
| local affine hard int32 | **9522** | schedule=sorted: 9512 | 9202 (1.03x) | 9399 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **7933** | schedule=sorted: 8224 | 8017 (0.99x) | 8170 (0.97x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **4431** | schedule=sorted: 4397 | 4405 (1.01x) | 4309 (1.03x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **986** | fill=striped: 986 | 1006 (0.98x) | 978 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 | **18758** | schedule=sorted: 19817 | 18181 (1.03x) | 20162 (0.93x) | 12726 (1.47x) | >10x | - | - | - | - | - | - | - |
| local linear score float32 | **16292** | schedule=sorted: 16704 | 15837 (1.03x) | 17176 (0.95x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **8869** | fill=striped: 10185 | 8267 (1.07x) | 9281 (0.96x) | - | - | - | - | - | - | 3801 (2.33x) | - | - |
| local linear path int32 | **12270** | schedule=sorted: 12513 | 11250 (1.09x) | 12326 (1.00x) | 8983 (1.37x) | 5590 (2.19x) | - | - | - | - | - | - | - |
| local linear path float32 | **11417** | schedule=sorted: 11701 | 10384 (1.10x) | 11571 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **6820** | schedule=sorted: 6839 | 5727 (1.19x) | 6710 (1.02x) | - | - | - | - | - | - | 816 (8.36x) | 989 (6.90x) | - |
| local linear hard int32 | **9399** | schedule=sorted: 14189 | 12370 (0.76x) | 14590 (0.64x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **12755** | schedule=sorted: 13462 | 11236 (1.14x) | 12931 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **7070** | schedule=sorted: 6981 | 5868 (1.20x) | 6912 (1.02x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **1943** | schedule=sorted: 1936 | 1644 (1.18x) | 1894 (1.03x) | - | - | - | - | - | - | - | - | - |

#### wloczykij · dna_L100 · 8:numa0 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 (load!) | **3053** | schedule=sorted: 3057 | 2388 (1.28x) | 3057 (1.00x) | 3250 (0.94x) | 4638 (0.66x) | - | 1382 (2.21x) | 1210 (2.52x) | - | - | - | - |
| global affine score float32 (load!) | **2846** | schedule=sorted: 2847 | 2498 (1.14x) | 2852 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 (load!) | **1357** | fill=striped: 1692 | 1201 (1.13x) | 1358 (1.00x) | - | - | - | - | - | - | 701 (1.94x) | - | - |
| global affine path int32 (load!) | **1942** | schedule=sorted: 1943 | 1712 (1.13x) | 1957 (0.99x) | 1629 (1.19x) | 1460 (1.33x) | - | 1283 (1.51x) | 1245 (1.56x) | - | - | - | - |
| global affine path float32 (load!) | **1994** | schedule=sorted: 1989 | 1578 (1.26x) | 1975 (1.01x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **1012** | schedule=sorted: 1016 | 783 (1.29x) | 1011 (1.00x) | - | - | - | - | - | - | 184 (5.49x) | 185 (5.47x) | - |
| global affine hard int32 (load!) | **2110** | schedule=sorted: 2106 | 1917 (1.10x) | 2129 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **2089** | schedule=sorted: 2092 | 1753 (1.19x) | 2087 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1061** | schedule=sorted: 1058 | 819 (1.29x) | 1045 (1.02x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **195** | schedule=sorted: 194 | 195 (1.00x) | 195 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 (load!) | **4088** | schedule=sorted: 4082 | 3132 (1.31x) | 4088 (1.00x) | 3226 (1.27x) | 4619 (0.88x) | - | 1378 (2.97x) | 15285 (0.27x) | 4310 (0.95x) | - | - | - |
| global linear score float32 (load!) | **4472** | schedule=sorted: 4467 | 3121 (1.43x) | 4471 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 (load!) | **1845** | schedule=sorted: 1851 | 1856 (0.99x) | 1905 (0.97x) | - | - | - | - | - | - | 1217 (1.52x) | - | - |
| global linear path int32 (load!) | **3063** | schedule=sorted: 3070 | 2377 (1.29x) | 3076 (1.00x) | 1630 (1.88x) | 1458 (2.10x) | - | 1295 (2.37x) | 12582 (0.24x) | 1959 (1.56x) | - | - | - |
| global linear path float32 (load!) | **3031** | schedule=sorted: 3037 | 2329 (1.30x) | 3036 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **1527** | schedule=sorted: 1543 | 1594 (0.96x) | 1547 (0.99x) | - | - | - | - | - | - | 335 (4.57x) | 223 (6.84x) | - |
| global linear hard int32 (load!) | **3694** | schedule=sorted: 3707 | 2737 (1.35x) | 3720 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **3688** | schedule=sorted: 3683 | 2698 (1.37x) | 3673 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1714** | schedule=sorted: 1714 | 1775 (0.97x) | 1712 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **398** | schedule=sorted: 399 | 353 (1.13x) | 398 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 (load!) | **2539** | fill=striped: 2638 | 2380 (1.07x) | 2536 (1.00x) | 3742 (0.68x) | 2134 (1.19x) | - | - | - | - | - | - | - |
| local affine score float32 | **2613** | fill=striped: 2988 | 2237 (1.17x) | 2615 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 (load!) | **1262** | fill=striped: 1753 | 1134 (1.11x) | 1266 (1.00x) | - | - | - | - | - | - | 393 (3.21x) | - | - |
| local affine path int32 (load!) | **1607** | schedule=sorted: 1611 | 1568 (1.03x) | 1632 (0.99x) | 1481 (1.09x) | 1282 (1.25x) | - | - | - | - | - | - | - |
| local affine path float32 (load!) | **1474** | schedule=sorted: 1475 | 1396 (1.06x) | 1505 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **809** | schedule=sorted: 813 | 698 (1.16x) | 832 (0.97x) | - | - | - | - | - | - | 118 (6.83x) | 180 (4.49x) | - |
| local affine hard int32 | **1756** | schedule=sorted: 1759 | 1735 (1.01x) | 1762 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **1612** | schedule=sorted: 1603 | 1526 (1.06x) | 1623 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **847** | schedule=sorted: 851 | 727 (1.16x) | 861 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **179** | schedule=sorted: 180 | 182 (0.98x) | 179 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 (load!) | **3177** | schedule=sorted: 3182 | 2925 (1.09x) | 3178 (1.00x) | 2405 (1.32x) | 2150 (1.48x) | - | - | - | - | - | - | - |
| local linear score float32 (load!) | **3137** | schedule=sorted: 3149 | 2718 (1.15x) | 3143 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **1594** | fill=striped: 1771 | 1383 (1.15x) | 1598 (1.00x) | - | - | - | - | - | - | 924 (1.73x) | - | - |
| local linear path int32 (load!) | **2255** | schedule=sorted: 2250 | 1827 (1.23x) | 2273 (0.99x) | 1450 (1.56x) | 1278 (1.76x) | - | - | - | - | - | - | - |
| local linear path float32 (load!) | **2212** | schedule=sorted: 2213 | 1883 (1.17x) | 2226 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 (load!) | **1250** | schedule=sorted: 1248 | 1018 (1.23x) | 1198 (1.04x) | - | - | - | - | - | - | 194 (6.45x) | 216 (5.80x) | - |
| local linear hard int32 | **2570** | schedule=sorted: 2572 | 2037 (1.26x) | 2581 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **2529** | schedule=sorted: 2535 | 2128 (1.19x) | 2522 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **1334** | schedule=sorted: 1333 | 969 (1.38x) | 1276 (1.05x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **364** | schedule=sorted: 364 | 308 (1.18x) | 360 (1.01x) | - | - | - | - | - | - | - | - | - |

#### wloczykij · dna_L1000 · 32 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **11347** | fill=striped: 18424 | 19604 (0.58x) | 18875 (0.60x) | 21068 (0.54x) | 21442 (0.53x) | - | 10627 (1.07x) | 20856 (0.54x) | - | - | - | - |
| global affine score float32 | **14478** | schedule=sorted: 15801 | 16769 (0.86x) | 15812 (0.92x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **10349** | schedule=sorted: 10316 | 10212 (1.01x) | 10454 (0.99x) | - | - | - | - | - | - | 3490 (2.97x) | - | - |
| global affine path int32 | **7515** | schedule=sorted: 7638 | 7912 (0.95x) | 7689 (0.98x) | 4860 (1.55x) | 2467 (3.05x) | - | 9345 (0.80x) | 6208 (1.21x) | - | - | - | - |
| global affine path float32 | **11390** | fill=striped: 11865 | 11525 (0.99x) | 11668 (0.98x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 | **7373** | fill=striped: 7926 | 7800 (0.95x) | 7781 (0.95x) | - | - | - | - | - | - | 918 (8.04x) | 1294 (5.70x) | - |
| global affine hard int32 | **7707** | fill=striped: 7645 | 8086 (0.95x) | 7581 (1.02x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **11872** | schedule=sorted: 11928 | 11822 (1.00x) | 11840 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **7355** | fill=striped: 7755 | 7830 (0.94x) | 5204 (1.41x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **208** | schedule=sorted: 205 | 200 (1.04x) | 204 (1.02x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 | **18868** | fill=striped: 18792 | 17887 (1.05x) | 10286 (1.83x) | 27678 (0.68x) | 20673 (0.91x) | - | 9064 (2.08x) | 104766 (0.18x) | 17861 (1.06x) | - | - | - |
| global linear score float32 | **19863** | schedule=sorted: 21036 | 21172 (0.94x) | 20850 (0.95x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **10897** | fill=striped: 11191 | 11112 (0.98x) | 11077 (0.98x) | - | - | - | - | - | - | 8066 (1.35x) | - | - |
| global linear path int32 | **11767** | fill=striped: 11836 | 12415 (0.95x) | 11742 (1.00x) | 7421 (1.59x) | 2810 (4.19x) | - | 9770 (1.20x) | 106330 (0.11x) | 15617 (0.75x) | - | - | - |
| global linear path float32 | **13256** | schedule=sorted: 13533 | 12625 (1.05x) | 13806 (0.96x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 | **6697** | fill=striped: 6697 | 6615 (1.01x) | 6668 (1.00x) | - | - | - | - | - | - | 1684 (3.98x) | 4124 (1.62x) | - |
| global linear hard int32 | **10377** | schedule=sorted: 10687 | 10938 (0.95x) | 10356 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **13519** | fill=striped: 13367 | 12551 (1.08x) | 13904 (0.97x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **6638** | fill=striped: 6555 | 6514 (1.02x) | 6657 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **641** | schedule=sorted: 635 | 605 (1.06x) | 630 (1.02x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **16041** | schedule=sorted: 16923 | 16871 (0.95x) | 15912 (1.01x) | 22097 (0.73x) | 8362 (1.92x) | - | - | - | - | - | - | - |
| local affine score float32 | **12953** | fill=striped: 18655 | 19305 (0.67x) | 19043 (0.68x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **10729** | fill=striped: 10686 | 10754 (1.00x) | 10553 (1.02x) | - | - | - | - | - | - | 2040 (5.26x) | - | - |
| local affine path int32 | **3899** | fill=striped: 6132 | 6518 (0.60x) | 6153 (0.63x) | 7539 (0.52x) | 2380 (1.64x) | - | - | - | - | - | - | - |
| local affine path float32 | **6570** | fill=striped: 6954 | 7011 (0.94x) | 6709 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **3488** | schedule=sorted: 3548 | 3637 (0.96x) | 3213 (1.09x) | - | - | - | - | - | - | 586 (5.95x) | 1130 (3.09x) | - |
| local affine hard int32 | **6167** | schedule=sorted: 6161 | 6633 (0.93x) | 6020 (1.02x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **6894** | schedule=sorted: 7026 | 6675 (1.03x) | 6945 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **3404** | fill=striped: 3524 | 3641 (0.93x) | 3478 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **79** | fill=striped: 80 | 80 (0.98x) | 80 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 | **15704** | schedule=sorted: 15883 | 16237 (0.97x) | 15481 (1.01x) | 25794 (0.61x) | 7943 (1.98x) | - | - | - | - | - | - | - |
| local linear score float32 | **11880** | fill=striped: 19326 | 19334 (0.61x) | 19372 (0.61x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **10397** | fill=striped: 10421 | 10099 (1.03x) | 10082 (1.03x) | - | - | - | - | - | - | 6141 (1.69x) | - | - |
| local linear path int32 | **9247** | schedule=sorted: 9381 | 8776 (1.05x) | 9335 (0.99x) | 7908 (1.17x) | 2462 (3.76x) | - | - | - | - | - | - | - |
| local linear path float32 | **9755** | fill=striped: 9895 | 10531 (0.93x) | 9835 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **4981** | schedule=sorted: 5014 | 5024 (0.99x) | 4818 (1.03x) | - | - | - | - | - | - | 906 (5.50x) | 2588 (1.93x) | - |
| local linear hard int32 | **9504** | fill=striped: 9475 | 9953 (0.95x) | 9320 (1.02x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **9924** | schedule=sorted: 9903 | 10733 (0.92x) | 9860 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **5023** | schedule=sorted: 5012 | 5077 (0.99x) | 5024 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **557** | fill=striped: 552 | 708 (0.79x) | 554 (1.01x) | - | - | - | - | - | - | - | - | - |

#### wloczykij · dna_L1000 · 8:numa0 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 (load!) | **2946** | schedule=sorted: 2950 | 3035 (0.97x) | 2930 (1.01x) | 3794 (0.78x) | 4887 (0.60x) | - | 1951 (1.51x) | 4818 (0.61x) | - | - | - | - |
| global affine score float32 (load!) | **2992** | schedule=sorted: 2998 | 3012 (0.99x) | 2980 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 (load!) | **1706** | schedule=sorted: 1703 | 1706 (1.00x) | 1703 (1.00x) | - | - | - | - | - | - | 905 (1.89x) | - | - |
| global affine path int32 (load!) | **1226** | schedule=sorted: 1231 | 1249 (0.98x) | 1233 (0.99x) | 1529 (0.80x) | 918 (1.34x) | - | 1823 (0.67x) | 1709 (0.72x) | - | - | - | - |
| global affine path float32 (load!) | **2251** | fill=striped: 2255 | 2270 (0.99x) | 2246 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **1371** | schedule=sorted: 1372 | 1325 (1.03x) | 1362 (1.01x) | - | - | - | - | - | - | 207 (6.61x) | 198 (6.91x) | - |
| global affine hard int32 (load!) | **1234** | schedule=sorted: 1233 | 1255 (0.98x) | 1237 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **2255** | fill=striped: 2255 | 2281 (0.99x) | 2267 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1369** | schedule=sorted: 1370 | 1325 (1.03x) | 1371 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **48** | fill=striped: 48 | 47 (1.00x) | 48 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 (load!) | **2942** | fill=striped: 2944 | 2985 (0.99x) | 2927 (1.01x) | 5264 (0.56x) | 4901 (0.60x) | - | 1957 (1.50x) | 52774 (0.06x) | 39335 (0.07x) | - | - | - |
| global linear score float32 (load!) | **3559** | schedule=sorted: 3554 | 3543 (1.00x) | 3559 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 (load!) | **1803** | schedule=sorted: 1811 | 1799 (1.00x) | 1816 (0.99x) | - | - | - | - | - | - | 1985 (0.91x) | - | - |
| global linear path int32 (load!) | **1912** | schedule=sorted: 1938 | 1997 (0.96x) | 1941 (0.98x) | 1528 (1.25x) | 909 (2.10x) | - | 1830 (1.04x) | 39527 (0.05x) | 16674 (0.11x) | - | - | - |
| global linear path float32 (load!) | **2591** | schedule=sorted: 2613 | 2575 (1.01x) | 2634 (0.98x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **1254** | schedule=sorted: 1250 | 1230 (1.02x) | 1248 (1.00x) | - | - | - | - | - | - | 407 (3.08x) | 892 (1.41x) | - |
| global linear hard int32 (load!) | **1913** | fill=striped: 1933 | 2012 (0.95x) | 1938 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **2471** | schedule=sorted: 2482 | 2511 (0.98x) | 2509 (0.98x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **1252** | fill=striped: 1254 | 1237 (1.01x) | 1259 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **149** | fill=striped: 149 | 142 (1.04x) | 150 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **2740** | schedule=sorted: 2755 | 2817 (0.97x) | 2761 (0.99x) | 5405 (0.51x) | 2175 (1.26x) | - | - | - | - | - | - | - |
| local affine score float32 | **3320** | schedule=sorted: 3296 | 3301 (1.01x) | 3280 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **1740** | schedule=sorted: 1735 | 1733 (1.00x) | 1738 (1.00x) | - | - | - | - | - | - | 479 (3.64x) | - | - |
| local affine path int32 | **979** | fill=striped: 980 | 1014 (0.97x) | 980 (1.00x) | 1389 (0.70x) | 893 (1.10x) | - | - | - | - | - | - | - |
| local affine path float32 | **1216** | fill=striped: 1227 | 1231 (0.99x) | 1211 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **620** | fill=striped: 621 | 620 (1.00x) | 628 (0.99x) | - | - | - | - | - | - | 128 (4.85x) | 192 (3.22x) | - |
| local affine hard int32 (load!) | **981** | fill=striped: 984 | 1014 (0.97x) | 985 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1225** | schedule=sorted: 1236 | 1240 (0.99x) | 1216 (1.01x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **624** | schedule=sorted: 622 | 622 (1.00x) | 629 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **18** | schedule=sorted: 18 | 19 (0.97x) | 18 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 (load!) | **2762** | schedule=sorted: 2748 | 2834 (0.97x) | 2747 (1.01x) | 5829 (0.47x) | 2183 (1.27x) | - | - | - | - | - | - | - |
| local linear score float32 (load!) | **3438** | fill=striped: 3463 | 3468 (0.99x) | 3463 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 (load!) | **1736** | fill=striped: 1743 | 1750 (0.99x) | 1747 (0.99x) | - | - | - | - | - | - | 1545 (1.12x) | - | - |
| local linear path int32 (load!) | **1666** | schedule=sorted: 1669 | 1659 (1.00x) | 1668 (1.00x) | 1507 (1.11x) | 882 (1.89x) | - | - | - | - | - | - | - |
| local linear path float32 (load!) | **2139** | fill=striped: 2132 | 2142 (1.00x) | 2140 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 (load!) | **944** | fill=striped: 951 | 1005 (0.94x) | 945 (1.00x) | - | - | - | - | - | - | 217 (4.35x) | 591 (1.60x) | - |
| local linear hard int32 (load!) | **1678** | fill=striped: 1685 | 1693 (0.99x) | 1684 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **2157** | schedule=sorted: 2159 | 2180 (0.99x) | 2157 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **954** | fill=striped: 953 | 1010 (0.94x) | 955 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **128** | fill=striped: 129 | 166 (0.78x) | 128 (1.00x) | - | - | - | - | - | - | - | - | - |

#### wloczykij · mirna · 32 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **11828** | schedule=sorted: 11475 | 10586 (1.12x) | 11454 (1.03x) | 9661 (1.22x) | >10x | - | 4073 (2.90x) | >10x | - | - | - | >10x |
| global affine score float32 | **6956** | schedule=sorted: 6911 | 10040 (0.69x) | 7936 (0.88x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **6098** | fill=striped: 3315 | 5551 (1.10x) | 5882 (1.04x) | - | - | - | - | - | - | 1020 (5.98x) | - | - |
| global affine path int32 | **4013** | schedule=sorted: 4129 | 4008 (1.00x) | 5283 (0.76x) | 6015 (0.67x) | >10x | - | 2957 (1.36x) | 429 (9.35x) | - | - | - | - |
| global affine path float32 | **3697** | schedule=sorted: 3862 | 3874 (0.95x) | 5111 (0.72x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 | **2751** | schedule=sorted: 2656 | 2777 (0.99x) | 3391 (0.81x) | - | - | - | - | - | - | 371 (7.42x) | >10x | - |
| global affine hard int32 | **3961** | schedule=sorted: 3763 | 3790 (1.04x) | 4381 (0.90x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **3853** | schedule=sorted: 4153 | 3990 (0.97x) | 3687 (1.05x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 | **2795** | schedule=sorted: 2693 | 2668 (1.05x) | 2642 (1.06x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 | **1179** | schedule=sorted: 1076 | 1040 (1.13x) | 1174 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 | **13674** | schedule=sorted: 14336 | 10915 (1.25x) | 14362 (0.95x) | 12818 (1.07x) | >10x | - | 4456 (3.07x) | 4321 (3.16x) | 1616 (8.46x) | - | - | >10x |
| global linear score float32 | **13588** | schedule=sorted: 13446 | 11423 (1.19x) | 13207 (1.03x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **3469** | fill=striped: 3671 | 2427 (1.43x) | 3326 (1.04x) | - | - | - | - | - | - | 1217 (2.85x) | - | - |
| global linear path int32 | **3926** | schedule=sorted: 4030 | 4057 (0.97x) | 5480 (0.72x) | 5477 (0.72x) | >10x | - | 3795 (1.03x) | 3522 (1.11x) | 1429 (2.75x) | - | - | - |
| global linear path float32 | **3856** | schedule=sorted: 3818 | 3766 (1.02x) | 5029 (0.77x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 | **2499** | fill=striped: 2682 | 2637 (0.95x) | 2524 (0.99x) | - | - | - | - | - | - | 550 (4.54x) | >10x | - |
| global linear hard int32 | **4022** | schedule=sorted: 4090 | 3941 (1.02x) | 3844 (1.05x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **3853** | schedule=sorted: 3838 | 3963 (0.97x) | 5962 (0.65x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **3101** | schedule=sorted: 3059 | 2938 (1.06x) | 3071 (1.01x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **1676** | schedule=sorted: 1666 | 1497 (1.12x) | 1706 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 | **9787** | schedule=sorted: 7028 | 10644 (0.92x) | 9565 (1.02x) | 8540 (1.15x) | >10x | - | - | - | - | - | - | >10x |
| local affine score float32 | **10045** | schedule=sorted: 6924 | 8734 (1.15x) | 7553 (1.33x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 | **2985** | schedule=sorted: 3016 | 5030 (0.59x) | 5396 (0.55x) | - | - | - | - | - | - | 837 (3.57x) | - | - |
| local affine path int32 | **4570** | schedule=sorted: 4647 | 4384 (1.04x) | 6527 (0.70x) | 6553 (0.70x) | >10x | - | - | - | - | - | - | - |
| local affine path float32 | **4343** | schedule=sorted: 4688 | 4285 (1.01x) | 6025 (0.72x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 | **3218** | schedule=sorted: 3209 | 2966 (1.09x) | 3049 (1.06x) | - | - | - | - | - | - | 393 (8.20x) | >10x | - |
| local affine hard int32 | **4277** | schedule=sorted: 4117 | 4003 (1.07x) | 5191 (0.82x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **4263** | schedule=sorted: 3952 | 4153 (1.03x) | 4795 (0.89x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **2674** | schedule=sorted: 2956 | 2824 (0.95x) | 3018 (0.89x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 | **1181** | schedule=sorted: 1101 | 1063 (1.11x) | 1053 (1.12x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 | **12525** | schedule=sorted: 6790 | 11672 (1.07x) | 12548 (1.00x) | 8840 (1.42x) | >10x | - | - | - | - | - | - | >10x |
| local linear score float32 | **6983** | schedule=sorted: 7029 | 10599 (0.66x) | 8059 (0.87x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **3184** | schedule=sorted: 3091 | 5880 (0.54x) | 3016 (1.06x) | - | - | - | - | - | - | 1108 (2.87x) | - | - |
| local linear path int32 | **4885** | schedule=sorted: 4428 | 4595 (1.06x) | 6925 (0.71x) | 6379 (0.77x) | >10x | - | - | - | - | - | - | - |
| local linear path float32 | **4848** | schedule=sorted: 4991 | 4361 (1.11x) | 4639 (1.05x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **3115** | schedule=sorted: 3469 | 3139 (0.99x) | 4224 (0.74x) | - | - | - | - | - | - | 470 (6.62x) | >10x | - |
| local linear hard int32 | **4522** | schedule=sorted: 4315 | 4284 (1.06x) | 6245 (0.72x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **4111** | schedule=sorted: 4222 | 4128 (1.00x) | 4632 (0.89x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **3320** | schedule=sorted: 3198 | 2764 (1.20x) | 3860 (0.86x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **1382** | schedule=sorted: 1371 | 1291 (1.07x) | 1486 (0.93x) | - | - | - | - | - | - | - | - | - |

#### wloczykij · mirna · 8:numa0 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | wfa2 | edlib | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 (load!) | **2524** | schedule=sorted: 2524 | 2033 (1.24x) | 2554 (0.99x) | 2289 (1.10x) | 789 (3.20x) | - | 846 (2.98x) | >10x | - | - | - | >10x |
| global affine score float32 | **2394** | schedule=sorted: 2388 | 2124 (1.13x) | 2393 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine score float64 | **1178** | fill=striped: 1339 | 1064 (1.11x) | 1173 (1.00x) | - | - | - | - | - | - | 246 (4.80x) | - | - |
| global affine path int32 (load!) | **1114** | schedule=sorted: 1118 | 987 (1.13x) | 1134 (0.98x) | 1116 (1.00x) | 845 (1.32x) | - | 721 (1.55x) | >10x | - | - | - | - |
| global affine path float32 (load!) | **1120** | schedule=sorted: 1118 | 926 (1.21x) | 1130 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **726** | schedule=sorted: 724 | 570 (1.27x) | 733 (0.99x) | - | - | - | - | - | - | 81 (8.99x) | >10x | - |
| global affine hard int32 | **1415** | schedule=sorted: 1425 | 1325 (1.07x) | 1435 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine hard float32 | **1451** | schedule=sorted: 1458 | 1235 (1.17x) | 1448 (1.00x) | - | - | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **864** | schedule=sorted: 864 | 658 (1.31x) | 873 (0.99x) | - | - | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **238** | schedule=sorted: 237 | 198 (1.20x) | 239 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear score int32 (load!) | **3200** | schedule=sorted: 3190 | 2551 (1.25x) | 3205 (1.00x) | 2864 (1.12x) | 912 (3.51x) | - | 866 (3.70x) | 890 (3.60x) | 804 (3.98x) | - | - | >10x |
| global linear score float32 (load!) | **3363** | schedule=sorted: 3429 | 2549 (1.32x) | 3431 (0.98x) | - | - | - | - | - | - | - | - | - |
| global linear score float64 | **1422** | schedule=sorted: 1427 | 1447 (0.98x) | 1461 (0.97x) | - | - | - | - | - | - | 277 (5.14x) | - | - |
| global linear path int32 (load!) | **1219** | schedule=sorted: 1223 | 1085 (1.12x) | 1255 (0.97x) | 1094 (1.11x) | 848 (1.44x) | - | 720 (1.69x) | 820 (1.49x) | 359 (3.39x) | - | - | - |
| global linear path float32 | **1124** | schedule=sorted: 1112 | 997 (1.13x) | 1165 (0.97x) | - | - | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **891** | fill=striped: 899 | 882 (1.01x) | 908 (0.98x) | - | - | - | - | - | - | 121 (7.36x) | >10x | - |
| global linear hard int32 (load!) | **1907** | schedule=sorted: 1892 | 1610 (1.18x) | 1921 (0.99x) | - | - | - | - | - | - | - | - | - |
| global linear hard float32 | **1794** | schedule=sorted: 1796 | 1568 (1.14x) | 1852 (0.97x) | - | - | - | - | - | - | - | - | - |
| global linear hard float64 | **1264** | schedule=sorted: 1272 | 1265 (1.00x) | 1265 (1.00x) | - | - | - | - | - | - | - | - | - |
| global linear soft float64 | **349** | schedule=sorted: 347 | 306 (1.14x) | 349 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine score int32 (load!) | **2188** | schedule=sorted: 2197 | 2022 (1.08x) | 2180 (1.00x) | 2445 (0.89x) | 1055 (2.07x) | - | - | - | - | - | - | >10x |
| local affine score float32 (load!) | **2240** | schedule=sorted: 2242 | 1943 (1.15x) | 2254 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine score float64 (load!) | **1124** | fill=striped: 1492 | 1016 (1.11x) | 1123 (1.00x) | - | - | - | - | - | - | 186 (6.05x) | - | - |
| local affine path int32 (load!) | **1408** | schedule=sorted: 1408 | 1339 (1.05x) | 1417 (0.99x) | 1190 (1.18x) | 815 (1.73x) | - | - | - | - | - | - | - |
| local affine path float32 (load!) | **1307** | schedule=sorted: 1312 | 1215 (1.08x) | 1329 (0.98x) | - | - | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **731** | schedule=sorted: 732 | 623 (1.17x) | 748 (0.98x) | - | - | - | - | - | - | 83 (8.82x) | >10x | - |
| local affine hard int32 (load!) | **1430** | schedule=sorted: 1420 | 1391 (1.03x) | 1433 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float32 | **1359** | schedule=sorted: 1361 | 1259 (1.08x) | 1359 (1.00x) | - | - | - | - | - | - | - | - | - |
| local affine hard float64 | **743** | schedule=sorted: 741 | 632 (1.18x) | 754 (0.99x) | - | - | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **227** | schedule=sorted: 226 | 196 (1.15x) | 229 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear score int32 (load!) | **2660** | schedule=sorted: 2659 | 2437 (1.09x) | 2665 (1.00x) | 1872 (1.42x) | 1008 (2.64x) | - | - | - | - | - | - | >10x |
| local linear score float32 (load!) | **2602** | schedule=sorted: 2608 | 2292 (1.14x) | 2588 (1.01x) | - | - | - | - | - | - | - | - | - |
| local linear score float64 | **1337** | fill=striped: 1492 | 1212 (1.10x) | 1353 (0.99x) | - | - | - | - | - | - | 253 (5.28x) | - | - |
| local linear path int32 (load!) | **1665** | schedule=sorted: 1667 | 1384 (1.20x) | 1679 (0.99x) | 1126 (1.48x) | 801 (2.08x) | - | - | - | - | - | - | - |
| local linear path float32 | **1617** | schedule=sorted: 1621 | 1405 (1.15x) | 1617 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear path float64 | **977** | schedule=sorted: 978 | 828 (1.18x) | 948 (1.03x) | - | - | - | - | - | - | 101 (9.69x) | >10x | - |
| local linear hard int32 | **1890** | schedule=sorted: 1884 | 1531 (1.23x) | 1887 (1.00x) | - | - | - | - | - | - | - | - | - |
| local linear hard float32 | **1822** | schedule=sorted: 1794 | 1549 (1.18x) | 1844 (0.99x) | - | - | - | - | - | - | - | - | - |
| local linear hard float64 | **1044** | schedule=sorted: 1047 | 885 (1.18x) | 946 (1.10x) | - | - | - | - | - | - | - | - | - |
| local linear soft float64 | **262** | schedule=sorted: 279 | 258 (1.02x) | 294 (0.89x) | - | - | - | - | - | - | - | - | - |

#### wloczykij · protein_L100 · 32 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **14077** | fill=striped: 13887 | 15604 (0.90x) | 13852 (1.02x) | 8763 (1.61x) | 2050 (6.87x) | - | 5098 (2.76x) | - | - | - |
| global affine score float32 | **9859** | schedule=sorted: 10003 | 10771 (0.92x) | 9394 (1.05x) | - | - | - | - | - | - | - |
| global affine score float64 | **8223** | schedule=sorted: 8411 | 8335 (0.99x) | 8455 (0.97x) | - | - | - | - | 2944 (2.79x) | - | - |
| global affine path int32 | **8828** | schedule=sorted: 8857 | 7719 (1.14x) | 5679 (1.55x) | 7441 (1.19x) | 3704 (2.38x) | - | 6906 (1.28x) | - | - | - |
| global affine path float32 | **8364** | schedule=sorted: 8581 | 6480 (1.29x) | 8037 (1.04x) | - | - | - | - | - | - | - |
| global affine path float64 | **4144** | schedule=sorted: 5975 | 4425 (0.94x) | 6095 (0.68x) | - | - | - | - | 820 (5.05x) | 830 (4.99x) | - |
| global affine hard int32 | **4891** | fill=striped: 4566 | 4537 (1.08x) | 4253 (1.15x) | - | - | - | - | - | - | - |
| global affine hard float32 | **4998** | schedule=sorted: 4871 | 3630 (1.38x) | 4954 (1.01x) | - | - | - | - | - | - | - |
| global affine hard float64 | **4087** | schedule=sorted: 4159 | 3418 (1.20x) | 3471 (1.18x) | - | - | - | - | - | - | - |
| global affine soft float64 | **967** | schedule=sorted: 971 | 940 (1.03x) | 961 (1.01x) | - | - | - | - | - | - | - |
| global linear score int32 | **13925** | schedule=sorted: 13900 | 15102 (0.92x) | 13547 (1.03x) | 11285 (1.23x) | >10x | - | 6900 (2.02x) | - | - | - |
| global linear score float32 | **12604** | schedule=sorted: 12060 | 13225 (0.95x) | 12133 (1.04x) | - | - | - | - | - | - | - |
| global linear score float64 | **9380** | schedule=sorted: 9357 | 9297 (1.01x) | 8887 (1.06x) | - | - | - | - | 5069 (1.85x) | - | - |
| global linear path int32 | **9441** | fill=striped: 9554 | 10299 (0.92x) | 9581 (0.99x) | 6485 (1.46x) | 2419 (3.90x) | - | 6951 (1.36x) | - | - | - |
| global linear path float32 | **10055** | schedule=sorted: 10267 | 8956 (1.12x) | 10668 (0.94x) | - | - | - | - | - | - | - |
| global linear path float64 | **6513** | schedule=sorted: 6421 | 6846 (0.95x) | 6281 (1.04x) | - | - | - | - | 1670 (3.90x) | 1013 (6.43x) | - |
| global linear hard int32 | **5389** | schedule=sorted: 5876 | 5665 (0.95x) | 5322 (1.01x) | - | - | - | - | - | - | - |
| global linear hard float32 | **5151** | fill=striped: 5509 | 5072 (1.02x) | 4908 (1.05x) | - | - | - | - | - | - | - |
| global linear hard float64 | **4317** | fill=striped: 4613 | 4674 (0.92x) | 4451 (0.97x) | - | - | - | - | - | - | - |
| global linear soft float64 | **713** | fill=striped: 924 | 661 (1.08x) | 705 (1.01x) | - | - | - | - | - | - | - |
| local affine score int32 | **13249** | fill=striped: 13139 | 14401 (0.92x) | 11920 (1.11x) | 9515 (1.39x) | >10x | - | - | - | - | - |
| local affine score float32 | **11191** | fill=striped: 10877 | 11881 (0.94x) | 10724 (1.04x) | - | - | - | - | - | - | - |
| local affine score float64 | **8613** | fill=striped: 8727 | 8624 (1.00x) | 8684 (0.99x) | - | - | - | - | 2188 (3.94x) | - | - |
| local affine path int32 | **7355** | schedule=sorted: 7530 | 6995 (1.05x) | 7569 (0.97x) | 5103 (1.44x) | 4527 (1.62x) | - | - | - | - | - |
| local affine path float32 | **6670** | schedule=sorted: 6725 | 6169 (1.08x) | 4618 (1.44x) | - | - | - | - | - | - | - |
| local affine path float64 | **4407** | schedule=sorted: 4431 | 4196 (1.05x) | 4398 (1.00x) | - | - | - | - | 513 (8.60x) | 859 (5.13x) | - |
| local affine hard int32 | **4694** | schedule=sorted: 4578 | 3807 (1.23x) | 4611 (1.02x) | - | - | - | - | - | - | - |
| local affine hard float32 | **4298** | schedule=sorted: 3910 | 4216 (1.02x) | 4309 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 | **3496** | schedule=sorted: 2784 | 2996 (1.17x) | 3472 (1.01x) | - | - | - | - | - | - | - |
| local affine soft float64 | **839** | schedule=sorted: 896 | 885 (0.95x) | 760 (1.10x) | - | - | - | - | - | - | - |
| local linear score int32 | **12909** | schedule=sorted: 12711 | 13732 (0.94x) | 12808 (1.01x) | 9073 (1.42x) | 2998 (4.31x) | - | - | - | - | - |
| local linear score float32 | **11770** | schedule=sorted: 13914 | 13264 (0.89x) | 12943 (0.91x) | - | - | - | - | - | - | - |
| local linear score float64 | **8215** | fill=striped: 9015 | 5969 (1.38x) | 8973 (0.92x) | - | - | - | - | 3276 (2.51x) | - | - |
| local linear path int32 | **10653** | schedule=sorted: 10671 | 8576 (1.24x) | 10441 (1.02x) | 7119 (1.50x) | 4568 (2.33x) | - | - | - | - | - |
| local linear path float32 | **9521** | schedule=sorted: 9311 | 8017 (1.19x) | 9576 (0.99x) | - | - | - | - | - | - | - |
| local linear path float64 | **6735** | fill=striped: 5238 | 6227 (1.08x) | 6420 (1.05x) | - | - | - | - | 935 (7.20x) | 996 (6.77x) | - |
| local linear hard int32 | **5273** | fill=striped: 5491 | 4724 (1.12x) | 5202 (1.01x) | - | - | - | - | - | - | - |
| local linear hard float32 | **5304** | schedule=sorted: 5065 | 4776 (1.11x) | 5139 (1.03x) | - | - | - | - | - | - | - |
| local linear hard float64 | **4510** | schedule=sorted: 4203 | 4453 (1.01x) | 4771 (0.95x) | - | - | - | - | - | - | - |
| local linear soft float64 | **1569** | schedule=sorted: 1582 | 1554 (1.01x) | 1594 (0.98x) | - | - | - | - | - | - | - |

#### wloczykij · protein_L100 · 8:numa0 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 (load!) | **2468** | schedule=sorted: 2470 | 2745 (0.90x) | 2472 (1.00x) | 2335 (1.06x) | 2871 (0.86x) | - | 1371 (1.80x) | - | - | - |
| global affine score float32 | **1972** | schedule=sorted: 1973 | 2035 (0.97x) | 1974 (1.00x) | - | - | - | - | - | - | - |
| global affine score float64 | **1490** | fill=striped: 1495 | 1490 (1.00x) | 1532 (0.97x) | - | - | - | - | 719 (2.07x) | - | - |
| global affine path int32 (load!) | **1695** | schedule=sorted: 1709 | 1382 (1.23x) | 1711 (0.99x) | 1345 (1.26x) | 1124 (1.51x) | - | 1269 (1.34x) | - | - | - |
| global affine path float32 | **1642** | schedule=sorted: 1655 | 1287 (1.28x) | 1652 (0.99x) | - | - | - | - | - | - | - |
| global affine path float64 | **1088** | schedule=sorted: 1093 | 806 (1.35x) | 1085 (1.00x) | - | - | - | - | 181 (6.00x) | 191 (5.71x) | - |
| global affine hard int32 (load!) | **1640** | schedule=sorted: 1635 | 1385 (1.18x) | 1652 (0.99x) | - | - | - | - | - | - | - |
| global affine hard float32 | **1586** | schedule=sorted: 1594 | 1295 (1.23x) | 1587 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1057** | schedule=sorted: 1056 | 809 (1.31x) | 1068 (0.99x) | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **183** | schedule=sorted: 183 | 183 (1.00x) | 182 (1.00x) | - | - | - | - | - | - | - |
| global linear score int32 (load!) | **2464** | schedule=sorted: 2468 | 2690 (0.92x) | 2468 (1.00x) | 2331 (1.06x) | 2881 (0.86x) | - | 1385 (1.78x) | - | - | - |
| global linear score float32 (load!) | **2458** | schedule=sorted: 2461 | 2568 (0.96x) | 2462 (1.00x) | - | - | - | - | - | - | - |
| global linear score float64 (load!) | **1618** | schedule=sorted: 1619 | 1632 (0.99x) | 1666 (0.97x) | - | - | - | - | 1259 (1.29x) | - | - |
| global linear path int32 (load!) | **1897** | fill=striped: 1895 | 1962 (0.97x) | 1899 (1.00x) | 1344 (1.41x) | 1127 (1.68x) | - | 1274 (1.49x) | - | - | - |
| global linear path float32 (load!) | **2214** | schedule=sorted: 2216 | 1743 (1.27x) | 2219 (1.00x) | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **1346** | fill=striped: 1350 | 1436 (0.94x) | 1354 (0.99x) | - | - | - | - | 402 (3.35x) | 229 (5.89x) | - |
| global linear hard int32 (load!) | **1928** | fill=striped: 1928 | 1994 (0.97x) | 1935 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **2224** | schedule=sorted: 2223 | 1751 (1.27x) | 2203 (1.01x) | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1223** | schedule=sorted: 1381 | 1460 (0.84x) | 1379 (0.89x) | - | - | - | - | - | - | - |
| global linear soft float64 | **156** | fill=striped: 202 | 151 (1.03x) | 156 (1.00x) | - | - | - | - | - | - | - |
| local affine score int32 | **2260** | schedule=sorted: 2261 | 2453 (0.92x) | 2259 (1.00x) | 1937 (1.17x) | 1655 (1.37x) | - | - | - | - | - |
| local affine score float32 | **2120** | schedule=sorted: 2127 | 2202 (0.96x) | 2117 (1.00x) | - | - | - | - | - | - | - |
| local affine score float64 (load!) | **1486** | fill=striped: 1488 | 1473 (1.01x) | 1484 (1.00x) | - | - | - | - | 474 (3.14x) | - | - |
| local affine path int32 (load!) | **1427** | schedule=sorted: 1436 | 1244 (1.15x) | 1443 (0.99x) | 1219 (1.17x) | 1054 (1.35x) | - | - | - | - | - |
| local affine path float32 (load!) | **1291** | schedule=sorted: 1294 | 1165 (1.11x) | 1302 (0.99x) | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **810** | schedule=sorted: 811 | 714 (1.13x) | 824 (0.98x) | - | - | - | - | 115 (7.03x) | 185 (4.39x) | - |
| local affine hard int32 (load!) | **1421** | schedule=sorted: 1431 | 1250 (1.14x) | 1426 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1286** | schedule=sorted: 1288 | 1175 (1.09x) | 1293 (0.99x) | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **808** | schedule=sorted: 810 | 719 (1.12x) | 814 (0.99x) | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **169** | fill=striped: 168 | 170 (0.99x) | 168 (1.00x) | - | - | - | - | - | - | - |
| local linear score int32 (load!) | **2257** | schedule=sorted: 2263 | 2457 (0.92x) | 2272 (0.99x) | 1832 (1.23x) | 1654 (1.36x) | - | - | - | - | - |
| local linear score float32 (load!) | **2525** | fill=striped: 2536 | 2607 (0.97x) | 2529 (1.00x) | - | - | - | - | - | - | - |
| local linear score float64 (load!) | **1593** | fill=striped: 1598 | 1571 (1.01x) | 1594 (1.00x) | - | - | - | - | 789 (2.02x) | - | - |
| local linear path int32 (load!) | **1949** | schedule=sorted: 1959 | 1499 (1.30x) | 1971 (0.99x) | 1212 (1.61x) | 1053 (1.85x) | - | - | - | - | - |
| local linear path float32 (load!) | **1811** | schedule=sorted: 1808 | 1492 (1.21x) | 1823 (0.99x) | - | - | - | - | - | - | - |
| local linear path float64 (load!) | **1263** | schedule=sorted: 1264 | 1103 (1.15x) | 1278 (0.99x) | - | - | - | - | 215 (5.86x) | 221 (5.72x) | - |
| local linear hard int32 (load!) | **1961** | schedule=sorted: 1953 | 1506 (1.30x) | 1959 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **1804** | schedule=sorted: 1810 | 1506 (1.20x) | 1817 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **1259** | schedule=sorted: 1260 | 1111 (1.13x) | 1277 (0.99x) | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **345** | schedule=sorted: 344 | 328 (1.05x) | 346 (1.00x) | - | - | - | - | - | - | - |

#### wloczykij · protein_L1000 · 32 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **18385** | fill=striped: 18334 | 17064 (1.08x) | 17509 (1.05x) | 16688 (1.10x) | 13434 (1.37x) | - | 10734 (1.71x) | - | - | - |
| global affine score float32 | **13586** | schedule=sorted: 13057 | 13647 (1.00x) | 13215 (1.03x) | - | - | - | - | - | - | - |
| global affine score float64 | **9106** | fill=striped: 9357 | 9336 (0.98x) | 9134 (1.00x) | - | - | - | - | 3511 (2.59x) | - | - |
| global affine path int32 | **7300** | fill=striped: 7290 | 7565 (0.96x) | 7321 (1.00x) | 6397 (1.14x) | 2460 (2.97x) | - | 9260 (0.79x) | - | - | - |
| global affine path float32 | **9587** | fill=striped: 10029 | 9987 (0.96x) | 9955 (0.96x) | - | - | - | - | - | - | - |
| global affine path float64 | **6387** | schedule=sorted: 6699 | 7000 (0.91x) | 7025 (0.91x) | - | - | - | - | 928 (6.88x) | 1711 (3.73x) | - |
| global affine hard int32 | **7192** | fill=striped: 6773 | 7476 (0.96x) | 7086 (1.01x) | - | - | - | - | - | - | - |
| global affine hard float32 | **9989** | schedule=sorted: 9625 | 9721 (1.03x) | 9492 (1.05x) | - | - | - | - | - | - | - |
| global affine hard float64 | **6761** | fill=striped: 7148 | 6524 (1.04x) | 7071 (0.96x) | - | - | - | - | - | - | - |
| global affine soft float64 | **249** | schedule=sorted: 252 | 254 (0.98x) | 247 (1.01x) | - | - | - | - | - | - | - |
| global linear score int32 | **17155** | fill=striped: 17837 | 18251 (0.94x) | 14640 (1.17x) | 23541 (0.73x) | 12745 (1.35x) | - | 10426 (1.65x) | - | - | - |
| global linear score float32 | **18745** | schedule=sorted: 19477 | 19930 (0.94x) | 19475 (0.96x) | - | - | - | - | - | - | - |
| global linear score float64 | **6677** | fill=striped: 10959 | 10674 (0.63x) | 10503 (0.64x) | - | - | - | - | 8178 (0.82x) | - | - |
| global linear path int32 | **11659** | fill=striped: 11697 | 12150 (0.96x) | 11662 (1.00x) | 6109 (1.91x) | 2648 (4.40x) | - | 9670 (1.21x) | - | - | - |
| global linear path float32 | **13115** | schedule=sorted: 13369 | 13259 (0.99x) | 12438 (1.05x) | - | - | - | - | - | - | - |
| global linear path float64 | **6250** | fill=striped: 6487 | 6262 (1.00x) | 6406 (0.98x) | - | - | - | - | 2205 (2.83x) | 4768 (1.31x) | - |
| global linear hard int32 | **11248** | schedule=sorted: 11548 | 11951 (0.94x) | 11488 (0.98x) | - | - | - | - | - | - | - |
| global linear hard float32 | **13146** | schedule=sorted: 13216 | 13144 (1.00x) | 13447 (0.98x) | - | - | - | - | - | - | - |
| global linear hard float64 | **6473** | schedule=sorted: 6548 | 6374 (1.02x) | 6384 (1.01x) | - | - | - | - | - | - | - |
| global linear soft float64 | **637** | fill=striped: 646 | 604 (1.05x) | 629 (1.01x) | - | - | - | - | - | - | - |
| local affine score int32 | **13953** | fill=striped: 16310 | 15979 (0.87x) | 16363 (0.85x) | 14906 (0.94x) | 7167 (1.95x) | - | - | - | - | - |
| local affine score float32 | **14928** | fill=striped: 14972 | 13842 (1.08x) | 14780 (1.01x) | - | - | - | - | - | - | - |
| local affine score float64 | **5933** | fill=striped: 9459 | 9385 (0.63x) | 9729 (0.61x) | - | - | - | - | 2596 (2.29x) | - | - |
| local affine path int32 | **4125** | schedule=sorted: 6055 | 6072 (0.68x) | 6121 (0.67x) | 5145 (0.80x) | 2185 (1.89x) | - | - | - | - | - |
| local affine path float32 | **5977** | fill=striped: 6349 | 6309 (0.95x) | 6245 (0.96x) | - | - | - | - | - | - | - |
| local affine path float64 | **3393** | schedule=sorted: 3339 | 3499 (0.97x) | 2450 (1.38x) | - | - | - | - | 545 (6.23x) | 1618 (2.10x) | - |
| local affine hard int32 | **6047** | schedule=sorted: 6064 | 6530 (0.93x) | 6126 (0.99x) | - | - | - | - | - | - | - |
| local affine hard float32 | **5424** | fill=striped: 6123 | 6056 (0.90x) | 5913 (0.92x) | - | - | - | - | - | - | - |
| local affine hard float64 | **3359** | fill=striped: 3316 | 2580 (1.30x) | 2581 (1.30x) | - | - | - | - | - | - | - |
| local affine soft float64 | **247** | fill=striped: 253 | 253 (0.98x) | 245 (1.01x) | - | - | - | - | - | - | - |
| local linear score int32 | **16204** | fill=striped: 15787 | 16383 (0.99x) | 14407 (1.12x) | 23415 (0.69x) | 7378 (2.20x) | - | - | - | - | - |
| local linear score float32 | **11648** | fill=striped: 19702 | 18892 (0.62x) | 18703 (0.62x) | - | - | - | - | - | - | - |
| local linear score float64 | **10605** | fill=striped: 10343 | 10402 (1.02x) | 10361 (1.02x) | - | - | - | - | 5668 (1.87x) | - | - |
| local linear path int32 | **9605** | fill=striped: 9327 | 9870 (0.97x) | 8805 (1.09x) | 7199 (1.33x) | 1962 (4.90x) | - | - | - | - | - |
| local linear path float32 | **9775** | fill=striped: 10018 | 10566 (0.93x) | 10074 (0.97x) | - | - | - | - | - | - | - |
| local linear path float64 | **5035** | schedule=sorted: 5099 | 5094 (0.99x) | 4945 (1.02x) | - | - | - | - | 1069 (4.71x) | 3934 (1.28x) | - |
| local linear hard int32 | **9713** | schedule=sorted: 9367 | 7114 (1.37x) | 8980 (1.08x) | - | - | - | - | - | - | - |
| local linear hard float32 | **7663** | schedule=sorted: 9976 | 10597 (0.72x) | 9966 (0.77x) | - | - | - | - | - | - | - |
| local linear hard float64 | **5048** | schedule=sorted: 5072 | 5131 (0.98x) | 5073 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 | **496** | schedule=sorted: 507 | 506 (0.98x) | 497 (1.00x) | - | - | - | - | - | - | - |

#### wloczykij · protein_L1000 · 8:numa0 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **2888** | schedule=sorted: 2890 | 2988 (0.97x) | 2874 (1.00x) | 3699 (0.78x) | 3096 (0.93x) | - | 1927 (1.50x) | - | - | - |
| global affine score float32 | **2479** | schedule=sorted: 2514 | 2509 (0.99x) | 2495 (0.99x) | - | - | - | - | - | - | - |
| global affine score float64 | **1567** | fill=striped: 1572 | 1579 (0.99x) | 1576 (0.99x) | - | - | - | - | 899 (1.74x) | - | - |
| global affine path int32 | **1204** | schedule=sorted: 1207 | 1225 (0.98x) | 1215 (0.99x) | 1471 (0.82x) | 833 (1.45x) | - | 1826 (0.66x) | - | - | - |
| global affine path float32 | **1930** | schedule=sorted: 1949 | 1983 (0.97x) | 1949 (0.99x) | - | - | - | - | - | - | - |
| global affine path float64 | **1268** | schedule=sorted: 1269 | 1253 (1.01x) | 1266 (1.00x) | - | - | - | - | 200 (6.35x) | 544 (2.33x) | - |
| global affine hard int32 | **1205** | schedule=sorted: 1210 | 1223 (0.99x) | 1213 (0.99x) | - | - | - | - | - | - | - |
| global affine hard float32 | **1940** | fill=striped: 1937 | 1983 (0.98x) | 1944 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float64 | **1268** | schedule=sorted: 1267 | 1246 (1.02x) | 1272 (1.00x) | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **61** | fill=striped: 61 | 60 (1.02x) | 60 (1.02x) | - | - | - | - | - | - | - |
| global linear score int32 | **2898** | fill=striped: 2894 | 2925 (0.99x) | 2877 (1.01x) | 4786 (0.61x) | 3107 (0.93x) | - | 1937 (1.50x) | - | - | - |
| global linear score float32 | **3396** | schedule=sorted: 3433 | 3439 (0.99x) | 3432 (0.99x) | - | - | - | - | - | - | - |
| global linear score float64 | **1774** | schedule=sorted: 1775 | 1769 (1.00x) | 1774 (1.00x) | - | - | - | - | 2057 (0.86x) | - | - |
| global linear path int32 (load!) | **1915** | fill=striped: 1919 | 1972 (0.97x) | 1932 (0.99x) | 1472 (1.30x) | 819 (2.34x) | - | 1818 (1.05x) | - | - | - |
| global linear path float32 (load!) | **2549** | fill=striped: 2544 | 2589 (0.98x) | 2585 (0.99x) | - | - | - | - | - | - | - |
| global linear path float64 | **1207** | schedule=sorted: 1206 | 1196 (1.01x) | 1209 (1.00x) | - | - | - | - | 524 (2.30x) | 1199 (1.01x) | - |
| global linear hard int32 | **1909** | fill=striped: 1926 | 1990 (0.96x) | 1926 (0.99x) | - | - | - | - | - | - | - |
| global linear hard float32 | **2531** | fill=striped: 2533 | 2595 (0.98x) | 2551 (0.99x) | - | - | - | - | - | - | - |
| global linear hard float64 | **1208** | fill=striped: 1214 | 1200 (1.01x) | 1212 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **146** | schedule=sorted: 147 | 137 (1.07x) | 146 (1.00x) | - | - | - | - | - | - | - |
| local affine score int32 (load!) | **2701** | schedule=sorted: 2699 | 2739 (0.99x) | 2706 (1.00x) | 3226 (0.84x) | 1806 (1.50x) | - | - | - | - | - |
| local affine score float32 (load!) | **2639** | schedule=sorted: 2649 | 2664 (0.99x) | 2669 (0.99x) | - | - | - | - | - | - | - |
| local affine score float64 | **1587** | schedule=sorted: 1592 | 1583 (1.00x) | 1588 (1.00x) | - | - | - | - | 566 (2.80x) | - | - |
| local affine path int32 (load!) | **984** | fill=striped: 985 | 1015 (0.97x) | 985 (1.00x) | 1332 (0.74x) | 811 (1.21x) | - | - | - | - | - |
| local affine path float32 (load!) | **1087** | fill=striped: 1090 | 1110 (0.98x) | 1073 (1.01x) | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **597** | schedule=sorted: 595 | 595 (1.00x) | 597 (1.00x) | - | - | - | - | 122 (4.91x) | 472 (1.27x) | - |
| local affine hard int32 (load!) | **985** | fill=striped: 992 | 1025 (0.96x) | 994 (0.99x) | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1089** | schedule=sorted: 1089 | 1116 (0.98x) | 1074 (1.01x) | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **597** | schedule=sorted: 597 | 597 (1.00x) | 600 (1.00x) | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **59** | schedule=sorted: 59 | 60 (0.98x) | 58 (1.01x) | - | - | - | - | - | - | - |
| local linear score int32 | **2708** | fill=striped: 2702 | 2777 (0.97x) | 2713 (1.00x) | 5105 (0.53x) | 1793 (1.51x) | - | - | - | - | - |
| local linear score float32 | **3356** | schedule=sorted: 3360 | 3369 (1.00x) | 3354 (1.00x) | - | - | - | - | - | - | - |
| local linear score float64 | **1714** | fill=striped: 1719 | 1718 (1.00x) | 1729 (0.99x) | - | - | - | - | 1389 (1.23x) | - | - |
| local linear path int32 (load!) | **1626** | schedule=sorted: 1631 | 1662 (0.98x) | 1638 (0.99x) | 1330 (1.22x) | 806 (2.02x) | - | - | - | - | - |
| local linear path float32 (load!) | **2093** | fill=striped: 2082 | 2105 (0.99x) | 2084 (1.00x) | - | - | - | - | - | - | - |
| local linear path float64 (load!) | **935** | schedule=sorted: 939 | 1004 (0.93x) | 937 (1.00x) | - | - | - | - | 250 (3.74x) | 911 (1.03x) | - |
| local linear hard int32 (load!) | **1629** | fill=striped: 1637 | 1664 (0.98x) | 1640 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **2104** | schedule=sorted: 2107 | 2129 (0.99x) | 2116 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **943** | fill=striped: 943 | 1007 (0.94x) | 942 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **114** | fill=striped: 114 | 115 (0.99x) | 114 (1.00x) | - | - | - | - | - | - | - |

#### wloczykij · protein_L300 · 32 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **19219** | fill=striped: 19828 | 21132 (0.91x) | 19376 (0.99x) | 16975 (1.13x) | 13921 (1.38x) | - | 6303 (3.05x) | - | - | - |
| global affine score float32 | **9231** | fill=striped: 13566 | 13429 (0.69x) | 13541 (0.68x) | - | - | - | - | - | - | - |
| global affine score float64 | **9575** | schedule=sorted: 9376 | 9565 (1.00x) | 9478 (1.01x) | - | - | - | - | 3556 (2.69x) | - | - |
| global affine path int32 | **7690** | fill=striped: 7917 | 7936 (0.97x) | 7974 (0.96x) | 10278 (0.75x) | 5849 (1.31x) | - | 9393 (0.82x) | - | - | - |
| global affine path float32 | **6871** | schedule=sorted: 7043 | 6905 (1.00x) | 7044 (0.98x) | - | - | - | - | - | - | - |
| global affine path float64 | **4196** | fill=striped: 4192 | 4140 (1.01x) | 3938 (1.07x) | - | - | - | - | 921 (4.56x) | 2165 (1.94x) | - |
| global affine hard int32 | **4665** | fill=striped: 7150 | 6879 (0.68x) | 7145 (0.65x) | - | - | - | - | - | - | - |
| global affine hard float32 | **6505** | fill=striped: 6500 | 6412 (1.01x) | 6362 (1.02x) | - | - | - | - | - | - | - |
| global affine hard float64 | **2851** | schedule=sorted: 3922 | 3681 (0.77x) | 3992 (0.71x) | - | - | - | - | - | - | - |
| global affine soft float64 | **323** | schedule=sorted: 328 | 313 (1.03x) | 324 (1.00x) | - | - | - | - | - | - | - |
| global linear score int32 | **19718** | schedule=sorted: 19783 | 20827 (0.95x) | 19257 (1.02x) | 16984 (1.16x) | 14686 (1.34x) | - | 9987 (1.97x) | - | - | - |
| global linear score float32 | **18412** | fill=striped: 12538 | 18364 (1.00x) | 18235 (1.01x) | - | - | - | - | - | - | - |
| global linear score float64 | **8671** | schedule=sorted: 7132 | 10734 (0.81x) | 10198 (0.85x) | - | - | - | - | 6942 (1.25x) | - | - |
| global linear path int32 | **14150** | fill=striped: 14726 | 15414 (0.92x) | 14566 (0.97x) | 10491 (1.35x) | 5726 (2.47x) | - | 9385 (1.51x) | - | - | - |
| global linear path float32 | **14494** | schedule=sorted: 14469 | 14520 (1.00x) | 14304 (1.01x) | - | - | - | - | - | - | - |
| global linear path float64 | **8376** | schedule=sorted: 8603 | 9022 (0.93x) | 8727 (0.96x) | - | - | - | - | 2269 (3.69x) | 3620 (2.31x) | - |
| global linear hard int32 | **11782** | fill=striped: 13034 | 13811 (0.85x) | 12932 (0.91x) | - | - | - | - | - | - | - |
| global linear hard float32 | **12682** | fill=striped: 12778 | 13703 (0.93x) | 12021 (1.06x) | - | - | - | - | - | - | - |
| global linear hard float64 | **7890** | fill=striped: 8114 | 8548 (0.92x) | 8086 (0.98x) | - | - | - | - | - | - | - |
| global linear soft float64 | **429** | schedule=sorted: 432 | 394 (1.09x) | 424 (1.01x) | - | - | - | - | - | - | - |
| local affine score int32 | **9913** | schedule=sorted: 17310 | 18138 (0.55x) | 16265 (0.61x) | 14066 (0.70x) | 7931 (1.25x) | - | - | - | - | - |
| local affine score float32 | **14996** | fill=striped: 14300 | 9308 (1.61x) | 14978 (1.00x) | - | - | - | - | - | - | - |
| local affine score float64 | **10058** | schedule=sorted: 10032 | 8553 (1.18x) | 6246 (1.61x) | - | - | - | - | 2575 (3.91x) | - | - |
| local affine path int32 | **6423** | fill=striped: 6489 | 7100 (0.90x) | 6384 (1.01x) | 8259 (0.78x) | 5118 (1.25x) | - | - | - | - | - |
| local affine path float32 | **6360** | schedule=sorted: 6346 | 6418 (0.99x) | 6314 (1.01x) | - | - | - | - | - | - | - |
| local affine path float64 | **2512** | fill=striped: 3546 | 3539 (0.71x) | 3486 (0.72x) | - | - | - | - | 572 (4.39x) | 2079 (1.21x) | - |
| local affine hard int32 | **5925** | fill=striped: 4245 | 6420 (0.92x) | 5980 (0.99x) | - | - | - | - | - | - | - |
| local affine hard float32 | **5949** | fill=striped: 5798 | 6017 (0.99x) | 5751 (1.03x) | - | - | - | - | - | - | - |
| local affine hard float64 | **3405** | fill=striped: 3421 | 3489 (0.98x) | 3426 (0.99x) | - | - | - | - | - | - | - |
| local affine soft float64 | **150** | fill=striped: 150 | 153 (0.98x) | 150 (1.00x) | - | - | - | - | - | - | - |
| local linear score int32 | **15015** | schedule=sorted: 18002 | 18202 (0.82x) | 17944 (0.84x) | 13063 (1.15x) | 8165 (1.84x) | - | - | - | - | - |
| local linear score float32 | **12053** | schedule=sorted: 19321 | 19892 (0.61x) | 19748 (0.61x) | - | - | - | - | - | - | - |
| local linear score float64 | **10849** | fill=striped: 10806 | 10969 (0.99x) | 10931 (0.99x) | - | - | - | - | 4542 (2.39x) | - | - |
| local linear path int32 | **11781** | fill=striped: 11730 | 12721 (0.93x) | 12229 (0.96x) | 8961 (1.31x) | 4812 (2.45x) | - | - | - | - | - |
| local linear path float32 | **11635** | fill=striped: 11749 | 11273 (1.03x) | 11738 (0.99x) | - | - | - | - | - | - | - |
| local linear path float64 | **6694** | schedule=sorted: 6672 | 6964 (0.96x) | 6921 (0.97x) | - | - | - | - | 1064 (6.29x) | 3445 (1.94x) | - |
| local linear hard int32 | **10493** | schedule=sorted: 10755 | 10629 (0.99x) | 10478 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float32 | **10191** | fill=striped: 10535 | 10590 (0.96x) | 10728 (0.95x) | - | - | - | - | - | - | - |
| local linear hard float64 | **6443** | fill=striped: 6390 | 6622 (0.97x) | 6557 (0.98x) | - | - | - | - | - | - | - |
| local linear soft float64 | **248** | schedule=sorted: 248 | 251 (0.99x) | 244 (1.01x) | - | - | - | - | - | - | - |

#### wloczykij · protein_L300 · 8:numa0 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 (load!) | **3122** | fill=striped: 3122 | 3363 (0.93x) | 3121 (1.00x) | 3341 (0.93x) | 3157 (0.99x) | - | 1774 (1.76x) | - | - | - |
| global affine score float32 | **2549** | schedule=sorted: 2551 | 2575 (0.99x) | 2552 (1.00x) | - | - | - | - | - | - | - |
| global affine score float64 | **1658** | schedule=sorted: 1657 | 1575 (1.05x) | 1643 (1.01x) | - | - | - | - | 906 (1.83x) | - | - |
| global affine path int32 (load!) | **1234** | fill=striped: 1250 | 1286 (0.96x) | 1246 (0.99x) | 1684 (0.73x) | 1059 (1.17x) | - | 1682 (0.73x) | - | - | - |
| global affine path float32 | **1186** | fill=striped: 1185 | 1196 (0.99x) | 1191 (1.00x) | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **698** | schedule=sorted: 698 | 703 (0.99x) | 708 (0.99x) | - | - | - | - | 205 (3.40x) | 529 (1.32x) | - |
| global affine hard int32 (load!) | **1244** | fill=striped: 1248 | 1302 (0.96x) | 1243 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **1185** | schedule=sorted: 1189 | 1204 (0.98x) | 1196 (0.99x) | - | - | - | - | - | - | - |
| global affine hard float64 | **700** | fill=striped: 698 | 708 (0.99x) | 710 (0.99x) | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **68** | fill=striped: 68 | 66 (1.03x) | 68 (1.00x) | - | - | - | - | - | - | - |
| global linear score int32 | **3123** | schedule=sorted: 3124 | 3362 (0.93x) | 3123 (1.00x) | 3346 (0.93x) | 3116 (1.00x) | - | 1779 (1.76x) | - | - | - |
| global linear score float32 | **3460** | fill=striped: 3465 | 3523 (0.98x) | 3452 (1.00x) | - | - | - | - | - | - | - |
| global linear score float64 | **1784** | schedule=sorted: 1786 | 1764 (1.01x) | 1791 (1.00x) | - | - | - | - | 1909 (0.93x) | - | - |
| global linear path int32 (load!) | **2595** | fill=striped: 2589 | 2639 (0.98x) | 2618 (0.99x) | 1686 (1.54x) | 1034 (2.51x) | - | 1677 (1.55x) | - | - | - |
| global linear path float32 (load!) | **2988** | fill=striped: 2985 | 3146 (0.95x) | 2994 (1.00x) | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **1702** | schedule=sorted: 1719 | 1779 (0.96x) | 1728 (0.98x) | - | - | - | - | 528 (3.23x) | 856 (1.99x) | - |
| global linear hard int32 (load!) | **2628** | fill=striped: 2628 | 2711 (0.97x) | 2639 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float32 | **3095** | schedule=sorted: 3099 | 3275 (0.94x) | 3097 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float64 | **1757** | fill=striped: 1760 | 1817 (0.97x) | 1755 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **93** | schedule=sorted: 93 | 86 (1.08x) | 92 (1.00x) | - | - | - | - | - | - | - |
| local affine score int32 (load!) | **2763** | schedule=sorted: 2763 | 2988 (0.92x) | 2763 (1.00x) | 2866 (0.96x) | 1800 (1.54x) | - | - | - | - | - |
| local affine score float32 (load!) | **2686** | schedule=sorted: 2687 | 2735 (0.98x) | 2690 (1.00x) | - | - | - | - | - | - | - |
| local affine score float64 | **1652** | fill=striped: 1652 | 1641 (1.01x) | 1651 (1.00x) | - | - | - | - | 569 (2.90x) | - | - |
| local affine path int32 (load!) | **1064** | fill=striped: 1073 | 1134 (0.94x) | 1066 (1.00x) | 1498 (0.71x) | 997 (1.07x) | - | - | - | - | - |
| local affine path float32 (load!) | **1052** | fill=striped: 1051 | 1068 (0.98x) | 1049 (1.00x) | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **582** | fill=striped: 582 | 581 (1.00x) | 590 (0.99x) | - | - | - | - | 126 (4.62x) | 500 (1.16x) | - |
| local affine hard int32 (load!) | **1068** | fill=striped: 1076 | 1148 (0.93x) | 1075 (0.99x) | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1054** | schedule=sorted: 1055 | 1075 (0.98x) | 1053 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **583** | fill=striped: 583 | 582 (1.00x) | 591 (0.99x) | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **32** | fill=striped: 32 | 33 (0.97x) | 32 (1.00x) | - | - | - | - | - | - | - |
| local linear score int32 (load!) | **2763** | schedule=sorted: 2764 | 2985 (0.93x) | 2762 (1.00x) | 2481 (1.11x) | 1804 (1.53x) | - | - | - | - | - |
| local linear score float32 (load!) | **3360** | schedule=sorted: 3362 | 3411 (0.99x) | 3368 (1.00x) | - | - | - | - | - | - | - |
| local linear score float64 (load!) | **1773** | fill=striped: 1775 | 1750 (1.01x) | 1803 (0.98x) | - | - | - | - | 1155 (1.54x) | - | - |
| local linear path int32 | **2185** | schedule=sorted: 2180 | 2188 (1.00x) | 2178 (1.00x) | 1486 (1.47x) | 987 (2.21x) | - | - | - | - | - |
| local linear path float32 | **2338** | fill=striped: 2365 | 2341 (1.00x) | 2350 (0.99x) | - | - | - | - | - | - | - |
| local linear path float64 | **1362** | schedule=sorted: 1361 | 1367 (1.00x) | 1366 (1.00x) | - | - | - | - | 251 (5.42x) | 823 (1.65x) | - |
| local linear hard int32 | **2234** | schedule=sorted: 2236 | 2243 (1.00x) | 2215 (1.01x) | - | - | - | - | - | - | - |
| local linear hard float32 | **2387** | fill=striped: 2425 | 2396 (1.00x) | 2392 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float64 | **1376** | schedule=sorted: 1378 | 1390 (0.99x) | 1379 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **54** | schedule=sorted: 54 | 54 (1.00x) | 53 (1.00x) | - | - | - | - | - | - | - |

#### wloczykij · protein_L3000 · 32 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 | **11731** | fill=striped: 11353 | 12080 (0.97x) | 11625 (1.01x) | 11651 (1.01x) | 4684 (2.50x) | - | 7300 (1.61x) | - | - | - |
| global affine score float32 | **9437** | fill=striped: 9513 | 9407 (1.00x) | 9485 (0.99x) | - | - | - | - | - | - | - |
| global affine score float64 | **6251** | schedule=sorted: 6377 | 6241 (1.00x) | 6422 (0.97x) | - | - | - | - | 2425 (2.58x) | - | - |
| global affine path int32 | **4546** | schedule=sorted: 4707 | 4759 (0.96x) | 4637 (0.98x) | 4472 (1.02x) | 978 (4.65x) | - | 6544 (0.69x) | - | - | - |
| global affine path float32 | **8022** | schedule=sorted: 8109 | 7943 (1.01x) | 8088 (0.99x) | - | - | - | - | - | - | - |
| global affine path float64 | **5238** | schedule=sorted: 5487 | 5451 (0.96x) | 5277 (0.99x) | - | - | - | - | 654 (8.01x) | 979 (5.35x) | - |
| global affine hard int32 | **4636** | schedule=sorted: 4556 | 4903 (0.95x) | 4586 (1.01x) | - | - | - | - | - | - | - |
| global affine hard float32 | **7904** | schedule=sorted: 8172 | 7872 (1.00x) | 8000 (0.99x) | - | - | - | - | - | - | - |
| global affine hard float64 | **5440** | fill=striped: 5449 | 5407 (1.01x) | 5471 (0.99x) | - | - | - | - | - | - | - |
| global affine soft float64 | **218** | schedule=sorted: 212 | 207 (1.06x) | 234 (0.93x) | - | - | - | - | - | - | - |
| global linear score int32 | **11076** | schedule=sorted: 11624 | 11854 (0.93x) | 11651 (0.95x) | 11760 (0.94x) | 4887 (2.27x) | - | 7312 (1.51x) | - | - | - |
| global linear score float32 | **14349** | schedule=sorted: 14159 | 14576 (0.98x) | 14397 (1.00x) | - | - | - | - | - | - | - |
| global linear score float64 | **7417** | schedule=sorted: 7329 | 7297 (1.02x) | 7348 (1.01x) | - | - | - | - | 5536 (1.34x) | - | - |
| global linear path int32 | **6321** | schedule=sorted: 7474 | 7696 (0.82x) | 7161 (0.88x) | 5278 (1.20x) | 1128 (5.61x) | - | 6434 (0.98x) | - | - | - |
| global linear path float32 | **7915** | schedule=sorted: 8474 | 8639 (0.92x) | 8653 (0.91x) | - | - | - | - | - | - | - |
| global linear path float64 | **4464** | schedule=sorted: 4443 | 4474 (1.00x) | 4419 (1.01x) | - | - | - | - | 1702 (2.62x) | 2958 (1.51x) | - |
| global linear hard int32 | **6453** | schedule=sorted: 7288 | 7097 (0.91x) | 6659 (0.97x) | - | - | - | - | - | - | - |
| global linear hard float32 | **8488** | fill=striped: 8740 | 8413 (1.01x) | 8869 (0.96x) | - | - | - | - | - | - | - |
| global linear hard float64 | **4458** | schedule=sorted: 4485 | 4450 (1.00x) | 4399 (1.01x) | - | - | - | - | - | - | - |
| global linear soft float64 | **683** | fill=striped: 715 | 675 (1.01x) | 689 (0.99x) | - | - | - | - | - | - | - |
| local affine score int32 | **10676** | schedule=sorted: 10466 | 9995 (1.07x) | 10162 (1.05x) | 9136 (1.17x) | 2537 (4.21x) | - | - | - | - | - |
| local affine score float32 | **10571** | fill=striped: 10219 | 9825 (1.08x) | 10464 (1.01x) | - | - | - | - | - | - | - |
| local affine score float64 | **6560** | schedule=sorted: 6703 | 6619 (0.99x) | 6617 (0.99x) | - | - | - | - | 1920 (3.42x) | - | - |
| local affine path int32 | **4075** | schedule=sorted: 4096 | 4355 (0.94x) | 4103 (0.99x) | 4424 (0.92x) | 1123 (3.63x) | - | - | - | - | - |
| local affine path float32 | **4177** | fill=striped: 4213 | 4159 (1.00x) | 4231 (0.99x) | - | - | - | - | - | - | - |
| local affine path float64 | **2266** | schedule=sorted: 2296 | 2324 (0.97x) | 2179 (1.04x) | - | - | - | - | 403 (5.63x) | 997 (2.27x) | - |
| local affine hard int32 | **4061** | fill=striped: 3960 | 4240 (0.96x) | 4064 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float32 | **4205** | schedule=sorted: 4193 | 4152 (1.01x) | 4189 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 | **2219** | schedule=sorted: 2270 | 2348 (0.94x) | 2273 (0.98x) | - | - | - | - | - | - | - |
| local affine soft float64 | **159** | schedule=sorted: 159 | 153 (1.04x) | 154 (1.03x) | - | - | - | - | - | - | - |
| local linear score int32 | **10488** | schedule=sorted: 10642 | 10509 (1.00x) | 10708 (0.98x) | 9630 (1.09x) | 2433 (4.31x) | - | - | - | - | - |
| local linear score float32 | **13953** | fill=striped: 13949 | 14317 (0.97x) | 14160 (0.99x) | - | - | - | - | - | - | - |
| local linear score float64 | **6908** | fill=striped: 7156 | 7135 (0.97x) | 6820 (1.01x) | - | - | - | - | 4637 (1.49x) | - | - |
| local linear path int32 | **6114** | fill=striped: 6133 | 6484 (0.94x) | 6179 (0.99x) | 4494 (1.36x) | 1088 (5.62x) | - | - | - | - | - |
| local linear path float32 | **6537** | fill=striped: 6492 | 6644 (0.98x) | 6296 (1.04x) | - | - | - | - | - | - | - |
| local linear path float64 | **3413** | schedule=sorted: 3405 | 3465 (0.98x) | 3414 (1.00x) | - | - | - | - | 805 (4.24x) | 2525 (1.35x) | - |
| local linear hard int32 | **6209** | schedule=sorted: 6271 | 6304 (0.98x) | 6159 (1.01x) | - | - | - | - | - | - | - |
| local linear hard float32 | **6397** | fill=striped: 6605 | 6867 (0.93x) | 6378 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float64 | **3431** | schedule=sorted: 3412 | 3457 (0.99x) | 3315 (1.03x) | - | - | - | - | - | - | - |
| local linear soft float64 | **566** | schedule=sorted: 606 | 562 (1.01x) | 580 (0.98x) | - | - | - | - | - | - | - |

#### wloczykij · protein_L3000 · 8:numa0 threads

| cell | **nwgrad (default)** | best nwgrad variant | clang | C++ | parasail | seqan3 | seqan2 | ksw2 | biopython | scikit-bio | pyopal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global affine score int32 (load!) | **2706** | fill=striped: 2708 | 2734 (0.99x) | 2713 (1.00x) | 3250 (0.83x) | 2259 (1.20x) | - | 1926 (1.41x) | - | - | - |
| global affine score float32 (load!) | **2477** | schedule=sorted: 2482 | 2508 (0.99x) | 2475 (1.00x) | - | - | - | - | - | - | - |
| global affine score float64 (load!) | **1550** | fill=striped: 1551 | 1553 (1.00x) | 1559 (0.99x) | - | - | - | - | 839 (1.85x) | - | - |
| global affine path int32 (load!) | **1159** | schedule=sorted: 1159 | 1179 (0.98x) | 1162 (1.00x) | 1021 (1.13x) | 423 (2.74x) | - | 1822 (0.64x) | - | - | - |
| global affine path float32 | **2240** | schedule=sorted: 2228 | 2251 (1.00x) | 2230 (1.00x) | - | - | - | - | - | - | - |
| global affine path float64 (load!) | **1391** | schedule=sorted: 1387 | 1337 (1.04x) | 1393 (1.00x) | - | - | - | - | 194 (7.17x) | 282 (4.93x) | - |
| global affine hard int32 (load!) | **1147** | fill=striped: 1156 | 1175 (0.98x) | 1161 (0.99x) | - | - | - | - | - | - | - |
| global affine hard float32 (load!) | **2224** | fill=striped: 2244 | 2240 (0.99x) | 2224 (1.00x) | - | - | - | - | - | - | - |
| global affine hard float64 (load!) | **1396** | schedule=sorted: 1381 | 1352 (1.03x) | 1393 (1.00x) | - | - | - | - | - | - | - |
| global affine soft float64 (load!) | **72** | schedule=sorted: 72 | 73 (1.00x) | 72 (1.00x) | - | - | - | - | - | - | - |
| global linear score int32 (load!) | **2721** | fill=striped: 2709 | 2730 (1.00x) | 2698 (1.01x) | 4307 (0.63x) | 2228 (1.22x) | - | 1922 (1.42x) | - | - | - |
| global linear score float32 (load!) | **3465** | fill=striped: 3483 | 3426 (1.01x) | 3483 (0.99x) | - | - | - | - | - | - | - |
| global linear score float64 (load!) | **1759** | fill=striped: 1755 | 1744 (1.01x) | 1762 (1.00x) | - | - | - | - | 2003 (0.88x) | - | - |
| global linear path int32 (load!) | **1722** | fill=striped: 1740 | 1821 (0.95x) | 1736 (0.99x) | 1003 (1.72x) | 420 (4.10x) | - | 1799 (0.96x) | - | - | - |
| global linear path float32 (load!) | **2272** | schedule=sorted: 2291 | 2275 (1.00x) | 2282 (1.00x) | - | - | - | - | - | - | - |
| global linear path float64 (load!) | **1171** | schedule=sorted: 1168 | 1150 (1.02x) | 1174 (1.00x) | - | - | - | - | 507 (2.31x) | 856 (1.37x) | - |
| global linear hard int32 (load!) | **1751** | fill=striped: 1737 | 1826 (0.96x) | 1717 (1.02x) | - | - | - | - | - | - | - |
| global linear hard float32 (load!) | **2294** | fill=striped: 2297 | 2279 (1.01x) | 2295 (1.00x) | - | - | - | - | - | - | - |
| global linear hard float64 (load!) | **1168** | schedule=sorted: 1170 | 1163 (1.00x) | 1173 (1.00x) | - | - | - | - | - | - | - |
| global linear soft float64 (load!) | **214** | fill=striped: 216 | 205 (1.04x) | 216 (0.99x) | - | - | - | - | - | - | - |
| local affine score int32 (load!) | **2575** | fill=striped: 2606 | 2588 (0.99x) | 2624 (0.98x) | 3087 (0.83x) | 1306 (1.97x) | - | - | - | - | - |
| local affine score float32 (load!) | **2634** | schedule=sorted: 2646 | 2656 (0.99x) | 2649 (0.99x) | - | - | - | - | - | - | - |
| local affine score float64 (load!) | **1563** | schedule=sorted: 1568 | 1563 (1.00x) | 1563 (1.00x) | - | - | - | - | 553 (2.83x) | - | - |
| local affine path int32 (load!) | **951** | schedule=sorted: 952 | 985 (0.97x) | 959 (0.99x) | 978 (0.97x) | 407 (2.34x) | - | - | - | - | - |
| local affine path float32 (load!) | **1010** | fill=striped: 1022 | 1051 (0.96x) | 1007 (1.00x) | - | - | - | - | - | - | - |
| local affine path float64 (load!) | **578** | schedule=sorted: 578 | 580 (1.00x) | 577 (1.00x) | - | - | - | - | 119 (4.87x) | 265 (2.18x) | - |
| local affine hard int32 (load!) | **953** | fill=striped: 958 | 985 (0.97x) | 961 (0.99x) | - | - | - | - | - | - | - |
| local affine hard float32 (load!) | **1013** | fill=striped: 1021 | 1064 (0.95x) | 1014 (1.00x) | - | - | - | - | - | - | - |
| local affine hard float64 (load!) | **577** | schedule=sorted: 577 | 578 (1.00x) | 578 (1.00x) | - | - | - | - | - | - | - |
| local affine soft float64 (load!) | **49** | schedule=sorted: 49 | 50 (0.97x) | 49 (1.00x) | - | - | - | - | - | - | - |
| local linear score int32 | **2615** | schedule=sorted: 2594 | 2627 (1.00x) | 2561 (1.02x) | 4417 (0.59x) | 1304 (2.00x) | - | - | - | - | - |
| local linear score float32 | **3382** | fill=striped: 3370 | 3366 (1.00x) | 3380 (1.00x) | - | - | - | - | - | - | - |
| local linear score float64 | **1698** | schedule=sorted: 1700 | 1675 (1.01x) | 1695 (1.00x) | - | - | - | - | 1488 (1.14x) | - | - |
| local linear path int32 (load!) | **1519** | fill=striped: 1536 | 1537 (0.99x) | 1540 (0.99x) | 953 (1.59x) | 406 (3.74x) | - | - | - | - | - |
| local linear path float32 (load!) | **1843** | schedule=sorted: 1864 | 1867 (0.99x) | 1851 (1.00x) | - | - | - | - | - | - | - |
| local linear path float64 (load!) | **903** | fill=striped: 917 | 960 (0.94x) | 916 (0.98x) | - | - | - | - | 243 (3.72x) | 723 (1.25x) | - |
| local linear hard int32 (load!) | **1539** | schedule=sorted: 1532 | 1540 (1.00x) | 1549 (0.99x) | - | - | - | - | - | - | - |
| local linear hard float32 (load!) | **1843** | fill=striped: 1854 | 1869 (0.99x) | 1848 (1.00x) | - | - | - | - | - | - | - |
| local linear hard float64 (load!) | **908** | fill=striped: 913 | 956 (0.95x) | 909 (1.00x) | - | - | - | - | - | - | - |
| local linear soft float64 (load!) | **179** | schedule=sorted: 180 | 179 (1.00x) | 180 (0.99x) | - | - | - | - | - | - | - |

</details>

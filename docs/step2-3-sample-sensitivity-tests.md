# Step 2.3 Optimizer Sensitivity Testing — Combined Results

**Version:** 1.1 | **Updated:** 2026-05-01 | **Status:** draft (Tests 1–2 complete, Tests 3–4 pending)

Four tests characterize how LHS sample size and beam diversity affect the Step 2.3 adaptive pathway optimizer’s cost outcomes. All tests use ERCOT, cost mode 2 (incremental clean cost − gas savings + stranding shadow), base scenario, Medium demand growth.

-----

## Test 1: LHS Sample-Size Sensitivity (Single Beam)

**Question:** Does increasing LHS sample count per solver step improve cost optimality on a fixed beam?

**Setup:** Beam 0 only (cheapest seed per pathway). LHS: 500–40,000. Seeds are LHS-independent; `scaled_samples` multiplies nominal by `(n_dims/9)^1.5`.

### Pathway A (VRE + storage, 9 dims)

|LHS      |Scaled   |Time     |90% ($B)  |95% ($B)  |99% ($B)    |99.9% ($B)   |Pk Gas (MW)|Pk H2 (MW)|
|--------:|--------:|--------:|---------:|---------:|-----------:|------------:|----------:|---------:|
|500      |500      |1.6s     |577.78    |1,190.89  |1,839.12    |2,025.18     |99,868     |42,270    |
|1,000    |1,000    |3.2s     |572.80    |1,102.64  |1,689.28    |1,872.60     |100,382    |17,163    |
|2,500    |2,500    |7.6s     |344.28    |839.76    |2,212.55    |2,212.55*    |92,570     |16,750    |
|**5,000**|**5,000**|**15.0s**|**346.37**|**864.78**|**2,073.81**|**2,073.81***|**91,871** |**5,679** |
|10,000   |10,000   |30.1s    |350.29    |867.89    |2,145.65    |2,145.65*    |92,717     |13,693    |
|20,000   |20,000   |59.0s    |324.74    |748.91    |1,584.49    |1,867.54     |92,131     |4,993     |
|40,000   |40,000   |206.6s   |321.37    |760.02    |1,835.57    |1,835.57*    |92,130     |55,912    |

** 99% and 99.9% achieved in same solver year (overshot)*

### Pathway B (all resources, 10 dims)

|LHS      |Scaled   |Time     |90% ($B)  |95% ($B)  |99% ($B)    |99.9% ($B)  |Pk Gas (MW)|Pk H2 (MW)|
|--------:|--------:|--------:|---------:|---------:|-----------:|-----------:|----------:|---------:|
|500      |585      |2.2s     |993.51    |1,665.48  |2,339.34    |2,533.90    |83,396     |24,019    |
|1,000    |1,171    |4.4s     |943.50    |1,618.90  |2,299.55    |2,493.90    |85,685     |19,972    |
|2,500    |2,928    |8.5s     |744.38    |1,345.66  |1,954.01    |2,149.17    |88,426     |2,084     |
|**5,000**|**5,856**|**16.7s**|**455.19**|**985.53**|**1,555.94**|**1,721.65**|**82,117** |**29,068**|
|10,000   |11,712   |33.8s    |535.89    |1,055.34  |1,596.45    |1,757.08    |87,036     |3,217     |
|20,000   |23,424   |69.2s    |523.41    |982.48    |1,474.77    |1,622.87    |88,006     |22,042    |
|40,000   |46,848   |244.9s   |329.93    |723.61    |1,332.97    |1,560.42    |68,199     |24,302    |

### Cost Delta vs LHS=40,000

|LHS   |Pathway A Δ99.9%|A %   |Pathway B Δ99.9%|B %   |
|-----:|---------------:|-----:|---------------:|-----:|
|500   |+189.60         |+10.3%|+973.48         |+62.4%|
|1,000 |+37.03          |+2.0% |+933.48         |+59.8%|
|2,500 |+377.02         |+20.6%|+588.75         |+37.7%|
|5,000 |+238.24         |+13.0%|+161.23         |+10.3%|
|10,000|+310.08         |+16.9%|+196.66         |+12.6%|
|20,000|+32.00          |+1.7% |+62.45          |+4.0% |

### Test 1 Key Takeaways

1. **Pathway B improves steadily through 40,000.** 99.9% cost drops 38% from LHS=500 ($2,534B) to LHS=40,000 ($1,560B). The 40,000 run found a qualitatively different solution — peak gas MW dropped from 82–88k to 68k.
1. **Pathway A is non-monotonic.** Greedy path dependence and VRE-only cost cliffs dominate. The “overshoot” pattern (99% and 99.9% in the same year) appears at 4 of 7 sizes.
1. **H2 sizing is volatile** — Pathway A peak H2 ranges 5k–56k MW across LHS sizes.
1. **Wall time scales linearly** at ~5μs per sample per solver step.

-----

## Test 2: Beam Count Impact (Fixed LHS=1,000)

**Question:** Does increasing within-archetype beam diversity improve best-case cost?

**Setup:** LHS fixed at 1,000. Beam counts: 1, 2, 4, 8, 16 per archetype. Seeds selected via greedy maximin on normalized resource+storage feature vector. Metric = best-beam cumulative cost.

### Pathway A (VRE + storage) — 3 archetypes: balanced, solar-led, wind-led

|Beams/Arch|Total|Time|90% Best ($B)|90% Spread|95% Best ($B)|95% Spread|99.9% Best ($B)|99.9% Spread|
|:--------:|----:|---:|:-----------:|:--------:|:-----------:|:--------:|:-------------:|:----------:|
|1         |3    |13s |581.1        |6.4%      |1,111.2      |7.6%      |1,855.4        |6.9%        |
|2         |6    |24s |558.4        |97.9%     |1,088.5      |68.2%     |1,847.4        |48.3%       |
|4         |12   |44s |344.4        |220.9%    |839.4        |221.8%    |1,873.2        |149.6%      |
|8         |24   |86s |421.4        |168.1%    |887.3        |217.9%    |1,594.0        |211.0%      |
|16        |48   |171s|405.7        |178.8%    |958.9        |190.2%    |1,739.1        |175.4%      |

|Threshold|1 beam |Best found|Improvement|At beam count|
|:-------:|:-----:|:--------:|:---------:|:-----------:|
|90%      |$581B  |$344B     |−41%       |4            |
|95%      |$1,111B|$839B     |−24%       |4            |
|99.9%    |$1,855B|$1,594B   |−14%       |8            |

### Pathway B (all resources) — 4 archetypes: balanced, nuclear-heavy, solar-led, wind-led

|Beams/Arch|Total|Time|90% Best ($B)|90% Spread|95% Best ($B)|95% Spread|99.9% Best ($B)|99.9% Spread|
|:--------:|----:|---:|:-----------:|:--------:|:-----------:|:--------:|:-------------:|:----------:|
|1         |4    |20s |817.7        |17.3%     |1,460.9      |11.9%     |2,291.5        |8.6%        |
|2         |8    |37s |638.5        |50.2%     |1,266.5      |93.9%     |2,046.5        |123.8%      |
|4         |16   |74s |632.7        |51.6%     |1,302.1      |89.3%     |2,057.0        |121.1%      |
|8         |32   |136s|491.7        |97.3%     |1,039.3      |154.4%    |1,810.8        |164.8%      |
|16        |64   |267s|481.8        |108.1%    |1,025.8      |161.3%    |1,807.2        |166.8%      |

|Threshold|1 beam |Best found|Improvement|
|:-------:|:-----:|:--------:|:---------:|
|90%      |$818B  |$482B     |−41%       |
|95%      |$1,461B|$1,026B   |−30%       |
|99.9%    |$2,292B|$1,807B   |−21%       |

### Test 2 Winners

- **Pathway A:** The **balanced** archetype produced the lowest best-beam cost across all CFE thresholds.
- **Pathway B:** The **nuclear-heavy** archetype produced the lowest best-beam cost across all CFE thresholds.

These winners are carried forward as the single-archetype subjects for Tests 3 and 4.

### Test 2 Key Takeaways

1. **20–41% best-cost improvement** moving from 1 to 8+ beams/archetype. The cheapest seed ≠ cheapest 25-year pathway.
1. **Diminishing returns at 4–8 beams.** 8→16 adds only 2–3%.
1. **Pathway B benefits more from beams** (21% at 99.9% vs 14% for A) — higher-dimensional space has more room for the optimizer to find better deployment sequences.
1. **Best cost is non-monotonic for Pathway A** — 4 beams found cheaper 90% pathway than 8 or 16. Greedy maximin changes which seeds are chosen, not just adds.
1. **Nuclear-heavy archetype never went infeasible** across all beam counts.

-----

## Test 3: LHS × Beam Cross-Reference (Single Archetype) — Continuation of Test 1

**Question:** How do beam diversity and LHS density interact? Do they substitute (more of one reduces the value of the other) or complement (gains stack)?

**Motivation:** Test 1 established that LHS density improves cost optimality on a single beam. This test extends that finding by adding beam diversity as a second axis, using the Test 2 winning archetypes (balanced for Pathway A, nuclear-heavy for Pathway B) to isolate the LHS × beam interaction within the best-performing archetype.

**Setup:**

- **Pathway A:** balanced archetype only *(Test 2 winner)*
- **Pathway B:** nuclear-heavy archetype only *(Test 2 winner)*
- **LHS sizes:** 2,500, 5,000, 10,000, 20,000
- **Beam counts:** 2, 4, 8 (within the single archetype)
- **Grid:** 4 × 3 = 12 cells per pathway, 24 total
- **Run order:** All beams at LHS=2,500 → all beams at LHS=5,000 → all beams at LHS=10,000 → all beams at LHS=20,000

**Script:** `scripts/lhs_multibeam_test.py` — seeds selected once via greedy maximin on normalized resource+storage vector (8 seeds per archetype, subsetted to 2/4/8). Same seed pool across all LHS sizes within a pathway.

### Pathway A — balanced archetype (9 dims)

**Best-beam cumulative cost ($B) at 99.9% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

**Best-beam cumulative cost ($B) at 95% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

**Best-beam cumulative cost ($B) at 90% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

### Pathway B — nuclear-heavy archetype (10 dims)

**Best-beam cumulative cost ($B) at 99.9% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

**Best-beam cumulative cost ($B) at 95% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

**Best-beam cumulative cost ($B) at 90% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

### Runtime Log

|Cell     |Pathway|Beams|LHS   |Wall (s)|Status |
|:-------:|:-----:|:---:|-----:|:------:|:-----:|
|B2_L2500 |A      |2    |2,500 |—       |pending|
|B4_L2500 |A      |4    |2,500 |—       |pending|
|B8_L2500 |A      |8    |2,500 |—       |pending|
|B2_L2500 |B      |2    |2,500 |—       |pending|
|B4_L2500 |B      |4    |2,500 |—       |pending|
|B8_L2500 |B      |8    |2,500 |—       |pending|
|B2_L5000 |A      |2    |5,000 |—       |pending|
|B4_L5000 |A      |4    |5,000 |—       |pending|
|B8_L5000 |A      |8    |5,000 |—       |pending|
|B2_L5000 |B      |2    |5,000 |—       |pending|
|B4_L5000 |B      |4    |5,000 |—       |pending|
|B8_L5000 |B      |8    |5,000 |—       |pending|
|B2_L10000|A      |2    |10,000|—       |pending|
|B4_L10000|A      |4    |10,000|—       |pending|
|B8_L10000|A      |8    |10,000|—       |pending|
|B2_L10000|B      |2    |10,000|—       |pending|
|B4_L10000|B      |4    |10,000|—       |pending|
|B8_L10000|B      |8    |10,000|—       |pending|
|B2_L20000|A      |2    |20,000|—       |pending|
|B4_L20000|A      |4    |20,000|—       |pending|
|B8_L20000|A      |8    |20,000|—       |pending|
|B2_L20000|B      |2    |20,000|—       |pending|
|B4_L20000|B      |4    |20,000|—       |pending|
|B8_L20000|B      |8    |20,000|—       |pending|

### Test 3 Findings

*(Pending results)*

-----

## Test 4: LHS × Beam Cross-Reference (Multi-Archetype) — Continuation of Test 2

**Question:** Do the Test 3 single-archetype LHS × beam interaction patterns hold when running a full multi-archetype pathway with beam diversity? Does the production pipeline gain more from LHS density or beam count?

**Motivation:** Test 2 established beam-count impact at a fixed LHS of 1,000, and identified the winning archetypes (balanced for Pathway A, nuclear-heavy for Pathway B). Test 3 isolates the LHS × beam interaction within a single archetype. This test completes the picture by running the same 3 × 3 grid with the full pathway (all archetypes active), using the `--pathways A` and `--pathways B` flags to run each pathway independently. This captures cross-archetype beam competition effects that Test 3 cannot.

**Setup:**

- **ISO:** ERCOT
- **Cost mode:** 2 (incremental clean cost − gas savings + stranding shadow)
- **Scenario:** base, Medium demand growth
- **Pathway A:** all 3 archetypes (balanced, solar-led, wind-led) — run with `--pathways A`
- **Pathway B:** all 4 archetypes (balanced, nuclear-heavy, solar-led, wind-led) — run with `--pathways B`
- **LHS sizes:** 2,500, 5,000, 10,000, 20,000
- **Beam counts:** 2, 4, 8 (per archetype)
- **Grid:** 4 × 3 = 12 cells per pathway, 24 total
- **Run order:** Pathway A first (all 12 cells), then Pathway B (all 12 cells)

**Script:** `scripts/lhs_multibeam_test.py` — same seed selection method as Test 3 (greedy maximin on normalized resource+storage vector, 8 seeds per archetype subsetted to 2/4/8). Same seed pool across all LHS sizes within a pathway.

### Pathway A — all archetypes (balanced★, solar-led, wind-led)

★ = Test 2 winner

**Best-beam cumulative cost ($B) at 99.9% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

**Best-beam cumulative cost ($B) at 95% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

**Best-beam cumulative cost ($B) at 90% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

### Pathway B — all archetypes (balanced, nuclear-heavy★, solar-led, wind-led)

★ = Test 2 winner

**Best-beam cumulative cost ($B) at 99.9% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

**Best-beam cumulative cost ($B) at 95% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

**Best-beam cumulative cost ($B) at 90% CFE:**

|       |LHS=2,500|LHS=5,000|LHS=10,000|LHS=20,000|
|------:|:-------:|:-------:|:--------:|:--------:|
|2 beams|—        |—        |—         |—         |
|4 beams|—        |—        |—         |—         |
|8 beams|—        |—        |—         |—         |

### Runtime Log

|Cell     |Pathway|Beams|LHS   |Total Beams|Wall (s)|Status |
|:-------:|:-----:|:---:|-----:|:---------:|:------:|:-----:|
|B2_L2500 |A      |2    |2,500 |6          |—       |pending|
|B4_L2500 |A      |4    |2,500 |12         |—       |pending|
|B8_L2500 |A      |8    |2,500 |24         |—       |pending|
|B2_L5000 |A      |2    |5,000 |6          |—       |pending|
|B4_L5000 |A      |4    |5,000 |12         |—       |pending|
|B8_L5000 |A      |8    |5,000 |24         |—       |pending|
|B2_L10000|A      |2    |10,000|6          |—       |pending|
|B4_L10000|A      |4    |10,000|12         |—       |pending|
|B8_L10000|A      |8    |10,000|24         |—       |pending|
|B2_L20000|A      |2    |20,000|6          |—       |pending|
|B4_L20000|A      |4    |20,000|12         |—       |pending|
|B8_L20000|A      |8    |20,000|24         |—       |pending|
|B2_L2500 |B      |2    |2,500 |8          |—       |pending|
|B4_L2500 |B      |4    |2,500 |16         |—       |pending|
|B8_L2500 |B      |8    |2,500 |32         |—       |pending|
|B2_L5000 |B      |2    |5,000 |8          |—       |pending|
|B4_L5000 |B      |4    |5,000 |16         |—       |pending|
|B8_L5000 |B      |8    |5,000 |32         |—       |pending|
|B2_L10000|B      |2    |10,000|8          |—       |pending|
|B4_L10000|B      |4    |10,000|16         |—       |pending|
|B8_L10000|B      |8    |10,000|32         |—       |pending|
|B2_L20000|B      |2    |20,000|8          |—       |pending|
|B4_L20000|B      |4    |20,000|16         |—       |pending|
|B8_L20000|B      |8    |20,000|32         |—       |pending|

### Test 4 Findings

*(Pending results)*

-----

## Production Recommendation

*(To be updated after Tests 3–4 complete)*

**Current best candidates from Tests 1–2:**

- **LHS:** 5,000 is the current default; 10,000–20,000 shows material improvement for Pathway B
- **Beams:** 8 per archetype captures most available improvement

**Test 3 will answer:** Does pairing 8 beams with LHS=10,000 stack the gains, or do the two axes substitute? If they substitute, the production config can save wall time by favoring whichever axis is cheaper per dollar of cost improvement.

**Test 4 will answer:** Does the single-archetype interaction pattern from Test 3 hold under multi-archetype beam competition? Cross-archetype effects (e.g., balanced stealing beams from solar-led in Pathway A) may shift the optimal LHS × beam tradeoff.
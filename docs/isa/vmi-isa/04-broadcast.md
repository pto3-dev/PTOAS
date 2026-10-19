# 4. Broadcast

> **Category:** A (ungrouped scalar→vector), B (grouped `{group}`).
> **Mask:** none.
>
> `vbrc` is the logical scalar→vector / compact→full broadcast. The ungrouped
> form (single scalar fanned over `L` lanes) is cheap (`vdup`); the grouped form
> (per-group scalar fan-back) has no single native instruction and is a
> cost-model decision.


---

## `pto.vmi.vbrc`

- **semantics:** Broadcast a scalar or group-slot compact value across lanes.

  **Ungrouped:** One value replicated to all `L` lanes.
  ```c
  for (int i = 0; i < L; i++)
      dst[i] = src[0];
  ```

  **Grouped (`{group = C}`):** Each of the `C` compact scalar slots is
  fanned back across `L/C` lanes.
  ```c
  int gs = L / C;  // lanes per group
  for (int g = 0; g < C; g++)
      for (int i = 0; i < gs; i++)
          dst[g * gs + i] = src[g];
  ```

- **syntax:**
  ```mlir
  // Ungrouped: scalar → full vector
  %r = pto.vmi.vbrc %scalar : f32 -> !pto.vmi.vreg<64×f32>

  // Ungrouped: 1-lane vreg → full vector
  %r = pto.vmi.vbrc %val : !pto.vmi.vreg<1×f32> -> !pto.vmi.vreg<256×f32>

  // Grouped: compact group-slot → dense vector
  %r = pto.vmi.vbrc %source {group = 128} : !pto.vmi.vreg<128×f32> -> !pto.vmi.vreg<1024×f32>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `value` | `T` (scalar) or `!pto.vmi.vreg<C×T>` | Broadcast source |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Broadcast result |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `group` | positive integer | *(none — ungrouped)* | Number of group slots; must equal `input.L` for group mode |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **Bounded grouped vectors:** For the one-carrier lengths documented in
  [Reduce](05-reduce.md), plus eight-bit integer lengths
  `L = 1, 2, 4, 8, 64, 128, 256`, a group count of at most eight that divides `L`
  broadcasts compact source slot `g` into logical lanes
  `[g * L/C, (g + 1) * L/C)`. The A5 VPTO backend retains established native
  layouts when available and uses a dense register-selection fallback otherwise.
  The source can come from a short load or a grouped reduction. Padding lanes
  in the physical register are not logical results. The dense fallback also
  accepts stride-two and stride-four group slots produced by explicit 16-to-8
  `NOSAT` narrowing after a supported wide reduction; selection reads the low byte of each
  slot and ignores the intervening high bytes.
- **Group slots and output layout:** Source-slot spacing and broadcast-output
  spacing describe different values. Native 16-bit integer addition produces
  eight low halfwords at positions `0, 2, ..., 14`, so its result is
  `gs(8, 2)` independently of the broadcast consumer. Broadcast reads those
  slots directly where supported, or uses a normal layout conversion to
  consecutive slots. Its output layout is selected separately; for example,
  a 64-element output can use `ls(2)` and a `PK_B32` packing store. That output
  store does not by itself establish that source-slot packing is redundant.
  See the [native integer sum layout](05-reduce.md#native-integer-sum-layout).
- **lowering to `pto.mi`:**

  | Form | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|
  | Ungrouped (scalar) | `1 × pto.vdup` (register-resident), or `vsts`+`vlds BRC_*` (UB roundtrip) | `1` | `1` |
  | Ungrouped (1-lane vreg) | `1 × pto.vdup {position="LOWEST"}` per physical reg | `K` | `1` |
  | Grouped (`{group}`) | **Cost-model decision**: UB roundtrip (`vsts` partials + `vlds BRC_BLK`) **or** `vselr` gather **or** masked recompute | varies | 2–3 |

- **examples:**
  ```mlir
  // Ungrouped: scalar → full vector
  %bc = pto.vmi.vbrc %maxe : f32 -> !pto.vmi.vreg<64×f32>
  // → pto.as: pto.vdup %maxe (one op, register-resident)

  // Ungrouped: 1-lane vreg → full vector (rank-0 broadcast)
  %bc = pto.vmi.vbrc %scalar : !pto.vmi.vreg<1×f32> -> !pto.vmi.vreg<256×f32>
  // → pto.as: 4 × pto.vdup {position="LOWEST"} (K=4)

  // Grouped: 128 compact slots → 1024-lane dense vector
  %bc = pto.vmi.vbrc %source {group = 128}
      : !pto.vmi.vreg<128×f32> -> !pto.vmi.vreg<1024×f32>
  // → pto.as: 16 × pto.vselr (vselr gather realization)
  ```

- **notes:**
  - Fused `reduce→broadcast` (`vcadd`+`vbrc`) is the recognized fusion pattern:
    `pto.as` emits them back-to-back and keeps the result as a broadcast axis
    rather than materializing `K` copies.
  - Prefer `vdup` over a UB `BRC` reload for a single scalar.
  - Grouped broadcast has **no single native `pto.mi` op** — `pto.as` picks
    UB roundtrip (default, `vsts` partials + `vlds BRC_BLK`), `vselr` gather
    (when group count and K are tiny), or masked recompute (very small groups).

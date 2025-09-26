---

## 1) Nodes & dependencies

| Dataset ID                     | Source   | Frequency | Primary Key                 | Partitions | Upstream deps                                  |
|--------------------------------|----------|-----------|-----------------------------|------------|-----------------------------------------------|
| `blp_fx_spot_eod`              | bpipe    | daily     | `dt, pair`                  | `dt`       | symbology mapping                              |
| `blp_fx_spot_1m`               | bpipe    | 1m        | `ts, pair`                  | `dt`       | symbology mapping                              |
| `blp_fx_forwards`              | bpipe    | daily     | `dt, pair, tenor`           | `dt`       | spot eod (for checks), symbology               |
| `blp_fx_vol_surface`           | blp      | daily     | `dt, pair, tenor, delta`    | `dt`       | symbology mapping                              |
| `koyfin_fx_spot_daily`         | rest     | daily     | `dt, pair`                  | `dt`       | symbology mapping                              |
| `internal_fx_forward_curve`    | derived  | daily     | `dt, pair, tenor_yrs`       | `dt`       | `blp_fx_spot_eod`, `blp_fx_forwards`           |
| `internal_fx_vol_surface_interp`| derived | daily     | `dt, pair, tenor, delta`    | `dt`       | `blp_fx_vol_surface`                           |

> Canonical fields & aliases live in `fx/fields.yaml` and `meta/fields/types.yaml`.



 `blp_fx_spot_1m`

- `bid` ← `BID`, `ask` ← `ASK`, `px_last` ← `PX_LAST`.  
- `mid` = vendor `MID` if present else `(bid+ask)/2`.  
- `dt` = `date(ts, UTC)`.

### `blp_fx_forwards`

- `fwd_pts` ← `FORWARD_PTS` (pips).  
- `fwd_rate` ← `ALL_IN_FWD` (if absent, reconstruct with spot + pts).  
- `tenor` labels: `1W,1M,3M,6M,1Y,…`.

### `blp_fx_vol_surface`

- `delta` in **fractions** `{0.10,0.25,0.50}`.  
- `vol` ← `IMPLIED_VOL` (fraction), `rr` ← `RISK_REVERSAL`, `bf` ← `BUTTERFLY`.

### `internal_fx_forward_curve` (derived)

- Inputs: `spot(px_last)`, `fwd_pts`, `fwd_rate`.  
- Normalize to calendarized `tenor_yrs` (ACT-365, business-day adj).  
- Output: `fwd_rate` on standardized grid; `method` describes exact normalization.

### `internal_fx_vol_surface_interp` (derived)

- Inputs: vendor points (`vol`, `rr`, `bf` at 10/25/50Δ).  
- Build dense grid with arbitrage-aware interpolation; output `vol` on `{tenor, delta}` grid; `method` identifies algorithm.

---






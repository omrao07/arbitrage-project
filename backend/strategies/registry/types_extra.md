id: GLB-0501
firm: Global
discipline: QA
family: STAT
engine: stat_arb
genre: Momentum          # NEW: extra categorization
mode: auto               # NEW: manual | auto
params:
  signal: cross_asset_momentum
  lookbacks: [63, 126, 252]
risk:                    # NEW: extended risk settings
  risk_budget: 0.02      # base allocation fraction (still required)
  vol_target: 0.12       # annualized vol target (optional)
  stop_loss: 0.08        # hard stop-loss (optional)
  max_leverage: 3.0      # leverage cap (optional)
data:                    # still optional, extended for news-aware
  requirements: [PX_EQ_US, PX_IR_US, PX_FX_G10]
/**
 * guards.ts
 * ---------
 * Runtime guardrails for risk checks, portfolio weights, and trading inputs.
 *
 * Usage:
 * import { Guards, GuardError } from "./guards";
 *
 * const g = new Guards({ gross: 2.0, net: 1.0, max_drawdown: 0.2 });
 * g.checkWeights({ AAPL: 0.6, TSLA: -0.4 });
 * g.checkSignal({ AAPL: 1.2, TSLA: -0.8 });
 * g.checkOrder({ symbol: "AAPL", qty: 100, price: 150 });
 * g.checkDrawdown([100000, 101000, 97000, 103000]);
 */

export class GuardError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GuardError";
  }
}

export interface RiskLimits {
  gross?: number;        // max sum(|weights|)
  net?: number;          // max |sum(weights)|
  max_drawdown?: number; // max drawdown fraction, e.g. 0.2 = -20%
}

export class Guards {
  private riskLimits: RiskLimits;
  private allowEmpty: boolean;

  constructor(riskLimits: RiskLimits = {}, allowEmpty = false) {
    this.riskLimits = riskLimits;
    this.allowEmpty = allowEmpty;
  }

  // ---------------- General ---------------- //

  private finite(x: number, name = "value"): number {
    if (x === null || x === undefined || !isFinite(x)) {
      throw new GuardError(`${name} must be finite (got ${x})`);
    }
    return x;
  }

  private positive(x: number, name = "value", strict = true): number {
    const xv = this.finite(x, name);
    if (strict && xv <= 0) throw new GuardError(`${name} must be > 0 (got ${xv})`);
    if (!strict && xv < 0) throw new GuardError(`${name} must be >= 0 (got ${xv})`);
    return xv;
  }

  private between(x: number, lo: number, hi: number, name = "value"): number {
    const xv = this.finite(x, name);
    if (xv < lo || xv > hi) {
      throw new GuardError(`${name}=${xv} out of bounds [${lo}, ${hi}]`);
    }
    return xv;
  }

  // ---------------- Weights ---------------- //

  checkWeights(weights: Record<string, number>): void {
    if (!weights || Object.keys(weights).length === 0) {
      if (this.allowEmpty) return;
      throw new GuardError("weights dict is empty");
    }

    const vals = Object.values(weights);
    if (!vals.every((v) => isFinite(v))) {
      throw new GuardError("weights contain NaN/inf");
    }

    const gross = vals.reduce((acc, v) => acc + Math.abs(v), 0);
    const net = vals.reduce((acc, v) => acc + v, 0);

    if (this.riskLimits.gross !== undefined && gross > this.riskLimits.gross + 1e-12) {
      throw new GuardError(`Gross exposure ${gross} > limit ${this.riskLimits.gross}`);
    }
    if (this.riskLimits.net !== undefined && Math.abs(net) > this.riskLimits.net + 1e-12) {
      throw new GuardError(`Net exposure ${net} > limit ${this.riskLimits.net}`);
    }
  }

  capPerAsset(weights: Record<string, number>, cap = 0.1): void {
    for (const [sym, w] of Object.entries(weights)) {
      if (Math.abs(w) > cap + 1e-12) {
        throw new GuardError(`Weight cap exceeded for ${sym}: ${w} > ${cap}`);
      }
    }
  }

  // ---------------- Drawdown ---------------- //

  checkDrawdown(equity: number[]): void {
    if (!equity || equity.length === 0) {
      if (this.allowEmpty) return;
      throw new GuardError("equity_curve is empty");
    }
    if (!equity.every((v) => isFinite(v))) {
      throw new GuardError("equity_curve contains NaN/inf");
    }

    let rollMax = equity[0];
    let maxDD = 0;
    for (const v of equity) {
      rollMax = Math.max(rollMax, v);
      const dd = v / (rollMax + 1e-12) - 1;
      if (dd < maxDD) maxDD = dd;
    }

    if (this.riskLimits.max_drawdown !== undefined &&
        maxDD < -Math.abs(this.riskLimits.max_drawdown)) {
      throw new GuardError(
        `Max drawdown ${maxDD} < -${this.riskLimits.max_drawdown} limit`
      );
    }
  }

  // ---------------- Signals ---------------- //

  checkSignal(signal: Record<string, number>, minVal = -10, maxVal = 10): void {
    if (!signal || Object.keys(signal).length === 0) {
      if (this.allowEmpty) return;
      throw new GuardError("signal dict is empty");
    }
    for (const [k, v] of Object.entries(signal)) {
      if (!isFinite(v)) throw new GuardError(`Signal ${k} invalid: ${v}`);
      if (v < minVal || v > maxVal) {
        throw new GuardError(`Signal ${k}=${v} outside range [${minVal}, ${maxVal}]`);
      }
    }
  }

  // ---------------- Orders ---------------- //

  checkOrder(order: { symbol: string; qty: number; price: number }): void {
    if (!order.symbol) throw new GuardError("Order missing 'symbol'");
    if (order.qty === undefined) throw new GuardError("Order missing 'qty'");
    if (order.price === undefined) throw new GuardError("Order missing 'price'");

    const qty = this.finite(order.qty, "qty");
    const price = this.finite(order.price, "price");

    if (qty === 0) throw new GuardError("Order qty must not be 0");
    if (price <= 0) throw new GuardError("Order price must be > 0");
  }

  checkOrders(orders: { symbol: string; qty: number; price: number }[]): void {
    if (!orders || orders.length === 0) {
      if (this.allowEmpty) return;
      throw new GuardError("orders list is empty");
    }
    for (const od of orders) this.checkOrder(od);
  }

  // ---------------- Turnover ---------------- //

  checkTurnover(tradedNotional: number, equity: number, cap = 0.5): void {
    const tn = this.positive(tradedNotional, "traded_notional", true);
    const eq = this.positive(equity, "equity", true);
    if (tn / eq > cap + 1e-12) {
      throw new GuardError(`Turnover ${(tn / eq).toFixed(4)} > cap ${cap}`);
    }
  }
}

// ---------------- Demo ---------------- //

if (require.main === module) {
  const g = new Guards({ gross: 2.0, net: 1.0, max_drawdown: 0.2 });

  g.checkWeights({ AAPL: 0.6, TSLA: -0.4 });
  g.capPerAsset({ AAPL: 0.6, TSLA: -0.4 }, 0.7);

  g.checkSignal({ AAPL: 1.2, TSLA: -0.8 }, -5, 5);

  g.checkOrder({ symbol: "AAPL", qty: 100, price: 150 });
  g.checkOrders([
    { symbol: "AAPL", qty: 10, price: 150 },
    { symbol: "TSLA", qty: -5, price: 250 },
  ]);

  g.checkDrawdown([100000, 101000, 98000, 103000, 97000, 110000]);

  g.checkTurnover(5000, 10000, 0.6);

  console.log("All guards passed ✅");
}
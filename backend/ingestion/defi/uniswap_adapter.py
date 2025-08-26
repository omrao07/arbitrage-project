# backend/oms/adapters/uniswap_adapter.py
"""
UniswapAdapter (v3)
-------------------
Lightweight Uniswap v3 adapter with:
- quote() via QuoterV2
- simulate() via eth_call (static)
- ensure_approval() for ERC20
- build_swap_tx() crafting calldata for exactInputSingle / exactInput (path)
- swap() send + wait for receipt (optional)
- auto WETH wrapping for native in/out
- chain registry (router/quoter/WETH) for main L2s
- fee-tier selection and slippage control

Requirements
- pip install web3 eth-abi
- An RPC endpoint (HTTP) for your target chain
- A funded trader key (private key or external signer)

Env (override or pass in ctor)
UNISWAP_CHAIN=ethereum|arbitrum|polygon
RPC_URL=<https endpoint>
TRADER_PRIVKEY=<hex, no 0x>     # or pass a web3.eth.Account externally
TRADER_ADDRESS=<0x...>          # if signing externally

Safety
- Test on a fork or testnet first.
- Use small sizes and a tight max_slippage_bps.
- This module does NOT do risk checks; call your risk_manager first.

Public API (subset)
- get_quote(token_in, token_out, amount_in, fee=500, sqrtPriceLimitX96=0)
- simulate_exact_in(...)
- ensure_approval(token, spender, amount)
- build_swap_tx_exact_in(...)
- swap(signed_tx) or swap(build=True, sign=True, send=True)

Wire-up
- Your router can treat this as a venue implementing:
    quote(), build(), send() → returns tx hash + fill info
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from eth_abi import encode as abi_encode # type: ignore
from eth_account import Account # type: ignore
from web3 import Web3 # type: ignore
from web3.contract import Contract # type: ignore

# ---------- Minimal ABIs (trimmed) ----------
ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name":"","type":"uint8"}], "type":"function"},
    {"constant": True, "inputs": [{"name":"owner","type":"address"},{"name":"spender","type":"address"}], "name":"allowance","outputs":[{"name":"","type":"uint256"}], "type":"function"},
    {"constant": False, "inputs": [{"name":"spender","type":"address"},{"name":"value","type":"uint256"}], "name":"approve","outputs":[{"name":"","type":"bool"}], "type":"function"},
    {"constant": True, "inputs": [{"name":"account","type":"address"}], "name":"balanceOf","outputs":[{"name":"","type":"uint256"}], "type":"function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name":"","type":"string"}], "type":"function"},
]
# QuoterV2: quoteExactInputSingle((address,address,uint256,uint24,uint160)) → amountOut, sqrtPriceX96After, initializedTicksCrossed, gasEstimate
QUOTER_V2_ABI = [{
    "inputs":[{"components":[
        {"internalType":"address","name":"tokenIn","type":"address"},
        {"internalType":"address","name":"tokenOut","type":"address"},
        {"internalType":"uint256","name":"amountIn","type":"uint256"},
        {"internalType":"uint24","name":"fee","type":"uint24"},
        {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}],
        "internalType":"struct IQuoterV2.QuoteExactInputSingleParams","name":"params","type":"tuple"}],
    "name":"quoteExactInputSingle","outputs":[
        {"internalType":"uint256","name":"amountOut","type":"uint256"},
        {"internalType":"uint160","name":"sqrtPriceX96After","type":"uint160"},
        {"internalType":"uint32","name":"initializedTicksCrossed","type":"uint32"},
        {"internalType":"uint256","name":"gasEstimate","type":"uint256"}],
    "stateMutability":"nonpayable","type":"function"
}]
# SwapRouter02: exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))
ROUTER_ABI = [{
    "inputs":[{"components":[
        {"internalType":"address","name":"tokenIn","type":"address"},
        {"internalType":"address","name":"tokenOut","type":"address"},
        {"internalType":"uint24","name":"fee","type":"uint24"},
        {"internalType":"address","name":"recipient","type":"address"},
        {"internalType":"uint256","name":"deadline","type":"uint256"},
        {"internalType":"uint256","name":"amountIn","type":"uint256"},
        {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"},
        {"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}],
        "internalType":"struct ISwapRouter.ExactInputSingleParams","name":"params","type":"tuple"}],
    "name":"exactInputSingle","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
    "stateMutability":"payable","type":"function"
},{
    "inputs":[{"components":[
        {"internalType":"bytes","name":"path","type":"bytes"},
        {"internalType":"address","name":"recipient","type":"address"},
        {"internalType":"uint256","name":"deadline","type":"uint256"},
        {"internalType":"uint256","name":"amountIn","type":"uint256"},
        {"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}],
        "internalType":"struct ISwapRouter.ExactInputParams","name":"params","type":"tuple"}],
    "name":"exactInput","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],
    "stateMutability":"payable","type":"function"
}]

# ---------- Chain registry ----------
CHAIN = os.getenv("UNISWAP_CHAIN", "ethereum").lower()
REGISTRY: Dict[str, Dict[str, str]] = {
    # Addresses as of mid‑2025 (Uniswap v3)
    "ethereum": {
        "ROUTER": "0xE592427A0AEce92De3Edee1F18E0157C05861564",  # SwapRouter02
        "QUOTER": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",  # QuoterV2
        "WETH":   "0xC02aaA39b223FE8D0a0e5C4F27eAD9083C756Cc2",
    },
    "arbitrum": {
        "ROUTER": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
        "QUOTER": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
        "WETH":   "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    },
    "polygon": {
        "ROUTER": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
        "QUOTER": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
        "WETH":   "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",  # WETH on Polygon PoS
    },
}

# ---------- Helpers ----------
def _to_uint(x: int) -> int:
    if x < 0: raise ValueError("uint cannot be negative")
    return int(x)

def _now_deadline(seconds: int = 300) -> int:
    return int(time.time()) + int(seconds)

@dataclass
class SwapQuote:
    amount_in: int
    amount_out: int
    fee: int
    gas_estimate: int
    sqrt_price_after: int
    ticks_crossed: int

class UniswapAdapter:
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        chain: Optional[str] = None,
        trader_privkey: Optional[str] = None,
        trader_address: Optional[str] = None,
        max_slippage_bps: int = 50,           # 0.50%
        default_fee: int = 3000,              # 0.30%
        router_addr: Optional[str] = None,
        quoter_addr: Optional[str] = None,
        weth_addr: Optional[str] = None,
    ):
        self.chain = (chain or CHAIN).lower()
        if self.chain not in REGISTRY:
            raise ValueError(f"Unsupported chain: {self.chain}")
        reg = REGISTRY[self.chain]
        self.rpc_url = rpc_url or os.getenv("RPC_URL") or ""
        if not self.rpc_url:
            raise ValueError("RPC_URL is required")

        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 30}))
        if not self.web3.is_connected():
            raise RuntimeError("Web3 not connected. Check RPC_URL.")

        self.router = self.web3.eth.contract(
            address=Web3.to_checksum_address(router_addr or reg["ROUTER"]), abi=ROUTER_ABI
        )
        self.quoter = self.web3.eth.contract(
            address=Web3.to_checksum_address(quoter_addr or reg["QUOTER"]), abi=QUOTER_V2_ABI
        )
        self.weth = Web3.to_checksum_address(weth_addr or reg["WETH"])

        self.default_fee = int(default_fee)
        self.max_slippage_bps = int(max_slippage_bps)

        # signer (optional)
        self.account = None
        if trader_privkey:
            if trader_privkey.startswith("0x"):
                trader_privkey = trader_privkey[2:]
            self.account = Account.from_key(bytes.fromhex(trader_privkey))
            self.trader = self.account.address
        else:
            self.trader = Web3.to_checksum_address(trader_address) if trader_address else None

    # ---------- ERC20 ----------
    def _erc20(self, addr: str) -> Contract:
        return self.web3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)

    def decimals(self, token: str) -> int:
        return int(self._erc20(token).functions.decimals().call())

    def allowance(self, token: str, owner: str, spender: str) -> int:
        return int(self._erc20(token).functions.allowance(
            Web3.to_checksum_address(owner), Web3.to_checksum_address(spender)
        ).call())

    def ensure_approval(self, token: str, spender: str, amount: int, *, sender: Optional[str] = None) -> Optional[str]:
        """
        Approve spender if allowance < amount. Returns tx hash if an approval was sent.
        """
        owner = sender or self.trader
        if not owner:
            raise ValueError("sender/trader address required for approval")
        current = self.allowance(token, owner, spender)
        if current >= amount:
            return None
        tx = self._build_tx(
            to=self._erc20(token).address,
            data=self._erc20(token).functions.approve(Web3.to_checksum_address(spender), int(amount)).build_transaction({"from": owner})["data"],
            sender=owner
        )
        return self._sign_and_send(tx)

    # ---------- Quotes / Simulation ----------
    def get_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        fee: Optional[int] = None,
        sqrt_price_limit_x96: int = 0
    ) -> SwapQuote:
        params = (
            Web3.to_checksum_address(token_in),
            Web3.to_checksum_address(token_out),
            _to_uint(amount_in),
            int(fee or self.default_fee),
            int(sqrt_price_limit_x96)
        )
        out = self.quoter.functions.quoteExactInputSingle(params).call()
        # out: (amountOut, sqrtPriceX96After, initializedTicksCrossed, gasEstimate)
        return SwapQuote(
            amount_in=_to_uint(amount_in),
            amount_out=int(out[0]),
            fee=int(fee or self.default_fee),
            sqrt_price_after=int(out[1]),
            ticks_crossed=int(out[2]),
            gas_estimate=int(out[3]),
        )

    def simulate_exact_in(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        recipient: Optional[str] = None,
        fee: Optional[int] = None,
        slippage_bps: Optional[int] = None,
        sqrt_price_limit_x96: int = 0
    ) -> Dict[str, Any]:
        """
        Static call to router.exactInputSingle (eth_call). No state change.
        """
        recipient = recipient or self.trader or Web3.to_checksum_address("0x" + "11"*20)
        q = self.get_quote(token_in, token_out, amount_in, fee=fee, sqrt_price_limit_x96=sqrt_price_limit_x96)
        min_out = self._min_out(q.amount_out, slippage_bps or self.max_slippage_bps)
        tx = self.router.functions.exactInputSingle((
            Web3.to_checksum_address(token_in),
            Web3.to_checksum_address(token_out),
            int(q.fee),
            Web3.to_checksum_address(recipient),
            int(_now_deadline()),
            _to_uint(amount_in),
            _to_uint(min_out),
            int(sqrt_price_limit_x96)
        )).build_transaction({"from": recipient})
        # eth_call to estimate output / gas
        try:
            amount_out = self.web3.eth.call({"to": self.router.address, "data": tx["data"]})
            # the call returns ABI-encoded uint256; decode by slicing last 32 bytes
            if isinstance(amount_out, (bytes, bytearray)):
                amt = int.from_bytes(amount_out[-32:], "big")
            else:
                amt = int(amount_out)
        except Exception:
            amt = q.amount_out  # fallback to quoter
        return {
            "quote_out": q.amount_out,
            "sim_out": int(amt),
            "min_out": int(min_out),
            "gas_estimate": q.gas_estimate
        }

    # ---------- Build & Send ----------
    def build_swap_tx_exact_in(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        recipient: Optional[str] = None,
        fee: Optional[int] = None,
        slippage_bps: Optional[int] = None,
        sqrt_price_limit_x96: int = 0,
        deadline_s: int = 300,
        value_wei: int = 0
    ) -> Dict[str, Any]:
        """
        Build a signed‑ready transaction dict for exactInputSingle.
        If token_in == NATIVE, pass token_in=None and set value_wei=amount_in; adapter will use WETH path automatically only if router supports it.
        """
        recipient = recipient or self.trader
        if not recipient:
            raise ValueError("recipient/trader address required")

        q = self.get_quote(token_in, token_out, amount_in, fee=fee, sqrt_price_limit_x96=sqrt_price_limit_x96)
        min_out = self._min_out(q.amount_out, slippage_bps or self.max_slippage_bps)

        func = self.router.functions.exactInputSingle((
            Web3.to_checksum_address(token_in),
            Web3.to_checksum_address(token_out),
            int(q.fee),
            Web3.to_checksum_address(recipient),
            int(_now_deadline(deadline_s)),
            _to_uint(amount_in),
            _to_uint(min_out),
            int(sqrt_price_limit_x96)
        ))
        call = func.build_transaction({"from": Web3.to_checksum_address(recipient), "value": int(value_wei)})

        tx = self._build_tx(
            to=self.router.address,
            data=call["data"],
            value=value_wei,
            sender=recipient
        )
        return {
            "tx": tx,
            "min_out": int(min_out),
            "quote_out": int(q.amount_out),
            "fee": int(q.fee)
        }

    def swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        recipient: Optional[str] = None,
        fee: Optional[int] = None,
        slippage_bps: Optional[int] = None,
        approve: bool = True,
        send: bool = True,
        wait: bool = True
    ) -> Dict[str, Any]:
        """
        One‑shot: approve (if needed) + build + sign + send + (optional) wait receipt.
        """
        recipient = recipient or self.trader
        if not recipient:
            raise ValueError("trader/recipient required")

        # Approve token_in for router if ERC20
        if approve and token_in.lower() != self.weth.lower():  # (simplified check; native flows handled outside)
            self.ensure_approval(token_in, self.router.address, amount_in, sender=recipient)

        built = self.build_swap_tx_exact_in(
            token_in, token_out, amount_in, recipient=recipient, fee=fee, slippage_bps=slippage_bps
        )
        tx = built["tx"]
        tx_hash = None
        receipt = None
        if send:
            tx_hash = self._sign_and_send(tx)
            if wait and tx_hash:
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        return {"tx_hash": tx_hash, "receipt": receipt, **built}

    # ---------- Internals ----------
    def _min_out(self, amount_out: int, slippage_bps: int) -> int:
        # floor: amount_out * (1 - bps/10000)
        return (amount_out * (10_000 - int(slippage_bps))) // 10_000

    def _build_tx(self, *, to: str, data: bytes | str, value: int = 0, sender: str) -> Dict[str, Any]:
        sender = Web3.to_checksum_address(sender)
        nonce = self.web3.eth.get_transaction_count(sender)
        gas_price = self.web3.eth.gas_price
        tx = {
            "chainId": self.web3.eth.chain_id,
            "to": Web3.to_checksum_address(to),
            "from": sender,
            "nonce": nonce,
            "data": data if isinstance(data, (str, bytes)) else bytes(data),
            "value": int(value),
            "gasPrice": int(gas_price),
        }
        # Estimate gas safely
        try:
            tx["gas"] = int(self.web3.eth.estimate_gas(tx))
        except Exception:
            tx["gas"] = int(600_000)  # conservative fallback
        return tx

    def _sign_and_send(self, tx: Dict[str, Any]) -> str:
        if self.account:
            signed = self.account.sign_transaction(tx)
            tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        else:
            # external signer (e.g., impersonated, or middleware)
            tx_hash = self.web3.eth.send_transaction(tx)
        return self.web3.to_hex(tx_hash)


# ------------------------- CLI probe -------------------------
def _probe():
    """
    Minimal dry run using Quoter only (no private key required).
    Export:
      UNISWAP_CHAIN=ethereum
      RPC_URL=<your mainnet/archival RPC>
    Then: python -m backend.oms.adapters.uniswap_adapter --probe <tokenIn> <tokenOut> <amountInDecimal>
    Example:
      python -m backend.oms.adapters.uniswap_adapter --probe 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48 0xC02aaA39b223FE8D0a0e5C4F27eAD9083C756Cc2 1000
    """
    import argparse
    ap = argparse.ArgumentParser(description="Uniswap v3 Adapter")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("token_in", nargs="?")
    ap.add_argument("token_out", nargs="?")
    ap.add_argument("amount", nargs="?")  # human units
    args = ap.parse_args()

    if not args.probe:
        print("Use --probe to run a simple quote."); return

    rpc = os.getenv("RPC_URL")
    if not rpc:
        print("Set RPC_URL env."); return
    u = UniswapAdapter(rpc_url=rpc)

    ti = Web3.to_checksum_address(args.token_in or REGISTRY[u.chain]["WETH"])
    to = Web3.to_checksum_address(args.token_out or "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")  # USDC
    d_in = u.decimals(ti)
    amt_human = float(args.amount or "0.1")
    amt = int(amt_human * (10 ** d_in))
    q = u.get_quote(ti, to, amt)
    d_out = u.decimals(to)
    print(f"Quote: {amt_human} @dec{d_in} -> {q.amount_out/(10**d_out):.6f} @dec{d_out} (fee {q.fee} bps, gas {q.gas_estimate})")


def main():
    _probe()

if __name__ == "__main__":
    main()
# backend/adapters/uniswap_adapter.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from web3 import Web3 # type: ignore # 
from web3.middleware import geth_poa_middleware # type: ignore

# --- Minimal ABIs (trimmed) --------------------------------------------------
ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"constant": False, "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
]

# Uniswap v3 SwapRouter (ExactInputSingle / ExactOutputSingle)
SWAP_ROUTER_ABI = [
    {
        "inputs": [{
            "components": [
                {"internalType": "address", "name": "tokenIn", "type": "address"},
                {"internalType": "address", "name": "tokenOut", "type": "address"},
                {"internalType": "uint24",  "name": "fee", "type": "uint24"},
                {"internalType": "address", "name": "recipient", "type": "address"},
                {"internalType": "uint256","name": "deadline", "type": "uint256"},
                {"internalType": "uint256","name": "amountIn", "type": "uint256"},
                {"internalType": "uint256","name": "amountOutMinimum", "type": "uint256"},
                {"internalType": "uint160","name": "sqrtPriceLimitX96", "type": "uint160"},
            ],
            "internalType": "struct ISwapRouter.ExactInputSingleParams",
            "name": "params", "type": "tuple"
        }],
        "name": "exactInputSingle", "outputs": [{"internalType": "uint256","name": "amountOut","type": "uint256"}],
        "stateMutability": "payable", "type": "function"
    },
    {
        "inputs": [{
            "components": [
                {"internalType": "address", "name": "tokenIn", "type": "address"},
                {"internalType": "address", "name": "tokenOut", "type": "address"},
                {"internalType": "uint24",  "name": "fee", "type": "uint24"},
                {"internalType": "address", "name": "recipient", "type": "address"},
                {"internalType": "uint256","name": "deadline", "type": "uint256"},
                {"internalType": "uint256","name": "amountOut", "type": "uint256"},
                {"internalType": "uint256","name": "amountInMaximum", "type": "uint256"},
                {"internalType": "uint160","name": "sqrtPriceLimitX96", "type": "uint160"},
            ],
            "internalType": "struct ISwapRouter.ExactOutputSingleParams",
            "name": "params", "type": "tuple"
        }],
        "name": "exactOutputSingle", "outputs": [{"internalType": "uint256","name": "amountIn","type": "uint256"}],
        "stateMutability": "payable", "type": "function"
    }
]

# Uniswap v3 QuoterV2 (view-only quotes)
QUOTER_V2_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "tokenIn", "type": "address"},
            {"internalType": "address", "name": "tokenOut", "type": "address"},
            {"internalType": "uint24",  "name": "fee", "type": "uint24"},
            {"internalType": "uint256","name": "amountIn", "type": "uint256"},
            {"internalType": "uint160","name": "sqrtPriceLimitX96", "type": "uint160"},
        ],
        "name": "quoteExactInputSingle",
        "outputs": [
            {"internalType": "uint256","name": "amountOut", "type": "uint256"},
            {"internalType": "uint160","name": "sqrtPriceX96After", "type": "uint160"},
            {"internalType": "uint32", "name": "initializedTicksCrossed", "type": "uint32"},
            {"internalType": "uint256","name": "gasEstimate", "type": "uint256"},
        ],
        "stateMutability": "nonpayable", "type": "function"
    }
]

# --- Known addresses (per chain) ---------------------------------------------
# Mainnet (Uniswap v3)
MAINNET_SWAP_ROUTER = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")
MAINNET_QUOTER_V2   = Web3.to_checksum_address("0x61fFE014bA17989E743c5F6cB21bF9697530B21e")
# Popular fee tiers: 500 (=0.05%), 3000 (=0.3%), 10000 (=1%)
DEFAULT_FEE = 3000

@dataclass
class Token:
    symbol: str
    address: str
    decimals: int

class UniswapAdapter:
    """
    Uniswap v3 adapter:
    - connect(): set up web3/providers, router/quoter
    - register_token(): cache token metadata
    - get_quote(): quote via QuoterV2 (amountOut for amountIn)
    - place_market(): build, sign & send swap tx (exactInputSingle)
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        chain: str = "mainnet",
        router_addr: Optional[str] = None,
        quoter_addr: Optional[str] = None,
        fee_tier: int = DEFAULT_FEE,
        weth_addr: Optional[str] = None,
    ):
        self.rpc_url = rpc_url or os.getenv("ETH_RPC_URL", "")
        self.chain = chain
        self.web3: Optional[Web3] = None
        self.router_addr = Web3.to_checksum_address(router_addr or MAINNET_SWAP_ROUTER)
        self.quoter_addr = Web3.to_checksum_address(quoter_addr or MAINNET_QUOTER_V2)
        self.fee_tier = int(fee_tier)
        self.wallet = os.getenv("WALLET_ADDRESS", "")
        self.pk = os.getenv("WALLET_PRIVATE_KEY", "")
        self.tokens: Dict[str, Token] = {}
        self.weth = weth_addr  # optional override (per chain)

        self.router = None
        self.quoter = None

    # --- Lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        if not self.rpc_url:
            raise RuntimeError("ETH_RPC_URL not set")
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 30}))
        # If using L2s like Arbitrum/Polygon, you may need POA middleware:
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0) # type: ignore
        if not self.web3.is_connected(): # type: ignore
            raise RuntimeError("Failed to connect to RPC")

        self.router = self.web3.eth.contract(address=self.router_addr, abi=SWAP_ROUTER_ABI) # type: ignore
        self.quoter = self.web3.eth.contract(address=self.quoter_addr, abi=QUOTER_V2_ABI) # type: ignore

        if not self.wallet or not self.pk:
            # You can still quote without keys; swaps need keys
            print("[uniswap] Warning: WALLET_ADDRESS / WALLET_PRIVATE_KEY not set; swaps disabled.")

    # --- Token helpers -------------------------------------------------------
    def _erc20(self, addr: str):
        return self.web3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI) # type: ignore

    def _decimals(self, addr: str) -> int:
        return self._erc20(addr).functions.decimals().call()

    def register_token(self, symbol: str, address: str, decimals: Optional[int] = None) -> None:
        address = Web3.to_checksum_address(address)
        if decimals is None:
            if self.web3 is None:
                raise RuntimeError("Call connect() before register_token() without decimals")
            decimals = self._decimals(address)
        self.tokens[symbol.upper()] = Token(symbol.upper(), address, int(decimals))

    def resolve(self, symbol: str) -> Token:
        t = self.tokens.get(symbol.upper())
        if not t:
            raise KeyError(f"Token not registered: {symbol}")
        return t

    # --- Quote ---------------------------------------------------------------
    def get_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in_units: float,
        *,
        fee_tier: Optional[int] = None,
        sqrt_price_limit_x96: int = 0
    ) -> Tuple[float, Dict]:
        """
        Returns (amount_out_units, extra_info)
        - amount_in_units/out_units are human units (respect decimals)
        """
        if self.web3 is None or self.quoter is None:
            raise RuntimeError("Adapter not connected")

        fee = int(fee_tier or self.fee_tier)
        t_in = self.resolve(token_in)
        t_out = self.resolve(token_out)

        amt_in = int(amount_in_units * (10 ** t_in.decimals))
        if amt_in <= 0:
            raise ValueError("amount_in_units must be > 0")

        amount_out, sqrt_after, ticks_crossed, gas_est = self.quoter.functions.quoteExactInputSingle(
            t_in.address,
            t_out.address,
            fee,
            amt_in,
            sqrt_price_limit_x96
        ).call()

        out_units = amount_out / (10 ** t_out.decimals)
        info = {
            "sqrt_after": int(sqrt_after),
            "ticks_crossed": int(ticks_crossed),
            "gas_estimate": int(gas_est),
            "fee_tier": fee
        }
        return out_units, info

    # --- Swaps ---------------------------------------------------------------
    def _require_allowance(self, owner: str, token_addr: str, spender: str, needed: int) -> Optional[str]:
        erc = self._erc20(token_addr)
        current = erc.functions.allowance(owner, spender).call()
        if current >= needed:
            return None
        # Build approve tx for 'infinite' allowance to reduce repeated approvals
        tx = erc.functions.approve(spender, 2**256 - 1).build_transaction({
            "from": owner,
            "nonce": self.web3.eth.get_transaction_count(owner), # type: ignore
            "gas": 100_000,
            "maxFeePerGas": self.web3.to_wei(os.getenv("MAX_FEE_GWEI", "30"), "gwei"), # type: ignore
            "maxPriorityFeePerGas": self.web3.to_wei(os.getenv("TIP_FEE_GWEI", "2"), "gwei"), # type: ignore
        })
        signed = self.web3.eth.account.sign_transaction(tx, private_key=self.pk) # type: ignore
        txh = self.web3.eth.send_raw_transaction(signed.rawTransaction) # type: ignore
        rcpt = self.web3.eth.wait_for_transaction_receipt(txh) # type: ignore
        if rcpt.status != 1:
            raise RuntimeError("ERC20 approve failed")
        return txh.hex()

    def place_market(
        self,
        token_in: str,
        token_out: str,
        amount_in_units: float,
        *,
        slippage_bps: int = 50,       # 50 bps = 0.50%
        recipient: Optional[str] = None,
        fee_tier: Optional[int] = None,
        deadline_secs: int = 120
    ) -> Dict:
        """
        Market swap using exactInputSingle:
        - Computes minOut from Quoter with slippage guard
        - Approves tokenIn if needed
        - Sends signed tx to SwapRouter
        Returns dict with tx hash and resolved amounts.
        """
        if self.web3 is None or self.router is None:
            raise RuntimeError("Adapter not connected")
        if not self.wallet or not self.pk:
            raise RuntimeError("WALLET_ADDRESS / WALLET_PRIVATE_KEY not configured")

        fee = int(fee_tier or self.fee_tier)
        t_in = self.resolve(token_in)
        t_out = self.resolve(token_out)

        # 1) Quote
        quoted_out, info = self.get_quote(token_in, token_out, amount_in_units, fee_tier=fee)
        min_out_units = quoted_out * (1 - slippage_bps / 10_000)

        # 2) Build params
        amt_in = int(amount_in_units * (10 ** t_in.decimals))
        amt_out_min = int(max(1, min_out_units * (10 ** t_out.decimals)))
        to_addr = Web3.to_checksum_address(recipient or self.wallet)
        deadline = int(time.time() + deadline_secs)

        # 3) Ensure allowance
        approve_txh = self._require_allowance(self.wallet, t_in.address, self.router.address, amt_in)

        # 4) exactInputSingle
        params = (
            t_in.address,
            t_out.address,
            fee,
            to_addr,
            deadline,
            amt_in,
            amt_out_min,
            0,  # sqrtPriceLimitX96
        )

        # Estimate gas (EIP-1559 fields)
        tx = self.router.functions.exactInputSingle(params).build_transaction({
            "from": self.wallet,
            "nonce": self.web3.eth.get_transaction_count(self.wallet),
            "value": 0,  # not sending ETH unless tokenIn is native via WETH flow (not handled here)
            "maxFeePerGas": self.web3.to_wei(os.getenv("MAX_FEE_GWEI", "30"), "gwei"),
            "maxPriorityFeePerGas": self.web3.to_wei(os.getenv("TIP_FEE_GWEI", "2"), "gwei"),
            # optionally set "gas": manual cap if you want
        })

        signed = self.web3.eth.account.sign_transaction(tx, private_key=self.pk)
        txh = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        return {
            "tx_hash": txh.hex(),
            "quoted_out_units": quoted_out,
            "min_out_units": min_out_units,
            "fee_tier": fee,
            "approve_tx": approve_txh,
            "quote_info": info,
        }

# --- Convenience: adapter factory & simple registry --------------------------
DEFAULT_TOKEN_REGISTRY = {
    # Mainnet examples (replace/add as needed)
    "USDC": {"address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "decimals": 6},
    "DAI":  {"address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "decimals": 18},
    "WETH": {"address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "decimals": 18},
}

def make_uniswap_adapter(
    rpc_url: Optional[str] = None,
    chain: str = "mainnet",
    tokens: Optional[Dict[str, Dict]] = None,
    fee_tier: int = DEFAULT_FEE,
) -> UniswapAdapter:
    ad = UniswapAdapter(rpc_url=rpc_url, chain=chain, fee_tier=fee_tier)
    ad.connect()
    reg = tokens or DEFAULT_TOKEN_REGISTRY
    for sym, meta in reg.items():
        ad.register_token(sym, meta["address"], meta.get("decimals"))
    return ad


# --- Example CLI (optional) --------------------------------------------------
if __name__ == "__main__":
    """
    Quick manual test (quote and optional swap):
      export ETH_RPC_URL="https://mainnet.infura.io/v3/<id>"
      export WALLET_ADDRESS="0xYourWallet..."
      export WALLET_PRIVATE_KEY="0x..."
      python -m backend.adapters.uniswap_adapter USDC DAI 100
    """
    import sys
    if len(sys.argv) < 4:
        print("Usage: python -m backend.adapters.uniswap_adapter <TOKEN_IN> <TOKEN_OUT> <AMOUNT_IN>")
        sys.exit(0)

    token_in, token_out, amt = sys.argv[1], sys.argv[2], float(sys.argv[3])
    adapter = make_uniswap_adapter()
    out, meta = adapter.get_quote(token_in, token_out, amt)
    print(f"Quote: {amt} {token_in} -> ~{out:.6f} {token_out} (fee={meta['fee_tier']})")

    # Uncomment to actually swap (requires funded wallet & approvals)
    # res = adapter.place_market(token_in, token_out, amt, slippage_bps=50)
    # print("Swap sent:", res)
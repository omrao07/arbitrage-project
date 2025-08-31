# tests/test_fix_gateway.py
import importlib
import time
from typing import Optional, Callable, Dict, Any, List
import pytest # type: ignore

# ---------------------- Import candidates (edit if needed) ----------------------
IMPORT_PATH_CANDIDATES = [
    "backend.oms.fix_gateway",
    "backend.exec.fix_gateway",
    "backend.infrastructure.fix_gateway",
    "gateways.fix_gateway",
    "fix_gateway",
    "infra.fix_gateway",
]

# ---------------------- Utilities: load + minimal SOH parser --------------------
SOH = "\x01"

def load_mod():
    last = None
    for p in IMPORT_PATH_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import fix_gateway from {IMPORT_PATH_CANDIDATES} ({last})")

def default_parse(raw: str) -> Dict[str, str]:
    """Ultra-simple FIX parser (fallback)"""
    parts = raw.rstrip(SOH).split(SOH)
    out = {}
    for kv in parts:
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out

def calc_checksum(body: str) -> int:
    # Sum of all bytes % 256
    return sum(bytearray(body.encode("ascii"))) % 256

# ---------------------- Fake transport -----------------------------------------
class FakeTransport:
    """Captures outbound frames and lets tests inject inbound frames."""
    def __init__(self):
        self.sent: List[bytes] = []
        self.inbound_handlers: List[Callable[[bytes], None]] = []

    def send(self, data: bytes):
        self.sent.append(data)

    def on_incoming(self, data: bytes):
        for h in self.inbound_handlers:
            h(data)

    def subscribe(self, cb: Callable[[bytes], None]):
        self.inbound_handlers.append(cb)

# ---------------------- API resolver -------------------------------------------
class API:
    """
    Adapts to:
      A) class FixGateway(transport=..., **cfg) with methods:
         .start(), .stop(), .send(msg_type, fields), .on_bytes(bytes)
         Optional helpers: .encode(dict)->bytes, .parse(bytes)->dict
      B) functions: build(...), send(...), on_bytes(...), encode/parse
    """
    def __init__(self, mod):
        self.mod = mod
        self.gw = None
        self.encode = None
        self.parse = None

        # Prefer class
        if hasattr(mod, "FixGateway"):
            Cls = getattr(mod, "FixGateway")
            try:
                self.gw = Cls  # store class; instantiate in start()
            except TypeError:
                self.gw = Cls
        else:
            # Function style
            self.build = getattr(mod, "build", None)
            self.send_fn = getattr(mod, "send", None)
            self.on_bytes = getattr(mod, "on_bytes", None)

        # optional encoders
        self.encode = getattr(mod, "encode", None)
        self.parse = getattr(mod, "parse", None)

    def start(self, transport: FakeTransport, **cfg):
        if self.gw:
            inst = self.gw(transport=transport, **cfg) if "transport" in getattr(self.gw, "__init__").__code__.co_varnames else self.gw(**cfg)
            # Try standard callbacks
            if hasattr(inst, "on_bytes"):
                transport.subscribe(lambda b: inst.on_bytes(b))
            if hasattr(inst, "start"):
                inst.start()
            self.inst = inst
            return inst
        else:
            assert self.build and self.on_bytes and self.send_fn, "Function API incomplete (need build/on_bytes/send)"
            obj = self.build(transport=transport, **cfg) if "transport" in self.build.__code__.co_varnames else self.build(**cfg)
            transport.subscribe(lambda b: self.on_bytes(obj, b)) # type: ignore
            self.inst = obj
            return obj

    def stop(self):
        if hasattr(self.inst, "stop"):
            self.inst.stop()

    def send(self, msg_type: str, fields: Dict[str, Any]):
        if hasattr(self.inst, "send"):
            return self.inst.send(msg_type, fields)
        return self.send_fn(self.inst, msg_type, fields)  # type: ignore # function API

    def parser(self) -> Callable[[bytes], Dict[str, str]]:
        if self.parse:
            return lambda b: self.parse(b) # type: ignore
        return lambda b: default_parse(b.decode("ascii"))

# ---------------------- Fixtures -----------------------------------------------
@pytest.fixture()
def transport():
    return FakeTransport()

@pytest.fixture()
def api(transport):
    mod = load_mod()
    api = API(mod)
    api.start(
        transport,
        sender_comp="BUY_SIDE",
        target_comp="BROKER",
        fix_version="FIX.4.4",
        heartbeat_sec=30,
        reset_seq=True,
    )
    yield api
    api.stop()

# ---------------------- Assertions helpers -------------------------------------
def assert_header_and_checksum(raw: bytes):
    s = raw.decode("ascii")
    assert s.startswith(f"8=FIX.4.4{SOH}")
    # 10= checksum last
    assert f"{SOH}10=" in s
    body, _, tail = s.rpartition(f"{SOH}10=")
    cs_str = tail.rstrip(SOH)
    assert cs_str.isdigit() and 0 <= int(cs_str) <= 255
    # Recompute checksum
    computed = calc_checksum(body + SOH + "10=")  # per FIX, checksum covers up to before its value
    assert int(cs_str) == computed

def last_sent_frame(transport: FakeTransport) -> str:
    assert transport.sent, "No outbound frames captured"
    return transport.sent[-1].decode("ascii")

# ---------------------- Tests ---------------------------------------------------

def test_logon_flow_builds_valid_A(api, transport):
    # Request a Logon (A)
    api.send("A", {"98": "0", "108": "30"})  # 98=EncryptMethod=0; 108=HeartBtInt
    s = last_sent_frame(transport)
    assert_header_and_checksum(transport.sent[-1])
    msg = api.parser()(transport.sent[-1])
    assert msg["35"] == "A"
    assert msg["49"] == "BUY_SIDE" and msg["56"] == "BROKER"
    assert msg["98"] == "0" and msg["108"] in ("30", "29", "31")  # allow rounding
    assert msg["34"] == "1"  # first sequence

def test_heartbeat_and_testrequest(api, transport):
    # Simulate broker sending TestRequest(1) -> we should answer Heartbeat(0) with 112 TestReqID
    test_id = "PING123"
    incoming = f"8=FIX.4.4{SOH}9=20{SOH}35=1{SOH}49=BROKER{SOH}56=BUY_SIDE{SOH}34=7{SOH}112={test_id}{SOH}10=000{SOH}"
    # Fix body length/checksum not accurate—gateway parser should still handle; or your parser may enforce strictly.
    transport.on_incoming(incoming.encode("ascii"))
    # Expect last sent is 35=0 with 112
    out = last_sent_frame(transport)
    assert "35=0" in out and f"112={test_id}" in out

def test_new_order_single_D(api, transport):
    # Place a basic order: 35=D, 55 symbol, 54 side, 38 qty, 40 ordType, 44 price
    api.send("D", {"11": "OID-1", "55": "AAPL", "54": "1", "38": "1000", "40": "2", "44": "185.25", "59": "0"})
    s = last_sent_frame(transport)
    assert_header_and_checksum(transport.sent[-1])
    m = default_parse(s)
    assert m["35"] == "D"
    for tag in ("11","55","54","38","40"):
        assert tag in m
    # Sequence increments
    seq = int(m["34"])
    assert seq >= 2  # after logon

def test_logout_flow(api, transport):
    api.send("5", {"58": "Client requested logout"})
    s = last_sent_frame(transport)
    m = default_parse(s)
    assert m["35"] == "5"
    assert "58" in m

def test_resend_request_handling(api, transport):
    """
    Simulate broker asking for a resend; gateway should respond (implementation may gap-fill
    with SequenceReset or replay stored messages). We just verify it doesn't crash and emits admin.
    """
    rr = f"8=FIX.4.4{SOH}9=30{SOH}35=2{SOH}49=BROKER{SOH}56=BUY_SIDE{SOH}34=12{SOH}7=1{SOH}16=5{SOH}10=000{SOH}"
    transport.on_incoming(rr.encode("ascii"))
    # Expect at least one outbound admin frame: 35=4 (SequenceReset-GapFill) or 35=X (ResendResponse vendor-specific)
    out = last_sent_frame(transport)
    assert ("35=4" in out) or ("122=" in out) or ("36=" in out)  # flexible

def test_body_length_and_checksum_are_correct_for_every_outbound(api, transport):
    """
    Quick sweep: ensure every captured outbound has valid BodyLength(9) and CheckSum(10).
    """
    # fire a couple of messages to capture
    api.send("0", {})  # Heartbeat
    api.send("2", {"7": "1", "16": "2"})  # ResendRequest out
    for b in transport.sent:
        s = b.decode("ascii")
        # Check BodyLength: value equals length between 35=.. SOH through before 10= field
        assert s.startswith("8=FIX.4.4" + SOH) and (SOH + "9=") in s
        p9 = s.index(SOH + "9=") + 1
        p9_end = s.index(SOH, p9)
        bodylen = int(s[p9+2:p9_end])
        # find start of body (after 9=..SOH)
        body_start = p9_end + 1
        # up to but excluding "10="
        p10 = s.rindex(SOH + "10=")
        calc_body = s[body_start:p10]
        assert bodylen == len(calc_body.encode("ascii"))
        # checksum validated by helper above
        assert_header_and_checksum(b)
# MPC Signing

This package (`security/mpc_signing/`) scaffolds **multi-party computation (MPC) signing** for use in
secure trading, custody, and governance flows.

⚠️ **Disclaimer:**  
This repo **does not implement cryptography**. It only provides the orchestration
layer (Coordinator, Strategy, Transport). To be safe in production you must plug
in a vetted MPC library (e.g., [FROST Schnorr signatures](https://datatracker.ietf.org/doc/draft-irtf-cfrg-frost/),
GG18/CGGMP for ECDSA, or vendor-provided MPC custody SDKs).

---

## Concepts

- **Participants**  
  Each party holds a private share of the signing key. No one ever sees the full key.

- **Threshold (t of N)**  
  Only `t` participants out of `N` need to cooperate to produce a valid signature.

- **Coordinator**  
  Runs the round: keygen → presign → partial sign → aggregate.

- **Strategy**  
  Defines the protocol messages (how to talk to participants, how to aggregate).

- **Transport**  
  Defines how to deliver requests/responses (HTTP, gRPC, Redis, in-process).

---

## Example Flow

```python
import asyncio
from security.mpc_signing.coordinator import Coordinator, Participant, RoundConfig
from security.mpc_signing.protocol import FrostLikeStrategy
from my_transports import HttpTransport  # you implement Transport

async def main():
    participants = [
        Participant("A", meta={"url": "http://hostA:8000"}),
        Participant("B", meta={"url": "http://hostB:8000"}),
        Participant("C", meta={"url": "http://hostC:8000"}),
    ]

    strategy = FrostLikeStrategy(threshold=2, scheme="frost-ed25519", context="news-intel")
    coord = Coordinator(participants=participants,
                        transport=HttpTransport(),
                        strategy=strategy,
                        cfg=RoundConfig(threshold=2, timeout_s=5, max_retries=1))

    # Ensure key material exists (idempotent)
    pub = await coord.ensure_key_material()
    print("Public key:", pub)

    # Run a signing round
    result = await coord.sign(b"hello world")
    print("Signature:", result.signature)

asyncio.run(main())

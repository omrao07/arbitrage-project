#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eu_data_soverignty.py — EU data sovereignty & cross-border transfer gap analysis

What it does
------------
Given your systems, data flows, vendors, contracts, and (optionally) a legal catalog,
this script builds an EU data sovereignty assessment and remediation plan:

1) Classifies datasets and systems (personal/sensitive, criticality, sectoral scope)
2) Maps data locations (at rest / in transit), cloud regions, subprocessors, support access
3) Detects third-country transfers and flags high-risk combinations (e.g., US support access,
   non-EEA storage, remote admin from outside EEA, unencrypted transfers, lack of key control)
4) Suggests legal mechanisms (SCC module, BCR, adequacy, derogations) and checks contract coverage
5) Generates a Transfer Register & Residency Gap Register
6) Proposes mitigations (BYOK/HYOK, encryption, key location, access controls, data minimisation)
7) Produces system-/vendor-level action plan, plus TIA/DPIA pre-filled skeletons

Inputs (CSV; flexible headers, case-insensitive)
-----------------------------------------------
--systems systems.csv
  Columns (recommended):
    system_id, name, owner, processor_role(Controller/Processor/Sub-processor),
    criticality(High/Med/Low), sector, data_categories(comma),
    pii(0/1), special_categories(0/1), children_data(0/1),
    cloud(0/1), cloud_service(SaaS/PaaS/IaaS), cloud_provider, cloud_region_primary,
    key_management(Self/BYOK/HYOK/Provider), key_location, encryption_at_rest(0/1), encryption_in_transit(0/1),
    admin_access_countries(comma), support_access_countries(comma)

--vendors vendors.csv (optional but recommended)
  Columns:
    vendor_id, name, role(Processor/Sub-processor/Controller), service, country_of_incorporation,
    primary_processing_country, subprocessors(comma 'name|country'), dpa_signed(0/1), scc_signed(0/1),
    bcr(0/1), adequacy_country(0/1), certification(s), last_audit_date

--flows flows.csv (optional)
  Logical/physical data flows. Columns:
    flow_id, source_system_id, dest_system_id_or_vendor, purpose, lawful_basis,
    data_categories(comma), transfer_mechanism(SCC/BCR/Adequacy/Derogation/Unknown),
    network_path(EU-only/Internet/MPLS/VPN), in_transit_encrypted(0/1),
    at_rest_location_country, at_rest_region, scheduled_exports_to_country(comma)

--contracts contracts.csv (optional)
  Columns:
    counterparty(vendor_id or name), agreement_type(DPA/Controller-Processor/SCC/BCR/IDTA),
    module(1/2/3/4 or text), effective_date, expires, annex_TIAs(0/1), annex_TOMs(0/1)

--catalog catalog.csv (optional)
  Regulation/obligation hints. Columns:
    rule_id, topic, applies_if(keyword list), severity(1-5), text, fix_hint

Key options
-----------
--asof 2025-09-06
--treat_eea "AT,BE,BG,HR,CY,CZ,DK,EE,FI,FR,DE,GR,HU,IS,IE,IT,LV,LI,LT,LU,MT,NL,NO,PL,PT,RO,SK,SI,ES,SE"
--adequacy "AD,AR,CA,FO,GS,IL,JP,NZ,CH,UY,KR,UK,US"    # Editable list; keep current to your policy
--high_risk_countries "RU,CN,IR,IQ,SY,AF,..."           # Org-specific list to flag
--encryption_mandatory 1                                # Require enc-at-rest + in-transit for cross-border
--outdir out_sov

Outputs
-------
- transfer_register.csv        Per flow/system: origin, destination, legal basis/mechanism, encryption, keys, risks
- residency_gaps.csv           Systems/vendors with non-EEA storage or access without adequate safeguards
- vendor_posture.csv           Vendor-level summary (DPA/SCC/BCR/adequacy, subprocessors map)
- action_plan.csv              Prioritised remediation tasks (owner, effort, due)
- system_matrix.csv            System × obligations matrix (enc/key mgmt/access/records)
- tia_templates/               One CSV per risky transfer pre-filled with context (for TIA workflow)
- dpia_candidates.csv          Systems likely needing DPIA (special data, large scale, high risk)
- summary.json                 Headline KPIs
- config.json                  Run configuration for reproducibility
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------- Utilities ----------------
def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t: return c
    for c in df.columns:
        if t in c.lower(): return c
    return None

def to_list(cell) -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)): return []
    if isinstance(cell, list): return [str(x).strip() for x in cell]
    return [x.strip() for x in str(cell).split(",") if str(x).strip()]

def to_date(s): return pd.to_datetime(s, errors="coerce")

def yesno(x) -> int:
    try:
        return 1 if float(x) > 0 else 0
    except Exception:
        return 1 if str(x).strip().lower() in {"y","yes","true"} else 0

def norm_country(code: str) -> str:
    return str(code).strip().upper()

def any_in(text: str, keys: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keys if k)


# ---------------- Loaders ----------------
def load_systems(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"system_id") or "system_id"):"system_id",
        (ncol(df,"name") or "name"):"name",
        (ncol(df,"owner") or "owner"):"owner",
        (ncol(df,"processor_role") or "processor_role"):"processor_role",
        (ncol(df,"criticality") or "criticality"):"criticality",
        (ncol(df,"sector") or "sector"):"sector",
        (ncol(df,"data_categories") or "data_categories"):"data_categories",
        (ncol(df,"pii") or "pii"):"pii",
        (ncol(df,"special_categories") or "special_categories"):"special_categories",
        (ncol(df,"children_data") or "children_data"):"children_data",
        (ncol(df,"cloud") or "cloud"):"cloud",
        (ncol(df,"cloud_service") or "cloud_service"):"cloud_service",
        (ncol(df,"cloud_provider") or "cloud_provider"):"cloud_provider",
        (ncol(df,"cloud_region_primary") or "cloud_region_primary"):"cloud_region_primary",
        (ncol(df,"key_management") or "key_management"):"key_management",
        (ncol(df,"key_location") or "key_location"):"key_location",
        (ncol(df,"encryption_at_rest") or "encryption_at_rest"):"encryption_at_rest",
        (ncol(df,"encryption_in_transit") or "encryption_in_transit"):"encryption_in_transit",
        (ncol(df,"admin_access_countries") or "admin_access_countries"):"admin_access_countries",
        (ncol(df,"support_access_countries") or "support_access_countries"):"support_access_countries",
    }
    df = df.rename(columns=ren)
    for b in ["pii","special_categories","children_data","cloud","encryption_at_rest","encryption_in_transit"]:
        if b in df.columns: df[b] = df[b].apply(yesno)
    for c in ["admin_access_countries","support_access_countries","data_categories"]:
        if c in df.columns: df[c+"_list"] = df[c].apply(to_list)
    if "cloud_region_primary" in df.columns:
        df["cloud_region_primary"] = df["cloud_region_primary"].astype(str)
    if "key_management" in df.columns:
        df["key_management_norm"] = df["key_management"].astype(str).str.upper()
    return df

def load_vendors(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"vendor_id") or "vendor_id"):"vendor_id",
        (ncol(df,"name") or "name"):"name",
        (ncol(df,"role") or "role"):"role",
        (ncol(df,"service") or "service"):"service",
        (ncol(df,"country_of_incorporation") or "country_of_incorporation"):"country_of_incorporation",
        (ncol(df,"primary_processing_country") or "primary_processing_country"):"primary_processing_country",
        (ncol(df,"subprocessors") or "subprocessors"):"subprocessors",
        (ncol(df,"dpa_signed") or "dpa_signed"):"dpa_signed",
        (ncol(df,"scc_signed") or "scc_signed"):"scc_signed",
        (ncol(df,"bcr") or "bcr"):"bcr",
        (ncol(df,"adequacy_country") or "adequacy_country"):"adequacy_country",
        (ncol(df,"certification") or "certification"):"certification",
        (ncol(df,"last_audit_date") or "last_audit_date"):"last_audit_date",
    }
    df = df.rename(columns=ren)
    for b in ["dpa_signed","scc_signed","bcr","adequacy_country"]:
        if b in df.columns: df[b] = df[b].apply(yesno)
    df["subprocessors_list"] = df.get("subprocessors","").apply(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])
    df["last_audit_date"] = to_date(df.get("last_audit_date", pd.NA))
    return df

def load_flows(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"flow_id") or "flow_id"):"flow_id",
        (ncol(df,"source_system_id") or "source_system_id"):"source_system_id",
        (ncol(df,"dest_system_id_or_vendor") or "dest_system_id_or_vendor"):"dest_system_id_or_vendor",
        (ncol(df,"purpose") or "purpose"):"purpose",
        (ncol(df,"lawful_basis") or "lawful_basis"):"lawful_basis",
        (ncol(df,"data_categories") or "data_categories"):"data_categories",
        (ncol(df,"transfer_mechanism") or "transfer_mechanism"):"transfer_mechanism",
        (ncol(df,"network_path") or "network_path"):"network_path",
        (ncol(df,"in_transit_encrypted") or "in_transit_encrypted"):"in_transit_encrypted",
        (ncol(df,"at_rest_location_country") or "at_rest_location_country"):"at_rest_location_country",
        (ncol(df,"at_rest_region") or "at_rest_region"):"at_rest_region",
        (ncol(df,"scheduled_exports_to_country") or "scheduled_exports_to_country"):"scheduled_exports_to_country"
    }
    df = df.rename(columns=ren)
    df["in_transit_encrypted"] = df.get("in_transit_encrypted", 0).apply(yesno)
    for c in ["data_categories","scheduled_exports_to_country"]:
        df[c+"_list"] = df.get(c,"").apply(to_list)
    return df

def load_contracts(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"counterparty") or "counterparty"):"counterparty",
        (ncol(df,"agreement_type") or "agreement_type"):"agreement_type",
        (ncol(df,"module") or "module"):"module",
        (ncol(df,"effective_date") or "effective_date"):"effective_date",
        (ncol(df,"expires") or "expires"):"expires",
        (ncol(df,"annex_TIAs") or "annex_TIAs"):"annex_TIAs",
        (ncol(df,"annex_TOMs") or "annex_TOMs"):"annex_TOMs",
    }
    df = df.rename(columns=ren)
    df["effective_date"] = to_date(df.get("effective_date", pd.NA))
    df["expires"] = to_date(df.get("expires", pd.NA))
    for b in ["annex_TIAs","annex_TOMs"]:
        if b in df.columns: df[b] = df[b].apply(yesno)
    return df

def load_catalog(path: Optional[str]) -> pd.DataFrame:
    if not path:
        # Minimal seed rules
        rows = [
            ("R1","Encryption","pii or special; cross-border",5,"Ensure TLS in transit and AES-256 at rest; manage keys in EEA.","Enable TLS1.2+, enforce enc-at-rest; BYOK/HYOK; HSM in EEA."),
            ("R2","KeyMgmt","cloud and cross-border",4,"Use BYOK/HYOK with keys in EEA; restrict provider access.","Adopt KMS with geo-fencing; split roles."),
            ("R3","TransferMechanism","third-country",5,"Use valid transfer mechanism (SCC module, BCR, adequacy).","Execute SCCs, attach TOMs and TIA."),
            ("R4","AccessControl","remote support/admin outside EEA",4,"Control, log, approve, and minimise remote access; JIT with MFA.","JIT access, PAM, break-glass, logging."),
            ("R5","DataMinimisation","special categories",4,"Minimise, pseudonymise if possible; segregate datasets.","Tokenisation/pseudo; segregate."),
        ]
        return pd.DataFrame(rows, columns=["rule_id","topic","applies_if","severity","text","fix_hint"])
    df = pd.read_csv(path)
    return df


# ---------------- Core logic ----------------
@dataclass
class Params:
    asof: pd.Timestamp
    eea: List[str]
    adequacy: List[str]
    high_risk: List[str]
    encryption_mandatory: bool
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EU data sovereignty & transfer assessment")
    ap.add_argument("--systems", required=True)
    ap.add_argument("--vendors", default="")
    ap.add_argument("--flows", default="")
    ap.add_argument("--contracts", default="")
    ap.add_argument("--catalog", default="")
    ap.add_argument("--asof", default="")
    ap.add_argument("--treat_eea", default="AT,BE,BG,HR,CY,CZ,DK,EE,FI,FR,DE,GR,HU,IS,IE,IT,LV,LI,LT,LU,MT,NL,NO,PL,PT,RO,SK,SI,ES,SE")
    ap.add_argument("--adequacy", default="AD,AR,CA,FO,GS,IL,JP,NZ,CH,UY,KR,UK,US")
    ap.add_argument("--high_risk_countries", default="RU,CN,IR,IQ,SY,AF")
    ap.add_argument("--encryption_mandatory", type=int, default=1)
    ap.add_argument("--outdir", default="out_sov")
    return ap.parse_args()

def is_eea_or_adequate(country: str, p: Params) -> Tuple[bool, str]:
    c = norm_country(country)
    if c in p.eea: return True, "EEA"
    if c in p.adequacy: return True, "Adequacy"
    return False, "ThirdCountry"

def system_residency_risks(sys_row: pd.Series, p: Params) -> Dict[str, object]:
    region = str(sys_row.get("cloud_region_primary","")).upper()
    key_loc = norm_country(sys_row.get("key_location",""))
    enc_ok = bool(sys_row.get("encryption_at_rest",0)) and bool(sys_row.get("encryption_in_transit",0))
    km = str(sys_row.get("key_management_norm","")).upper()
    # infer country from region (very rough): assume last 2 letters if looks like code
    region_cc = region[-2:] if len(region) >= 2 else ""
    ok, basis = is_eea_or_adequate(region_cc, p) if region_cc else (True,"Unknown")
    support_countries = [norm_country(x) for x in sys_row.get("support_access_countries_list", [])]
    admin_countries = [norm_country(x) for x in sys_row.get("admin_access_countries_list", [])]
    remote = list({*support_countries, *admin_countries} - set(p.eea))
    kms_good = km in {"BYOK","HYOK","SELF"}
    key_loc_ok, key_loc_basis = is_eea_or_adequate(key_loc, p) if key_loc else (False, "Unknown")
    risk = 0
    notes = []
    if not ok: risk += 2; notes.append(f"Primary region outside EEA/adequacy ({region})")
    if not enc_ok and p.encryption_mandatory: risk += 2; notes.append("Missing mandatory encryption")
    if remote: risk += 1; notes.append(f"Remote access from {','.join(remote)}")
    if not kms_good: risk += 1; notes.append(f"Key mgmt not BYOK/HYOK (={km or 'N/A'})")
    if not key_loc_ok: risk += 1; notes.append(f"Keys not in EEA/adequacy ({key_loc or 'N/A'})")
    if any(c in p.high_risk for c in remote): risk += 1; notes.append("Remote access from high-risk country")
    return {
        "region": region, "region_status": basis,
        "encryption_ok": int(enc_ok),
        "key_mgmt": km or "N/A", "key_location": key_loc or "N/A", "key_location_status": key_loc_basis,
        "remote_access_outside_eea": ",".join(remote) if remote else "",
        "risk_score": int(risk),
        "risk_notes": "; ".join(notes)
    }

def decide_transfer_mechanism(dest_country: str, vendor_row: Optional[pd.Series], contracts: pd.DataFrame, p: Params) -> Tuple[str, str]:
    ok, basis = is_eea_or_adequate(dest_country, p)
    if ok:
        mech = "Adequacy" if basis == "Adequacy" else "EEA"
        return mech, "No SCC needed" if mech != "Adequacy" else "Adequacy covers transfer"
    # Try vendor posture / contracts
    scc = bool(vendor_row.get("scc_signed", 0)) if vendor_row is not None else False
    bcr = bool(vendor_row.get("bcr", 0)) if vendor_row is not None else False
    if bcr:
        return "BCR", "Binding Corporate Rules in place"
    if scc:
        # Pick SCC module by role (simplified)
        role = str(vendor_row.get("role","Processor")).lower() if vendor_row is not None else "processor"
        module = "Module 2 (Controller→Processor)" if "processor" in role else "Module 3 (Processor→Processor)"
        return f"SCC {module}", "Ensure TOMs + Annex II/III and TIA attached"
    # Last resort
    return "Unknown", "Execute SCCs or stop transfer until mechanism is set"

def flows_transfer_register(flows: pd.DataFrame, systems: pd.DataFrame, vendors: pd.DataFrame, contracts: pd.DataFrame, p: Params) -> pd.DataFrame:
    sys_map = systems.set_index("system_id")
    ven_by_name = vendors.set_index("name") if not vendors.empty else pd.DataFrame()
    ven_by_id   = vendors.set_index("vendor_id") if "vendor_id" in vendors.columns else pd.DataFrame()
    rows = []
    for _, f in flows.iterrows():
        src = str(f.get("source_system_id",""))
        dst = str(f.get("dest_system_id_or_vendor",""))
        at_rest_cc = norm_country(f.get("at_rest_location_country",""))
        exports = [norm_country(x) for x in f.get("scheduled_exports_to_country_list", [])]
        # resolve vendor row if any
        vrow = None
        if dst in ven_by_id.index: vrow = ven_by_id.loc[dst]
        elif dst in ven_by_name.index: vrow = ven_by_name.loc[dst]
        mech = str(f.get("transfer_mechanism","")).strip()
        mech_suggest = ""
        if not mech or mech.lower() == "unknown":
            mech, mech_suggest = decide_transfer_mechanism(at_rest_cc or (vrow.get("primary_processing_country","") if vrow is not None else ""), vrow, contracts, p)
        enc_tr = int(f.get("in_transit_encrypted", 0))
        # risk flags
        ok_rest, basis_rest = is_eea_or_adequate(at_rest_cc, p) if at_rest_cc else (True, "Unknown")
        risky_exports = [c for c in exports if c and c not in p.eea and c not in p.adequacy]
        risk = 0
        notes = []
        if not ok_rest: risk += 2; notes.append(f"Storage in {at_rest_cc} (non-EEA)")
        if not enc_tr: risk += 1; notes.append("Transit encryption missing")
        if risky_exports: risk += 1; notes.append(f"Scheduled exports to {','.join(risky_exports)}")
        rows.append({
            "flow_id": f.get("flow_id",""),
            "source_system_id": src,
            "dest": dst,
            "purpose": f.get("purpose",""),
            "lawful_basis": f.get("lawful_basis",""),
            "data_categories": ",".join(to_list(f.get("data_categories",""))),
            "at_rest_country": at_rest_cc or "",
            "at_rest_status": basis_rest,
            "in_transit_encrypted": enc_tr,
            "transfer_mechanism": mech,
            "mechanism_note": mech_suggest,
            "risky_exports_to": ",".join(risky_exports),
            "risk_score": int(risk),
            "risk_notes": "; ".join(notes)
        })
    return pd.DataFrame(rows)

def vendor_posture_table(vendors: pd.DataFrame, p: Params) -> pd.DataFrame:
    if vendors.empty: return pd.DataFrame()
    rows = []
    for _, v in vendors.iterrows():
        pp = norm_country(v.get("primary_processing_country",""))
        ok, basis = is_eea_or_adequate(pp, p) if pp else (True, "Unknown")
        subproc = []
        for sp in v.get("subprocessors_list", []):
            # format 'name|country'
            name, cc = (sp.split("|")+["",""])[:2]
            subproc.append((name.strip(), norm_country(cc)))
        risky_sub = [name for (name,cc) in subproc if cc not in p.eea and cc not in p.adequacy]
        rows.append({
            "vendor_id": v.get("vendor_id",""),
            "vendor_name": v.get("name",""),
            "role": v.get("role",""),
            "processing_country": pp or "",
            "country_status": basis,
            "dpa_signed": int(v.get("dpa_signed",0)),
            "scc_signed": int(v.get("scc_signed",0)),
            "bcr": int(v.get("bcr",0)),
            "adequacy_country": int(v.get("adequacy_country",0)),
            "risky_subprocessors": ";".join(risky_sub),
            "last_audit_date": str(v.get("last_audit_date","")) if pd.notna(v.get("last_audit_date", pd.NaT)) else "",
        })
    return pd.DataFrame(rows)

def residency_gaps_table(systems: pd.DataFrame, p: Params) -> pd.DataFrame:
    rows = []
    for _, s in systems.iterrows():
        r = system_residency_risks(s, p)
        rows.append({
            "system_id": s["system_id"], "system_name": s["name"],
            "pii": int(s.get("pii",0)), "special_categories": int(s.get("special_categories",0)),
            **r
        })
    df = pd.DataFrame(rows)
    # flag gaps
    df["gap_flag"] = ((df["region_status"] == "ThirdCountry") | (df["encryption_ok"] == 0) |
                      (df["key_location_status"].isin(["ThirdCountry","Unknown"])) |
                      (df["remote_access_outside_eea"] != "")).astype(int)
    return df.sort_values(["risk_score","gap_flag","special_categories","pii"], ascending=[False, False, False, False])

def dpia_candidates(systems: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    join = systems.merge(gaps[["system_id","risk_score","gap_flag"]], on="system_id", how="left")
    # simple heuristic: special categories OR children data OR high risk/gap
    crit = (
        (join.get("special_categories",0) > 0) |
        (join.get("children_data",0) > 0) |
        (join.get("pii",0) > 0) & (join.get("criticality","").astype(str).str.lower().isin(["high","critical"])) |
        (join.get("risk_score",0) >= 3) |
        (join.get("gap_flag",0) > 0)
    )
    return join.loc[crit, ["system_id","name","criticality","pii","special_categories","children_data","risk_score","gap_flag"]].drop_duplicates()

def action_plan(gaps: pd.DataFrame, transfers: pd.DataFrame, vendors_tbl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # System gaps
    for _, g in gaps.iterrows():
        fixes = []
        if g["encryption_ok"] == 0: fixes.append("Enable enc-at-rest & in-transit")
        if g["key_mgmt"] not in {"BYOK","HYOK","SELF"}: fixes.append("Adopt BYOK/HYOK & HSM")
        if g["key_location_status"] in {"ThirdCountry","Unknown"}: fixes.append("Move KMS keys to EEA/adequacy")
        if g["region_status"] == "ThirdCountry": fixes.append("Relocate data-at-rest to EEA/adequacy or establish mechanism")
        if g["remote_access_outside_eea"]: fixes.append("Gate remote access via JIT/MFA/PAM; EU-based break-glass")
        if not fixes: continue
        rows.append({
            "object": "system",
            "id": g["system_id"],
            "name": g["system_name"],
            "priority": int(g["risk_score"] >= 3) * 3 + int("ThirdCountry" in (g["region_status"], g["key_location_status"])) * 2 + 1,
            "actions": "; ".join(fixes),
            "owner_suggested": "System Owner / Security",
            "due": ""
        })
    # Transfer fixes
    for _, t in transfers.iterrows():
        fixes = []
        if t["transfer_mechanism"].startswith("Unknown"): fixes.append("Execute SCCs (proper module) or BCR; complete TIA + TOMs")
        if t["in_transit_encrypted"] == 0: fixes.append("Encrypt transfer channel (TLS/IPsec)")
        if t["at_rest_status"] == "ThirdCountry": fixes.append("Evaluate relocation or add EU mirror with minimisation")
        if t["risky_exports_to"]: fixes.append("Disable scheduled exports or route via EU processor with mechanism")
        if not fixes: continue
        rows.append({
            "object": "flow",
            "id": t["flow_id"],
            "name": f"{t['source_system_id']}→{t['dest']}",
            "priority": int(t["risk_score"] >= 2) * 3 + 1,
            "actions": "; ".join(fixes),
            "owner_suggested": "Privacy / Legal",
            "due": ""
        })
    # Vendor posture
    for _, v in vendors_tbl.iterrows():
        fixes = []
        if v.get("dpa_signed",0) == 0: fixes.append("Execute DPA (Art. 28)")
        if (v.get("country_status") == "ThirdCountry") and (v.get("scc_signed",0) == 0) and (v.get("bcr",0) == 0) and (v.get("adequacy_country",0) == 0):
            fixes.append("Put SCCs in place or switch to adequacy/EEA region")
        if v.get("risky_subprocessors"): fixes.append("Approve/restrict risky subprocessors or require EEA routing")
        if not fixes: continue
        rows.append({
            "object": "vendor",
            "id": v["vendor_id"],
            "name": v["vendor_name"],
            "priority": 3 if "SCCs" in " ".join(fixes) else 2,
            "actions": "; ".join(fixes),
            "owner_suggested": "Procurement / Legal",
            "due": ""
        })
    plan = pd.DataFrame(rows)
    if plan.empty: return plan
    # Normalise priority (1..5)
    plan["priority"] = plan["priority"].clip(lower=1)
    plan = plan.sort_values(["priority","object"], ascending=[False, True])
    return plan

def system_matrix(systems: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    g = gaps.set_index("system_id")
    rows = []
    for _, s in systems.iterrows():
        gid = s["system_id"]
        r = g.loc[gid] if gid in g.index else pd.Series()
        rows.append({
            "system_id": gid, "system_name": s["name"],
            "role": s.get("processor_role",""),
            "enc_at_rest": s.get("encryption_at_rest",0),
            "enc_in_transit": s.get("encryption_in_transit",0),
            "key_mgmt": s.get("key_management",""),
            "key_location": s.get("key_location",""),
            "cloud_region_primary": s.get("cloud_region_primary",""),
            "admin_access_countries": ",".join(s.get("admin_access_countries_list",[])),
            "support_access_countries": ",".join(s.get("support_access_countries_list",[])),
            "gap_flag": int(r.get("gap_flag",0)) if not r.empty else 0,
            "risk_score": int(r.get("risk_score",0)) if not r.empty else 0
        })
    return pd.DataFrame(rows)

def write_tia_templates(transfers: pd.DataFrame, outdir: Path):
    tia_dir = outdir / "tia_templates"
    tia_dir.mkdir(parents=True, exist_ok=True)
    risky = transfers[(transfers["at_rest_status"]=="ThirdCountry") | (transfers["transfer_mechanism"].str.startswith("SCC")) | (transfers["transfer_mechanism"]=="Unknown")]
    for _, r in risky.iterrows():
        df = pd.DataFrame([{
            "flow_id": r["flow_id"],
            "source": r["source_system_id"],
            "destination": r["dest"],
            "data_categories": r["data_categories"],
            "purpose": r["purpose"],
            "lawful_basis": r["lawful_basis"],
            "transfer_mechanism": r["transfer_mechanism"],
            "encryption": "Yes" if r["in_transit_encrypted"] else "No",
            "technical_measures": "",
            "contractual_measures": "",
            "organisational_measures": "",
            "destination_country": r["at_rest_country"],
            "risk_notes": r["risk_notes"],
            "assessment_summary": "",
            "decision": ""
        }])
        df.to_csv(tia_dir / f"TIA_{r['flow_id'] or (r['source_system_id']+'_'+r['dest'])}.csv", index=False)


# ---------------- Main ----------------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    asof = pd.to_datetime(args.asof) if args.asof else pd.Timestamp.today().normalize()

    systems = load_systems(args.systems)
    vendors = load_vendors(args.vendors)
    flows = load_flows(args.flows)
    contracts = load_contracts(args.contracts)
    catalog = load_catalog(args.catalog)

    p = Params(
        asof=asof,
        eea=[norm_country(x) for x in to_list(args.treat_eea)],
        adequacy=[norm_country(x) for x in to_list(args.adequacy)],
        high_risk=[norm_country(x) for x in to_list(args.high_risk_countries)],
        encryption_mandatory=bool(args.encryption_mandatory),
        outdir=args.outdir
    )

    # Build tables
    gaps = residency_gaps_table(systems, p)
    transfers = flows_transfer_register(flows, systems, vendors, contracts, p) if not flows.empty else pd.DataFrame(columns=[
        "flow_id","source_system_id","dest","purpose","lawful_basis","data_categories","at_rest_country","at_rest_status",
        "in_transit_encrypted","transfer_mechanism","mechanism_note","risky_exports_to","risk_score","risk_notes"
    ])
    vendors_tbl = vendor_posture_table(vendors, p)
    plan = action_plan(gaps, transfers, vendors_tbl)
    sysmat = system_matrix(systems, gaps)
    dpia = dpia_candidates(systems, gaps)

    # Write outputs
    gaps.to_csv(outdir / "residency_gaps.csv", index=False)
    transfers.to_csv(outdir / "transfer_register.csv", index=False)
    if not vendors_tbl.empty: vendors_tbl.to_csv(outdir / "vendor_posture.csv", index=False)
    if not plan.empty: plan.to_csv(outdir / "action_plan.csv", index=False)
    sysmat.to_csv(outdir / "system_matrix.csv", index=False)
    if not dpia.empty: dpia.to_csv(outdir / "dpia_candidates.csv", index=False)
    write_tia_templates(transfers, outdir)

    # KPIs
    kpi = {
        "asof": str(asof.date()),
        "systems": int(len(systems)),
        "vendors": int(len(vendors)) if not vendors.empty else 0,
        "flows": int(len(flows)) if not flows.empty else 0,
        "with_gaps": int(gaps["gap_flag"].sum()),
        "high_risk_systems": int((gaps["risk_score"] >= 3).sum()),
        "third_country_transfers": int((transfers["at_rest_status"] == "ThirdCountry").sum()) if not transfers.empty else 0,
        "unknown_mechanisms": int((transfers["transfer_mechanism"] == "Unknown").sum()) if not transfers.empty else 0,
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))
    (outdir / "config.json").write_text(json.dumps({
        "asof": str(asof.date()),
        "eea": p.eea, "adequacy": p.adequacy, "high_risk": p.high_risk,
        "encryption_mandatory": p.encryption_mandatory,
        "inputs": {"systems": args.systems, "vendors": args.vendors or None, "flows": args.flows or None, "contracts": args.contracts or None, "catalog": args.catalog or None}
    }, indent=2))

    # Console
    print("== EU Data Sovereignty ==")
    print(f"As of {kpi['asof']}: {kpi['systems']} systems, {kpi['vendors']} vendors, {kpi['flows']} flows")
    print(f"Gaps: {kpi['with_gaps']} systems  |  Third-country transfers: {kpi['third_country_transfers']}  |  Unknown mechanisms: {kpi['unknown_mechanisms']}")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
eu_cybersecurity_mandates.py — Assess EU cyber/operational resilience mandates (NIS2, DORA, CRA, CER, GDPR-security)
and generate a prioritized remediation plan for your organization.

What this does
--------------
Given your systems inventory, current controls, incidents, vendors, and (optionally) a regulation catalog,
this script will:

1) Determine applicability of EU mandates by entity type/sector/size (keyword heuristics)
2) Expand each mandate into concrete requirements ("controls library")
3) Map your existing controls to requirements (keyword/taxonomy match + confidence score)
4) Detect gaps, estimate implementation effort, and compute risk-weighted priority
5) Produce: gap register, action plan (who/when/cost), system-level requirements,
   vendor/third-party obligations, and an incident-reporting matrix.

This is a *framework*. Default catalogs are intentionally generic. Supply your own regs.csv to reflect
the latest legal texts, sector guidance, and member-state transposition dates.

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--systems systems.csv
    Your asset/service inventory. Suggested columns:
      system_id, name, owner, criticality(High/Med/Low), sector, function, data_classes (comma),
      internet_exposed(0/1), users_eu(0/1), cloud(0/1), third_parties (comma), rto_hours, rpo_hours

--controls controls.csv
    Your implemented controls (policies/tech/process). Suggested columns:
      control_id, name, domain, subdomain, description, status(Implemented/Planned/NA),
      evidence_url, owner, last_audit_date, maturity(1-5), iso27001(clauses), nist_csF, kill_chain_tags

--incidents incidents.csv (optional)
    Historical incidents for risk weighting. Suggested columns:
      date, system_id, severity(1-5), type (ransomware/phishing/etc), downtime_hours, data_loss, vendor_involved(0/1)

--vendors vendors.csv (optional)
    Third parties. Suggested columns:
      vendor_id, name, service, critical(0/1), data_processing(0/1), cloud(0/1), country, contract_renewal_date

--regs regs.csv (optional but recommended)
    Regulation catalog & requirements. Columns:
      mandate, version, requirement_id, requirement, category, keywords(comma), min_maturity(1-5),
      evidence(sample), role(Applicant/Financial/Operator-of-ES/Manufacturer/etc),
      applicability_keywords(comma), reporting_deadline_hours, due_date(YYYY-MM-DD), penalty_max_eur,
      references (urls or citation keys)

Key options
-----------
--asof 2025-09-06         Analysis date (affects overdue flags)
--base-standard ISO27001  Optional base taxonomy shown in crosswalk labels
--risk-weights "sev:0.5,downtime:0.3,exposure:0.2"
--effort-default 5        Default effort (pts) for a new control where not specified
--outdir out_mandates     Output directory

Outputs
-------
- gap_register.csv          Per requirement: coverage %, mapped controls, gap, priority, effort, owner suggestion
- action_plan.csv           Deduplicated actions with start-by/due-by, prerequisites, suggested owners
- system_requirements.csv   System × requirement matrix (critical systems first)
- vendor_requirements.csv   Vendor obligations and due diligence asks
- reporting_matrix.csv      Incident types × mandate reporting windows & contacts (where available)
- summary.json              Headline KPIs and risk summary
- config.json               Run configuration & knobs

Notes
-----
- No legal advice. Bring your own regs.csv to reflect authoritative obligations and dates.
- Matching is keyword-based with simple NLP; review the outputs before acting.
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


# ----------------------------- Utilities -----------------------------
def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None


def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def to_list(cell) -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell]
    return [x.strip() for x in str(cell).split(",") if str(x).strip()]


def norm_text(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()


def safe_num(x, default=0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


# ----------------------------- Inputs -----------------------------
def read_csv_any(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return pd.read_csv(path)


def load_systems(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df, "system_id") or "system_id"): "system_id",
        (ncol(df, "name") or "name"): "name",
        (ncol(df, "owner") or "owner"): "owner",
        (ncol(df, "criticality") or "criticality"): "criticality",
        (ncol(df, "sector") or "sector"): "sector",
        (ncol(df, "function") or "function"): "function",
        (ncol(df, "data_classes") or "data_classes"): "data_classes",
        (ncol(df, "internet_exposed") or "internet_exposed"): "internet_exposed",
        (ncol(df, "users_eu") or "users_eu"): "users_eu",
        (ncol(df, "cloud") or "cloud"): "cloud",
        (ncol(df, "third_parties") or "third_parties"): "third_parties",
        (ncol(df, "rto_hours") or "rto_hours"): "rto_hours",
        (ncol(df, "rpo_hours") or "rpo_hours"): "rpo_hours",
    })
    for b in ["internet_exposed", "users_eu", "cloud"]:
        if b in df.columns:
            df[b] = (pd.to_numeric(df[b], errors="coerce") > 0).astype(int)
    for c in ["rto_hours", "rpo_hours"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["data_classes_list"] = df["data_classes"].apply(to_list) if "data_classes" in df.columns else [[] for _ in range(len(df))]
    df["third_parties_list"] = df["third_parties"].apply(to_list) if "third_parties" in df.columns else [[] for _ in range(len(df))]
    return df


def load_controls(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df, "control_id") or "control_id"): "control_id",
        (ncol(df, "name") or "name"): "name",
        (ncol(df, "domain") or "domain"): "domain",
        (ncol(df, "subdomain") or "subdomain"): "subdomain",
        (ncol(df, "description") or "description"): "description",
        (ncol(df, "status") or "status"): "status",
        (ncol(df, "evidence_url") or "evidence_url"): "evidence_url",
        (ncol(df, "owner") or "owner"): "owner",
        (ncol(df, "last_audit_date") or "last_audit_date"): "last_audit_date",
        (ncol(df, "maturity") or "maturity"): "maturity",
        (ncol(df, "iso27001") or "iso27001"): "iso27001",
        (ncol(df, "nist_csf") or "nist_csf"): "nist_csf",
        (ncol(df, "kill_chain_tags") or "kill_chain_tags"): "kill_chain_tags",
    })
    df["maturity"] = pd.to_numeric(df.get("maturity", np.nan), errors="coerce")
    df["desc_norm"] = (df["name"].fillna("") + " " + df["description"].fillna("")).apply(norm_text)
    df["tags"] = df.get("kill_chain_tags", "").apply(to_list)
    return df


def load_incidents(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df, "date") or df.columns[0]): "date",
        (ncol(df, "system_id") or "system_id"): "system_id",
        (ncol(df, "severity") or "severity"): "severity",
        (ncol(df, "type") or "type"): "type",
        (ncol(df, "downtime_hours") or "downtime_hours"): "downtime_hours",
        (ncol(df, "data_loss") or "data_loss"): "data_loss",
        (ncol(df, "vendor_involved") or "vendor_involved"): "vendor_involved",
    })
    df["date"] = to_date(df["date"])
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
    df["downtime_hours"] = pd.to_numeric(df["downtime_hours"], errors="coerce")
    df["vendor_involved"] = (pd.to_numeric(df["vendor_involved"], errors="coerce") > 0).astype(int)
    return df


def load_vendors(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df, "vendor_id") or "vendor_id"): "vendor_id",
        (ncol(df, "name") or "name"): "name",
        (ncol(df, "service") or "service"): "service",
        (ncol(df, "critical") or "critical"): "critical",
        (ncol(df, "data_processing") or "data_processing"): "data_processing",
        (ncol(df, "cloud") or "cloud"): "cloud",
        (ncol(df, "country") or "country"): "country",
        (ncol(df, "contract_renewal_date") or "contract_renewal_date"): "contract_renewal_date",
    })
    for b in ["critical", "data_processing", "cloud"]:
        if b in df.columns:
            df[b] = (pd.to_numeric(df[b], errors="coerce") > 0).astype(int)
    df["contract_renewal_date"] = to_date(df.get("contract_renewal_date", pd.Series([None]*len(df))))
    return df


def default_catalog() -> pd.DataFrame:
    """
    Minimal seed catalog with broad buckets. You should replace/augment with regs.csv.
    """
    rows = [
        # mandate, version, requirement_id, requirement, category, keywords, min_maturity, evidence, role, applicability, reporting_deadline_hours, due_date, penalty_max_eur, references
        ("NIS2", "seed", "NIS2-IR-01", "Incident response plan incl. reporting workflow & contact points", "Incident Response",
         "incident,response,report,contact,playbook,csirt", 3, "IRP doc + on-call roster", "Operator-of-essential|Important Entity",
         "network,critical,infrastructure,service,operator,essential,important", 72, "", "", ""),
        ("NIS2", "seed", "NIS2-RM-02", "Risk management: asset inventory, risk assessment, treatment plan", "Risk",
         "risk,assessment,register,asset inventory,ra,rtp", 3, "Risk register + last review", "All",
         "operator,essential,important,network,service", "", "", "", ""),
        ("NIS2", "seed", "NIS2-SC-03", "Supply chain security: third-party risk management & contractual clauses", "Third-Party",
         "vendor,third party,supplier,contract,clauses,assurance,sca", 3, "TPRM policy + sample DD report", "All",
         "vendor,third,outsourc,cloud,processor,service provider", "", "", "", ""),
        ("DORA", "seed", "DORA-TR-01", "ICT testing incl. threat-led penetration testing for critical functions", "Testing",
         "pentest,tlpt,threat led,red team,testing,vulnerability", 3, "RT/TLPT report", "Financial Entity",
         "bank,insurance,investment,psp,fintech,financial", "", "", "", ""),
        ("DORA", "seed", "DORA-IR-02", "Register & classify ICT incidents; report major incidents to competent authority", "Incident Response",
         "ict incident,register,classification,threshold,major,report", 3, "Incident register + taxonomy", "Financial Entity",
         "financial,bank,insurance,investment", 72, "", "", ""),
        ("CRA", "seed", "CRA-SD-01", "Secure development lifecycle incl. vulnerability handling and updates", "Secure Dev",
         "secure development,sdla,vulnerability handling,coordinated disclosure,updates,patch", 3, "SDLC policy + backlog linkage", "Manufacturer",
         "product,software,device,manufacturer,firmware,embedded", "", "", "", ""),
        ("CER", "seed", "CER-BC-01", "Business continuity & resilience measures for critical entities", "Resilience",
         "business continuity,bcms,dr,continuity,exercise", 3, "BCP + exercise results", "Critical Entity",
         "critical entity,essential service,operator", "", "", "", ""),
        ("GDPR", "seed", "GDPR-S-32", "Security of processing (Art. 32): appropriate technical and organisational measures", "Data Protection",
         "gdpr,security of processing,art 32,encryption,pseudonymisation", 3, "SoA + TOMs", "Controller|Processor",
         "personal data,controller,processor,eu residents", "", "", "", ""),
    ]
    cols = ["mandate","version","requirement_id","requirement","category","keywords","min_maturity","evidence",
            "role","applicability_keywords","reporting_deadline_hours","due_date","penalty_max_eur","references"]
    df = pd.DataFrame(rows, columns=cols)
    df["keywords"] = df["keywords"].apply(lambda s: [k.strip() for k in str(s).split(",") if k.strip()])
    df["applicability_keywords"] = df["applicability_keywords"].apply(lambda s: [k.strip() for k in str(s).split(",") if k.strip()])
    df["min_maturity"] = pd.to_numeric(df["min_maturity"], errors="coerce").fillna(3)
    df["reporting_deadline_hours"] = pd.to_numeric(df["reporting_deadline_hours"], errors="coerce")
    return df


def load_catalog(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return default_catalog()
    df = pd.read_csv(path)
    # Normalize essential columns
    ren = {
        (ncol(df, "mandate") or "mandate"): "mandate",
        (ncol(df, "version") or "version"): "version",
        (ncol(df, "requirement_id") or "requirement_id"): "requirement_id",
        (ncol(df, "requirement") or "requirement"): "requirement",
        (ncol(df, "category") or "category"): "category",
        (ncol(df, "keywords") or "keywords"): "keywords",
        (ncol(df, "min_maturity") or "min_maturity"): "min_maturity",
        (ncol(df, "evidence") or "evidence"): "evidence",
        (ncol(df, "role") or "role"): "role",
        (ncol(df, "applicability_keywords") or "applicability_keywords"): "applicability_keywords",
        (ncol(df, "reporting_deadline_hours") or "reporting_deadline_hours"): "reporting_deadline_hours",
        (ncol(df, "due_date") or "due_date"): "due_date",
        (ncol(df, "penalty_max_eur") or "penalty_max_eur"): "penalty_max_eur",
        (ncol(df, "references") or "references"): "references",
    }
    df = df.rename(columns=ren)
    for c in ["keywords","applicability_keywords"]:
        if c in df.columns:
            df[c] = df[c].apply(to_list)
        else:
            df[c] = [[] for _ in range(len(df))]
    df["min_maturity"] = pd.to_numeric(df.get("min_maturity", 3), errors="coerce").fillna(3)
    df["reporting_deadline_hours"] = pd.to_numeric(df.get("reporting_deadline_hours", np.nan), errors="coerce")
    return df


# ----------------------------- Applicability & Matching -----------------------------
def entity_roles_from_systems(systems: pd.DataFrame) -> List[str]:
    """
    Crude heuristics to infer org roles vis-à-vis mandates.
    """
    roles = set()
    sec = " ".join(systems.get("sector", "").astype(str).str.lower().tolist())
    func = " ".join(systems.get("function", "").astype(str).str.lower().tolist())
    if any(k in sec + func for k in ["bank", "insurance", "payments", "psp", "securities", "broker", "fund"]):
        roles.add("Financial Entity")
    if any(k in sec + func for k in ["grid", "power", "energy", "water", "transport", "health", "telecom", "operator", "critical"]):
        roles.add("Operator-of-essential")
        roles.add("Critical Entity")
    if any(k in func for k in ["manufacturer", "firmware", "device", "iot", "embedded", "product"]):
        roles.add("Manufacturer")
    # Most orgs are Controllers/Processors if handling EU personal data
    if systems.get("users_eu", pd.Series(dtype=int)).fillna(0).max() > 0:
        roles.add("Controller"); roles.add("Processor")
    if not roles:
        roles.add("All")
    return sorted(roles)


def applicability_score(requirement_row, roles: List[str], systems: pd.DataFrame) -> float:
    """
    0..1 score combining role match and keyword presence across systems metadata.
    """
    role = str(requirement_row.get("role", "All"))
    role_ok = any(r in role for r in roles) or role == "All"
    base = 0.5 if role_ok else 0.1
    # keywords signal strength
    text = (" ".join(systems.get("sector","").astype(str)) + " " +
            " ".join(systems.get("function","").astype(str))).lower()
    kw = requirement_row.get("applicability_keywords", []) or []
    hit = sum(1 for k in kw if k and k.lower() in text)
    if kw:
        base += 0.5 * min(1.0, hit / max(1, len(kw)//2 or 1))
    return float(max(0.0, min(1.0, base)))


def match_controls(requirement_row, controls: pd.DataFrame) -> pd.DataFrame:
    """
    Return controls with match score 0..1 by keyword overlap.
    """
    req = norm_text(requirement_row["requirement"])
    kw = [norm_text(k) for k in (requirement_row.get("keywords") or [])]
    rows = []
    for _, c in controls.iterrows():
        txt = c["desc_norm"]
        score = 0.0
        if req:
            # simple Jaccard on tokens
            a = set(req.split()); b = set(txt.split())
            j = len(a & b) / max(1, len(a | b))
            score = max(score, j)
        # keyword containment
        if kw:
            hits = sum(1 for k in kw if k and k in txt)
            score = max(score, hits / max(1, len(kw)))
        # bonus for domain/category overlap
        dom = norm_text(str(c.get("domain","")) + " " + str(c.get("subdomain","")))
        if any(k and k in dom for k in kw):
            score = max(score, 0.6 + 0.4 * score)
        if score > 1e-6:
            rows.append({"control_id": c["control_id"], "name": c["name"], "status": c.get("status",""), "maturity": c.get("maturity", np.nan), "score": float(score)})
    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False).head(10)


# ----------------------------- Risk & Priority -----------------------------
def system_risk_weight(systems: pd.DataFrame, incidents: pd.DataFrame) -> pd.Series:
    """
    Compute a 0..1 risk weight per system using criticality, exposure and incident history.
    """
    df = systems.copy()
    crit = df.get("criticality","Low").astype(str).str.lower().map({"high": 1.0, "med": 0.6, "medium":0.6, "low": 0.3}).fillna(0.3)
    exposure = (df.get("internet_exposed", 0).fillna(0)*0.6 + df.get("cloud", 0).fillna(0)*0.4).clip(0,1)
    # incidents severity avg scaled
    sev = pd.Series(0.0, index=df["system_id"])
    if not incidents.empty:
        s = incidents.groupby("system_id")["severity"].mean().clip(lower=0, upper=5) / 5.0
        sev = sev.add(s, fill_value=0.0)
    w = (0.5*crit + 0.3*exposure + 0.2*sev).clip(0,1)
    return w.reindex(df["system_id"]).fillna(0.3)


def requirement_priority(applicability: float, coverage: float, min_maturity: float,
                         sys_weight: float, risk_weights: Dict[str,float]) -> float:
    """
    Priority (higher = do sooner). Combines applicability↑, gap (1-coverage)↑, min maturity↑, and system risk↑.
    """
    gap = max(0.0, 1.0 - coverage)
    base = (0.45 * applicability + 0.35 * gap + 0.20 * (min_maturity/5.0))
    # amplify by risk of affected systems (proxy)
    prio = base * (0.5 + 0.5 * sys_weight)
    return float(max(0.0, min(1.0, prio)))


# ----------------------------- Orchestration -----------------------------
@dataclass
class Config:
    systems: str
    controls: str
    incidents: Optional[str]
    vendors: Optional[str]
    regs: Optional[str]
    asof: str
    base_standard: str
    risk_weights: Dict[str, float]
    effort_default: float
    outdir: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EU cybersecurity mandates — gap analysis & action planning")
    ap.add_argument("--systems", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--incidents", default="")
    ap.add_argument("--vendors", default="")
    ap.add_argument("--regs", default="")
    ap.add_argument("--asof", default="")
    ap.add_argument("--base-standard", default="ISO27001")
    ap.add_argument("--risk-weights", default="sev:0.5,downtime:0.3,exposure:0.2")
    ap.add_argument("--effort-default", type=float, default=5.0)
    ap.add_argument("--outdir", default="out_mandates")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    asof = pd.to_datetime(args.asof) if args.asof else pd.Timestamp.today().normalize()

    systems = load_systems(args.systems)
    controls = load_controls(args.controls)
    incidents = load_incidents(args.incidents)
    vendors = load_vendors(args.vendors)
    catalog = load_catalog(args.regs)

    roles = entity_roles_from_systems(systems)
    sys_risk = system_risk_weight(systems, incidents)
    org_sys_weight = float(sys_risk.mean())

    # For each requirement, do matching
    recs = []
    sys_matrix_rows = []
    vend_rows = []
    report_rows = []
    for _, r in catalog.iterrows():
        appl = applicability_score(r, roles, systems)
        matches = match_controls(r, controls)
        # Coverage = best score adjusted by control status/maturity
        cov = 0.0
        if not matches.empty:
            m = matches.iloc[0]
            status = str(m.get("status","")).lower()
            mat = safe_num(m.get("maturity", 0), 0.0) / 5.0
            status_mult = 1.0 if "implement" in status or "operat" in status else (0.5 if "plan" in status else 0.25)
            cov = float(m["score"]) * (0.6 + 0.4*mat) * status_mult
            cov = max(0.0, min(1.0, cov))
        prio = requirement_priority(appl, cov, r.get("min_maturity", 3), org_sys_weight, {})
        effort = args.effort_default * (1.2 if r.get("min_maturity",3) >= 4 else 1.0) * (1.1 if cov < 0.25 else 0.9)
        overdue = False
        if str(r.get("due_date","")).strip():
            try:
                overdue = pd.to_datetime(r["due_date"]) < asof and cov < 0.9
            except Exception:
                overdue = False

        recs.append({
            "mandate": r["mandate"],
            "requirement_id": r["requirement_id"],
            "requirement": r["requirement"],
            "category": r.get("category",""),
            "applicability": round(appl,3),
            "coverage": round(cov,3),
            "priority": round(prio,3),
            "min_maturity_req": safe_num(r.get("min_maturity",3),3),
            "effort_points": round(float(effort),1),
            "best_control_id": (matches.iloc[0]["control_id"] if not matches.empty else ""),
            "best_control_name": (matches.iloc[0]["name"] if not matches.empty else ""),
            "best_control_status": (matches.iloc[0]["status"] if not matches.empty else ""),
            "reporting_deadline_hours": r.get("reporting_deadline_hours",""),
            "due_date": r.get("due_date",""),
            "overdue": int(bool(overdue)),
            "penalty_max_eur": r.get("penalty_max_eur",""),
            "references": r.get("references",""),
        })

        # System requirement rows (attach to critical systems first)
        for _, srow in systems.sort_values("criticality", ascending=False).iterrows():
            sys_matrix_rows.append({
                "system_id": srow["system_id"],
                "system_name": srow["name"],
                "criticality": srow.get("criticality",""),
                "mandate": r["mandate"],
                "requirement_id": r["requirement_id"],
                "requirement": r["requirement"],
                "applicability": round(appl,3),
                "coverage": round(cov,3),
                "priority": round(prio,3),
            })

        # Vendor obligations (if requirement hints at TPRM or vendor)
        kw_all = " ".join([*(r.get("keywords") or []), *(r.get("applicability_keywords") or [])]).lower()
        if any(k in kw_all for k in ["vendor","third","supplier","outsourc","contract","cloud","service provider","processor"]):
            for _, v in vendors.iterrows():
                vend_rows.append({
                    "vendor_id": v.get("vendor_id",""),
                    "vendor_name": v.get("name",""),
                    "critical": v.get("critical",0),
                    "mandate": r["mandate"],
                    "requirement_id": r["requirement_id"],
                    "requirement": r["requirement"],
                    "ask": "Insert contractual clause / provide assurance evidence",
                    "renewal_date": v.get("contract_renewal_date",""),
                })

        # Reporting matrix
        if r.get("reporting_deadline_hours", np.nan) == r.get("reporting_deadline_hours", np.nan):  # not NaN
            report_rows.append({
                "mandate": r["mandate"],
                "requirement_id": r["requirement_id"],
                "category": r.get("category",""),
                "incident_type_example": "major ICT incident" if r["mandate"] == "DORA" else "significant incident",
                "deadline_hours": r.get("reporting_deadline_hours"),
                "contact": "competent authority / CSIRT (per member state) — fill from playbook",
            })

    gaps = pd.DataFrame(recs).sort_values(["priority","applicability"], ascending=[False, False])
    sys_matrix = pd.DataFrame(sys_matrix_rows)
    vend_tbl = pd.DataFrame(vend_rows).drop_duplicates()
    report_tbl = pd.DataFrame(report_rows).drop_duplicates()

    # Action plan: from gaps where coverage < 0.8 and applicability >= 0.4
    plan_rows = []
    for _, g in gaps.iterrows():
        if g["coverage"] >= 0.8 or g["applicability"] < 0.4:
            continue
        owner = g.get("best_control_name","")
        owner_guess = "CISO" if "incident" in g["category"].lower() else ("Head of IT Ops" if "resilience" in g["category"].lower() else "Risk & Compliance")
        plan_rows.append({
            "mandate": g["mandate"],
            "requirement_id": g["requirement_id"],
            "action": f"Implement/Enhance: {g['requirement']}",
            "owner_suggested": owner_guess,
            "effort_points": g["effort_points"],
            "priority": g["priority"],
            "start_by": str(asof.date()),
            "due_by": g["due_date"] or "",
            "dependencies": g["best_control_id"],
            "evidence_to_collect": "policy/procedure + technical evidence" if "policy" in g["requirement"].lower() else "test report / config export",
        })
    plan = pd.DataFrame(plan_rows).sort_values(["priority","effort_points"], ascending=[False, True])

    # KPIs
    kpi = {
        "asof": str(asof.date()),
        "roles_inferred": roles,
        "requirements_total": int(len(gaps)),
        "requirements_high_prio": int((gaps["priority"] >= 0.7).sum()),
        "avg_coverage": float(round(gaps["coverage"].mean() if len(gaps) else 0.0, 3)),
        "gaps_needing_action": int((gaps["coverage"] < 0.8).sum()),
        "top_mandates_by_priority": gaps.groupby("mandate")["priority"].mean().sort_values(ascending=False).round(3).head(5).to_dict(),
    }

    # Write outputs
    gaps.to_csv(outdir / "gap_register.csv", index=False)
    plan.to_csv(outdir / "action_plan.csv", index=False)
    sys_matrix.to_csv(outdir / "system_requirements.csv", index=False)
    if not vend_tbl.empty:
        vend_tbl.to_csv(outdir / "vendor_requirements.csv", index=False)
    if not report_tbl.empty:
        report_tbl.to_csv(outdir / "reporting_matrix.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))
    (outdir / "config.json").write_text(json.dumps(asdict(Config(
        systems=args.systems, controls=args.controls, incidents=args.incidents or None,
        vendors=args.vendors or None, regs=args.regs or None, asof=str(asof.date()),
        base_standard=args.base_standard,
        risk_weights={kv.split(":")[0]: float(kv.split(":")[1]) for kv in str(args.risk_weights).split(",") if ":" in kv},
        effort_default=float(args.effort_default),
        outdir=args.outdir
    )), indent=2))

    # Console
    print("== EU Cybersecurity Mandates ==")
    print(f"As of: {kpi['asof']}  Roles inferred: {', '.join(roles)}")
    print(f"Requirements: {kpi['requirements_total']}  Avg coverage: {kpi['avg_coverage']:.2f}  Need action: {kpi['gaps_needing_action']}")
    print("Top mandates by avg priority:", kpi["top_mandates_by_priority"])
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()

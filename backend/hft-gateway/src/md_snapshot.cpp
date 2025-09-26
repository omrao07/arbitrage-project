// src/md_snapshot.cpp
#include "md_snapshot.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <sstream>

namespace {

// trim helpers
inline void ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char ch){ return !std::isspace(ch); }));
}
inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](unsigned char ch){ return !std::isspace(ch); }).base(), s.end());
}
inline void trim(std::string& s) { ltrim(s); rtrim(s); }

// fast double/uint64 parse (fallback to stod/stoull if from_chars fails)
inline bool parse_double(const std::string& s, double& out) {
    const char* b = s.data();
    const char* e = b + s.size();
    auto res = std::from_chars(b, e, out);
    if (res.ec == std::errc()) return true;
    try { out = std::stod(s); return true; } catch (...) { return false; }
}
inline bool parse_u64(const std::string& s, std::uint64_t& out) {
    const char* b = s.data();
    const char* e = b + s.size();
    auto res = std::from_chars(b, e, out);
    if (res.ec == std::errc()) return true;
    try { out = static_cast<std::uint64_t>(std::stoull(s)); return true; } catch (...) { return false; }
}

} // namespace

namespace md {

Snapshot::Snapshot(OnQuote cb) : cb_(std::move(cb)) {}

void Snapshot::ingest_line(const std::string& line) {
    // Expected formats:
    //   "SYM,bid,ask,ts"
    //   "SYM,bid,ask"        (ts defaults to 0)
    // Whitespace around fields is ignored. Extra fields are ignored.
    std::string sym, sbid, sask, sts;

    {
        std::stringstream ss(line);
        if (!std::getline(ss, sym, ',')) return;
        if (!std::getline(ss, sbid, ',')) return;
        if (!std::getline(ss, sask, ',')) return;
        std::getline(ss, sts, ','); // optional
    }

    trim(sym); trim(sbid); trim(sask); trim(sts);

    double bid = 0.0, ask = 0.0;
    std::uint64_t ts = 0;

    if (!parse_double(sbid, bid)) return;
    if (!parse_double(sask, ask)) return;
    if (!sts.empty() && !parse_u64(sts, ts)) ts = 0;

    Quote q{bid, ask, ts};

    {
        std::lock_guard<std::mutex> g(mu_);
        book_[sym] = q;
    }

    if (cb_) cb_(sym, q);
}

bool Snapshot::get_quote(const std::string& sym, Quote& out) const {
    std::lock_guard<std::mutex> g(mu_);
    auto it = book_.find(sym);
    if (it == book_.end()) return false;
    out = it->second;
    return true;
}

std::size_t Snapshot::size() const {
    std::lock_guard<std::mutex> g(mu_);
    return book_.size();
}

} // namespace md
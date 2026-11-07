"""Duplicate / template family detection for leakage-aware splitting.

A *family* is a connected component over rows linked by any of:

1. identical normalised fingerprint (``cleaning_fingerprint``: NFKC, lowercase, collapsed
   whitespace),
2. identical template stem (text with a trailing ``(example N)`` marker removed - the
   dataset contains ~200 benign stems repeated with different numbers),
3. MinHash / Jaccard similarity >= ``near_threshold`` on word 3-gram shingles.

Rows whose text has no word tokens (pure punctuation / escape sequences) are **not** linked
by the near-duplicate rule: an empty shingle set would otherwise make every such row look
identical (Jaccard of two empty MinHashes is 1.0) - this is the over-reporting bug seen in
the historical ``leakage_report.json``.

``source`` is never used as a family key.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

from src.data.cleaning import cleaning_fingerprint

DEFAULT_TEMPLATE_PATTERN = r"\s*\(example\s+\d+\)\s*$"
_WORD_RE = re.compile(r"\w+", re.UNICODE)
NUM_PERM = 128


def word_shingles(text: str, n: int = 3) -> set[str]:
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def minhash_for(text: str, num_perm: int = NUM_PERM) -> MinHash | None:
    shingles = word_shingles(text)
    if not shingles:
        return None
    mh = MinHash(num_perm=num_perm)
    for sh in shingles:
        mh.update(sh.encode("utf-8"))
    return mh


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def template_stem(fingerprint: str, pattern: str = DEFAULT_TEMPLATE_PATTERN) -> str | None:
    """Return the stem if the (normalised) text ends with a template marker, else None."""
    stem, n = re.subn(pattern, "", fingerprint, flags=re.IGNORECASE)
    stem = stem.strip()
    return stem if n and stem else None


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class FamilyReport:
    rows: int = 0
    families: int = 0
    singleton_families: int = 0
    largest_family: int = 0
    exact_links: int = 0
    template_links: int = 0
    near_links: int = 0
    rows_without_shingles: int = 0
    near_threshold: float = 0.85
    family_size_histogram: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_families(
    texts: list[str],
    near_threshold: float = 0.85,
    template_pattern: str = DEFAULT_TEMPLATE_PATTERN,
    num_perm: int = NUM_PERM,
    use_near: bool = True,
) -> tuple[list[str], list[str], FamilyReport]:
    """Assign a family id to every text.

    Returns ``(family_ids, fingerprints, report)`` aligned with ``texts``.
    """
    n = len(texts)
    uf = UnionFind(n)
    report = FamilyReport(rows=n, near_threshold=near_threshold)

    fingerprints = [cleaning_fingerprint(t) for t in texts]

    # 1) exact normalised fingerprint
    first_by_fp: dict[str, int] = {}
    for i, fp in enumerate(fingerprints):
        if fp in first_by_fp:
            uf.union(first_by_fp[fp], i)
            report.exact_links += 1
        else:
            first_by_fp[fp] = i

    # 2) template stem
    first_by_stem: dict[str, int] = {}
    for i, fp in enumerate(fingerprints):
        stem = template_stem(fp, template_pattern)
        if stem is None:
            continue
        if stem in first_by_stem:
            uf.union(first_by_stem[stem], i)
            report.template_links += 1
        else:
            first_by_stem[stem] = i
        # A stem that also exists as a standalone text must join the same family.
        if stem in first_by_fp:
            uf.union(first_by_fp[stem], i)

    # 3) near duplicates (MinHash LSH) - one representative per exact fingerprint is enough
    if use_near:
        try:
            lsh = MinHashLSH(threshold=near_threshold, num_perm=num_perm)
        except ValueError:
            lsh = None
        if lsh is not None:
            reps: dict[int, MinHash] = {}
            for fp, idx in first_by_fp.items():
                mh = minhash_for(texts[idx], num_perm=num_perm)
                if mh is None:
                    report.rows_without_shingles += 1
                    continue
                reps[idx] = mh
                lsh.insert(str(idx), mh)
            for idx, mh in reps.items():
                for key in lsh.query(mh):
                    other = int(key)
                    if other != idx and uf.find(other) != uf.find(idx):
                        uf.union(other, idx)
                        report.near_links += 1

    # Deterministic family ids: hash of the lexicographically smallest fingerprint in the component
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        members[uf.find(i)].append(i)
    family_ids = [""] * n
    sizes: Counter[int] = Counter()
    for root, idxs in members.items():
        canonical = min(fingerprints[i] for i in idxs)
        fid = f"fam-{_short_hash(canonical)}"
        for i in idxs:
            family_ids[i] = fid
        sizes[len(idxs)] += 1

    report.families = len(members)
    report.singleton_families = sizes.get(1, 0)
    report.largest_family = max((len(v) for v in members.values()), default=0)
    buckets = {"1": 0, "2-5": 0, "6-20": 0, "21-100": 0, ">100": 0}
    for size, count in sizes.items():
        if size == 1:
            buckets["1"] += count
        elif size <= 5:
            buckets["2-5"] += count
        elif size <= 20:
            buckets["6-20"] += count
        elif size <= 100:
            buckets["21-100"] += count
        else:
            buckets[">100"] += count
    report.family_size_histogram = buckets
    return family_ids, fingerprints, report

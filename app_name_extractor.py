"""
App Name Extractor using rapidfuzz.

Strategy for maximum accuracy + minimum weight:
─────────────────────────────────────────────────
TIER 1 — Curated Dictionary (~500 entries)
  • Top apps worldwide + all their OCR aliases (misspellings, abbreviations).
  • Exact match via dict lookup: O(1), zero false positives.
  
TIER 2 — Full fuzzy index (~41k cleaned names)
  • All unique clean app names from android_apps_names.json.
  • rapidfuzz.process.extractOne with score_cutoff threshold.
  • Only used if Tier 1 misses. Slightly more expensive but still <1ms.

Key design choices:
  • Use rapidfuzz.fuzz.WRatio (weighted ratio) — best for OCR text which may
    have extra/missing chars, different casing, or partial matches.
  • Score cutoff of 85+ for tier2 prevents garbage matches.
  • Minimum text length of 3 chars — prevents matching "OK", "NO", etc.
  • Blocklist of generic words prevents "Settings" → "Settings" (app) matches.
"""

import json
import os
from typing import Optional, Dict, Tuple

# Try to import rapidfuzz
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("WARNING: rapidfuzz not installed. Install with: pip install rapidfuzz")


class AppNameExtractor:
    """Lightweight app-name extractor using rapidfuzz fuzzy matching."""

    def __init__(
        self,
        lookup_path: str = None,
        search_index_path: str = None,
        tier2_score_cutoff: int = 85,
        min_text_length: int = 3,
    ):
        """
        Args:
            lookup_path: Path to app_lookup.json (structured data with aliases).
            search_index_path: Path to app_search_index.json (flat term→app mapping).
            tier2_score_cutoff: Minimum rapidfuzz score for tier2 fuzzy match (0-100).
            min_text_length: Minimum text length to attempt matching.
        """
        self.tier2_score_cutoff = tier2_score_cutoff
        self.min_text_length = min_text_length

        # Resolve default paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if lookup_path is None:
            lookup_path = os.path.join(base_dir, 'app_lookup.json')
        if search_index_path is None:
            search_index_path = os.path.join(base_dir, 'app_search_index.json')

        # ── TIER 1: Exact lookup dictionary ──
        # Maps lowercase search term → canonical app name
        self.exact_lookup: Dict[str, str] = {}
        
        # ── TIER 2: Fuzzy search choices ──
        # List of all unique search terms for rapidfuzz
        self.fuzzy_choices: list = []
        self.fuzzy_choice_to_app: Dict[str, str] = {}  # choice → canonical name

        self._load_data(lookup_path, search_index_path)

    def _load_data(self, lookup_path: str, search_index_path: str):
        """Load and index both data sources."""
        # Load search index for both tier1 (exact) and tier2 (fuzzy)
        if os.path.exists(search_index_path):
            with open(search_index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            for term, info in index.items():
                canonical = info['name']
                self.exact_lookup[term.lower()] = canonical
                self.fuzzy_choice_to_app[term] = canonical
            
            self.fuzzy_choices = list(self.fuzzy_choice_to_app.keys())
            print(f"[AppNameExtractor] Loaded {len(self.exact_lookup)} exact terms, "
                  f"{len(self.fuzzy_choices)} fuzzy choices")
        else:
            print(f"[AppNameExtractor] WARNING: {search_index_path} not found")

    # ─── Blocklist: words that should NEVER match an app ───────────────
    BLOCKLIST = frozenset({
        # UI elements
        'ok', 'yes', 'no', 'cancel', 'done', 'next', 'back', 'open',
        'close', 'send', 'share', 'save', 'delete', 'edit', 'add',
        'new', 'more', 'menu', 'info', 'help', 'about', 'loading',
        'error', 'warning', 'success', 'retry', 'submit', 'confirm',
        'accept', 'decline', 'allow', 'deny', 'skip', 'continue',
        # Login
        'login', 'log in', 'sign in', 'sign up', 'register',
        'password', 'username', 'forgot', 'remember',
        # Status bar
        'am', 'pm', 'wifi', 'lte', '4g', '5g', 'battery',
        # Generic labels
        'home', 'search', 'settings', 'notifications', 'profile',
        'account', 'inbox', 'feed', 'explore', 'discover',
        'trending', 'popular', 'recent', 'favorites', 'history',
    })

    def extract(self, text: str) -> Optional[Dict]:
        """
        Try to match the given OCR text to a known application name.

        Args:
            text: Raw OCR text to match.

        Returns:
            Dict with keys: 'app_name', 'matched_term', 'score', 'tier'
            or None if no match found.
        """
        if not text or not isinstance(text, str):
            return None

        cleaned = text.strip()
        
        # Skip too-short texts
        if len(cleaned) < self.min_text_length:
            return None

        normalized = cleaned.lower()

        # Skip blocklisted terms
        if normalized in self.BLOCKLIST:
            return None

        # ── TIER 1: Exact match ──
        if normalized in self.exact_lookup:
            return {
                'app_name': self.exact_lookup[normalized],
                'matched_term': cleaned,
                'score': 100,
                'tier': 1,
            }

        # ── TIER 2: Fuzzy match ──
        if not RAPIDFUZZ_AVAILABLE or not self.fuzzy_choices:
            return None

        result = process.extractOne(
            normalized,
            self.fuzzy_choices,
            scorer=fuzz.WRatio,
            score_cutoff=self.tier2_score_cutoff,
        )

        if result is not None:
            matched_term, score, _ = result
            return {
                'app_name': self.fuzzy_choice_to_app[matched_term],
                'matched_term': matched_term,
                'score': round(score, 1),
                'tier': 2,
            }

        return None

    def extract_batch(self, texts: list) -> list:
        """
        Extract app names from a batch of OCR texts.

        Args:
            texts: List of OCR text strings.

        Returns:
            List of results (each is a dict or None).
        """
        return [self.extract(t) for t in texts]


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    extractor = AppNameExtractor()

    test_cases = [
        # Exact matches (tier 1)
        "Instagram",
        "WhatsApp",
        "YouTube",
        "Netflix",
        "Spotify",
        "Gmail",
        
        # Aliases (tier 1)
        "IG",
        "WA",
        "YT",
        "FB",
        "Chrome",
        
        # OCR misspellings (tier 2 fuzzy)
        "lnstagram",      # l instead of I
        "Whats App",      # space
        "Spotlfy",        # typo
        "Faceb00k",       # zero instead of O
        "Googl Chrome",   # missing e
        "Tik Tok",        # space
        
        # Should NOT match
        "OK",
        "Cancel",
        "Loading...",
        "Sign In",
        "Settings",
        "12:45 PM",
        "Hello World",
    ]

    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    start = time.perf_counter()
    for text in test_cases:
        result = extractor.extract(text)
        if result:
            print(f"  '{text:20s}' → {result['app_name']:25s} "
                  f"(score={result['score']}, tier={result['tier']})")
        else:
            print(f"  '{text:20s}' → No match")
    elapsed = time.perf_counter() - start
    
    print(f"\n  Total time for {len(test_cases)} lookups: {elapsed*1000:.1f}ms")
    print(f"  Average per lookup: {elapsed/len(test_cases)*1000:.2f}ms")

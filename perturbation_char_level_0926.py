import random
import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import logging
import math
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import string   
import re
# --- helpers 
_WORD = r"[A-Za-z]+(?:[-'][A-Za-z]+)*"          # handles hyphenated words & apostrophes: policy-maker, bank's
_PLACEHOLDER = r"__PHRASE_\d+__"
_PUNCT = r"[^\w\s]"                             # any single non-word, non-space (.,;:!?()[]{}"”’ etc.)
_NUMBER = r"\d+(?:[.,]\d+)*%?|\$\d+(?:[.,]\d+)*"
TOKEN_RX = re.compile(fr"{_PLACEHOLDER}|{_WORD}|{_PUNCT}|{_NUMBER}")

def tokenize(text: str):
    # Returns a list of tokens: words/placeholders/punctuation
    return TOKEN_RX.finditer(text)



def _choose_positions(s: str, 
                      n_edits: int, 
                      max_per_word: int = 2,
                      boundary_skip_p: float = 0.8,
                      perturbation_type: str = "deletion") -> List[int]:
    """
    Select up to n_edits character positions in s to perturb.
    No word contributes more than max_per_word positions.

    boundary_skip_p: probability to skip a boundary character within a word (start/end index).
    """
    if n_edits <= 0 or max_per_word <= 0:
        return []

    # 1) Find non-space "words" as spans (handles multiple spaces/tabs cleanly)

    spans = []  # (start, end, tok)
    for m in TOKEN_RX.finditer(s):
        start, end = m.span()
        tok = s[start:end]
        spans.append((start, end, tok))

    # 2) Collect candidate positions per word
    per_word_positions = []  # list[list[int]]
    for start, end, tok in spans:
        if end <= start:
            per_word_positions.append([])
            continue

        if  _is_mostly_numeric(tok, level = 0.6):
            per_word_positions.append([])
            continue
        if is_punctuation(tok):
            per_word_positions.append([])
            continue
        
        # Helper: is deleting idx going to zero out its token?
        def _would_zero_token(idx: int, start_idx: int, end_idx: int) -> bool:
            if start_idx <= idx < end_idx:
                # token length after deletion
                new_len = (end_idx - start_idx) - 1
                return new_len <= 0
            return False  # idx not in a token (e.g., whitespace between tokens)


        positions = []
        for i in range(start, end):
            ch = s[i]
            if ch.isspace():
                continue
            if _would_zero_token(i, start, end) and perturbation_type == "deletion":
                continue
            if i < end-1:
                if s[i] == s[i+1]:
                    continue

            at_boundary = (i == start) or (i == end - 1)

            # Optionally downweight very short tokens
            if len(tok) <= 2 and random.random() < 0.8:
                continue

            # Optionally skip boundaries
            if at_boundary and random.random() < boundary_skip_p:
                continue

            positions.append(i)

        # 3) Shuffle and hard-cap per word to avoid oversampling later
        if positions:
            random.shuffle(positions)
            per_word_positions.append(positions[:max_per_word])
        else:
            per_word_positions.append([])

    random.shuffle(per_word_positions)
    # 4) Round-robin selection for fairness, respecting global n_edits
    total_available = sum(len(lst) for lst in per_word_positions)
    target = min(n_edits, total_available)

    selected = []
    round_idx = 0
    while len(selected) < target:
        progressed = False
        for lst in per_word_positions:
            if round_idx < len(lst):
                selected.append(lst[round_idx])
                if len(selected) >= target:
                    break
                progressed = True
        if not progressed:
            break
        round_idx += 1

    # Positions are unique by construction; return as-is
    return selected

def _is_mostly_numeric(tok: str, level: float = 0.6) -> bool:
    digits = sum(c.isdigit() for c in tok)
    return digits >= level * max(1, len(tok))


def is_punctuation(token: str) -> bool:
    """Return True if the token is a punctuation character."""
    import string
    # return token in string.punctuation
    return all(ch in string.punctuation for ch in token)


def solve_a_b(n1, n2, d1, d2):

    # Solve for a, b from two anchors for Δ(n) = (a + b ln n)/n
    b = (d1*n1 - d2*n2) / (np.log(n1) - np.log(n2))
    a = d1*n1 - b*np.log(n1)
    return a, b


def cer_estimation(n, i, n0, level="Conservative"):
    mu, sigma = 0.0117, 0.0143 
    scenarios = {
        "Conservative": {"Delta1": 0.050, "Delta2": 0.010},
        "Balanced":     {"Delta1": 0.010, "Delta2": 0.012},
        "Aggressive":   {"Delta1": 0.015, "Delta2": 0.015},
    }
    d1, d2 = scenarios[level]["Delta1"], scenarios[level]["Delta2"]
    a, b = solve_a_b(20, 40, d1, d2)
    # Option B: CER(n|i) = mu + i*sigma + i * (a + b ln n) / n
    return  mu + i*sigma + i * (a + b*np.log(n+n0)) / n



def _expected_edits(n: int, intensity: str) -> float:
    if n <= 1:
        return 0.0
    # Ensure expected edits never decrease as n increases
    n0: int = 5
    mult = {"low": 1.0, "moderate": 2.0, "high": 3.0}.get(intensity, 1.0)
    

    lam = cer_estimation(n, mult, n0) * n
    
    # B: float = 1.5
    # n0: int = 30
    # max_cer: float = 0.05
    # # Core curve: sublinear growth; damped for ultra-short by n0
    # lam = mult * B * math.log(1.0 + n / float(n0))
    return lam

def _integerize_edits(lam: float, mode: str = "stochastic") -> int:
        # Integerize
    if mode == "poisson":
        # Knuth sampler (small lambdas); replace with numpy if available
        if lam <= 0: k = 0
        else:
            L = math.exp(-lam)
            k, p = 0, 1.0
            while p > L:
                k += 1
                p *= random.random()
            k = max(0, k - 1)
    else:
        base = math.floor(lam)
        frac = lam - base
        k = base + (1 if random.random() < frac else 0)
    return k

# --- 3) insertion char from context ---
def _insert_char_for_gap(s: str, # text
                         g: int, # idx
                         typo_map: dict) -> str:
    """Pick an inserted char using local context + typo_map when possible."""

    left  = s[g-1] if g-1 >= 0 else ''
    right = s[g] if g < len(s) else ''
    # Prioritize: duplicate left, else right, else 'e'
    if random.random() < 0.8:
        if left: return left
        if right: return right
    else:
        base  = left if left else right
        lower = base.lower() if base else None

        if lower and typo_map.get(lower):
            c = random.choice(typo_map[lower])
            return c.upper() if (left and left.isupper()) else c

    return 'e'



import numpy as np
import random
import string
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict

class PerturbationType(Enum):
    """Types of character-level perturbations"""
    TYPO = "typo"
    DELETION = "deletion"
    INSERTION = "insertion"
    SUBSTITUTION = "substitution"
    TRANSPOSITION = "transposition"
    DUPLICATION = "duplication"
    CASE_CHANGE = "case_change"
    UNICODE_SUBSTITUTION = "unicode_substitution"
    WHITESPACE_NOISE = "whitespace_noise"
    PUNCTUATION_NOISE = "punctuation_noise"

@dataclass
class TestCase:
    """Represents a single test case"""
    original_query: str
    perturbed_query: str
    perturbation_type: PerturbationType
    intensity: float
    expected_degradation: float

class CharacterPerturbator:
    """Handles character-level perturbations with adjustable intensity"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Common typo mappings
        self.typo_map = {
            'a': ['s', 'q', 'z'],
            'b': ['v', 'n', 'g'],
            'c': ['x', 'v', 'd'],
            'd': ['s', 'f', 'c'],
            'e': ['w', 'r', '3'],
            'f': ['d', 'g', 'r'],
            'g': ['f', 'h', 't'],
            'h': ['g', 'j', 'y'],
            'i': ['u', 'o', '8'],
            'j': ['h', 'k', 'u'],
            'k': ['j', 'l', 'i'],
            'l': ['k', 'p', '1'],
            'm': ['n'],
            'n': ['b', 'm', 'h'],
            'o': ['i', 'p', '0'],
            'p': ['o', 'l', '0'],
            'q': ['w', 'a', '1'],
            'r': ['e', 't', '4'],
            's': ['a', 'd', 'z'],
            't': ['r', 'y', '5'],
            'u': ['y', 'i', '7'],
            'v': ['c', 'b', 'f'],
            'w': ['q', 'e', '2'],
            'x': ['z', 'c', 's'],
            'y': ['t', 'u', '6'],
            'z': ['a', 'x', 's']
        }
        
        # Unicode lookalikes
        self.unicode_map = {
            'a': ['а', 'ɑ', 'α'],
            'e': ['е', 'ε', 'ɛ'],
            'i': ['і', 'ι', 'ɪ'],
            'o': ['о', 'ο', 'σ'],
            'c': ['с', 'ϲ'],
            'p': ['р', 'ρ'],
            'x': ['х', 'χ'],
            'y': ['у', 'γ'],
        }
    
    def apply_perturbation(self, 
                           text: str,
                           perturbation_type: PerturbationType, 
                           intensity: float) -> str:
        """Apply a specific perturbation type with given intensity"""
        if perturbation_type == PerturbationType.TYPO:
            return self._apply_typo(text, intensity)
        elif perturbation_type == PerturbationType.DELETION:
            return self._apply_deletion(text, intensity)
        elif perturbation_type == PerturbationType.INSERTION:
            return self._apply_insertion(text, intensity)
        elif perturbation_type == PerturbationType.SUBSTITUTION:
            return self._apply_substitution(text, intensity)
        elif perturbation_type == PerturbationType.TRANSPOSITION:
            return self._apply_transposition(text, intensity)
        elif perturbation_type == PerturbationType.DUPLICATION:
            return self._apply_duplication(text, intensity)
        elif perturbation_type == PerturbationType.CASE_CHANGE:
            return self._apply_case_change(text, intensity)
        elif perturbation_type == PerturbationType.UNICODE_SUBSTITUTION:
            return self._apply_unicode_substitution(text, intensity)
        elif perturbation_type == PerturbationType.WHITESPACE_NOISE:
            return self._apply_whitespace_noise(text, intensity)
        elif perturbation_type == PerturbationType.PUNCTUATION_NOISE:
            return self._apply_punctuation_noise(text, intensity)
        else:
            return text
    

    def _apply_typo(self, text: str, intensity: str, mode: str = "stochastic") -> str:
        """Simulate keyboard typos with length-aware intensity and realism."""
        if not text:
            return text

        # ---- tunables (adjust to taste) ----
        max_cer = 0.08       # hard cap on edits as a fraction of length
        min_len_for_forced_edit = 5  # don't force edits for very short strings
        boundary_skip_p = 0.8 # avoid first/last char of a word ~70% of the time
        # ------------------------------------

        n_chars = len(text) 

        # Mutate a single character using typo_map; preserve case
        def _mutate_char(orig: str) -> str:
            lower = orig.lower()
            if lower in getattr(self, "typo_map", {}) and self.typo_map[lower]:
                repl = random.choice(self.typo_map[lower])
                return repl.upper() if orig.isupper() else repl
            return orig  # no mapping -> leave unchanged

        # 1) sample number of edits
        lam = _expected_edits(n_chars, intensity)
        edits = _integerize_edits(lam, mode)
        # 2) clamp by global CER cap
        max_edits = max(1, int(n_chars * max_cer))

        edits = min(edits, max_edits)
  
        # 3) avoid over-perturbing very short queries
        if n_chars < min_len_for_forced_edit:
            edits = 1 if random.random() < 0.1 else 0

        if edits <= 0:
            return text

        # 4) pick positions and apply edits
        positions = _choose_positions(text, edits)
        if not positions:
            return text

        chars = list(text)
        for idx in positions:
            chars[idx] = _mutate_char(chars[idx])

        return ''.join(chars)



    def _apply_deletion(self, text: str, 
                        intensity: str, 
                        max_cer: float = 0.08) -> str:

        min_len_for_forced_edit = 10

        n_chars = len(text)

        lam = _expected_edits(n_chars, intensity)

        edits = _integerize_edits(lam)

        max_edits = max(1, int(n_chars * max_cer))

        edits = min(edits, max_edits)

        if n_chars < min_len_for_forced_edit:
            edits = 1 if random.random() < 0.1 else 0
     
        if edits <= 0:
            return text

        positions = _choose_positions(text, edits, boundary_skip_p=0.9)
        chars = list(text)
        for idx in sorted(positions, reverse=True):  # prevent index shift
            if 0 <= idx < len(chars):
                del chars[idx]
        out = ''.join(chars)

        return out


    def _apply_insertion(self, 
                         text: str, 
                         intensity: str, 
                         mode: str = "stochastic") -> str:
        if not text:
            return text
        """Insert realistic characters (adjacent on keyboard)"""
        min_len_for_forced_edit = 10
        max_cer = 0.08       # hard cap on edits as a fraction of length

        n_chars = len(text)

        # 1) sample number of edits
        lam = _expected_edits(n_chars, intensity)
        edits = _integerize_edits(lam, mode)
        # 2) clamp by global CER cap
        max_edits = max(1, int(n_chars * max_cer))
        edits = min(edits, max_edits)
        # 3) avoid over-perturbing very short queries
        if n_chars < min_len_for_forced_edit:
            edits = 1 if random.random() < 0.1 else 0

        if edits <= 0:
            return text

        # 4) pick positions and apply edits
        positions = _choose_positions(text, edits, boundary_skip_p=1.0)
        if not positions:
            return text
        
        chars = list(text)
        for idx in sorted(positions, reverse=True):  # prevent index shift
            if 0 <= idx < len(chars):
                insert_char = _insert_char_for_gap(text, idx, self.typo_map)
                chars.insert(idx, insert_char)

        return ''.join(chars)


    def _apply_substitution(self, text: str, intensity: float) -> str:
        """Substitute random characters"""
        chars = list(text)
        num_substitutions = max(1, int(len(chars) * intensity))
        
        for _ in range(num_substitutions):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = random.choice(string.ascii_letters)
        
        return ''.join(chars)
    
    def _apply_transposition(self, 
                             text: str, 
                             intensity: str,
                             mode: str = "stochastic",
                             boundary_skip_p: float = 0.9) -> str:
        """Transpose adjacent characters"""
        if not text:
            return text
        """Insert realistic characters (adjacent on keyboard)"""
        min_len_for_forced_edit = 4
        max_cer = 0.08       # hard cap on edits as a fraction of length
        max_per_word: int = 2  # max transpositions per word
        n_chars = len(text)

        # 1) sample number of edits
        lam = _expected_edits(n_chars, intensity)
        edits = _integerize_edits(lam, mode)
        # 2) clamp by global CER cap
        max_edits = max(1, int(n_chars * max_cer))
        edits = min(edits, max_edits)
        # 3) avoid over-perturbing very short queries
        if n_chars < min_len_for_forced_edit:
            edits = 1 if random.random() < 0.8 else 0

        if edits <= 0:
            return text


        spans = []  # (start, end, tok)
        for m in TOKEN_RX.finditer(text):
            start, end = m.span()
            tok = text[start:end]
            spans.append((start, end, tok))

        per_word_pairs = []  # each entry: list of valid start indices i to swap (i,i+1)
        for start, end, tok in spans:
            if end <= start:
               per_word_pairs.append([])
               continue
            # If token is mostly digits, always allow transposition
            if _is_mostly_numeric(tok, level=0.6):
                starts = []
                for i in range(start, end - 1):
                    if text[i].isspace() or text[i+1].isspace():
                        continue
                    starts.append(i)
                random.shuffle(starts)
                chosen = []
                blocked = set()
                for i in starts:
                    if i in blocked or (i - 1) in blocked or (i + 1) in blocked:
                        continue
                    chosen.append(i)
                    blocked.update({i - 1, i, i + 1})
                    if len(chosen) >= max_per_word:
                        break
                per_word_pairs.append(chosen)
                continue


            if end - start < 2 or is_punctuation(tok):
                per_word_pairs.append([])
                continue

            starts = []
            for i in range(start, end - 1):
                # avoid whitespace & boundaries
                if text[i].isspace() or text[i+1].isspace():
                    continue
                at_left_boundary  = (i == start)
                at_right_boundary = (i + 1 == end - 1)
                if (at_left_boundary or at_right_boundary) and random.random() < boundary_skip_p:
                    continue
                # skip identical pair to increase visible effect, e.g., "ee"
                if text[i] == text[i+1]:
                    continue
                starts.append(i)

            random.shuffle(starts)
            # take up to max_per_word *non-overlapping* starts
            chosen = []
            blocked = set()
            for i in starts:
                if i in blocked or (i - 1) in blocked or (i + 1) in blocked:
                    continue
                chosen.append(i)
                # block neighbors so pairs don't overlap (i,i+1) conflicts with i-1 and i+1
                blocked.update({i - 1, i, i + 1})
                if len(chosen) >= max_per_word:
                    break

            per_word_pairs.append(chosen)
        
        random.shuffle(per_word_pairs)
        total_avail = sum(len(lst) for lst in per_word_pairs)
        target = min(edits, total_avail)
        selected, rr = [], 0
        while len(selected) < target:
            progressed = False
            for lst in per_word_pairs:
                if rr < len(lst):
                    selected.append(lst[rr])
                    if len(selected) >= target:
                        break
                    progressed = True
            if not progressed:
                break
            rr += 1

        if not selected:
            return text
        # ----- apply swaps -----
        # Swapping adjacent chars does not change length → no index shift issues.
        chars = list(text)
        # Sort ascending just for determinism; overlaps already prevented
        # print("Selected positions for transposition:", selected)
        for i in sorted(selected):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

        return ''.join(chars)
    
    def _apply_duplication(self, text: str, intensity: float) -> str:
        """Duplicate random characters"""
        chars = list(text)
        num_duplications = max(1, int(len(chars) * intensity))
        
        for _ in range(num_duplications):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            chars.insert(idx, chars[idx])
        
        return ''.join(chars)
    
    def _apply_case_change(self, text: str, intensity: float) -> str:
        """Randomly change case of characters"""
        chars = list(text)
        num_changes = max(1, int(len(chars) * intensity))
        
        for _ in range(num_changes):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            if chars[idx].isalpha():
                chars[idx] = chars[idx].swapcase()
        
        return ''.join(chars)
    
    def _apply_unicode_substitution(self, text: str, intensity: float) -> str:
        """Replace characters with unicode lookalikes"""
        chars = list(text)
        num_substitutions = max(1, int(len(chars) * intensity))
        
        for _ in range(num_substitutions):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            char = chars[idx].lower()
            
            if char in self.unicode_map:
                chars[idx] = random.choice(self.unicode_map[char])
        
        return ''.join(chars)
    
    def _apply_whitespace_noise(self, text: str, intensity: float) -> str:
        """Add or remove whitespace"""
        words = text.split()
        
        if random.random() < 0.5:
            # Add extra spaces
            num_additions = max(1, int(len(words) * intensity))
            for _ in range(num_additions):
                if not words:
                    break
                idx = random.randint(0, len(words) - 1)
                words[idx] = words[idx] + ' '
        else:
            # Remove spaces
            if len(words) > 1:
                num_merges = max(1, int((len(words) - 1) * intensity))
                for _ in range(num_merges):
                    if len(words) <= 1:
                        break
                    idx = random.randint(0, len(words) - 2)
                    words[idx] = words[idx] + words[idx + 1]
                    del words[idx + 1]
        
        return ' '.join(words)
    
    def _apply_punctuation_noise(self, text: str, intensity: float) -> str:
        """Add, remove, or change punctuation"""
        chars = list(text)
        punctuation = string.punctuation
        num_changes = max(1, int(len(chars) * intensity))
        
        for _ in range(num_changes):
            action = random.choice(['add', 'remove', 'change'])
            
            if action == 'add':
                idx = random.randint(0, len(chars))
                chars.insert(idx, random.choice(punctuation))
            elif action == 'remove':
                punct_indices = [i for i, c in enumerate(chars) if c in punctuation]
                if punct_indices:
                    idx = random.choice(punct_indices)
                    del chars[idx]
            elif action == 'change':
                punct_indices = [i for i, c in enumerate(chars) if c in punctuation]
                if punct_indices:
                    idx = random.choice(punct_indices)
                    chars[idx] = random.choice(punctuation)
        
        return ''.join(chars)

class HybridIRModelTester:
    """Main testing framework for hybrid IR models"""
    
    def __init__(self, model_interface: Callable):
        """
        Initialize tester with model interface
        
        Args:
            model_interface: Function that takes query and returns ranked results
                            Should return List[Tuple[doc_id, score]]
        """
        self.model = model_interface
        self.perturbator = CharacterPerturbator()
        self.test_results = defaultdict(list)
    
    def generate_test_cases(self, queries: List[str], 
                           perturbation_types: List[PerturbationType] = None,
                           intensity_levels: List[float] = None) -> List[TestCase]:
        """Generate test cases with various perturbations"""
        if perturbation_types is None:
            perturbation_types = list(PerturbationType)
        
        if intensity_levels is None:
            intensity_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
        
        test_cases = []
        
        for query in queries:
            for p_type, intensity_levels in perturbation_types.items():
                for intensity in intensity_levels:
                    perturbed = self.perturbator.apply_perturbation(
                        query, p_type, intensity
                    )
                    
                    # Expected degradation is proportional to intensity
                    # This is a simple heuristic; adjust based on your model
                    expected_degradation = intensity
                    
                    test_cases.append(TestCase(
                        original_query=query,
                        perturbed_query=perturbed,
                        perturbation_type=p_type,
                        intensity=intensity,
                        expected_degradation=expected_degradation
                    ))
        
        return test_cases
    
    

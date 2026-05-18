"""
==============================================================
  ViBroBraille - Braille Encoder
==============================================================

WHAT THIS DOES:
  Converts plain text into Unicode Braille characters.
  Each letter, number, and symbol maps to a Unicode Braille cell
  in the range U+2800–U+28FF.

BRAILLE BASICS:
  A Braille cell is a 2x3 grid of 6 dots:
      Dot 1  Dot 4
      Dot 2  Dot 5
      Dot 3  Dot 6

  We support:
    - Grade 1 Braille: letter-for-letter mapping (implemented here)
    - Numbers with numeric indicator prefix (⠼)
    - Capital indicator prefix (⠠) for uppercase letters
    - Basic punctuation

UNICODE BRAILLE:
  Python represents Braille as Unicode characters U+2800–U+28FF.
  Each character's bit pattern encodes which dots are raised.

  Bit layout:
    bit 0 → dot 1
    bit 1 → dot 2
    bit 2 → dot 3
    bit 3 → dot 4
    bit 4 → dot 5
    bit 5 → dot 6
    bit 6 → dot 7  (not used in Grade 1)
    bit 7 → dot 8  (not used in Grade 1)

  Example:
    Letter 'a' = dot 1 only → bit 0 set → 0b000001 → U+2801 → ⠁

NO EXTRA DEPENDENCIES NEEDED: This module uses only Python stdlib.
"""

from typing import List


class BrailleEncoder:
    """
    Encodes plain text strings into Unicode Braille.

    Usage:
        encoder = BrailleEncoder()
        result  = encoder.encode("Hello World")

        print(result["unicode"])        # ⠠⠓⠑⠇⠇⠕ ⠠⠺⠕⠗⠇⠙
        print(result["cells"])          # [{'char':'H','braille':'⠠⠓'}, ...]
        print(result["dot_patterns"])   # [{'char':'H','dots':[1,2,3,...]}, ...]
    """

    # ── Grade 1 Braille Alphabet (Unicode code point offsets from U+2800) ──
    # Each value is the integer that, when added to 0x2800, gives the Braille char
    # Dots encoded as bitmask: bit0=dot1, bit1=dot2, bit2=dot3, bit3=dot4, bit4=dot5, bit5=dot6

    ALPHA_MAP = {
        'a': 0b000001,   # ⠁  dot 1
        'b': 0b000011,   # ⠃  dots 1,2
        'c': 0b001001,   # ⠉  dots 1,4
        'd': 0b011001,   # ⠙  dots 1,4,5
        'e': 0b010001,   # ⠑  dots 1,5
        'f': 0b001011,   # ⠋  dots 1,2,4
        'g': 0b011011,   # ⠛  dots 1,2,4,5
        'h': 0b010011,   # ⠓  dots 1,2,5
        'i': 0b001010,   # ⠊  dots 2,4
        'j': 0b011010,   # ⠚  dots 2,4,5
        'k': 0b000101,   # ⠅  dots 1,3
        'l': 0b000111,   # ⠇  dots 1,2,3
        'm': 0b001101,   # ⠍  dots 1,3,4
        'n': 0b011101,   # ⠝  dots 1,3,4,5
        'o': 0b010101,   # ⠕  dots 1,3,5
        'p': 0b001111,   # ⠏  dots 1,2,3,4
        'q': 0b011111,   # ⠟  dots 1,2,3,4,5
        'r': 0b010111,   # ⠗  dots 1,2,3,5
        's': 0b001110,   # ⠎  dots 2,3,4
        't': 0b011110,   # ⠞  dots 2,3,4,5
        'u': 0b100101,   # ⠥  dots 1,3,6
        'v': 0b100111,   # ⠧  dots 1,2,3,6
        'w': 0b111010,   # ⠺  dots 2,4,5,6
        'x': 0b101101,   # ⠭  dots 1,3,4,6
        'y': 0b111101,   # ⠽  dots 1,3,4,5,6
        'z': 0b110101,   # ⠵  dots 1,3,5,6
    }

    # Numbers use the same patterns as a–j but with a numeric indicator prefix
    # Numeric indicator = dots 3,4,5,6 = 0b111100
    NUMERIC_INDICATOR = 0b111100  # ⠼

    NUMBER_MAP = {
        '1': ALPHA_MAP['a'],
        '2': ALPHA_MAP['b'],
        '3': ALPHA_MAP['c'],
        '4': ALPHA_MAP['d'],
        '5': ALPHA_MAP['e'],
        '6': ALPHA_MAP['f'],
        '7': ALPHA_MAP['g'],
        '8': ALPHA_MAP['h'],
        '9': ALPHA_MAP['i'],
        '0': ALPHA_MAP['j'],
    }

    # Capital indicator = dot 6 = 0b100000
    CAPITAL_INDICATOR = 0b100000  # ⠠

    # Punctuation mappings
    PUNCT_MAP = {
        ' ':  None,         # space → Braille space (U+2800, empty cell)
        '.':  0b010110,     # ⠲  dots 2,5,6
        ',':  0b000010,     # ⠂  dot 2
        '?':  0b100110,     # ⠦  dots 2,3,6
        '!':  0b010110,     # ⠖  dots 2,3,5  (same as period in many systems)
        ';':  0b000110,     # ⠆  dots 2,3
        ':':  0b010010,     # ⠒  dots 2,5
        '-':  0b100100,     # ⠤  dots 3,6
        "'":  0b000100,     # ⠄  dot 3
        '"':  0b010110,     # ⠶  dots 2,3,5,6
        '(':  0b110110,     # ⠶  opening paren
        ')':  0b110110,     # ⠶  closing paren
        '\n': None,         # newline → space in Braille
    }

    def __init__(self):
        """Initialize encoder with combined character lookup."""
        # Build a unified lookup: lowercase char → bitmask
        self._lookup = {}
        self._lookup.update(self.ALPHA_MAP)
        self._lookup.update(self.PUNCT_MAP)

    # ── Public API ────────────────────────────────────────

    def encode(self, text: str) -> dict:
        """
        Encode a text string into Unicode Braille.

        Args:
            text: Plain text string (any case)

        Returns:
            dict:
                unicode      → full Braille string (Unicode chars)
                cells        → list of {'char': X, 'braille': '⠠⠓', 'index': N}
                dot_patterns → list of {'char': X, 'dots': [1,2,3]}
                char_count   → number of input characters
                cell_count   → number of Braille cells produced
        """
        cells         = []
        unicode_parts = []
        dot_patterns  = []
        index         = 0

        i = 0
        while i < len(text):
            char = text[i]

            # ── Handle numbers ────────────────────────────
            if char.isdigit():
                # Collect the whole number sequence
                num_str = ""
                while i < len(text) and text[i].isdigit():
                    num_str += text[i]
                    i += 1

                # Add numeric indicator before number sequence
                indicator_char = chr(0x2800 + self.NUMERIC_INDICATOR)
                unicode_parts.append(indicator_char)
                cells.append({
                    "char":    "#",
                    "braille": indicator_char,
                    "index":   index,
                    "type":    "numeric_indicator"
                })
                index += 1

                # Encode each digit
                for digit in num_str:
                    bitmask = self.NUMBER_MAP.get(digit)
                    if bitmask is not None:
                        braille_char = chr(0x2800 + bitmask)
                        unicode_parts.append(braille_char)
                        dots         = self._bitmask_to_dots(bitmask)
                        cells.append({
                            "char":    digit,
                            "braille": braille_char,
                            "index":   index,
                            "type":    "number",
                            "dots":    dots
                        })
                        dot_patterns.append({"char": digit, "dots": dots})
                        index += 1

                continue  # already advanced i

            # ── Handle uppercase letters ──────────────────
            if char.isupper():
                # Insert capital indicator
                cap_char = chr(0x2800 + self.CAPITAL_INDICATOR)
                unicode_parts.append(cap_char)
                cells.append({
                    "char":    "^",
                    "braille": cap_char,
                    "index":   index,
                    "type":    "capital_indicator"
                })
                index += 1
                char = char.lower()  # now encode as lowercase

            # ── Handle space ──────────────────────────────
            if char == " " or char == "\n":
                space_char = chr(0x2800)  # empty Braille cell = space
                unicode_parts.append(space_char)
                cells.append({
                    "char":    " ",
                    "braille": space_char,
                    "index":   index,
                    "type":    "space"
                })
                index += 1
                i += 1
                continue

            # ── Handle letters ────────────────────────────
            if char in self.ALPHA_MAP:
                bitmask      = self.ALPHA_MAP[char]
                braille_char = chr(0x2800 + bitmask)
                dots         = self._bitmask_to_dots(bitmask)
                unicode_parts.append(braille_char)
                cells.append({
                    "char":    char,
                    "braille": braille_char,
                    "index":   index,
                    "type":    "letter",
                    "dots":    dots
                })
                dot_patterns.append({"char": char, "dots": dots})
                index += 1
                i += 1
                continue

            # ── Handle punctuation ────────────────────────
            if char in self.PUNCT_MAP:
                bitmask = self.PUNCT_MAP[char]
                if bitmask is not None:
                    braille_char = chr(0x2800 + bitmask)
                    dots         = self._bitmask_to_dots(bitmask)
                    unicode_parts.append(braille_char)
                    cells.append({
                        "char":    char,
                        "braille": braille_char,
                        "index":   index,
                        "type":    "punctuation",
                        "dots":    dots
                    })
                    index += 1
                i += 1
                continue

            # ── Unknown character → skip ──────────────────
            i += 1

        return {
            "unicode":      "".join(unicode_parts),
            "cells":        cells,
            "dot_patterns": dot_patterns,
            "char_count":   len(text),
            "cell_count":   len(cells)
        }

    def encode_char(self, char: str) -> str:
        """
        Encode a single character to its Braille Unicode equivalent.

        Args:
            char: Single character string

        Returns:
            Braille Unicode character string (may be 2 chars if capital/number)
        """
        result = self.encode(char)
        return result["unicode"]

    def get_dot_pattern(self, char: str) -> List[int]:
        """
        Get the dot numbers raised for a given character.

        Args:
            char: Single lowercase letter or digit

        Returns:
            List of dot numbers (1–6) that are raised
        """
        char = char.lower()
        if char in self.ALPHA_MAP:
            return self._bitmask_to_dots(self.ALPHA_MAP[char])
        if char in self.NUMBER_MAP:
            return self._bitmask_to_dots(self.NUMBER_MAP[char])
        return []

    # ── Helpers ───────────────────────────────────────────

    def _bitmask_to_dots(self, bitmask: int) -> List[int]:
        """
        Convert a 6-bit bitmask to a list of dot numbers (1-indexed).

        Example:
            0b000001 → [1]        (dot 1 only → letter 'a')
            0b000011 → [1, 2]     (dots 1,2 → letter 'b')
            0b001001 → [1, 4]     (dots 1,4 → letter 'c')
        """
        dot_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
        dots    = []
        for bit_pos, dot_num in dot_map.items():
            if bitmask & (1 << bit_pos):
                dots.append(dot_num)
        return sorted(dots)

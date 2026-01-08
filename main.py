"""
Poke Battle pygame driver.

This file contains the pygame UI, animation, and audio glue that sits on top of the battle engine.
Edits made by assistant: added explanatory comments and small safe cleanup of a few redundant lines.
"""

# Standard library
import copy
import io
import math
import os
from pathlib import Path
import random
import threading
import operator as op

# Third party
import pygame
import requests
import torch

# Local project modules
import data
from env import BattleEnv, PROC
from model import QNet

# ========= Global constants and configuration =========

# ASCII hyphen character. Used elsewhere to normalize text without typing a literal dash.
# Example use: replacing it in move names or log text.
DASH = chr(45)

# UI sizing
ICON_PX = 120        # Square size in pixels for small icons (party icons, move icons, etc).
ROW_H = 100          # Height in pixels for one UI row (menus, logs, selection rows).
ANIM_DELAY_MS = 1250 # Delay used to pace battle animation steps, in milliseconds.

# Asset directories
SPRITE_DIR = "sprites"              # Base folder for sprite assets (general).
SPRITE_FRONT_DIR = "sprites_front"  # Folder for front facing Pokémon sprites (opponent facing you).
SPRITE_BACK_DIR = "sprites_back"    # Folder for back facing Pokémon sprites (your Pokémon from behind).

# Battle rendering size reference
BATTLE_PX = 220      # Base pixel size used as a scale reference for battle sprites and FX sizing.

# Sound effects folder
SFX_DIR = Path("sound_effects")     # Directory containing sound effect files.

# FX surface caches
# _fx_raw: original loaded images (usually with alpha), stored once to avoid disk reloads.
# _fx_surfaces: resized variants of those images keyed by (name, width) or similar, to avoid re scaling every frame.
_fx_surfaces = {}
_fx_raw = {}

# Cache for pre rendered "steel link" surfaces used by the steel animation.
# Key is typically (link_w, link_h, alpha) so the same link graphics are reused without re drawing.
_steel_link_cache = {}

# 1
def _draw_capsule(surf, x, y, w, h, col):
    """
    Draw a pill / capsule shape (a rectangle with fully rounded ends).

    Used all over the UI and FX because it looks cleaner than sharp rectangles:
    HP bars, buttons, and the "steel link" pieces are basically capsules stacked.
    """
    # Defensive sizing: pygame does not like 0 or negative sizes.
    w = int(max(1, w))
    h = int(max(1, h))

    # Capsule radius is half the smaller side:
    # if w >= h, the ends are circles of radius h/2.
    # if h > w, the ends are circles of radius w/2 (vertical pill).
    rad = int(max(1, op.truediv(min(w, h), 2)))

    if w >= h:
        # Horizontal capsule:
        # 1) draw the middle rectangle (no rounding)
        # 2) draw a circle at the left end
        # 3) draw a circle at the right end
        pygame.draw.rect(
            surf,
            col,
            pygame.Rect(
                int(op.add(x, rad)),
                int(y),
                int(op.sub(w, op.mul(2, rad))),
                int(h),
            ),
        )
        pygame.draw.circle(surf, col, (int(op.add(x, rad)), int(op.add(y, rad))), rad)
        pygame.draw.circle(surf, col, (int(op.add(x, op.sub(w, rad))), int(op.add(y, rad))), rad)
    else:
        # Vertical capsule (same logic, rotated):
        # 1) draw the middle rectangle
        # 2) draw a circle at the top end
        # 3) draw a circle at the bottom end
        pygame.draw.rect(
            surf,
            col,
            pygame.Rect(
                int(x),
                int(op.add(y, rad)),
                int(w),
                int(op.sub(h, op.mul(2, rad))),
            ),
        )
        pygame.draw.circle(surf, col, (int(op.add(x, rad)), int(op.add(y, rad))), rad)
        pygame.draw.circle(surf, col, (int(op.add(x, rad)), int(op.add(y, op.sub(h, rad)))), rad)


# 2
def steel_link_surface(w, h, alpha):
    """
    Build (or fetch from cache) a small "steel chain link" sprite.

    This is used in the Steel move animation to draw a ring of interlocking
    links around the target. Caching matters because this function can be
    called many times per frame.
    """
    # Clamp sizes so the link never degenerates into a tiny unreadable blob.
    w = int(max(6, w))
    h = int(max(6, h))

    # Clamp alpha to pygame valid range.
    a = int(max(0, min(255, alpha)))

    # Cache key includes size and alpha, since those change the actual pixels.
    key = (w, h, a)
    if key in _steel_link_cache:
        return _steel_link_cache[key]

    # Create a transparent surface for the link.
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    # Three tone palette to fake metal:
    # outer: dark rim, mid: main body, hi: highlight strip.
    outer = (55, 60, 72, a)
    mid = (120, 128, 145, a)
    hi = (210, 215, 225, int(max(0, min(255, op.mul(a, 0.65)))))

    # Outer capsule is the link boundary.
    _draw_capsule(surf, 0, 0, w, h, outer)

    # Inset pass: draw a slightly smaller capsule inside for thickness shading.
    inset1 = int(max(1, op.truediv(min(w, h), 8)))
    _draw_capsule(
        surf,
        inset1,
        inset1,
        int(op.sub(w, op.mul(2, inset1))),
        int(op.sub(h, op.mul(2, inset1))),
        mid,
    )

    # Inner hole: draw a fully transparent capsule to carve out the center.
    # inset2 is larger than inset1 to ensure a visible "ring" thickness.
    inset2 = int(max(op.add(inset1, 2), op.truediv(min(w, h), 4)))
    hole = (0, 0, 0, 0)
    _draw_capsule(
        surf,
        inset2,
        inset2,
        int(op.sub(w, op.mul(2, inset2))),
        int(op.sub(h, op.mul(2, inset2))),
        hole,
    )

    # Small highlight line near the top edge to sell the metal specular.
    pygame.draw.line(
        surf,
        hi,
        (int(op.add(inset1, 1)), int(op.add(inset1, 2))),
        (int(op.sub(w, op.add(inset1, 2))), int(op.add(inset1, 2))),
        2,
    )

    # Store in cache and return.
    _steel_link_cache[key] = surf
    return surf


# 3
def make_near_color_transparent(surf, key_rgb, tol):
    """
    Simple chroma key.

    Any pixel whose (r,g,b) is within tol of key_rgb gets alpha set to 0.
    This is a cheap way to remove solid colored backgrounds from PNGs
    when the asset is not already exported with transparency.
    """
    # Target color to remove.
    key_r, key_g, key_b = int(key_rgb[0]), int(key_rgb[1]), int(key_rgb[2])

    # Image dimensions.
    w = surf.get_width()
    h = surf.get_height()

    # Ensure we have per pixel alpha (RGBA) to edit.
    if surf.get_flags() & pygame.SRCALPHA:
        out = surf.copy()
    else:
        out = surf.convert_alpha()

    # Walk pixels and knock out those close to the key color.
    # Note: this is O(w*h), so keep assets small or cache results.
    for y in range(h):
        for x in range(w):
            r, g, b, a = out.get_at((x, y))

            # Per channel difference check.
            if (
                abs(op.sub(int(r), key_r)) <= tol
                and abs(op.sub(int(g), key_g)) <= tol
                and abs(op.sub(int(b), key_b)) <= tol
            ):
                # Keep RGB but set alpha to 0 (fully transparent).
                out.set_at((x, y), (r, g, b, 0))

    return out


# 4
def get_fx_raw_surface(name):
    """
    Load and cache an effect image (original size).

    This returns the "raw" surface (not scaled), but still applies chroma key
    transparency using the top left pixel as the background color.
    """
    # Normalized cache key so "Leaf" and "leaf" resolve to the same asset.
    key = str(name).strip().lower()
    if key in _fx_raw:
        return _fx_raw[key]

    # FX assets live under special_effects/<name>.png
    p = Path("special_effects") / (key + ".png")
    if not p.exists():
        # Cache the miss so we do not hit disk every frame.
        _fx_raw[key] = None
        return None

    try:
        # Load then convert to a per pixel alpha surface for fast blits.
        raw = pygame.image.load(str(p))
        surf = raw.convert_alpha()

        # Use the top left pixel as the "background" color to remove.
        # This works if the image has a uniform background.
        bg = surf.get_at((0, 0))
        bg_rgb = (int(bg.r), int(bg.g), int(bg.b))

        # Remove pixels near the background color.
        surf = make_near_color_transparent(surf, bg_rgb, tol=40)

        # Cache and return.
        _fx_raw[key] = surf
        return surf
    except Exception:
        # Fail safe: cache None to avoid repeated crashing loads.
        _fx_raw[key] = None
        return None


# 5
def get_fx_surface(name, target_w):
    """
    Load + cache a scaled FX surface.

    This is the "give me a sprite at width target_w" helper.
    It loads the PNG, chroma keys the background, then smooth scales it.
    The result is cached by (name, target_w).
    """
    # Cache key includes the normalized filename + the requested width.
    key = (str(name).strip().lower(), int(target_w))
    if key in _fx_surfaces:
        return _fx_surfaces[key]

    # Asset path.
    p = Path("special_effects") / (key[0] + ".png")
    if not p.exists():
        _fx_surfaces[key] = None
        return None

    try:
        # Load + alpha convert for faster blit.
        raw = pygame.image.load(str(p))
        surf = raw.convert_alpha()

        # Top left pixel is treated as background for chroma key.
        bg = surf.get_at((0, 0))
        bg_rgb = (int(bg.r), int(bg.g), int(bg.b))

        # Remove background with tolerance.
        surf = make_near_color_transparent(surf, bg_rgb, tol=40)

    except Exception:
        _fx_surfaces[key] = None
        return None

    # If the surface is broken for some reason, do not scale.
    w, h = surf.get_size()
    if w <= 0 or h <= 0:
        _fx_surfaces[key] = surf
        return surf

    # Scale uniformly based on width.
    scale = float(target_w) / float(w)
    new_w = int(target_w)
    new_h = int(h * scale)

    # Smoothscale gives nicer results than scale for sprites.
    surf2 = pygame.transform.smoothscale(surf, (new_w, new_h))

    # Cache and return the scaled sprite.
    _fx_surfaces[key] = surf2
    return surf2


# 6
def find_sfx_file(stem):
    """
    Find the first existing SFX file for a given stem inside SFX_DIR.

    Tries wav then mp3 then ogg. Returns a Path or None.
    """
    # Normalize input so " Steel_Snap " still works.
    stem = str(stem).strip().lower()

    # Try extensions in priority order.
    for ext in ("wav", "mp3", "ogg"):
        p = SFX_DIR / f"{stem}.{ext}"
        if p.exists():
            return p

    # Nothing found.
    return None


# 7
def safe_sound(path_obj):
    """
    Safely load a pygame sound.

    Returns a pygame.mixer.Sound if it works, otherwise returns None.
    This lets the game run even if some SFX files are missing or broken.
    """
    if path_obj is None:
        return None
    try:
        return pygame.mixer.Sound(str(path_obj))
    except Exception:
        return None


# 8
def build_status_sfx(status_dict):
    """
    Preload SFX for status effects.

    Input: a dictionary whose keys are status names (paralyze, burn, etc).
    Output: a dictionary mapping those same keys to Sound objects or None.
    """
    out = {}

    # For each status name, resolve a file and load it safely.
    for key in status_dict.keys():
        out[key] = safe_sound(find_sfx_file(key))

    return out


# Caches and shared icon loading state.
_battle_surfaces = {}

_icon_surfaces = {}       # decoded pygame surfaces by icon key
_icon_inflight = set()    # keys currently being downloaded / decoded
_icon_ready_bytes = {}    # raw bytes fetched and ready to convert
_icon_lock = threading.Lock()  # mutex for cross thread icon state
_icon_fail_until = {}     # rate limit: key -> timestamp until we retry


# 9
def clean_name(x):
    """
    Normalize a string for display and basic lookups.

    Converts DASH to space, strips whitespace, and returns a plain string.
    """
    return str(x).replace(DASH, " ").strip()


# 10
def title_name(x):
    """
    Convert a slug style name into a nicer title case.

    Example: "mr mime" -> "Mr Mime"
    """
    s = clean_name(x)
    if len(s) == 0:
        return s

    # Capitalize the first letter of each word.
    return " ".join([w[:1].upper() + w[1:] for w in s.split()])


# 11
def clamp01(x):
    """
    Clamp to the closed interval [0, 1].

    Used everywhere for animation fractions so easing math stays sane.
    """
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


# 12
def draw_text(surface, text, font, x, y, color):
    """
    Render text and blit it at pixel coordinates.

    This is a thin wrapper so the rest of the UI code stays clean.
    """
    img = font.render(str(text), True, color)
    surface.blit(img, (int(x), int(y)))


# 13
def draw_rect(surface, rect, color, width=0):
    """
    Wrapper around pygame.draw.rect.

    width=0 fills the rectangle.
    width>0 draws an outline of that thickness.
    """
    pygame.draw.rect(surface, color, rect, int(width))


# 14
def draw_hp_bar(surface, x, y, w, h, cur_hp, max_hp, label, font, colors):
    """
    Draw a labeled HP bar.

    - Background panel
    - Foreground fill scaled by cur_hp/max_hp
    - Border outline
    - Label and numeric HP text
    """
    # Prevent division by zero or negative max HP.
    mx = float(max_hp) if float(max_hp) > 0.0 else 1.0

    # Clamp current HP into [0, mx].
    cur_raw = float(cur_hp)
    cur = max(0.0, min(cur_raw, mx))

    # Fraction used to compute fill width.
    frac = clamp01(cur / mx)

    # Background and filled rectangles.
    bg = pygame.Rect(int(x), int(y), int(w), int(h))
    fill_w = int(op.mul(float(w), frac))
    fg = pygame.Rect(int(x), int(y), int(fill_w), int(h))

    # Draw base panel then fill then border.
    draw_rect(surface, bg, colors["panel"])
    draw_rect(surface, fg, colors["ok"])
    draw_rect(surface, bg, colors["border"], 2)

    # Text: label above, numbers inside the bar.
    hp_txt = f"{int(cur)} / {int(mx)}"
    draw_text(surface, label, font, x, op.sub(y, 20), colors["text"])
    draw_text(surface, hp_txt, font, op.add(x, 6), op.add(y, 6), colors["text"])


# 15
def approach_value(cur, target, step):
    """
    Smoothly move a value toward a target by a fixed step per call.

    Used for UI easing like bar animations:
    - If the remaining difference is smaller than step, snap to target.
    - Otherwise move by step in the correct direction.
    """
    diff = float(op.sub(target, cur))

    # If we're close enough, finish cleanly instead of oscillating.
    if abs(diff) <= float(step):
        return float(target)

    # Move toward the target with a fixed velocity.
    if diff > 0.0:
        return float(op.add(cur, step))
    return float(op.sub(cur, step))

class Button:
    """
    Minimal clickable UI button.

    It is intentionally simple:
    - A rectangle background
    - A border
    - A text label
    - An enabled flag that disables clicks + changes color
    """

    def __init__(self, rect, text, font):  # 16
        # Store geometry as a pygame.Rect so collision checks and drawing are easy.
        self.rect = pygame.Rect(rect)

        # Button label text (forced to string so numbers and None do not break rendering).
        self.text = str(text)

        # Font used to render the label.
        self.font = font

        # If False:
        # - draw() uses "button_off" color
        # - hit() always returns False
        self.enabled = True

        # Optional override colors:
        # bg: custom background color for this button
        # fg: custom text color for this button
        # If these are None we fall back to the global theme colors dict.
        self.bg = None   # expected format: (r, g, b) or (r, g, b, a)
        self.fg = None   # expected format: (r, g, b) or (r, g, b, a)

    def draw(self, surface, colors):  # 17
        """
        Draw the button.

        colors is the global theme dictionary (button, button_off, border, text, etc).
        """
        # Choose background color:
        # - enabled buttons use per button bg override if provided
        # - otherwise use theme button color
        # - disabled buttons always use theme button_off color
        if self.enabled:
            col = self.bg if self.bg is not None else colors["button"]
        else:
            col = colors["button_off"]

        # Draw background rectangle.
        draw_rect(surface, self.rect, col)

        # Draw border outline (fixed thickness 2) to give it definition.
        draw_rect(surface, self.rect, colors["border"], 2)

        # Choose text color:
        # - per button override if provided
        # - otherwise theme text color
        txt_col = self.fg if self.fg is not None else colors["text"]

        # Render and blit text with a small padding offset inside the rect.
        draw_text(
            surface,
            self.text,
            self.font,
            op.add(self.rect.x, 10),
            op.add(self.rect.y, 10),
            txt_col,
        )

    def hit(self, pos):  # 18
        """
        Return True if a click position is inside the button AND the button is enabled.
        """
        # Disabled buttons should never trigger actions.
        if not self.enabled:
            return False

        # pygame.Rect.collidepoint handles point in rect logic.
        return bool(self.rect.collidepoint(pos))


class TextInput:
    """
    Minimal single line text input.

    Behaviors:
    - Click inside to activate (focus)
    - Type characters when active
    - Backspace deletes one character
    - Enter does nothing (caller can decide what Enter means)
    - Input is filtered to a safe character set
    - Text is capped at 30 chars to avoid UI overflow
    """

    def __init__(self, rect, font, placeholder="type to search"):  # 19
        # Geometry for click focusing and drawing.
        self.rect = pygame.Rect(rect)

        # Font for text rendering.
        self.font = font

        # Current entered text.
        self.text = ""

        # Placeholder text shown when self.text is empty.
        self.placeholder = str(placeholder)

        # Focus state:
        # True means keyboard input is accepted.
        self.active = False

    def draw(self, surface, colors):  # 20
        """
        Draw the input box.

        When empty:
        - show placeholder
        - use muted color
        When not empty:
        - show typed text
        - use normal text color
        """
        # Background and border.
        draw_rect(surface, self.rect, colors["input_bg"])
        draw_rect(surface, self.rect, colors["border"], 2)

        # Decide what text to show (typed text or placeholder).
        shown = self.text if len(self.text) > 0 else self.placeholder

        # Decide color: muted when placeholder is showing.
        col = colors["text"] if len(self.text) > 0 else colors["muted"]

        # Draw with a small padding so text does not touch the border.
        draw_text(
            surface,
            shown,
            self.font,
            op.add(self.rect.x, 8),
            op.add(self.rect.y, 8),
            col,
        )

    def handle_event(self, event):  # 21
        """
        Process pygame events relevant to text input.

        Call this from your main event loop:
        for event in pygame.event.get():
            text_input.handle_event(event)
        """
        # Mouse click toggles focus:
        # - click inside: active True
        # - click outside: active False
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = bool(self.rect.collidepoint(event.pos))
            return

        # Only consume keyboard events while focused.
        if event.type == pygame.KEYDOWN and self.active:
            # Backspace deletes one character.
            if event.key == pygame.K_BACKSPACE:
                if len(self.text) > 0:
                    # Slice off the last character.
                    self.text = self.text[: op.sub(len(self.text), 1)]
                return

            # Enter is ignored here (caller can decide to trigger a search, submit, etc).
            if event.key == pygame.K_RETURN:
                return

            # event.unicode is the typed character, respecting keyboard layout and shift.
            ch = event.unicode
            if ch is None:
                return
            ch = str(ch)
            if len(ch) == 0:
                return

            # Filter characters to keep the input predictable and safe for lookups:
            # allow letters, numbers, space, underscore, dot.
            ok = True
            for c in ch:
                if c.isalnum() or c == " " or c == "_" or c == ".":
                    continue
                ok = False
                break

            # Append if valid and within max length.
            if ok and len(self.text) < 30:
                self.text = self.text + ch


def list_matches(name_pool, query):  # 22
    """
    Return a list of pokemon names that match a query string.

    Used by the selection UI search:
    - If query is empty, return everything
    - Otherwise normalize query to our internal naming style
    - Then keep only names that start with the query prefix
    """
    # Normalize the query into a lowercase string (safe even if query is None or not a string).
    q = str(query).strip().lower()

    # Empty search means "show the full list".
    if len(q) == 0:
        return list(name_pool)

    # Our pokemon names often use a DASH separator instead of spaces or underscores.
    # Example: "mr mime" becomes "mr-mime" (since DASH is chr(45)).
    q = q.replace(" ", DASH).replace("_", DASH)

    # Prefix match so the list feels like a fast typeahead.
    # Using startswith instead of "in" makes results less chaotic.
    out = [n for n in name_pool if str(n).startswith(q)]
    return out


def clamp_scroll(offset, total, visible):  # 23
    """
    Clamp a scroll offset so the visible window stays inside list bounds.

    offset: current scroll index (top item index)
    total: total items in the list
    visible: how many items can be shown at once
    """
    # Convert totals to ints so math is stable even if floats sneak in.
    total_i = int(total)

    # visible must be at least 1 so we never divide by zero or create negative windows.
    visible_i = max(1, int(visible))

    # Maximum allowed offset is total minus visible.
    # If total < visible, max offset becomes 0 (no scrolling needed).
    max_off = max(0, int(op.sub(total_i, visible_i)))

    # Clamp offset into [0, max_off].
    return int(max(0, min(int(offset), max_off)))


def safe_filename(name_lower):  # 24
    """
    Return a filesystem safe filename fragment derived from a string.

    Rules:
    - letters and digits are kept
    - everything else becomes underscore
    - avoid empty names by falling back to "poke"
    - always append ".png"
    """
    # Normalize input to lowercase; safe even if input is weird.
    s = str(name_lower).strip().lower()

    # Build a cleaned character list for speed and simplicity.
    out = []
    for ch in s:
        # Keep alphanumeric characters.
        if ch.isalnum():
            out.append(ch)
        else:
            # Replace spaces, dashes, punctuation, accents, etc with underscores.
            out.append("_")

    # Join into a string and trim underscores from ends.
    fname = "".join(out).strip("_")

    # If everything got stripped, use a default.
    if len(fname) == 0:
        fname = "poke"

    # Return as png filename.
    return fname + ".png"


def try_load_local_icon(name_lower):  # 25
    """
    Try to load a cached pokemon icon from disk.

    Returns:
    - pygame.Surface (scaled to ICON_PX x ICON_PX) if found and loadable
    - None if the file does not exist or loading fails
    """
    # Build icon path from the sprite cache folder + sanitized name.
    path = os.path.join(SPRITE_DIR, safe_filename(name_lower))

    # Fast fail if file is missing.
    if not os.path.exists(path):
        return None

    try:
        # Load and convert to include alpha channel for transparency.
        surf = pygame.image.load(path).convert_alpha()

        # Force a consistent icon size for the UI list.
        surf = pygame.transform.smoothscale(surf, (ICON_PX, ICON_PX))
        return surf
    except Exception:
        # Do not crash the whole UI because one file is broken.
        return None


def try_load_local_sprite(folder, name_lower, size_px):  # 26
    """
    Try to load a cached pokemon sprite from disk.

    folder: directory path containing sprites
    name_lower: pokemon name in lowercase
    size_px: target square size (width and height)

    Returns a scaled Surface or None.
    """
    # Build sprite path from folder + sanitized name.
    path = os.path.join(str(folder), safe_filename(name_lower))

    # If it is not on disk, do not attempt loading.
    if not os.path.exists(path):
        return None

    try:
        # Load sprite with alpha (important for cutout edges).
        surf = pygame.image.load(path).convert_alpha()

        # Normalize to a square size so battle layout stays consistent.
        surf = pygame.transform.smoothscale(surf, (int(size_px), int(size_px)))
        return surf
    except Exception:
        # Missing file, invalid png, etc. We just skip it.
        return None


def load_scaled_front(rel_path, target_h):  # 27
    """
    Load a front sprite and scale it for the selection screen.

    This keeps aspect ratio:
    - height is forced to target_h
    - width is computed from the original sprite ratio
    """
    try:
        # Load sprite and ensure alpha is preserved.
        img = pygame.image.load(Path(rel_path)).convert_alpha()
    except Exception:
        # File missing or corrupted.
        return None

    # Read original size.
    w = img.get_width()
    h = img.get_height()

    # Avoid divide by zero and nonsense surfaces.
    if h <= 0:
        return None

    # Scale factor based on desired height.
    scale = float(target_h) / float(h)

    # Compute new dimensions (keep aspect ratio).
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Reject invalid results.
    if new_w <= 0 or new_h <= 0:
        return None

    # Smooth scale so small sprites look less crunchy.
    return pygame.transform.smoothscale(img, (new_w, new_h))


def build_select_background():  # 28
    """
    Build background row data used in the pokemon selection screen.

    Output format: list of row dictionaries.
    Each row stores:
    - y: vertical position
    - dir: scroll direction (1 or 0)
    - speed: pixels per second
    - gap: spacing between sprites
    - sprites: list of preloaded sprite Surfaces
    - strip_len: total looping length in pixels
    - anchor: current scroll offset in pixels
    """
    # Sprite size in the background strips.
    target_h = 100

    # Gap between sprites in starter rows.
    gap = 10

    # Slightly larger gap for legendary rows so it feels less cluttered.
    gap_l = 14

    def load_list(names):  # 28.1
        """
        Helper: load a list of front sprites by name.
        Missing files are skipped (so the background still builds).
        """
        out = []
        for n in names:
            # Each sprite is expected under sprites_front/<name>.png
            s = load_scaled_front(Path("sprites_front") / (str(n) + ".png"), target_h)
            if s is not None:
                out.append(s)
        return out

    # Starter evolution lines for each region.
    kanto_starters = load_list([
        "bulbasaur", "ivysaur", "venusaur",
        "charmander", "charmeleon", "charizard",
        "squirtle", "wartortle", "blastoise",
    ])

    johto_starters = load_list([
        "chikorita", "bayleef", "meganium",
        "cyndaquil", "quilava", "typhlosion",
        "totodile", "croconaw", "feraligatr",
    ])

    hoenn_starters = load_list([
        "treecko", "grovyle", "sceptile",
        "torchic", "combusken", "blaziken",
        "mudkip", "marshtomp", "swampert",
    ])

    sinnoh_starters = load_list([
        "turtwig", "grotle", "torterra",
        "chimchar", "monferno", "infernape",
        "piplup", "prinplup", "empoleon",
    ])

    # Regional legendary groups.
    kanto_legends = load_list([
        "articuno", "zapdos", "moltres", "mewtwo",
    ])

    johto_legends = load_list([
        "lugia", "ho_oh",
        "raikou", "entei", "suicune",
    ])

    # Larger legends across later gens.
    big_legends = []
    for nm in ["kyogre", "groudon", "rayquaza", "dialga", "palkia"]:
        s = load_scaled_front(Path("sprites_front") / (str(nm) + ".png"), target_h)
        if s is not None:
            big_legends.append(s)

    # Special filename cases that do not match plain "<name>.png".
    s = load_scaled_front(Path("sprites_front") / "giratina_altered.png", target_h)
    if s is not None:
        big_legends.append(s)

    s = load_scaled_front(Path("sprites_front") / "deoxys_normal.png", target_h)
    if s is not None:
        big_legends.append(s)

    # Define animated rows.
    # dir:
    # - 1 means move right to left by increasing anchor
    # - 0 means move left to right by decreasing anchor (we mod wrap anyway)
    rows = [
        {"y": 10,  "dir": 1, "speed": 90, "gap": gap,   "sprites": kanto_starters},
        {"y": 90,  "dir": 0, "speed": 90, "gap": gap,   "sprites": johto_starters},
        {"y": 170, "dir": 1, "speed": 90, "gap": gap,   "sprites": hoenn_starters},
        {"y": 250, "dir": 0, "speed": 90, "gap": gap,   "sprites": sinnoh_starters},
        {"y": 350, "dir": 1, "speed": 80, "gap": gap_l, "sprites": kanto_legends},
        {"y": 440, "dir": 0, "speed": 80, "gap": gap_l, "sprites": johto_legends},
        {"y": 530, "dir": 0, "speed": 70, "gap": gap_l, "sprites": big_legends},
    ]

    # Precompute strip lengths for looping animation.
    for r in rows:
        sprs = r["sprites"]
        g = int(r["gap"])

        # If there are no sprites, give a dummy length so modulo math never breaks.
        if len(sprs) == 0:
            r["strip_len"] = 1
        else:
            # strip_len is the total pixel width of the repeating strip:
            # sum(sprite widths + gap)
            total = 0
            for im in sprs:
                total = int(op.add(total, int(op.add(im.get_width(), g))))
            r["strip_len"] = max(1, total)

        # anchor is the current scroll position in pixels for that row.
        r["anchor"] = 0.0

    return rows


def draw_select_background(screen, rows, w, h, dt):  # 29
    """
    Draw the selection screen background.

    It does:
    - Light solid base fill
    - Several horizontally scrolling sprite strips (looped with modulo)
    - Optional overlay layer (currently transparent, but kept as a hook)
    """
    # Fill the whole screen with a light background color.
    screen.fill((245, 245, 248))

    # Draw each row as a looping strip of sprites.
    for r in rows:
        sprs = r["sprites"]
        if len(sprs) == 0:
            continue

        # Convert speed (pixels per second) into a per frame step using dt (seconds).
        step = float(op.mul(float(r["speed"]), float(dt)))

        # Update anchor based on direction.
        # We wrap with modulo strip_len so it loops forever without growing.
        if int(r["dir"]) == 1:
            r["anchor"] = float((r["anchor"] + step) % float(r["strip_len"]))
        else:
            r["anchor"] = float((r["anchor"] - step) % float(r["strip_len"]))

        # Start drawing at negative anchor so the strip appears shifted.
        x = float(op.sub(0.0, r["anchor"]))
        y = int(r["y"])
        g = int(r["gap"])

        # Keep blitting sprites until we cover the full screen width.
        while x < float(w):
            for im in sprs:
                # Draw one sprite at current x.
                screen.blit(im, (int(x), y))

                # Advance x by sprite width plus the row gap.
                x = float(op.add(x, float(op.add(im.get_width(), g))))

    # Optional overlay so the UI stays readable.
    # Right now it is fully transparent, but you can tint it if you want a dimming effect.
    overlay = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 0))
    screen.blit(overlay, (0, 0))


# ============================================================
# Persistent FX state (kept across frames)
# ============================================================

# Fairy particle system state.
# fairy_fx_particles holds per sparkle data so the animation looks continuous
# instead of spawning brand new random sparkles every frame.
fairy_fx_particles = []

# One time init flag for the fairy particles.
# We set this to True after we create the initial particle list.
fairy_fx_inited = False

# Steel snap sound gate.
# This prevents the snap SFX from playing every frame during the snap phase.
steel_snap_sfx_played = False

# Steel snap sound object (pygame.mixer.Sound or None if missing).
# Assigned during your SFX loading step.
steel_snap_sfx = None

# Player 1 attack state.
# p1_attack_active tells the renderer to draw the current attack FX.
p1_attack_active = False

# Player 1 attack local timer in frames (or ticks depending on your usage).
# This increments each frame while p1_attack_active is True.
p1_attack_t = 0.0

# Player 2 attack state.
# Same idea as player 1, but for the opponent.
p2_attack_active = False

# Player 2 attack local timer.
# Increments while p2_attack_active is True.
p2_attack_t = 0.0


# ---------------------------------------------------------------------------
# Global timing knobs
# ---------------------------------------------------------------------------

attack_duration = 176  # total animation length in frames for the current move FX (overall timeline)

LUNGE_T_STEP = 4.0     # how fast the attacker "lunges" forward (bigger = faster, fewer frames)
FX_T_STEP = 2.0        # how fast the generic attack FX timeline advances (bigger = faster)
FAINT_T_STEP = 1.0     # how fast faint animation advances (bigger = faster)

FLASH_PERIOD = 10      # frames per flash cycle for any flashing overlay (smaller = faster flashing)


# ---------------------------------------------------------------------------
# POISON FX tuning
# ---------------------------------------------------------------------------

POISON_BUBBLE_COUNT = 7    # number of poison bubbles drawn per cycle / per burst
POISON_RISE_SPEED = 0.5    # upward speed of bubbles (pixels per frame)
POISON_WIGGLE_AMP = 32     # sideways wobble amplitude in pixels
POISON_WIGGLE_FREQ = 0.04  # wobble speed (higher = faster wiggle)
POISON_DELAY = 8           # delay between bubble spawns or steps (frames)
POISON_MAX_RISE = 220      # cap on how high bubbles rise before disappearing (pixels)
POISON_SIZE_W = 52         # base bubble sprite / bubble width reference (pixels)


# ---------------------------------------------------------------------------
# ROCK (ground rock fall) FX tuning
# ---------------------------------------------------------------------------

ROCK_FALL_SPEED = 5            # downward speed of the falling rock (pixels per frame)
ROCK_BOUNCE_AMP = 108          # bounce height / vertical oscillation amplitude after impact (pixels)
ROCK_BOUNCE_FREQ = 0.01        # bounce oscillation speed (higher = faster bouncing)
ROCK_DELAY = 10                # delay before the rock starts / before impact phase (frames)

ROCK_SPLATTER_COUNT = 10       # number of small debris particles emitted on impact
ROCK_SPLATTER_LIFE = 48        # particle lifetime in frames (how long debris stays visible)
ROCK_SPLATTER_SPEED = 1.2      # initial particle speed scale (pixels per frame)
ROCK_SPLATTER_GRAV = 0.32      # gravity applied to debris (pixels per frame squared)
ROCK_SPLATTER_RADIUS_MIN = 3   # min debris particle radius (pixels)
ROCK_SPLATTER_RADIUS_MAX = 6   # max debris particle radius (pixels)
ROCK_SPLATTER_ALPHA = 230      # debris opacity (0 to 255)
ROCK_SPLATTER_SPREAD_X = 64    # horizontal spread of debris spawn / velocities (pixels)
ROCK_SPLATTER_SPREAD_Y = 26    # vertical spread of debris spawn / velocities (pixels)
ROCK_SPLATTER_START_DELAY = 0  # extra delay before debris spawns (frames)


# ---------------------------------------------------------------------------
# PUNCH (impact flash) FX tuning
# ---------------------------------------------------------------------------

PUNCH_START_W = 260         # starting width of the punch impact sprite / ring (pixels)
PUNCH_END_W = 140           # ending width after it shrinks / eases (pixels)
PUNCH_HIT_FRAC = 0.35       # normalized time fraction when the "hit moment" happens (0 to 1)
PUNCH_SHAKE_PX = 10         # screen shake strength on hit (pixels)
PUNCH_ALPHA = 230           # punch overlay opacity (0 to 255)
PUNCH_WIGGLE_PX = 6         # small jitter applied to the punch effect (pixels)


# ---------------------------------------------------------------------------
# FIRE FX tuning
# ---------------------------------------------------------------------------

FIRE_RAY_W = 60            # width of each fire ray sprite / strip (pixels)
FIRE_RAY_COUNT = 10        # how many ray slices are drawn along the beam
FIRE_RAY_SPACING = 0.10    # spacing between ray slices in normalized beam space (smaller = denser)
FIRE_RAY_WAVES = 1.2       # wave cycles along the beam (higher = more oscillation in the beam)

FIRE_BLAST_W = 180         # size of the impact blast sprite / circle at target (pixels)
FIRE_BLAST_COUNT = 6       # how many blast puffs / rings are drawn at impact
FIRE_BLAST_JITTER = 18     # random offset applied to blast puffs to feel chaotic (pixels)


# ---------------------------------------------------------------------------
# ELECTRIC FX tuning
# ---------------------------------------------------------------------------

ELEC_LINE_W = 9                # thickness of lightning lines (pixels)
ELEC_BOLT_COUNT = 4            # number of main lightning bolts drawn
ELEC_SEGMENTS = 7              # number of segments per bolt (higher = more jagged detail)
ELEC_MAIN_FRAC = 0.52          # fraction of fx time spent on the main bolt phase (0 to 1)

ELEC_RESIDUAL_SPARKS = 48      # number of small sparks after the main bolt
ELEC_RESIDUAL_ARCS = 56        # number of small arc lines after the main bolt

ELEC_GLOW_R0 = 18              # starting glow radius around target at impact (pixels)
ELEC_GLOW_R1 = 190             # ending glow radius as it expands / fades (pixels)


# ---------------------------------------------------------------------------
# WATER JET FX tuning
# ---------------------------------------------------------------------------

WATER_MAIN_FRAC = 0.72      # fraction of atk_fx_t spent firing the jet before the splash phase (0 to 1)
WATER_JET_W0 = 16           # jet thickness at start (pixels)
WATER_JET_W1 = 10           # jet thickness at end (pixels)
WATER_RIBBONS = 5           # number of parallel "streams" that form the jet (more = fuller jet)

WATER_WIGGLE_AMP = 30       # sideways wiggle amplitude of the jet path (pixels)
WATER_WIGGLE_FREQ = 0.55    # wiggle speed (higher = faster wiggle)

WATER_SPLASH_COUNT = 18     # number of splash droplets / particles at impact
WATER_SPLASH_R0 = 20        # starting splash radius (pixels)
WATER_SPLASH_R1 = 106       # ending splash radius as it expands (pixels)
WATER_JET_SEGMENTS = 12     # line segments used to approximate the jet curve (higher = smoother)


# ---------------------------------------------------------------------------
# GRASS (leaf swirl) FX tuning
# ---------------------------------------------------------------------------

LEAF_W = 56                    # leaf sprite base width (pixels)
LEAF_COUNT = 8                 # number of leaves spawned
LEAF_DELAY = 10                # delay between leaf spawns / stagger (frames)

LEAF_WIGGLE_AMP = 12           # sideways wiggle amplitude while flying (pixels)
LEAF_WIGGLE_FREQ = 0.13        # wiggle speed (higher = faster wobble)
LEAF_SPIN_DEG_PER_FRAME = 4.0  # leaf rotation speed (degrees per frame)
LEAF_ALPHA = 220               # leaf opacity (0 to 255)

LEAF_RISE_FRAMES = 22          # frames spent rising from attacker side
LEAF_FALL_FRAMES = 250         # frames spent falling / drifting down (longer = floatier)
LEAF_LAUNCH_FRAMES = 18        # frames spent launching outward toward target

LEAF_APEX_Y = 130              # vertical apex height reference (pixels from some origin, depends on your coords)
LEAF_START_SPREAD = 102        # initial spread of leaf spawn positions (pixels)
LEAF_TARGET_SPREAD = 34        # how tight the leaves converge near the target (pixels)
LEAF_FALL_DRIFT = 40           # max sideways drift during fall (pixels)


# ---------------------------------------------------------------------------
# PSYCHIC (ring projectile + impact pulse) FX tuning
# ---------------------------------------------------------------------------

PSY_CIRCLE_COUNT = 12          # number of traveling rings / circles
PSY_SPACING = 0.11             # spacing between rings along the travel path (normalized)
PSY_RING_R_MIN = 7             # min ring radius (pixels)
PSY_RING_R_MAX = 15            # max ring radius (pixels)
PSY_RING_W = 3                 # ring stroke width (pixels)

PSY_RING_WIGGLE = 1.4          # lateral wiggle strength for rings (pixels or scale factor depending on code)
PSY_IMPACT_SPAN = 0.55         # fraction of fx time devoted to impact expansion (0 to 1)

PSY_IMPACT_R0 = 18             # impact ring start radius (pixels)
PSY_IMPACT_R1 = 150            # impact ring end radius (pixels)
PSY_IMPACT_W0 = 6              # impact ring start stroke width (pixels)
PSY_IMPACT_W1 = 2              # impact ring end stroke width (pixels)


# ---------------------------------------------------------------------------
# ICE (falling flakes + shatter bits) FX tuning
# ---------------------------------------------------------------------------

ICE_FLAKE_W = 54                # ice flake sprite width (pixels)
ICE_COUNT = 8                   # number of flakes spawned
ICE_DELAY_MAX = 16              # max random spawn delay per flake (frames)
ICE_LIFE_FRAMES = 100           # flake lifetime (frames)

ICE_SIDE_SPREAD = 16.0          # horizontal spawn spread (pixels)
ICE_GRAVITY = 10.0              # downward acceleration applied to flakes (pixels per frame squared)
ICE_SPIN_DEG_PER_FRAME = 4.0    # flake rotation speed (degrees per frame)

ICE_SCALE_MIN = 0.75            # min scale factor for flake size
ICE_SCALE_MAX = 1.15            # max scale factor for flake size
ICE_ALPHA = 240                 # flake opacity (0 to 255)

ICE_STAGGER_FRAMES = 9          # spacing between flake spawns (frames)
ICE_JITTER_MAX = 2              # small random positional jitter per frame (pixels)

ICE_BITS_COUNT = 4              # number of shatter bits emitted at impact
ICE_BITS_LIFE_FRAMES = 74       # shatter bit lifetime (frames)
ICE_BITS_SPEED = 8.2            # initial shatter bit speed (pixels per frame)
ICE_BITS_GRAVITY = 0.15         # gravity for shatter bits (pixels per frame squared)

ICE_BITS_RADIUS_MIN = 2         # min shatter bit radius (pixels)
ICE_BITS_RADIUS_MAX = 4         # max shatter bit radius (pixels)
ICE_BITS_ALPHA = 230            # shatter bit opacity (0 to 255)

ICE_BITS_SPREAD_JITTER = 0.42   # randomness factor for bit directions / spread (bigger = messier burst)


# ---------------------------------------------------------------------------
# DARK (shrinking ring + orbiting motes + shake) FX tuning
# ---------------------------------------------------------------------------

DARK_RING_R0 = 120          # dark ring start radius (pixels)
DARK_RING_R1 = 44           # dark ring end radius (pixels)
DARK_RING_W0 = 12           # ring stroke start width (pixels)
DARK_RING_W1 = 3            # ring stroke end width (pixels)
DARK_RING_ALPHA = 220       # ring opacity (0 to 255)

DARK_MOTE_COUNT = 14        # number of orbiting dark motes
DARK_MOTE_ORBIT_R = 34      # orbit radius around target (pixels)
DARK_MOTE_R_MIN = 2         # min mote radius (pixels)
DARK_MOTE_R_MAX = 5         # max mote radius (pixels)
DARK_MOTE_ALPHA = 200       # mote opacity (0 to 255)

DARK_SHAKE_PX = 7           # screen shake strength during dark hit (pixels)
DARK_SHAKE_PERIOD = 4       # frames per shake cycle (smaller = faster shake)
DARK_SHAKE_FREQ = 0.9       # shake oscillation speed scale (higher = faster shake)


# ---------------------------------------------------------------------------
# GROUND (shake + debris + cracks) FX tuning
# ---------------------------------------------------------------------------

GROUND_FOOT_Y_OFFSET = 70       # vertical offset to aim the effect near feet / ground contact (pixels)
GROUND_SHAKE_PX = 8             # shake strength during ground hit (pixels)
GROUND_SHAKE_FREQ = 0.95        # shake speed (higher = faster shake)

GROUND_BITS_COUNT = 22          # number of dirt bits / particles
GROUND_BITS_LIFE_FRAMES = 46    # particle lifetime (frames)
GROUND_BITS_SPEED = 7.6         # initial speed (pixels per frame)
GROUND_BITS_GRAVITY = 0.85      # gravity for bits (pixels per frame squared)

GROUND_BITS_RADIUS_MIN = 2      # min bit radius (pixels)
GROUND_BITS_RADIUS_MAX = 7      # max bit radius (pixels)
GROUND_BITS_ALPHA = 230         # bit opacity (0 to 255)

GROUND_BITS_ANG_SPREAD = 0.85   # angular spread for emitted bits (radians scale factor)
GROUND_BITS_X_SPREAD = 18       # extra horizontal randomness (pixels)
GROUND_BITS_Y_SPREAD = 8        # extra vertical randomness (pixels)

GROUND_CRACK_COUNT = 7          # number of crack polylines drawn
GROUND_CRACK_SEGMENTS = 7       # segments per crack (higher = more detailed)
GROUND_CRACK_LEN0 = 6           # crack length at start of growth (pixels)
GROUND_CRACK_LEN1 = 92          # crack length at full growth (pixels)

GROUND_CRACK_JITTER = 0.55      # crack jaggedness factor (bigger = more zigzag)
GROUND_CRACK_ALPHA = 210        # crack opacity (0 to 255)

GROUND_CRACK_W0 = 1             # crack line width at start (pixels)
GROUND_CRACK_W1 = 3             # crack line width at end (pixels)

GROUND_FOOT_Y_FRAC = 0.28       # foot placement as fraction of target sprite size (0 to 1)


# ---------------------------------------------------------------------------
# DRAGON (wavy beam) FX tuning
# ---------------------------------------------------------------------------

DRAGON_RAY_W = 72            # dragon beam width (pixels)
DRAGON_RAY_COUNT = 11        # number of beam slices drawn
DRAGON_RAY_SPACING = 0.10    # spacing between slices (normalized)
DRAGON_RAY_WAVES = 1.15      # number of wave cycles along beam length
DRAGON_RAY_WIGGLE = 12       # lateral wiggle amplitude (pixels)
DRAGON_RAY_ALPHA = 255       # beam opacity (0 to 255)


# ---------------------------------------------------------------------------
# STEEL (bullet line + tip + impact + sparks) FX tuning
# ---------------------------------------------------------------------------

STEEL_BULLET_MAIN_FRAC = 0.62  # fraction of fx time spent on the traveling bullet before impact (0 to 1)

STEEL_LINE_W0 = 12             # bullet trail thickness at start (pixels)
STEEL_LINE_W1 = 4              # bullet trail thickness at end (pixels)

STEEL_TIP_R0 = 6               # bullet tip radius at start (pixels)
STEEL_TIP_R1 = 12              # bullet tip radius at end (pixels)

STEEL_IMPACT_R0 = 14           # impact ring start radius (pixels)
STEEL_IMPACT_R1 = 92           # impact ring end radius (pixels)
STEEL_IMPACT_W0 = 6            # impact ring start stroke width (pixels)
STEEL_IMPACT_W1 = 2            # impact ring end stroke width (pixels)

STEEL_SPARK_COUNT = 22         # number of sparks on impact
STEEL_SPARK_LIFE = 26          # spark lifetime (frames)
STEEL_SPARK_SPEED = 9.2        # spark initial speed (pixels per frame)
STEEL_SPARK_GRAV = 0.55        # gravity for sparks (pixels per frame squared)


def ground_shake_offset(t, dur):  # 30
    """
    Small camera shake offset used by ground style attacks.

    Idea:
    - Shake is strongest at the start, then fades out to zero by the end.
    - We return (dx, dy) that you add to your draw positions (cheap camera shake).
    """
    # dur must be at least 1 so time normalization never divides by zero.
    dur = float(max(1, dur))

    # Normalize time into p in range 0 to 1.
    # p = 0 at start of animation, p = 1 at end.
    p = clamp01(float(op.truediv(float(t), dur)))

    # Inverse progress so the shake fades out over time.
    inv = float(op.sub(1.0, p))

    # Amplitude in pixels, scaled by inv so it decays to zero.
    amp = int(op.mul(float(GROUND_SHAKE_PX), inv))

    # If amplitude is gone, stop shaking.
    if amp <= 0:
        return 0, 0

    # Phase controls how fast the shake oscillates.
    # GROUND_SHAKE_FREQ is basically "wiggles per frame" in radians scaling land.
    phase = float(op.mul(float(t), float(GROUND_SHAKE_FREQ)))

    # Horizontal shake: sin wave.
    dx = int(op.mul(float(amp), math.sin(phase)))

    # Vertical shake: another sin wave with a phase shift so it is not identical to dx.
    dy = int(op.mul(float(amp), math.sin(op.add(phase, 1.9))))

    # Return pixel offsets.
    return dx, dy


def dark_shake_offset(t, dur):  # 31
    """
    Small camera shake offset used by dark style attacks.

    Same structure as ground_shake_offset:
    - Fade out over time
    - Two sine waves with a phase shift
    - Different constants so it "feels" different
    """
    # dur must be at least 1 so time normalization never divides by zero.
    dur = float(max(1, dur))

    # Normalize time into 0..1 progress.
    p = clamp01(float(op.truediv(float(t), dur)))

    # inv goes from 1 down to 0, used to decay intensity.
    inv = float(op.sub(1.0, p))

    # Dark shake amplitude in pixels (separate constant from ground).
    amp = int(op.mul(float(DARK_SHAKE_PX), inv))

    # If amplitude is gone, stop shaking.
    if amp <= 0:
        return 0, 0

    # Oscillation phase uses DARK_SHAKE_FREQ.
    phase = float(op.mul(float(t), float(DARK_SHAKE_FREQ)))

    # Horizontal shake.
    dx = int(op.mul(float(amp), math.sin(phase)))

    # Vertical shake with a slightly different phase shift than ground.
    dy = int(op.mul(float(amp), math.sin(op.add(phase, 1.7))))

    return dx, dy


# Per move type animation durations (in frames).
# These are default "base" lengths used by attack_duration_for_type.
# Some types (grass) are computed dynamically because they are multi phase and staggered.
ATTACK_DURATIONS = {
    "fire": 180,
    "water": 220,
    "electric": 180,
    "fighting": 160,
    "rock": 210,
    "poison": 290,
    "ground": 170,
    "unknown": 176,
    "ice": 300,
    "dark": 190,
    "bug": 175,
    "dragon": 200,
    "ghost": 185,
    "steel": 190,
    # grass is computed dynamically
}


def compute_grass_duration():  # 32
    """
    Compute grass effect duration based on leaf animation timing.

    Grass is special because:
    - Each leaf has 3 phases (rise, fall, launch)
    - Leaves are staggered by LEAF_DELAY
    - We want enough extra frames so the final leaf finishes cleanly
    """
    # Phase lengths for one leaf, in frames.
    rise_n = int(max(1, LEAF_RISE_FRAMES))
    fall_n = int(max(1, LEAF_FALL_FRAMES))
    launch_n = int(max(1, LEAF_LAUNCH_FRAMES))

    # Total length for one leaf to complete all phases.
    total_n = int(op.add(op.add(rise_n, fall_n), launch_n))

    # Stagger time added because leaves do not all start at t=0.
    # If LEAF_COUNT is 1, stagger becomes 0.
    stagger = int(op.mul(max(0, int(op.sub(LEAF_COUNT, 1))), int(LEAF_DELAY)))

    # Add a small buffer so the last leaves do not get cut off by the global duration.
    return int(op.add(op.add(total_n, stagger), 12))


def attack_duration_for_type(atype):  # 33
    """
    Pick an animation duration for the given attack type.

    Rules:
    - Grass uses compute_grass_duration (dynamic)
    - Otherwise use ATTACK_DURATIONS lookup
    - Unknown types fall back to ATTACK_DURATIONS["unknown"]
    """
    # Normalize input type string.
    k = str(atype).strip().lower()

    # Grass and plant share the leaf animation, so duration is computed.
    if k in ("grass", "plant"):
        return compute_grass_duration()

    # Lookup the base duration.
    base = ATTACK_DURATIONS.get(k)

    # Fallback to "unknown" if the type is not in the dict.
    if base is None:
        base = ATTACK_DURATIONS.get("unknown", 176)

    return int(base)


# --- Attack FX (move animation) state ---
# These globals track one currently playing move animation.
atk_fx_active = False      # True while an attack animation is currently playing
atk_fx_t = 0.0             # Time counter (frames). This is the "t" passed into draw_attack_fx.
atk_fx_type = "unknown"    # Current move type string (selects which animation branch to run)
atk_fx_who = "p1"          # "p1" or "p2" (used to choose attacker vs target coordinates)


# --- Grass leaf animation bookkeeping ---
# Grass has per leaf hit tracking, so we do not trigger hit SFX multiple times.
leaf_hit_seen = set()      # Leaf indices that already triggered the "hit" moment
leaf_hit_sfx = None        # Cached Sound for leaf hits (loaded once, reused)


# --- Status FX (status condition overlay) state ---
# Status effects are separate from move attacks.
status_fx_active = False   # True while a status effect animation is currently playing
status_fx_t = 0.0          # Status animation time counter in frames
status_fx_key = ""         # Status id string ("burn", "poisoned", etc), selects FX branch
status_fx_who = "p1"       # "p1" or "p2" (where to draw the status effect)


# --- Status FX timing constants ---
STATUS_FX_DURATION = 90    # Total duration of a status FX in frames
STATUS_FX_T_STEP = 2.0     # How much status_fx_t increases per frame while active


def start_status_fx(who, key):  # 34
    """
    Queue a status animation effect for a given target.

    This just sets the global status FX state.
    The main update loop will advance status_fx_t and stop it after STATUS_FX_DURATION.
    """
    # We mutate global status state, so declare globals.
    global status_fx_active, status_fx_t, status_fx_key, status_fx_who

    # Enable the status effect animation.
    status_fx_active = True

    # Reset time to the start.
    status_fx_t = 0.0

    # Normalize status key so the draw code can compare safely.
    status_fx_key = str(key).strip().lower()

    # Store which side the status applies to.
    status_fx_who = who


def start_attack(who, atype="unknown"):  # 35
    """
    Initialize attack animation state and prepare any per type resets.

    It does:
    - Resets the global attack FX timer and type
    - Resets per type state (fairy particles, leaf hit tracking, steel snap sound)
    - Sets attack_duration based on type
    - Turns on the per player attack flag (p1_attack_active or p2_attack_active)
    """
    # Per player attack flags and timers.
    global p1_attack_active, p1_attack_t, p2_attack_active, p2_attack_t

    # Global attack FX state.
    global atk_fx_active, atk_fx_t, atk_fx_type, atk_fx_who

    # Global duration used by the FX draw code to normalize time.
    global attack_duration

    # Fairy keeps a particle list that must reset per cast.
    global fairy_fx_particles, fairy_fx_inited

    # Steel snap sound should play once per steel attack.
    global steel_snap_sfx_played
    if str(atype).strip().lower() == "steel":
        steel_snap_sfx_played = False

    # Turn on attack FX and reset the frame timer.
    atk_fx_active = True
    atk_fx_t = 0.0

    # Store normalized type and who.
    atk_fx_type = str(atype).strip().lower()
    atk_fx_who = who

    # Fairy uses persistent particles, so reset them so every cast starts fresh.
    if atk_fx_type == "fairy":
        fairy_fx_particles = []
        fairy_fx_inited = False

    # Grass uses leaf hit bookkeeping per cast, so clear it.
    if atk_fx_type in ("grass", "plant"):
        leaf_hit_seen.clear()

    # Compute duration for this move type.
    attack_duration = attack_duration_for_type(atk_fx_type)

    # Flip per player flags so the UI knows which sprite is attacking.
    if who == "p2":
        p2_attack_active = True
        p2_attack_t = 0.0
        return

    p1_attack_active = True
    p1_attack_t = 0.0


# Faint animation state per player.
p1_faint_active = False
p1_faint_t = 0.0

p2_faint_active = False
p2_faint_t = 0.0

# How long the faint animation lasts (in frames).
faint_duration = 30


def start_faint(who):  # 36
    """
    Begin the faint animation for a pokemon sprite.

    This sets the faint state for the chosen side.
    The render/update loop will advance p1_faint_t or p2_faint_t until faint_duration.
    """
    global p1_faint_active, p1_faint_t, p2_faint_active, p2_faint_t

    # Start fainting for opponent sprite.
    if who == "p2":
        p2_faint_active = True
        p2_faint_t = 0.0
        return

    # Start fainting for player sprite.
    p1_faint_active = True
    p1_faint_t = 0.0


def faint_alpha(t):  # 37
    """
    Compute sprite alpha during faint animation.

    - At t=0 alpha is 255 (fully visible)
    - At t=faint_duration alpha goes to 0 (fully invisible)
    """
    # Convert time into normalized fraction 0..1.
    frac = float(t) / float(max(1, faint_duration))
    frac = clamp01(frac)

    # Invert so alpha decreases over time.
    a = int(op.mul(255.0, op.sub(1.0, frac)))

    # Clamp to a valid 0..255 alpha value.
    return max(0, min(255, a))


def faint_drop(t):  # 38
    """
    Compute sprite vertical drop during faint animation.

    - At start drop is 0
    - At end drop is 60 pixels (sprite falls down)
    """
    # Normalize time.
    frac = float(t) / float(max(1, faint_duration))
    frac = clamp01(frac)

    # Linear drop in pixels.
    return int(op.mul(60.0, frac))


def lunge_offset(t):  # 39
    """
    Compute attacker lunge offset during certain attacks.

    Pattern:
    - First half: move forward to max distance
    - Second half: move back to original position
    """
    # Halfway point in frames.
    half = float(attack_duration) / 2.0
    if half <= 0.0:
        return 0

    tf = float(t)

    # First half: ramp up to 40 pixels.
    if tf <= half:
        return int(op.mul(40.0, op.truediv(tf, half)))

    # Second half: ramp back down to 0.
    return int(op.mul(40.0, op.sub(1.0, op.truediv(op.sub(tf, half), half))))


def target_flash_alpha(t):  # 40
    """
    Compute target flash alpha during hit moments.

    This is a simple blinking effect:
    - For half the period: alpha 255
    - For the other half: alpha 120
    """
    ti = int(t)

    # Half period used for on vs off split.
    half = int(op.truediv(FLASH_PERIOD, 2))

    # Toggle based on modulo to get a repeating blink.
    return 255 if (ti % FLASH_PERIOD) < half else 120


def make_placeholder_battle_sprite(font, colors, size_px):  # 41
    """
    Generate a simple placeholder sprite when a real sprite is missing.

    Visual:
    - Solid panel background
    - Border outline
    - A big question mark centered
    """
    # Create an RGBA surface so it can blend into the battle scene.
    s = pygame.Surface((int(size_px), int(size_px)), pygame.SRCALPHA)

    # Fill with panel color.
    s.fill(colors["panel"])

    # Outline border for readability.
    pygame.draw.rect(s, colors["border"], s.get_rect(), 2)

    # Render a centered question mark.
    txt = font.render("?", True, colors["muted"])
    r = txt.get_rect(center=(int(size_px) // 2, int(size_px) // 2))
    s.blit(txt, r)

    return s


def get_battle_sprite(name_lower, folder, size_px, placeholder):  # 42
    """
    Return the correct battle sprite (front or back) with caching.

    It does:
    - Cache lookup by (folder, name, size)
    - If not cached, try disk load
    - If load fails, use the placeholder surface
    - Store result back into cache
    """
    # Cache key includes folder, normalized name, and size.
    key = (str(folder), str(name_lower).strip().lower(), int(size_px))

    # Fast path: return cached surface.
    if key in _battle_surfaces:
        return _battle_surfaces[key]

    # Try loading from local sprite cache.
    surf = try_load_local_sprite(folder, name_lower, size_px)

    # If loading failed, fall back to placeholder.
    if surf is None:
        surf = placeholder

    # Cache it so we do not hit disk repeatedly.
    _battle_surfaces[key] = surf
    return surf


def draw_star(surf, x, y, r, alpha):  # 43
    """
    Draw a simple star shape (used by some effects).

    Method:
    - Build a small temporary transparent surface
    - Compute a 10 point star polygon (outer and inner radii)
    - Fill polygon with a yellowish color and requested alpha
    - Blit it centered at (x, y)
    """
    # Ensure radius is at least 2 so the star is visible.
    rr = int(max(2, r))

    # Padding so the polygon does not clip at the edges.
    pad = 4

    # Temporary surface size.
    size = rr * 2 + pad
    tmp = pygame.Surface((size, size), pygame.SRCALPHA)

    # Local center point for star generation.
    cx = size * 0.5
    cy = size * 0.5

    # Outer and inner radii define the star spikes.
    outer = rr
    inner = rr * 0.45

    # Generate 10 points around the circle, alternating outer and inner radius.
    pts = []
    for i in range(10):
        ang = (math.pi * 2.0) * (i / 10.0)
        rad = outer if (i % 2 == 0) else inner
        px = cx + rad * math.cos(ang)
        py = cy + rad * math.sin(ang)
        pts.append((px, py))

    # Clamp alpha to 0..255 and build the star color.
    col = (255, 235, 90, int(max(0, min(255, alpha))))

    # Draw filled polygon star.
    pygame.draw.polygon(tmp, col, pts)

    # Blit so that star center lands on (x, y).
    surf.blit(tmp, (x - cx, y - cy))

def draw_attack_fx(screen, target_center, atype, t, src_center=None, target_size=None): #44
    """
    Draw the active attack animation on top of the battle scene.

    Inputs used by every animation:
      screen: pygame surface to draw onto
      target_center: (x, y) center of the target sprite
      atype: attack type string, selects which animation branch runs
      t: animation time counter (your code treats it like frames)
      src_center: (x, y) center of the attacker sprite, if provided
      target_size: size reference for effects that scale with the target sprite
    """
    cx, cy = int(target_center[0]), int(target_center[1])

    # If attacker center is not available, default to the target center.
    if src_center is None:
        sx, sy = cx, cy
    else:
        sx, sy = int(src_center[0]), int(src_center[1])

    kind = str(atype).strip().lower()

    # Electric animation
    # Phase split is controlled by attack_duration and ELEC_MAIN_FRAC.
    # Main phase draws fresh bolts each frame (so they flicker by randomness).
    # Residual phase draws a fading glow plus many short sparks and tiny arcs around the target.
    if kind == "electric":
        dur = float(max(1, attack_duration))
        main_t = float(op.mul(dur, float(ELEC_MAIN_FRAC)))
        tt = float(t)

        # Main bolt phase
        if tt < main_t:
            for k in range(int(ELEC_BOLT_COUNT)):
                # Alpha flicker depends on parity of int(t).
                a = int(op.add(120, op.mul(60, (int(t) % 2))))
                col = (255, 255, 120, a)

                # Build one jagged polyline starting at the target center.
                # Each segment moves upward (y decreases) with alternating left right x offsets.
                pts = []
                x = cx
                y = cy
                pts.append((x, y))

                for j in range(int(ELEC_SEGMENTS)):
                    dx = random.randint(16, 34)
                    dy = random.randint(22, 38)
                    x = int(op.add(x, dx if (j % 2 == 0) else op.sub(0, dx)))
                    y = int(op.sub(y, dy))
                    pts.append((x, y))

                # Line thickness is ELEC_LINE_W.
                if len(pts) >= 2:
                    pygame.draw.lines(screen, col, False, pts, int(ELEC_LINE_W))

            # Extra target centered glow that pulses with time.
            glow_r = int(op.add(ELEC_GLOW_R0, op.mul(2, int(t) % 6)))
            pygame.draw.circle(screen, (200, 220, 255, 90), (cx, cy), glow_r, 3)
            return

        # Residual fade phase from main_t to dur
        rfrac = 0.0
        denom = float(op.sub(dur, main_t))
        if denom > 0.0:
            rfrac = clamp01(float(op.truediv(op.sub(tt, main_t), denom)))

        fade = float(op.sub(1.0, rfrac))
        a2 = int(op.mul(170.0, fade))  # alpha for sparks and arcs
        a3 = int(op.mul(120.0, fade))  # alpha for glow ring

        # Expanding glow ring from ELEC_GLOW_R0 to ELEC_GLOW_R1
        glow_r2 = int(op.add(ELEC_GLOW_R0, op.mul(float(op.sub(ELEC_GLOW_R1, ELEC_GLOW_R0)), rfrac)))
        pygame.draw.circle(screen, (160, 210, 255, a3), (cx, cy), glow_r2, 4)

        # Short random spark lines around the target, spread increases with rfrac
        for s in range(int(ELEC_RESIDUAL_SPARKS)):
            lim = int(op.add(14, op.mul(28.0, rfrac)))
            dx = random.randint(int(op.sub(0, lim)), int(lim))
            dy = random.randint(int(op.sub(0, lim)), int(lim))

            x0 = int(op.add(cx, dx))
            y0 = int(op.add(cy, dy))

            dx2 = random.randint(int(op.sub(0, 10)), 10)
            dy2 = random.randint(int(op.sub(0, 10)), 10)

            x1 = int(op.add(x0, dx2))
            y1 = int(op.add(y0, dy2))

            pygame.draw.line(screen, (255, 255, 180, a2), (x0, y0), (x1, y1), 3)

        # Tiny jagged arc polylines around the target
        for j in range(int(ELEC_RESIDUAL_ARCS)):
            pts = []
            x = int(op.add(cx, random.randint(int(op.sub(0, 18)), 18)))
            y = int(op.add(cy, random.randint(int(op.sub(0, 18)), 18)))
            pts.append((x, y))

            for k in range(4):
                dx = random.randint(10, 22)
                dy = random.randint(10, 22)
                x = int(op.add(x, dx if (k % 2 == 0) else op.sub(0, dx)))
                y = int(op.add(y, dy if (k % 2 == 1) else op.sub(0, dy)))
                pts.append((x, y))

            pygame.draw.lines(screen, (255, 255, 140, a2), False, pts, 2)

        return

    # Fire animation
    # Uses optional sprites: fire_ray and fire_blast.
    # Rays are repeatedly stamped along the attacker to target line, with perpendicular wiggle.
    # Blast is a pulsing impact sprite at the target with jitter and scale pulse.
    if kind == "fire":
        ray_raw = get_fx_raw_surface("fire_ray")
        blast_raw = get_fx_raw_surface("fire_blast")

        # Fallback draws rotating hot dots around the target.
        if ray_raw is None and blast_raw is None:
            for j in range(10):
                ang = float(op.add(op.mul(j, 0.6), op.mul(t, 0.4)))
                r = int(op.add(6, (t % 6)))
                ox = int(op.mul(22.0, math.cos(ang)))
                oy = int(op.mul(22.0, math.sin(ang)))
                col = (255, 140, 60, 140)
                pygame.draw.circle(screen, col, (int(op.add(cx, ox)), int(op.add(cy, oy))), r)
            return

        # Direction from attacker to target
        dx = float(op.sub(cx, sx))
        dy = float(op.sub(cy, sy))
        dist = float(math.hypot(dx, dy))
        if dist < 1.0:
            dist = 1.0

        dir_x = float(op.truediv(dx, dist))
        dir_y = float(op.truediv(dy, dist))

        # Rotate ray sprites so they align with the attacker to target line.
        ang_deg = float(math.degrees(math.atan2(dy, dx)))
        rot_deg = float(op.sub(0.0, ang_deg))

        dur = float(max(1, attack_duration))
        frac = clamp01(float(op.truediv(float(t), dur)))

        # Fire ray stamps along the line
        if ray_raw is not None:
            bw = int(ray_raw.get_width())
            bh = int(ray_raw.get_height())
            if bw > 0 and bh > 0:
                # FIRE_RAY_W controls the scaled width of the ray sprite.
                scale = float(op.truediv(float(FIRE_RAY_W), float(bw)))
                ray_h = int(op.mul(float(bh), scale))
                if ray_h < 1:
                    ray_h = 1

                ray_scaled = pygame.transform.smoothscale(ray_raw, (int(FIRE_RAY_W), int(ray_h)))
                ray_rot = pygame.transform.rotate(ray_scaled, rot_deg)

                # Base parameter that scrolls along the path.
                # FIRE_RAY_WAVES controls how fast the stamps advance per full animation.
                base_u = float(op.mul(frac, float(FIRE_RAY_WAVES)))

                for i in range(int(FIRE_RAY_COUNT)):
                    # FIRE_RAY_SPACING shifts stamps behind each other.
                    u = float(op.sub(base_u, op.mul(float(i), float(FIRE_RAY_SPACING))))
                    u = float(u % 1.0)

                    along = float(op.mul(u, dist))
                    px = float(op.add(float(sx), op.mul(dir_x, along)))
                    py = float(op.add(float(sy), op.mul(dir_y, along)))

                    # Small perpendicular wobble, hard coded amplitude 10.0 here.
                    wig = float(op.mul(10.0, math.sin(op.add(op.mul(frac, 14.0), float(i)))))
                    px = float(op.add(px, op.mul(op.sub(0.0, dir_y), wig)))
                    py = float(op.add(py, op.mul(dir_x, wig)))

                    bx = int(op.sub(int(px), op.truediv(ray_rot.get_width(), 2)))
                    by = int(op.sub(int(py), op.truediv(ray_rot.get_height(), 2)))
                    screen.blit(ray_rot, (bx, by))

        # Fire blast pulses at target
        if blast_raw is not None:
            bw2 = int(blast_raw.get_width())
            bh2 = int(blast_raw.get_height())
            if bw2 > 0 and bh2 > 0:
                # FIRE_BLAST_W controls the base scaled width of the blast sprite.
                scale2 = float(op.truediv(float(FIRE_BLAST_W), float(bw2)))
                blast_h = int(op.mul(float(bh2), scale2))
                if blast_h < 1:
                    blast_h = 1

                blast_scaled = pygame.transform.smoothscale(blast_raw, (int(FIRE_BLAST_W), int(blast_h)))

                # FIRE_BLAST_COUNT sets how many pulse cycles occur across the whole animation.
                phase = float(op.mul(frac, float(FIRE_BLAST_COUNT)))
                idx = int(phase)
                local = float(op.sub(phase, float(idx)))

                # Only draw during the early part of each cycle.
                if local < 0.45:
                    # Pulse goes from 1 down to 0 within that window.
                    pulse = float(op.sub(1.0, op.truediv(local, 0.45)))

                    # FIRE_BLAST_JITTER controls random target offset, scaled by pulse.
                    jj = int(op.mul(float(FIRE_BLAST_JITTER), pulse))
                    jx = random.randint(int(op.sub(0, jj)), int(jj))
                    jy = random.randint(int(op.sub(0, jj)), int(jj))

                    # Scale up then shrink back: w2 starts larger when pulse is high.
                    w2 = int(op.add(float(FIRE_BLAST_W), op.mul(60.0, pulse)))
                    if w2 < 40:
                        w2 = 40

                    scale3 = float(op.truediv(float(w2), float(bw2)))
                    h2 = int(op.mul(float(bh2), scale3))
                    if h2 < 1:
                        h2 = 1

                    blast2 = pygame.transform.smoothscale(blast_raw, (int(w2), int(h2)))
                    blast2.set_alpha(int(op.add(120, int(op.mul(120.0, pulse)))))

                    bx2 = int(op.sub(int(op.add(cx, jx)), op.truediv(blast2.get_width(), 2)))
                    by2 = int(op.sub(int(op.add(cy, jy)), op.truediv(blast2.get_height(), 2)))
                    screen.blit(blast2, (bx2, by2))

        return

    # Water animation
    # Two phases, split by WATER_MAIN_FRAC of the total duration.
    # Main phase: draws WATER_RIBBONS polylines from attacker toward target, extending over time.
    # Splash phase: draws expanding rings at the target plus WATER_SPLASH_COUNT droplets.
    if kind == "water":
        # If no attacker center, fallback to a pulsing ring at target.
        if src_center is None:
            rr = int(14 + 3 * (int(t) % 10))
            pygame.draw.circle(screen, (120, 180, 255, 160), (cx, cy), rr, 3)
            return

        sx = int(src_center[0])
        sy = int(src_center[1])
        tx = int(cx)
        ty = int(cy)

        dur = float(max(1, attack_duration))
        p = clamp01(float(t) / dur)

        # main_p goes 0 to 1 during the main jet portion
        main_p = clamp01(float(p) / float(max(0.000001, WATER_MAIN_FRAC)))

        # splash_p goes 0 to 1 after the main jet portion ends
        splash_p = clamp01(
            float(op.sub(p, WATER_MAIN_FRAC)) / float(max(0.000001, float(op.sub(1.0, WATER_MAIN_FRAC))))
        )

        # Vector from attacker to target
        dx = float(op.sub(tx, sx))
        dy = float(op.sub(ty, sy))
        dist = float(math.hypot(dx, dy))
        if dist < 1.0:
            dist = 1.0

        ux = float(op.truediv(dx, dist))
        uy = float(op.truediv(dy, dist))

        # Perpendicular vector, used to offset points sideways for wiggle and droplet spread
        px = float(op.sub(0.0, uy))
        py = float(ux)

        # Jet thickness interpolates from WATER_JET_W0 to WATER_JET_W1 during main phase
        w = int(op.add(WATER_JET_W0, op.mul(op.sub(WATER_JET_W1, WATER_JET_W0), main_p)))
        w = max(4, w)

        # Main jet phase
        if p <= WATER_MAIN_FRAC:
            # End point advances from attacker toward target.
            # Starts at 0.15 of the path and grows up to 1.0 as main_p reaches 1.
            end_frac = clamp01(float(op.add(0.15, op.mul(0.95, main_p))))
            ex = float(op.add(sx, op.mul(dx, end_frac)))
            ey = float(op.add(sy, op.mul(dy, end_frac)))

            # Draw multiple ribbon polylines, each one a wiggly line along the path.
            for k in range(int(WATER_RIBBONS)):
                pts = []
                denom = float(max(1, int(op.sub(WATER_JET_SEGMENTS, 1))))

                for s in range(int(WATER_JET_SEGMENTS)):
                    # u is the normalized position along the current visible portion of the jet
                    u = float(s) / denom
                    u = float(op.mul(u, end_frac))

                    along = float(op.mul(u, dist))
                    bx = float(op.add(sx, op.mul(ux, along)))
                    by = float(op.add(sy, op.mul(uy, along)))

                    # Wiggle phase depends on time, ribbon index, and position along the jet
                    phase = float(op.add(op.mul(float(t), WATER_WIGGLE_FREQ), op.mul(k, 1.3)))
                    phase = float(op.add(phase, op.mul(u, 6.0)))

                    # Wiggle amplitude increases toward the front of the jet
                    amp = float(op.mul(WATER_WIGGLE_AMP, float(op.add(0.35, op.mul(0.65, u)))))
                    wig = float(op.mul(amp, math.sin(phase)))

                    # Offset point sideways
                    ox = float(op.mul(px, wig))
                    oy = float(op.mul(py, wig))

                    pts.append((int(op.add(bx, ox)), int(op.add(by, oy))))

                # Three layered strokes to fake a bright water core with darker outline.
                if len(pts) >= 2:
                    pygame.draw.lines(screen, (40, 120, 220, 170), False, pts, max(2, w))
                    pygame.draw.lines(screen, (80, 170, 255, 190), False, pts, max(2, int(op.sub(w, 4))))
                    pygame.draw.lines(screen, (230, 250, 255, 210), False, pts, max(2, int(op.sub(w, 8))))

            # Droplets sprinkled along the current visible part of the jet.
            # Offsets are perpendicular, scaled by w.
            for _ in range(10):
                u2 = random.random() * float(end_frac)
                bx2 = float(op.add(sx, op.mul(dx, u2)))
                by2 = float(op.add(sy, op.mul(dy, u2)))
                off = float(op.mul(op.sub(random.random(), 0.5), op.mul(w, 0.9)))
                rx = int(op.add(bx2, op.mul(px, off)))
                ry = int(op.add(by2, op.mul(py, off)))
                pygame.draw.circle(screen, (200, 240, 255, 180), (rx, ry), random.randint(1, 3))

            # Rounded cap at the jet front, centered at (ex, ey).
            # Radius is (w / 2) plus WATER_WIGGLE_AMP, so the head is always chunky.
            cap_r = int(op.add(op.truediv(w, 2), WATER_WIGGLE_AMP))
            cap_r = max(6, cap_r)

            exi = int(ex)
            eyi = int(ey)
            pygame.draw.circle(screen, (40, 120, 220, 190), (exi, eyi), int(op.add(cap_r, 3)))
            pygame.draw.circle(screen, (80, 170, 255, 210), (exi, eyi), int(op.add(cap_r, 1)))
            pygame.draw.circle(screen, (110, 190, 255, 230), (exi, eyi), cap_r)

            return

        # Splash phase at the target
        r = int(op.add(WATER_SPLASH_R0, op.mul(op.sub(WATER_SPLASH_R1, WATER_SPLASH_R0), splash_p)))
        r = max(6, r)

        # Two expanding rings
        pygame.draw.circle(screen, (180, 235, 255, 200), (tx, ty), r, 3)
        pygame.draw.circle(screen, (80, 170, 255, 160), (tx, ty), max(1, int(op.sub(r, 8))), 3)

        # Splash droplets distributed around the rings
        for _ in range(int(WATER_SPLASH_COUNT)):
            ang = random.random() * (math.pi * 2.0)
            rr2 = float(op.mul(r, float(op.add(0.4, op.mul(0.9, random.random())))))
            px2 = int(op.add(tx, op.mul(math.cos(ang), rr2)))
            py2 = int(op.add(ty, op.mul(math.sin(ang), rr2)))
            pygame.draw.circle(screen, (210, 245, 255, 190), (px2, py2), random.randint(1, 3))

        return

    # Ground animation
    # Dust puffs are always drawn near the computed foot position.
    # Cracks grow during the first 0.55 fraction of the animation (grow = p / 0.55).
    # Crack drawing uses deterministic Random seeds so crack shapes do not flicker frame to frame.
    if kind == "ground":
        try:
            dur = float(max(1, attack_duration))
            p = clamp01(float(op.truediv(float(t), dur)))

            # Foot anchor position derived from target center plus a fraction of target_size.
            ts = int(target_size) if target_size is not None else int(BATTLE_PX)
            foot_y = int(op.add(cy, int(op.mul(float(ts), float(GROUND_FOOT_Y_FRAC)))))
            ox = int(cx)
            oy = int(foot_y)

            # Dust alpha pulses with time, color is constant.
            a = int(op.add(80, op.mul(5, (int(t) % 10))))
            dust_col = (160, 120, 70, a)

            # Six dust circles around the foot anchor, random offsets each frame.
            for j in range(6):
                dx = random.randint(10, 50)
                dy = random.randint(0, 25)
                sx2 = dx if (j % 2 == 0) else int(op.sub(0, dx))
                px1 = int(op.add(ox, sx2))
                py1 = int(op.add(oy, dy))
                pygame.draw.circle(screen, dust_col, (px1, py1), random.randint(4, 9))

            # Crack growth is limited to the first 55 percent of the animation.
            grow = clamp01(float(op.truediv(p, 0.55)))

            # Crack length interpolates from GROUND_CRACK_LEN0 to GROUND_CRACK_LEN1 as grow increases.
            crack_len = int(op.add(GROUND_CRACK_LEN0, op.mul(op.sub(GROUND_CRACK_LEN1, GROUND_CRACK_LEN0), grow)))

            crack_w = 1

            # Crack alpha scales up with grow, using GROUND_CRACK_ALPHA as the base.
            a2 = int(op.mul(float(GROUND_CRACK_ALPHA), float(op.add(0.35, op.mul(0.65, grow)))))
            a2 = max(0, min(255, int(a2)))
            crack_col = (0, 0, 0, int(a2))

            segs = int(max(3, GROUND_CRACK_SEGMENTS))
            base_step = float(op.truediv(float(crack_len), float(segs)))

            # Clamp angles to keep sin(angle) non negative, so cracks move downward on screen.
            def clamp_angle_down(rad_ang):
                twopi = float(op.mul(2.0, math.pi))
                ang = float(rad_ang % twopi)
                lo = float(math.radians(5.0))
                hi = float(math.radians(175.0))
                if ang < lo:
                    ang = lo
                if ang > hi:
                    ang = hi
                return ang

            # Pick a perpendicular direction that still moves downward on screen.
            def choose_down_perp(base_ang, rng_obj):
                a1 = float(op.add(base_ang, math.radians(90.0)))
                a2b = float(op.sub(base_ang, math.radians(90.0)))
                a1 = clamp_angle_down(a1)
                a2b = clamp_angle_down(a2b)

                s1 = math.sin(a1)
                s2 = math.sin(a2b)

                ok1 = s1 >= 0.0
                ok2 = s2 >= 0.0

                if ok1 and ok2:
                    return a1 if rng_obj.random() < 0.5 else a2b
                if ok1:
                    return a1
                return a2b

            # rng is unused later, but keeping it does not change behavior.
            rng = random.Random(4242)

            for k in range(int(GROUND_CRACK_COUNT)):
                # Per crack deterministic RNG, so each crack has a stable shape.
                rngk = random.Random(int(op.add(4242, op.mul(k, 97))))

                # Crack start point near the foot anchor.
                sx0 = int(op.add(ox, rngk.randint(int(op.sub(0, 10)), 10)))
                sy0 = int(op.add(oy, rngk.randint(int(op.sub(0, 4)), 4)))

                x = float(sx0)
                y = float(sy0)
                pts = [(int(x), int(y))]

                # Base direction is between 15 and 165 degrees.
                base_ang = float(rngk.uniform(math.radians(15.0), math.radians(165.0)))

                # Jitter scales with grow, so early cracks are less chaotic.
                j0 = float(op.mul(float(GROUND_CRACK_JITTER), float(op.add(0.20, op.mul(0.80, grow)))))
                j1 = float(op.mul(j0, 0.75))

                for s in range(segs):
                    # Even segments follow the base direction with jitter.
                    # Odd segments sometimes switch to a perpendicular direction to form jagged crack shapes.
                    if (s % 2) == 0:
                        ang = float(op.add(base_ang, rngk.uniform(op.sub(0.0, j0), j0)))
                    else:
                        perp = choose_down_perp(base_ang, rngk)
                        ang = float(op.add(perp, rngk.uniform(op.sub(0.0, j1), j1)))

                    ang = clamp_angle_down(ang)

                    # Step length is base_step times randomness, then scaled by grow.
                    step = float(op.mul(base_step, rngk.uniform(0.75, 1.30)))
                    step = float(op.mul(step, float(op.add(0.55, op.mul(0.45, grow)))))

                    dx = float(op.mul(math.cos(ang), step))
                    dy = float(op.mul(math.sin(ang), step))

                    x = float(op.add(x, dx))
                    y = float(op.add(y, dy))
                    pts.append((int(x), int(y)))

                    # Occasional side branch line from the current crack point.
                    if rngk.random() < 0.28:
                        b_ang = choose_down_perp(ang, rngk)
                        b_ang = float(op.add(b_ang, rngk.uniform(op.sub(0.0, j1), j1)))
                        b_ang = clamp_angle_down(b_ang)

                        b_step = float(op.mul(step, rngk.uniform(0.22, 0.42)))
                        bx = int(x)
                        by = int(y)
                        tx = int(op.add(bx, op.mul(math.cos(b_ang), b_step)))
                        ty = int(op.add(by, op.mul(math.sin(b_ang), b_step)))

                        pygame.draw.line(screen, crack_col, (bx, by), (tx, ty), crack_w)

                if len(pts) >= 2:
                    pygame.draw.lines(screen, crack_col, False, pts, crack_w)
            return

        # If anything fails in the crack logic, fallback draws only dust around the target center.
        except Exception:
            col = (160, 120, 70, 140)
            for j in range(6):
                dx = random.randint(10, 50)
                dy = random.randint(0, 25)
                px1 = int(op.add(cx, dx if (j % 2 == 0) else op.sub(0, dx)))
                py1 = int(op.add(cy, dy))
                pygame.draw.circle(screen, col, (px1, py1), random.randint(4, 9))
            return

    # Rock animation
    # Three rocks fall from above, spaced horizontally.
    # After each rock impacts, it bounces using sin() and spawns splatter particles.
    if kind == "rock":
        rock = get_fx_surface("rock", 56)
        if rock is None:
            col = (170, 150, 120, 150)
            for j in range(3):
                pygame.draw.circle(
                    screen,
                    col,
                    (int(op.add(cx, op.mul(op.sub(j, 1), 35))), int(op.sub(cy, 20))),
                    10,
                )
            return

        rw, rh = rock.get_size()

        # ground_y ends up below the target center because it subtracts a negative number.
        ground_y = int(op.sub(cy, -40))
        fall_h = 170

        speed = ROCK_FALL_SPEED
        bounce_amp = ROCK_BOUNCE_AMP
        bounce_freq = ROCK_BOUNCE_FREQ

        for i in range(3):
            delay = int(op.mul(i, ROCK_DELAY))
            tt = float(op.sub(t, delay))
            if tt < 0.0:
                continue

            # horizontal spacing of the three rocks
            xoff = int(op.mul(op.sub(i, 1), 40))

            y0 = float(op.sub(ground_y, fall_h))
            y_raw = float(op.add(y0, op.mul(speed, tt)))

            impacted = y_raw >= float(ground_y)

            # after impact, y becomes ground_y minus a bouncing amount
            y = y_raw
            if impacted:
                over = float(op.sub(y_raw, float(ground_y)))
                phase = float(op.mul(over, bounce_freq))
                bounce = float(op.mul(bounce_amp, abs(math.sin(phase))))
                y = float(op.sub(float(ground_y), bounce))

            x = int(op.add(cx, xoff))
            bx = int(op.sub(x, op.truediv(rw, 2)))
            by = int(op.sub(int(y), op.truediv(rh, 2)))
            screen.blit(rock, (bx, by))

            # Splatter after impact
            if impacted:
                # time since impact, computed by subtracting the fall time
                fall_time = float(op.truediv(op.sub(ground_y, y0), max(1.0, speed)))
                t_imp = float(op.sub(tt, fall_time))
                t_imp = float(op.add(t_imp, ROCK_SPLATTER_START_DELAY))

                if 0.0 <= t_imp <= float(ROCK_SPLATTER_LIFE):
                    life = float(ROCK_SPLATTER_LIFE)
                    u = clamp01(float(op.truediv(t_imp, life)))
                    fade = float(op.sub(1.0, u))
                    a = int(op.mul(float(ROCK_SPLATTER_ALPHA), fade))
                    a = max(0, min(255, a))
                    base_col = (170, 150, 120, a)

                    # deterministic per rock index so splatter does not flicker frame to frame
                    rng = random.Random(1000 + i)

                    for k in range(int(ROCK_SPLATTER_COUNT)):
                        # angle fan and a little random spread
                        ang = float(op.add(0.4, op.mul(k, 0.75)))
                        ang = float(op.add(ang, rng.random() * 0.35))

                        vx = float(op.mul(ROCK_SPLATTER_SPEED, math.cos(ang)))
                        vy = float(op.mul(ROCK_SPLATTER_SPEED, math.sin(ang)))

                        vx = float(op.mul(vx, 1.2))

                        # ballistic motion, y uses ground_y minus upward travel and gravity term
                        px = float(op.add(x, op.mul(vx, t_imp)))
                        py = float(op.sub(ground_y, op.add(op.mul(vy, t_imp), op.mul(ROCK_SPLATTER_GRAV, t_imp * t_imp))))

                        # extra spread jitter for messy look
                        px = float(op.add(px, rng.uniform(-ROCK_SPLATTER_SPREAD_X, ROCK_SPLATTER_SPREAD_X) * 0.08))
                        py = float(op.add(py, rng.uniform(-ROCK_SPLATTER_SPREAD_Y, ROCK_SPLATTER_SPREAD_Y) * 0.05))

                        r = rng.randint(ROCK_SPLATTER_RADIUS_MIN, ROCK_SPLATTER_RADIUS_MAX)
                        pygame.draw.circle(screen, base_col, (int(px), int(py)), r)

        return

    # Poison animation
    # Bubble sprites rise from below the target, with sideways wiggle and fade as they rise.
    if kind == "poison":
        bub = get_fx_surface("poison_2", POISON_SIZE_W)
        if bub is None:
            col = (180, 90, 200, 140)
            for j in range(6):
                yy = int(op.add(cy, op.sub(70, op.mul(j, 22))))
                xx = int(op.add(cx, int(op.mul(18.0, math.sin(op.add(op.mul(t, 0.35), op.mul(j, 1.3)))))))
                pygame.draw.circle(screen, col, (xx, yy), 10, 2)
            return

        bw, bh = bub.get_size()

        start_y = int(op.add(cy, 80))
        rise_h = int(POISON_MAX_RISE)

        for i in range(int(POISON_BUBBLE_COUNT)):
            delay = int(op.mul(i, POISON_DELAY))
            tt = float(op.sub(t, delay))
            if tt < 0.0:
                continue

            dy = float(op.mul(POISON_RISE_SPEED, tt))
            if dy > float(rise_h):
                continue

            # wiggle and drift control x position
            wig = float(op.mul(POISON_WIGGLE_AMP, math.sin(op.add(op.mul(tt, POISON_WIGGLE_FREQ), op.mul(i, 1.7)))))
            drift = int(op.mul(op.sub(i, op.truediv(POISON_BUBBLE_COUNT, 2)), 6))
            x = int(op.add(cx, op.add(int(wig), drift)))

            y = int(op.sub(start_y, int(dy)))

            # alpha decreases with height: higher bubbles are more transparent
            frac = 1.0
            if rise_h > 0:
                frac = clamp01(float(op.truediv(dy, rise_h)))

            a = int(op.add(60, op.mul(180.0, op.sub(1.0, frac))))
            a = max(0, min(255, a))

            temp = bub.copy()
            temp.set_alpha(a)

            bx = int(op.sub(x, op.truediv(bw, 2)))
            by = int(op.sub(y, op.truediv(bh, 2)))
            screen.blit(temp, (bx, by))

        return

    # Fighting animation
    # Punch sprite scales down over time and jitters less over time.
    if kind == "fighting":
        base = get_fx_raw_surface("punch")
        if base is None:
            base = get_fx_raw_surface("puch")

        if base is None:
            col = (240, 240, 240, 160)
            rr = int(op.add(22, op.mul(2, t)))
            pygame.draw.circle(screen, col, (cx, cy), rr, 4)
            return

        dur = float(max(1, attack_duration))
        frac = clamp01(float(op.truediv(float(t), dur)))

        hit_t = float(PUNCH_HIT_FRAC)
        if frac < hit_t:
            local = float(op.truediv(frac, hit_t))
        else:
            # After the hit fraction, local is forced to 1.0 in this code.
            denom = float(max(0.000000001, op.sub(1.0, hit_t)))
            local = float(op.truediv(op.sub(frac, hit_t), denom))
            local = 1.0

        inv = float(op.sub(1.0, frac))

        # Width eases from PUNCH_START_W down to PUNCH_END_W as frac increases.
        w_float = float(op.add(PUNCH_END_W, op.mul(op.sub(PUNCH_START_W, PUNCH_END_W), inv)))
        w = int(op.mul(10, round(op.truediv(w_float, 10.0))))
        w = max(40, w)

        bw = base.get_width()
        bh = base.get_height()
        if bw <= 0 or bh <= 0:
            return

        scale = float(op.truediv(float(w), float(bw)))
        h = int(op.mul(float(bh), scale))
        if h <= 0:
            return

        fist = pygame.transform.smoothscale(base, (int(w), int(h)))
        fist.set_alpha(int(PUNCH_ALPHA))

        # wob is a horizontal sinusoidal sway.
        wob = int(op.mul(PUNCH_WIGGLE_PX, math.sin(op.mul(frac, 10.0))))

        # punch_in is an extra shove term driven by local.
        punch_in = int(op.mul(36.0, math.sin(op.mul(local, 1.2))))

        ox = int(op.add(wob, punch_in))
        oy = int(op.sub(0, int(op.mul(12.0, math.sin(op.mul(local, 1.6))))))

        # shake amplitude shrinks as inv shrinks.
        shake = int(op.mul(PUNCH_SHAKE_PX, inv))
        jx = random.randint(int(op.sub(0, shake)), int(shake))
        jy = random.randint(int(op.sub(0, shake)), int(shake))

        x = int(op.add(cx, op.add(ox, jx)))
        y = int(op.add(cy, op.add(oy, jy)))

        bx = int(op.sub(x, op.truediv(fist.get_width(), 2)))
        by = int(op.sub(y, op.truediv(fist.get_height(), 2)))
        screen.blit(fist, (bx, by))

        return

    # Grass animation
    # This effect is a three phase combo per leaf:
    # Phase A rises fast above the attacker
    # Phase B floats down slowly like a drifting leaf
    # Phase C launches fast to the target like a thrown leaf star
    # Leaves are staggered with LEAF_DELAY so the screen stays busy.
    
    if kind in ("grass", "plant"):
        # Load the raw leaf sprite (unscaled, with transparency).
        leaf_raw = get_fx_raw_surface("leaf_transparent")
    
        # Asset missing fallback: draw simple green rings around the target.
        if leaf_raw is None:
            col = (120, 200, 80, 180)  # soft green, semi transparent
            for j in range(10):  # draw multiple little circles
                ox = random.randint(int(op.sub(0, 40)), 40)  # random x offset
                oy = random.randint(int(op.sub(0, 40)), 40)  # random y offset
                pygame.draw.circle(screen, col, (int(cx + ox), int(cy + oy)), 6, 2)
            return
    
        # This effect needs attacker position, because the leaves rise from attacker.
        if src_center is None:
            return
    
        # Attacker center (source point for leaf motion).
        ax = int(src_center[0])
        ay = int(src_center[1])
    
        # Target center (impact point for launched leaves).
        tx = int(cx)
        ty = int(cy)
    
        # Read raw sprite size so we can scale it safely.
        bw = int(leaf_raw.get_width())
        bh = int(leaf_raw.get_height())
        if bw <= 0 or bh <= 0:
            return
    
        # Scale the leaf to a fixed width LEAF_W while keeping aspect ratio.
        scale0 = float(LEAF_W) / float(bw)
        leaf_h = int(float(bh) * scale0)
        if leaf_h < 1:
            leaf_h = 1
        leaf_base = pygame.transform.smoothscale(leaf_raw, (int(LEAF_W), int(leaf_h)))
    
        # Phase lengths in frames for one leaf.
        rise_n = float(max(1, LEAF_RISE_FRAMES))       # rise time
        fall_n = float(max(1, LEAF_FALL_FRAMES))       # float down time
        launch_n = float(max(1, LEAF_LAUNCH_FRAMES))   # launch time
        total_n = float(rise_n + fall_n + launch_n)    # total lifetime per leaf
    
        # Ease out makes motion fast early, slow near the end.
        def ease_out(x):
            x = clamp01(x)
            return 1.0 - (1.0 - x) * (1.0 - x)
    
        # Ease in makes motion slow early, fast near the end.
        def ease_in(x):
            x = clamp01(x)
            return x * x
    
        # Ease in out accelerates then decelerates.
        def ease_in_out(x):
            x = clamp01(x)
            if x < 0.5:
                return 2.0 * x * x
            return 1.0 - 2.0 * (1.0 - x) * (1.0 - x)
    
        # Each leaf is offset in time by LEAF_DELAY so they do not all move together.
        for i in range(int(LEAF_COUNT)):
            delay = int(i * int(LEAF_DELAY))   # per leaf start delay in frames
            tt = float(t - delay)              # local time for this leaf
            if tt < 0.0 or tt > total_n:
                continue
    
            # spread_center is centered around 0 so leaves distribute left to right.
            spread_center = float(i) - float(LEAF_COUNT - 1) / 2.0
    
            # start_spread defines the base horizontal offset for this leaf.
            start_spread = spread_center * (float(LEAF_START_SPREAD) / float(max(1, LEAF_COUNT - 1)))
    
            # Wiggle phase uses time plus leaf index so each leaf wiggles differently.
            wig_phase = float(tt * float(LEAF_WIGGLE_FREQ) + float(i) * 1.7)
    
            # Wiggle is a sine wave x motion to make it feel organic.
            wig = float(float(LEAF_WIGGLE_AMP) * math.sin(wig_phase))
    
            # sway_x is the total horizontal offset applied in all phases.
            sway_x = float(start_spread + wig)
    
            # fx, fy is the final draw position for the current frame.
            fx = float(ax)
            fy = float(ay)
    
            # Choose which phase to run based on local time tt.
            if tt <= rise_n:
                # Phase A rise fast
                u = float(tt / rise_n)         # normalized progress inside rise
                u = ease_out(u)                # fast rise then gentle stop
    
                apex_y = float(ay - int(LEAF_APEX_Y))  # leaf apex above attacker
    
                fx = float(ax + sway_x)                 # x with sway
                fy = float(ay + (apex_y - float(ay)) * u)  # interpolate y to apex
    
                rot = float(i * 25.0 + tt * 1.5)  # gentle spin during rise
                a = int(LEAF_ALPHA)               # constant alpha
    
            elif tt <= rise_n + fall_n:
                # Phase B fall slow and float
                u = float((tt - rise_n) / fall_n)  # normalized progress inside fall
                u = ease_in(u)                     # slow then faster downward
    
                apex_y = float(ay - int(LEAF_APEX_Y))  # start fall from apex
    
                # Add drift plus wobble so the leaf floats instead of dropping straight.
                fx = float(ax + sway_x + float(LEAF_FALL_DRIFT) * math.sin(tt * 0.08 + i))
                fy = float(apex_y + (float(ay) - apex_y) * u)  # move back to attacker y
    
                # Flutter rotation with a sine term.
                rot = float(i * 25.0 + (tt - rise_n) * 2.2 + 18.0 * math.sin(tt * 0.06 + i))
                a = int(LEAF_ALPHA)
    
            else:
                # Phase C launch fast to target
                u = float((tt - rise_n - fall_n) / launch_n)  # normalized progress inside launch
                u = ease_in_out(u)                             # accelerate then brake at impact
    
                # Play hit sound once per leaf near the end of its launch.
                if u >= 0.98:
                    if i not in leaf_hit_seen:
                        if leaf_hit_sfx is not None:
                            leaf_hit_sfx.play()
                        leaf_hit_seen.add(i)
    
                # Launch starts near attacker, with reduced sway so it aims cleaner.
                sx2 = float(ax + sway_x * 0.35)
                sy2 = float(ay)
    
                # Spread impacts so multiple leaves do not stack perfectly.
                spread2 = spread_center * (float(LEAF_TARGET_SPREAD) / float(max(1, LEAF_COUNT - 1)))
                ex2 = float(tx + spread2)            # end x around target
                ey2 = float(ty + 0.25 * spread2)     # tiny y spread too
    
                # Linear interpolation from start to end.
                fx = float(op.add(sx2, op.mul(op.sub(ex2, sx2), u)))
                fy = float(op.add(sy2, op.mul(op.sub(ey2, sy2), u)))
    
                # Spin harder in launch to look like a thrown projectile.
                rot = float(i * 40.0 + (tt - rise_n - fall_n) * float(LEAF_SPIN_DEG_PER_FRAME) * 6.0)
    
                # Small fade near the end so it does not look glued to the target.
                a = int(float(LEAF_ALPHA) * float(1.0 - 0.25 * u))
                if a < 0:
                    a = 0
    
            # Rotate the leaf sprite for this frame.
            leaf_rot = pygame.transform.rotate(leaf_base, rot)
    
            # Apply alpha for this frame.
            leaf_rot.set_alpha(max(0, min(255, int(a))))
    
            # Draw centered on (fx, fy).
            bx = int(fx - leaf_rot.get_width() / 2)
            by = int(fy - leaf_rot.get_height() / 2)
            screen.blit(leaf_rot, (bx, by))
    
        return
    
    # Psychic animation
    # A train of rings travels from attacker to target.
    # Rings are spaced by PSY_SPACING so multiple rings are visible at once.
    # While travelling, each ring is drawn along the source to target line.
    # After reaching the target, rings become expanding impact pulses.
    # A small wobble modulates radius so it feels alive.
    # If src_center is missing, use a simple pulsing target ring.
    
    if kind == "psychic" or kind == "physics":
        # Fallback if we do not know the attacker position.
        if src_center is None:
            rr = int(op.add(24, op.mul(2, int(t) % 12)))  # pulsing radius
            pygame.draw.circle(screen, (210, 120, 255), (cx, cy), rr, 4)
            return
    
        # Source and target points.
        sx = int(src_center[0])
        sy = int(src_center[1])
        tx = int(cx)
        ty = int(cy)
    
        # Vector from source to target.
        dx = float(op.sub(tx, sx))
        dy = float(op.sub(ty, sy))
    
        # Distance is used for normalization and safety.
        dist = float(math.hypot(dx, dy))
        if dist < 1.0:
            dist = 1.0
    
        # Convert time t to a normalized fraction across attack_duration.
        dur = float(max(1, attack_duration))
        frac = clamp01(float(op.truediv(float(t), dur)))
    
        # span defines how far the ring train head moves as frac goes from 0 to 1.
        span = float(op.add(1.0, op.mul(float(PSY_SPACING), float(op.sub(PSY_CIRCLE_COUNT, 1)))))
    
        # base is the head position along the train.
        base = float(op.mul(frac, span))
    
        # Draw PSY_CIRCLE_COUNT rings.
        for i in range(int(PSY_CIRCLE_COUNT)):
            # u_raw is where this ring sits relative to the head, spaced backwards.
            u_raw = float(op.sub(base, op.mul(float(PSY_SPACING), float(i))))
            if u_raw < 0.0:
                continue
    
            # wobble changes radius slightly so rings do not look static.
            wob = float(
                op.mul(
                    float(PSY_RING_WIGGLE),
                    math.sin(op.add(op.mul(u_raw, 14.0), op.mul(float(i), 1.7))),
                )
            )
    
            # Travel phase: u_raw below 1 means the ring is between attacker and target.
            if u_raw < 1.0:
                # Position moves along the source to target line using u_raw as lerp factor.
                px = float(op.add(float(sx), op.mul(dx, u_raw)))
                py = float(op.add(float(sy), op.mul(dy, u_raw)))
    
                # Radius grows while travelling.
                r0 = float(PSY_RING_R_MIN)
                r1 = float(PSY_RING_R_MAX)
                r = int(op.add(r0, op.mul(op.sub(r1, r0), clamp01(u_raw))))
                r = int(op.add(r, wob))
                if r < 2:
                    r = 2
    
                # Ring outline thickness.
                w = int(PSY_RING_W)
                if w < 1:
                    w = 1
    
                # Main ring plus small inner highlight ring.
                pygame.draw.circle(screen, (210, 120, 255), (int(px), int(py)), int(r), int(w))
                pygame.draw.circle(screen, (255, 235, 255), (int(px), int(py)), max(2, int(op.sub(r, 3))), 1)
                continue
    
            # Impact phase: u_raw above 1 means ring has reached the target and expands.
            impact_u = float(op.truediv(op.sub(u_raw, 1.0), float(PSY_IMPACT_SPAN)))
            impact_u = clamp01(impact_u)
    
            # Impact radius expands from PSY_IMPACT_R0 to PSY_IMPACT_R1.
            ir0 = float(PSY_IMPACT_R0)
            ir1 = float(PSY_IMPACT_R1)
            ir = int(op.add(ir0, op.mul(op.sub(ir1, ir0), impact_u)))
            if ir < 2:
                ir = 2
    
            # Impact ring thickness interpolates from PSY_IMPACT_W0 to PSY_IMPACT_W1.
            iw0 = float(PSY_IMPACT_W0)
            iw1 = float(PSY_IMPACT_W1)
            iw = int(op.add(iw0, op.mul(op.sub(iw1, iw0), impact_u)))
            if iw < 1:
                iw = 1
    
            # Draw the impact pulse centered on the target.
            pygame.draw.circle(screen, (210, 120, 255), (tx, ty), ir, iw)
            pygame.draw.circle(screen, (255, 235, 255), (tx, ty), max(2, int(op.sub(ir, 6))), 1)
    
        return
    
    # Ice animation
    # Two modes:
    # If src_center is missing, do a local swirl around the target.
    # If src_center exists, send shards from attacker to target with wobble and slight gravity.
    # Shards fade in and out using a sin curve so they pop less harshly.
    # After the travel lifetime ends, spawn a small burst of ice bits on the target.
    
    if kind == "ice":
        # Load and scale a snowflake sprite for the main shards.
        flake = get_fx_surface("snowflake_transparent", ICE_FLAKE_W)
    
        # Asset missing fallback: simple pulsing ring.
        if flake is None:
            rr = int(10 + 2 * (int(t) % 12))
            pygame.draw.circle(screen, (200, 240, 255, 160), (cx, cy), rr, 2)
            return
    
        # If attacker position unknown, run a target only swirl.
        if src_center is None:
            for i in range(int(ICE_COUNT)):
                rng = random.Random(9000 + i)  # stable randomness per shard
                ang = float(rng.uniform(0.0, math.pi * 2.0) + 0.25 * float(t))  # swirl angle
                rad = float(8.0 + 1.8 * float(t))  # radius grows over time
                x = float(cx) + rad * math.cos(ang)  # swirl x position
                y = float(cy) + rad * math.sin(ang)  # swirl y position
    
                rot = float(rng.uniform(0.0, 360.0) + float(t) * float(ICE_SPIN_DEG_PER_FRAME))  # rotation
                sc = float(rng.uniform(ICE_SCALE_MIN, ICE_SCALE_MAX))  # per shard scale
    
                img = pygame.transform.rotozoom(flake, rot, sc)  # rotate and scale sprite
                img.set_alpha(180)  # constant alpha for swirl mode
    
                bw, bh = img.get_size()  # size after rotozoom
                screen.blit(img, (int(x - bw / 2), int(y - bh / 2)))  # draw centered
    
            return
    
        # Source and target points for travel mode.
        sx = int(src_center[0])
        sy = int(src_center[1])
        tx = int(cx)
        ty = int(cy)
    
        # Vector from attacker to target.
        dx = float(op.sub(tx, sx))
        dy = float(op.sub(ty, sy))
    
        # Normalize direction.
        dist = float(math.hypot(dx, dy))
        if dist < 1.0:
            dist = 1.0
    
        ux = float(op.truediv(dx, dist))  # unit direction x
        uy = float(op.truediv(dy, dist))  # unit direction y
    
        # Perpendicular direction for sideways wobble.
        px = float(op.sub(0.0, uy))
        py = float(ux)
    
        # Duration fraction for the whole attack, used for consistent timing.
        dur = float(max(1, attack_duration))
        base_p = clamp01(float(op.truediv(float(t), dur)))
    
        # For each shard, we use stable randomness and per shard stagger.
        for i in range(int(ICE_COUNT)):
            rng = random.Random(1337 + i * 10007)  # stable shard identity
    
            # Stagger each shard so they are not a uniform line.
            delay = int(op.add(op.mul(i, ICE_STAGGER_FRAMES), rng.randint(0, ICE_JITTER_MAX)))
            tt = float(op.sub(float(t), float(delay)))  # local time for this shard
            if tt < 0.0:
                continue
    
            # life is travel time, bits_life is burst time after impact.
            life = float(max(1.0, float(ICE_LIFE_FRAMES)))
            bits_life = float(max(1.0, float(ICE_BITS_LIFE_FRAMES)))
    
            # tt2 loops travel motion so multiple shards can be active.
            tt2 = float(tt % life)
    
            # p is normalized progress in travel portion.
            p = float(op.truediv(tt2, life))
    
            # p2 is smoothstep easing so motion starts and ends smoothly.
            p2 = float(p * p * (3.0 - 2.0 * p))
    
            # along is distance traveled along the line.
            along = float(op.mul(p2, dist))
    
            # Base position along the line.
            x0 = float(op.add(float(sx), op.mul(ux, along)))
            y0 = float(op.add(float(sy), op.mul(uy, along)))
    
            # curve peaks mid flight, used for wobble and alpha.
            curve = float(math.sin(p * math.pi))
    
            # side picks a random wobble amount per shard.
            side = float(rng.uniform(op.sub(0.0, ICE_SIDE_SPREAD), ICE_SIDE_SPREAD))
    
            # wob is sideways offset, strongest in the middle.
            wob = float(op.mul(side, curve))
    
            # Apply sideways wobble.
            x = float(op.add(x0, op.mul(px, wob)))
            y = float(op.add(y0, op.mul(py, wob)))
    
            # gravity factor is zero at start and end, so it still hits the target.
            grav_fac = float(op.mul(p, op.sub(1.0, p)))
            y = float(op.add(y, op.mul(float(ICE_GRAVITY), grav_fac)))
    
            # spin values for shard rotation.
            rot0 = float(rng.uniform(0.0, 360.0))
            rot = float(op.add(rot0, op.mul(float(tt), float(ICE_SPIN_DEG_PER_FRAME))))
    
            # per shard scale.
            sc = float(rng.uniform(ICE_SCALE_MIN, ICE_SCALE_MAX))
    
            # Travel draw: only while tt is within one life.
            if tt <= life:
                img = pygame.transform.rotozoom(flake, rot, sc)  # rotate and scale
    
                # alpha fades in then out using curve.
                a = int(op.mul(float(ICE_ALPHA), curve))
                a = max(0, min(255, a))
                img.set_alpha(a)
    
                bw, bh = img.get_size()
                screen.blit(img, (int(op.sub(x, bw / 2)), int(op.sub(y, bh / 2))))
    
            # Burst draw: after travel life ends, spawn bits around target for bits_life frames.
            if tt >= life and tt <= float(op.add(life, bits_life)):
                dt_hit = float(op.sub(tt, life))  # time since impact
                u_hit = clamp01(float(op.truediv(dt_hit, bits_life)))  # normalized burst progress
                fade = float(op.sub(1.0, u_hit))  # fade out over burst time
    
                # Burst center is exactly target.
                x_hit = float(tx)
                y_hit = float(ty)
    
                # alpha for bits fades out.
                a_bits = int(op.mul(float(ICE_BITS_ALPHA), fade))
                a_bits = max(0, min(255, a_bits))
                col_bits = (190, 235, 255, a_bits)
    
                # Stable randomness for bit directions.
                rng2 = random.Random(77777 + i * 1009)
    
                for k in range(int(ICE_BITS_COUNT)):
                    # base angle spreads evenly in a circle.
                    base_ang = float(op.mul((math.pi * 2.0), float(op.truediv(float(k), float(max(1, ICE_BITS_COUNT))))))
    
                    # add jitter so it is not too symmetric.
                    ang = float(op.add(base_ang, rng2.uniform(op.sub(0.0, ICE_BITS_SPREAD_JITTER), ICE_BITS_SPREAD_JITTER)))
    
                    # randomize speed a bit.
                    sp = float(op.mul(float(ICE_BITS_SPEED), rng2.uniform(0.75, 1.15)))
    
                    # velocity components.
                    vx = float(op.mul(sp, math.cos(ang)))
                    vy = float(op.mul(sp, math.sin(ang)))
    
                    # dt scales motion so bits are not too fast.
                    dt = float(op.truediv(dt_hit, 6.0))
    
                    # gravity factor grows over time.
                    gfac = float(op.mul(u_hit, u_hit))
    
                    # ballistic position with gravity.
                    px2 = float(op.add(x_hit, op.mul(vx, dt)))
                    py2 = float(
                        op.add(
                            y_hit,
                            op.add(
                                op.mul(vy, dt),
                                op.mul(op.mul(float(ICE_BITS_GRAVITY), gfac), dt * dt),
                            ),
                        )
                    )
    
                    # random radius per bit.
                    rr2 = rng2.randint(int(ICE_BITS_RADIUS_MIN), int(ICE_BITS_RADIUS_MAX))
    
                    # draw one bit.
                    pygame.draw.circle(screen, col_bits, (int(px2), int(py2)), int(rr2))
    
        return
    
    # Dark animation
    # Creates a target shake plus a shrinking halo ring.
    # inv decreases from 1 to 0 across the animation, so shake fades out.
    # A pulse modulates ring alpha so it feels alive.
    # Orbiting motes add motion without heavy assets.
    
    if kind == "dark":
        dur = float(max(1, attack_duration))  # avoid divide by zero
        p = clamp01(float(op.truediv(float(t), dur)))  # normalized progress
        inv = float(op.sub(1.0, p))  # inverted progress, strong early, weak late
    
        # Shake amplitude fades out as inv goes to 0.
        shake_amp = int(op.mul(float(DARK_SHAKE_PX), inv))
        if shake_amp < 0:
            shake_amp = 0
    
        # Random jitter for screen shake illusion at the target.
        if shake_amp > 0:
            jx = random.randint(int(op.sub(0, shake_amp)), int(shake_amp))
            jy = random.randint(int(op.sub(0, shake_amp)), int(shake_amp))
        else:
            jx = 0
            jy = 0
    
        # Jittered center.
        x0 = int(op.add(cx, jx))
        y0 = int(op.add(cy, jy))
    
        # Ring radius shrinks from DARK_RING_R0 to DARK_RING_R1.
        r = int(op.add(float(DARK_RING_R1), op.mul(float(op.sub(DARK_RING_R0, DARK_RING_R1)), inv)))
    
        # Ring width transitions from DARK_RING_W0 to DARK_RING_W1.
        w = int(op.add(float(DARK_RING_W1), op.mul(float(op.sub(DARK_RING_W0, DARK_RING_W1)), inv)))
        if w < 1:
            w = 1
    
        # Pulse affects alpha so the ring breathes.
        pulse = float(op.mul(inv, float(0.35 + 0.65 * abs(math.sin(op.mul(float(t), 0.18))))))
    
        # Outer ring alpha scales with pulse.
        a1 = int(op.mul(float(DARK_RING_ALPHA), pulse))
        a1 = max(0, min(255, a1))
    
        # Outer and inner colors with alpha.
        col_outer = (70, 0, 110, a1)
        col_inner = (140, 40, 190, int(op.mul(float(a1), 0.7)))
    
        # Draw halo ring.
        pygame.draw.circle(screen, col_outer, (x0, y0), int(r), int(w))
    
        # Inner ring for depth.
        pygame.draw.circle(screen, col_inner, (x0, y0), max(1, int(op.sub(r, 10))), max(1, int(op.sub(w, 1))))
    
        # Orbiting mote alpha fades out over time.
        a2 = int(op.mul(float(DARK_MOTE_ALPHA), inv))
        a2 = max(0, min(255, a2))
        col_mote = (120, 30, 170, a2)
    
        # Draw motes in orbit so it feels alive.
        for k in range(int(DARK_MOTE_COUNT)):
            ang = float(
                op.add(
                    op.mul(float(k), 6.283185307179586 / float(max(1, DARK_MOTE_COUNT))),
                    op.mul(float(t), 0.14),
                )
            )
    
            # Orbit radius pulses slightly per mote.
            rr = float(op.mul(float(DARK_MOTE_ORBIT_R), float(0.55 + 0.45 * abs(math.sin(op.add(ang, 1.7))))))
    
            # Mote position on orbit.
            mx = int(op.add(x0, op.mul(rr, math.cos(ang))))
            my = int(op.add(y0, op.mul(rr, math.sin(ang))))
    
            # Stable size randomness per mote.
            rad = random.Random(90000 + k).randint(int(DARK_MOTE_R_MIN), int(DARK_MOTE_R_MAX))
    
            # Draw mote.
            pygame.draw.circle(screen, col_mote, (mx, my), int(rad))
    
        return
    
    # Bug animation
    # Uses a web sprite that scales up over time as if it is thrown onto the target.
    # Progress p is eased (p2) so the scale ramps smoothly.
    # If the asset is missing, use a simple expanding circle fallback.
    
    if kind == "bug":
        web_raw = get_fx_raw_surface("spider_web")  # raw web sprite
    
        # Asset missing fallback: expanding gray ring.
        if web_raw is None:
            rr = int(op.add(12, op.mul(2, int(t) % 14)))
            pygame.draw.circle(screen, (230, 230, 230, 170), (cx, cy), rr, 3)
            return
    
        dur = float(max(1, attack_duration))  # avoid divide by zero
    
        speed = 3.0  # speed multiplier so the web expands quicker
        p = clamp01(float(t) * speed / dur)  # normalized progress with speed
    
        # Smoothstep easing for clean start and stop.
        p2 = p * p * (3.0 - 2.0 * p)
    
        # Default target size if missing.
        if target_size is None:
            target_size = BATTLE_PX
    
        # max_w sets the final web width relative to target.
        max_w = int(op.mul(float(target_size), 1.6))
        if max_w < 1:
            max_w = 1
    
        # Current width interpolates from 0 to max_w.
        w = int(op.mul(float(max_w), p2))
        if w < 1:
            w = 1
    
        # Read raw sprite size.
        bw = int(web_raw.get_width())
        bh = int(web_raw.get_height())
        if bw <= 0 or bh <= 0:
            return
    
        # Scale sprite to width w while keeping aspect ratio.
        scale = float(op.truediv(float(w), float(bw)))
        h = int(op.mul(float(bh), scale))
        if h < 1:
            h = 1
    
        # Smooth scale makes it less pixelated.
        img = pygame.transform.smoothscale(web_raw, (int(w), int(h)))
    
        # Full alpha so it feels sticky and immediate.
        img.set_alpha(255)
    
        # Draw centered on target.
        bx = int(op.sub(cx, op.truediv(img.get_width(), 2)))
        by = int(op.sub(cy, op.truediv(img.get_height(), 2)))
        screen.blit(img, (bx, by))
        return
    
    # Dragon animation
    # Draws multiple ray sprites along the attacker to target line.
    # Rays are spaced by DRAGON_RAY_SPACING and loop using u modulo 1.
    # Each ray position wobbles sideways using a perpendicular vector.
    # Alpha pulses so the trail looks energetic.
    # If src_center or asset is missing, fallback to a simple target ring.
    
    if kind == "dragon":
        ray_raw = get_fx_raw_surface("dragon_attack")  # raw ray sprite
    
        # Need attacker position and the sprite asset.
        if src_center is None or ray_raw is None:
            rr = int(op.add(18, op.mul(2, int(t) % 10)))
            pygame.draw.circle(screen, (150, 90, 210, 170), (cx, cy), rr, 3)
            return
    
        # Define source coordinates explicitly so this block is self contained.
        sx = int(src_center[0])
        sy = int(src_center[1])
    
        # Vector from attacker to target.
        dx = float(op.sub(cx, sx))
        dy = float(op.sub(cy, sy))
    
        # Distance for normalization.
        dist = float(math.hypot(dx, dy))
        if dist < 1.0:
            dist = 1.0
    
        # Unit direction from attacker to target.
        dir_x = float(op.truediv(dx, dist))
        dir_y = float(op.truediv(dy, dist))
    
        # Rotation angle so the sprite points toward the target.
        ang_deg = float(math.degrees(math.atan2(dy, dx)))
        rot_deg = float(op.sub(0.0, ang_deg))  # invert to match sprite orientation
    
        # Time fraction used to animate pulses and wave motion.
        dur = float(max(1, attack_duration))
        frac = clamp01(float(op.truediv(float(t), dur)))
    
        # Read raw sprite size.
        bw = int(ray_raw.get_width())
        bh = int(ray_raw.get_height())
        if bw <= 0 or bh <= 0:
            return
    
        # Scale ray sprite to a fixed width DRAGON_RAY_W.
        scale = float(op.truediv(float(DRAGON_RAY_W), float(bw)))
        ray_h = int(op.mul(float(bh), scale))
        if ray_h < 1:
            ray_h = 1
    
        # Build rotated sprite once to reuse.
        ray_scaled = pygame.transform.smoothscale(ray_raw, (int(DRAGON_RAY_W), int(ray_h)))
        ray_rot = pygame.transform.rotate(ray_scaled, rot_deg)
    
        # Perpendicular vector for sideways wobble.
        perp_x = float(op.sub(0.0, dir_y))
        perp_y = float(dir_x)
    
        # base_u shifts the ray train forward over time.
        base_u = float(op.mul(frac, float(DRAGON_RAY_WAVES)))
    
        for i in range(int(DRAGON_RAY_COUNT)):
            # u is the position of this ray along the train, wrapped 0 to 1.
            u = float(op.sub(base_u, op.mul(float(i), float(DRAGON_RAY_SPACING))))
            u = float(u % 1.0)
    
            # along is distance traveled along the attacker to target line.
            along = float(op.mul(u, dist))
    
            # Base position on the line.
            px0 = float(op.add(float(sx), op.mul(dir_x, along)))
            py0 = float(op.add(float(sy), op.mul(dir_y, along)))
    
            # Wobble changes position sideways for energy feel.
            wig_phase = float(op.add(op.mul(frac, 18.0), op.mul(float(i), 0.9)))
            wig = float(op.mul(float(DRAGON_RAY_WIGGLE), math.sin(wig_phase)))
    
            # Apply wobble using perpendicular vector.
            px = float(op.add(px0, op.mul(perp_x, wig)))
            py = float(op.add(py0, op.mul(perp_y, wig)))
    
            # Copy sprite so we can set per ray alpha.
            img = ray_rot.copy()
    
            # Pulse alpha so rays flicker.
            pulse = float(0.55 + 0.45 * abs(math.sin(op.add(op.mul(frac, 10.0), float(i) * 0.7))))
            a = int(op.mul(float(DRAGON_RAY_ALPHA), pulse))
            a = max(0, min(255, a))
            img.set_alpha(a)
    
            # Draw centered at computed position.
            bx = int(op.sub(int(px), op.truediv(img.get_width(), 2)))
            by = int(op.sub(int(py), op.truediv(img.get_height(), 2)))
            screen.blit(img, (bx, by))
    
        return
    
    # Ghost animation
    # Scales a ghost face sprite up using smoothstep.
    # Alpha pulses so it feels eerie.
    # Small shake adds life, strongest early, weaker later.
    # If asset missing, fallback to a simple ring.
    
    if kind == "ghost":
        face_raw = get_fx_raw_surface("ghost_transparent")  # ghost sprite
    
        # Asset missing fallback: expanding pale ring.
        if face_raw is None:
            rr = int(op.add(14, op.mul(2, int(t) % 10)))
            pygame.draw.circle(screen, (220, 220, 255, 170), (cx, cy), rr, 3)
            return
    
        # Progress across the full attack duration.
        dur = float(max(1, attack_duration))
        p = clamp01(float(op.truediv(float(t), dur)))
    
        # Smoothstep easing for clean scale motion.
        smooth = float(op.mul(p, p))
        smooth = float(op.mul(smooth, op.sub(3.0, op.mul(2.0, p))))
    
        # Default target size if missing.
        if target_size is None:
            target_size = int(BATTLE_PX)
    
        ts = int(target_size)
    
        # Compute max and start width so it grows from small to near target size.
        max_w = int(op.mul(float(ts), 1.10))
        max_w = max(12, max_w)
    
        start_w = int(op.mul(float(max_w), 0.18))
        start_w = max(8, start_w)
    
        # Width interpolates with smooth.
        w = int(op.add(start_w, op.mul(op.sub(max_w, start_w), smooth)))
        w = max(1, w)
    
        # Snap width to multiples of 4 to reduce shimmer from constant resizing.
        snap = 4
        w = int(op.mul(snap, int(round(float(op.truediv(w, snap))))))
    
        # Read raw sprite size.
        bw = int(face_raw.get_width())
        bh = int(face_raw.get_height())
        if bw <= 0 or bh <= 0:
            return
    
        # Scale to width w.
        scale = float(op.truediv(float(w), float(bw)))
        h = int(op.mul(float(bh), scale))
        h = max(1, h)
    
        # Build scaled image.
        img = pygame.transform.smoothscale(face_raw, (int(w), int(h)))
    
        # Pulse alpha for eerie breathing.
        pulse = float(0.65 + 0.35 * abs(math.sin(op.add(op.mul(smooth, 10.0), 0.7))))
        a = int(op.mul(op.mul(245.0, smooth), pulse))
        a = max(0, min(255, a))
        img.set_alpha(int(a))
    
        # Shake amplitude is stronger early, weaker later.
        shake_amp = int(op.mul(8.0, op.sub(1.0, smooth)))
        shake_amp = max(0, shake_amp)
    
        # Deterministic wobble from sines, not random, so it looks smooth.
        jx = int(op.mul(float(shake_amp), math.sin(op.add(op.mul(smooth, 14.0), 1.3))))
        jy = int(op.mul(float(shake_amp), math.sin(op.add(op.mul(smooth, 12.0), 2.1))))
    
        # Draw centered on target plus jitter.
        bx = int(op.sub(cx, op.truediv(img.get_width(), 2)))
        by = int(op.sub(cy, op.truediv(img.get_height(), 2)))
        screen.blit(img, (int(op.add(bx, jx)), int(op.add(by, jy))))
        return
    
    # Steel animation
    # Three phases driven by p across attack_duration:
    # Phase 1 draws a link ring growing in, so it looks like a chain forming.
    # Phase 2 tightens the ring into a smaller radius with wobble fading out.
    # Phase 3 snap draws sharp metal streaks outward and plays the snap sound once.
    # steel_snap_sfx_played prevents sound spam.
    
    if kind == "steel":
        global steel_snap_sfx_played
        try:
            dur = float(max(1, attack_duration))  # avoid divide by zero
            p = clamp01(float(op.truediv(float(t), dur)))  # normalized progress
    
            # Reset snap sound trigger at the first frame.
            if int(t) == 0:
                steel_snap_sfx_played = False
    
            # Target size drives ring radius and link sizing.
            ts = int(target_size) if target_size is not None else int(BATTLE_PX)
    
            # Phase boundaries.
            draw_frac = 0.45     # ring draws in from 0 to this
            tighten_frac = 0.82  # ring tightens from draw_frac to this
    
            # Smoothstep easing for nicer phase transitions.
            def smoothstep(x):
                x = clamp01(x)
                return float(op.mul(op.mul(x, x), op.sub(3.0, op.mul(2.0, x))))
    
            # Base ring radius and tightened radius.
            base_r = float(op.mul(float(ts), 0.58))
            tight_r = float(op.mul(float(ts), 0.30))
    
            # Use an ellipse so it looks like it wraps the sprite more naturally.
            rx0 = base_r
            ry0 = float(op.mul(base_r, 0.65))
            rx1 = tight_r
            ry1 = float(op.mul(tight_r, 0.70))
    
            # Link count scales with target size but stays in a reasonable range.
            n_links = int(max(14, min(34, int(op.truediv(ts, 9)))))
    
            # Alpha ramps up a bit with time so the ring feels like it appears.
            base_alpha = 210
            metal_a = int(op.mul(base_alpha, clamp01(float(op.add(0.25, op.mul(0.75, p))))))
            metal_a = max(0, min(255, metal_a))
    
            # Metal line color, darker than white.
            col_line = (95, 100, 120, int(op.mul(metal_a, 0.85)))
    
            # Constant for circle angle.
            two_pi = float(op.mul(2.0, math.pi))
    
            # Phase 1 draw in
            if p <= draw_frac:
                # Normalize progress inside draw phase.
                u = 0.0
                if draw_frac > 0.000001:
                    u = clamp01(float(op.truediv(p, draw_frac)))
                u2 = smoothstep(u)
    
                # kmax is how many links to draw so the ring grows.
                kmax = int(op.mul(float(n_links), u2))
                if kmax < 1:
                    kmax = 1
                if kmax > n_links:
                    kmax = n_links
    
                prev = None  # previous link position for connecting line
    
                for i in range(kmax):
                    # Angle around the ring.
                    ang = float(op.mul(two_pi, float(op.truediv(float(i), float(n_links)))))
    
                    # Point on ellipse.
                    x = float(op.add(float(cx), op.mul(rx0, math.cos(ang))))
                    y = float(op.add(float(cy), op.mul(ry0, math.sin(ang))))
    
                    # Draw connecting line between links.
                    if prev is not None:
                        pygame.draw.line(screen, col_line, (int(prev[0]), int(prev[1])), (int(x), int(y)), 4)
    
                    # Link size based on target.
                    link_w = int(max(10, op.mul(ts, 0.16)))
                    link_h = int(max(8, op.mul(ts, 0.09)))
    
                    # Build link surface and alternate rotation to look interlocked.
                    img0 = steel_link_surface(link_w, link_h, metal_a)
                    img1 = pygame.transform.rotate(img0, 90)
                    img = img0 if (i % 2 == 0) else img1
    
                    # Draw link centered.
                    bx = int(op.sub(int(x), op.truediv(img.get_width(), 2)))
                    by = int(op.sub(int(y), op.truediv(img.get_height(), 2)))
                    screen.blit(img, (bx, by))
    
                    prev = (x, y)
    
                return
    
            # Phase 2 tighten
            if p <= tighten_frac:
                # Normalize progress inside tighten phase.
                u = 0.0
                denom = float(op.sub(tighten_frac, draw_frac))
                if denom > 0.000001:
                    u = clamp01(float(op.truediv(op.sub(p, draw_frac), denom)))
                u2 = smoothstep(u)
    
                # Interpolate ellipse radii.
                rx = float(op.add(rx0, op.mul(op.sub(rx1, rx0), u2)))
                ry = float(op.add(ry0, op.mul(op.sub(ry1, ry0), u2)))
    
                # Wobble fades out as u2 approaches 1.
                wob_amp = int(op.mul(4.0, op.sub(1.0, u2)))
                wob_phase = float(op.mul(float(t), 0.22))
    
                prev = None
    
                for i in range(n_links):
                    # Angle around the ring.
                    ang = float(op.mul(two_pi, float(op.truediv(float(i), float(n_links)))))
    
                    # Wobble is a sine offset, strongest early.
                    wob = 0.0
                    if wob_amp > 0:
                        wob = float(op.mul(float(wob_amp), math.sin(op.add(wob_phase, op.mul(float(i), 0.9)))))
    
                    # Apply wobble by adding wob to rx and ry.
                    x = float(op.add(float(cx), op.mul(op.add(rx, wob), math.cos(ang))))
                    y = float(op.add(float(cy), op.mul(op.add(ry, wob), math.sin(ang))))
    
                    # Connect links with a line for a solid chain look.
                    if prev is not None:
                        pygame.draw.line(screen, col_line, (int(prev[0]), int(prev[1])), (int(x), int(y)), 4)
    
                    # Link size.
                    link_w = int(max(10, op.mul(ts, 0.16)))
                    link_h = int(max(8, op.mul(ts, 0.09)))
    
                    # Alternate link orientation.
                    img0 = steel_link_surface(link_w, link_h, metal_a)
                    img1 = pygame.transform.rotate(img0, 90)
                    img = img0 if (i % 2 == 0) else img1
    
                    # Draw link centered.
                    bx = int(op.sub(int(x), op.truediv(img.get_width(), 2)))
                    by = int(op.sub(int(y), op.truediv(img.get_height(), 2)))
                    screen.blit(img, (bx, by))
    
                    prev = (x, y)
    
                # Extra fading ring to emphasize tightening.
                ring_a = int(op.mul(120.0, op.sub(1.0, u2)))
                ring_a = max(0, min(255, ring_a))
                pygame.draw.circle(screen, (220, 220, 235, ring_a), (cx, cy), int(op.mul(rx, 0.55)), 2)
    
                return
    
            # Phase 3 snap
            # Play snap sound once.
            if not steel_snap_sfx_played:
                global steel_snap_sfx
                if steel_snap_sfx is not None:
                    steel_snap_sfx.play()
                steel_snap_sfx_played = True
    
            # Normalize progress inside snap phase.
            u = 0.0
            denom = float(op.sub(1.0, tighten_frac))
            if denom > 0.000001:
                u = clamp01(float(op.truediv(op.sub(p, tighten_frac), denom)))
            u2 = smoothstep(u)
    
            # Alpha fades out as snap expands.
            snap_a = int(op.mul(220.0, op.sub(1.0, u2)))
            snap_a = max(0, min(255, snap_a))
    
            # Two tone snap lines.
            col_snap = (235, 235, 250, snap_a)
            col_snap2 = (140, 145, 170, snap_a)
    
            # Number of streaks.
            chunk_count = 8
    
            # Stable randomness for streak placement.
            rng = random.Random(77777)
    
            # Spread of streak angles.
            spread_half = float(op.mul(math.pi, 0.92))
    
            for c in range(chunk_count):
                # base is normalized index 0 to 1.
                base = float(op.truediv(float(c), float(max(1, chunk_count - 1))))
    
                # Angle centered around pi, spread across spread_half.
                ang = float(op.add(math.pi, op.mul(op.sub(base, 0.5), op.mul(2.0, spread_half))))
    
                # Add small random angle jitter.
                ang = float(op.add(ang, rng.uniform(op.sub(0.0, 0.28), 0.28)))
    
                # out_dist expands with u2.
                out_dist = float(op.mul(op.mul(float(ts), 0.85), u2))
    
                # Offset from target.
                ox2 = float(op.mul(out_dist, math.cos(ang)))
                oy2 = float(op.mul(out_dist, math.sin(ang)))
    
                # Streak start point.
                x0 = float(op.add(float(cx), ox2))
                y0 = float(op.add(float(cy), oy2))
    
                # Segment length randomization.
                seg_len = float(op.add(10.0, rng.uniform(0.0, 18.0)))
    
                # Streak end point.
                x1 = float(op.add(x0, op.mul(seg_len, math.cos(ang))))
                y1 = float(op.add(y0, op.mul(seg_len, math.sin(ang))))
    
                # Draw thick dark line then thin bright line for metal shine.
                pygame.draw.line(screen, col_snap2, (int(x0), int(y0)), (int(x1), int(y1)), 4)
                pygame.draw.line(screen, col_snap, (int(x0), int(y0)), (int(x1), int(y1)), 2)
    
                # Add small circles to make it feel like fragments.
                pygame.draw.circle(screen, col_snap, (int(x0), int(y0)), 4, 2)
                pygame.draw.circle(screen, col_snap2, (int(x1), int(y1)), 3, 2)
    
            return
    
        except Exception:
            # Hard fallback if something goes wrong, keep game alive.
            rr = int(op.add(14, op.mul(2, int(t) % 10)))
            pygame.draw.circle(screen, (220, 220, 235, 170), (cx, cy), rr, 3)
            return
    
    # Fairy animation
    # Uses a persistent particle list so sparkles feel continuous.
    # Particles drift downward and wobble sideways.
    # When a particle exits below the target, it is respawned above.
    # Alpha pulses per particle so the sparkle twinkles.
    
    if kind == "fairy":
        global fairy_fx_particles, fairy_fx_inited
    
        # Target size drives spawn spread and reset threshold.
        ts = int(target_size) if target_size is not None else int(BATTLE_PX)
    
        # Basic target geometry for spawn and reset bounds.
        tx = int(cx)
        ty_top = int(op.sub(cy, op.truediv(ts, 2)))  # top of target area
        ty_bot = int(op.add(cy, op.truediv(ts, 2)))  # bottom of target area
        spread_x = int(op.add(op.mul(ts, 0.7), 10))  # how wide particles spread
    
        # Initialize particle list once.
        if not fairy_fx_inited:
            fairy_fx_particles = []
            count = 34  # number of sparkles
    
            for i in range(count):
                # Spawn x around the target.
                px = int(op.add(tx, random.randint(int(op.sub(0, spread_x)), int(spread_x))))
    
                # Spawn y above the target so they fall into view.
                py = int(op.sub(ty_top, random.randint(20, 140)))
    
                # Downward speed per particle.
                vy = random.uniform(1.2, 2.4)
    
                # Side drift amplitude.
                drift = random.uniform(0.25, 0.85)
    
                # Phase offset so wobble differs per particle.
                phase = random.uniform(0.0, math.pi * 2.0)
    
                # Star size.
                size = random.uniform(5.0, 9.0)
    
                # Twinkle rate.
                tw = random.uniform(0.10, 0.22)
    
                # Store particle state.
                fairy_fx_particles.append([px, py, vy, drift, phase, size, tw])
    
            fairy_fx_inited = True
    
        tt = float(t)  # time for wobble and twinkle
    
        for pstate in fairy_fx_particles:
            px, py, vy, drift, phase, size, tw = pstate
    
            # Side wobble based on phase and time.
            wobble = float(op.mul(drift, math.sin(op.add(phase, op.mul(tt, 0.18)))))
            px2 = float(op.add(px, wobble))
    
            # Move downward.
            py = float(op.add(py, vy))
    
            # Reset if below the visible area near the target.
            if py > float(op.add(ty_bot, int(op.mul(ts, 0.55)))):
                py = float(op.sub(ty_top, random.randint(25, 120)))  # respawn above
                px = int(op.add(tx, random.randint(int(op.sub(0, spread_x)), int(spread_x))))  # new x
                phase = random.uniform(0.0, math.pi * 2.0)  # new wobble phase
    
            # Twinkle alpha uses a sine wave per particle.
            a = float(
                op.add(
                    110.0,
                    op.mul(
                        110.0,
                        op.add(
                            0.5,
                            op.mul(0.5, math.sin(op.add(phase, op.mul(tt, tw)))),
                        ),
                    ),
                )
            )
    
            # Draw sparkle star at current position.
            draw_star(screen, int(px2), int(py), float(size), a)
    
            # Write back updated state.
            pstate[0] = px
            pstate[1] = py
            pstate[4] = phase
    
        return


    rr = int(op.add(8, op.mul(t, 3)))
    col = (255, 255, 255, 130)
    pygame.draw.circle(screen, col, (cx, cy), rr, 3)


def make_placeholder_icon(font, colors):  # 45
    """
    Generate a placeholder icon surface when a real icon is missing.

    Visual:
    - Same size as normal icons (ICON_PX x ICON_PX)
    - Panel colored background with a border
    - A centered "?" so missing assets are obvious, not silent
    """
    # Create an RGBA surface so it blends cleanly over the UI.
    s = pygame.Surface((ICON_PX, ICON_PX), pygame.SRCALPHA)

    # Fill with the standard panel color so it matches the UI theme.
    s.fill(colors["panel"])

    # Draw an outline so the icon box is readable on light backgrounds.
    pygame.draw.rect(s, colors["border"], s.get_rect(), 2)

    # Render a question mark using the provided font and muted text color.
    txt = font.render("?", True, colors["muted"])

    # Center the text inside the icon box.
    r = txt.get_rect(center=(ICON_PX // 2, ICON_PX // 2))

    # Blit the text onto the placeholder surface.
    s.blit(txt, r)

    return s


def pokeapi_slug(name_lower):  # 46
    """
    Convert a display name into a PokeAPI slug string.

    Why this exists:
    - PokeAPI uses specific naming conventions (lowercase, dashes, no punctuation)
    - Some Pokemon have special gender symbols or weird punctuation in names
    - We normalize all that so requests consistently work
    """
    # Normalize input to lowercase trimmed text.
    s = str(name_lower).strip().lower()

    # Normalize punctuation variations:
    # - replace curly apostrophe with normal apostrophe
    # - remove dots (e.g., "mr. mime")
    # - remove apostrophes entirely (PokeAPI does not keep them)
    s = s.replace("’", "'").replace(".", "").replace("'", "")

    # Spaces become DASH (you defined DASH = chr(45)).
    s = s.replace(" ", DASH)

    # Gender symbols get mapped into slug fragments.
    # Example: "meowstic♀" should become something like "meowstic-f"
    s = s.replace("♀", DASH + "f").replace("♂", DASH + "m")

    # Special case: some sources use "nidoran_female" / "nidoran_male".
    # PokeAPI expects "nidoran-f" / "nidoran-m".
    if s == "nidoran_female":
        s = "nidoran" + DASH + "f"
    if s == "nidoran_male":
        s = "nidoran" + DASH + "m"

    # Underscores also become DASH to match PokeAPI style.
    s = s.replace("_", DASH)

    return s


def fetch_icon_worker(name_lower):  # 47
    """
    Background worker that downloads icons from the PokeAPI and stores them locally (in memory first).

    Thread behavior:
    - Runs in a separate thread so the UI does not freeze during network calls
    - Writes results into _icon_ready_bytes under a lock
    - Always removes the name from _icon_inflight in finally (no "stuck downloading" state)
    """
    try:
        # Convert name into the exact PokeAPI slug.
        slug = pokeapi_slug(name_lower)

        # Fetch pokemon JSON data (contains sprite URLs).
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{slug}", timeout=8)

        # Default: no sprite URL found.
        sprite_url = None

        # If API call succeeded, parse JSON and extract the "front_default" sprite URL.
        if r.status_code == 200:
            j = r.json()
            sprite_url = j.get("sprites", {}).get("front_default")

        # Default: no image bytes downloaded.
        img_bytes = None

        # If we have a sprite URL, fetch the image bytes.
        if sprite_url:
            r2 = requests.get(sprite_url, timeout=8)
            if r2.status_code == 200:
                img_bytes = r2.content

        # Store the raw bytes (or None) so the main thread can turn it into a pygame Surface.
        with _icon_lock:
            _icon_ready_bytes[name_lower] = img_bytes

    except Exception:
        # On any failure (network, JSON parse, etc), mark as failed with None.
        with _icon_lock:
            _icon_ready_bytes[name_lower] = None

    finally:
        # Always clear inflight status so future requests can retry.
        with _icon_lock:
            _icon_inflight.discard(name_lower)


def request_icon(name_lower, placeholder_icon):  # 48
    """
    Request an icon.

    Contract:
    - This function returns immediately (no blocking downloads).
    - If icon is already cached or on disk, load it now.
    - Otherwise start a worker thread (unless we are in cooldown from a recent failure).

    Note:
    - placeholder_icon is passed in but not used here because this function only queues work.
      The drawing code can still use placeholder while waiting.
    """
    # Normalize key for caches.
    n = str(name_lower).strip().lower()

    # If we already have a Surface for this icon, do nothing.
    if n in _icon_surfaces:
        return

    # Try local disk cache first (fast and offline friendly).
    local = try_load_local_icon(n)
    if local is not None:
        _icon_surfaces[n] = local
        return

    # Throttle retries: if we failed recently, skip network calls for a short time.
    now = pygame.time.get_ticks()
    fail_until = _icon_fail_until.get(n)
    if fail_until is not None and now < fail_until:
        return

    # Mark as inflight under a lock so we do not spawn duplicate threads.
    with _icon_lock:
        if n in _icon_inflight:
            return
        _icon_inflight.add(n)

    # Start the download worker as a daemon thread so it won't block program exit.
    t = threading.Thread(target=fetch_icon_worker, args=(n,), daemon=True)
    t.start()


def process_ready_icons(placeholder_icon):  # 49
    """
    Move finished downloads from worker threads into in memory caches.

    What it does:
    - Pulls (name, bytes) results from _icon_ready_bytes under lock
    - Converts bytes into a pygame Surface (main thread)
    - Stores the Surface in _icon_surfaces
    - On failure, stores placeholder and sets a short cooldown
    """
    # Collect finished items without holding the lock while processing images.
    to_process = []
    with _icon_lock:
        for k in list(_icon_ready_bytes.keys()):
            to_process.append((k, _icon_ready_bytes.pop(k)))

    # Current tick time used for cooldown scheduling.
    now = pygame.time.get_ticks()

    for name_lower, b in to_process:
        # If the worker reported failure or empty bytes, use placeholder and throttle retries.
        if not b:
            _icon_surfaces[name_lower] = placeholder_icon
            _icon_fail_until[name_lower] = int(op.add(now, 5000))
            continue

        try:
            # Load image bytes into a pygame Surface.
            surf = pygame.image.load(io.BytesIO(b)).convert_alpha()

            # Normalize icon size so UI layout stays consistent.
            surf = pygame.transform.smoothscale(surf, (ICON_PX, ICON_PX))

            # Store in memory cache.
            _icon_surfaces[name_lower] = surf

        except Exception:
            # If decode fails, store placeholder and throttle retries.
            _icon_surfaces[name_lower] = placeholder_icon
            _icon_fail_until[name_lower] = int(op.add(now, 5000))


def wrap_text(font, text, max_w):  # 50
    """
    Wrap a long string into multiple lines that fit a given pixel width.

    Approach:
    - Greedy wrap by words where possible
    - If a single word is too wide, split it into chunks by characters
    - Uses font.size(...) to measure real pixel width (not character count)
    """
    # Ensure we are working with a string.
    s = str(text)

    # Empty input returns a single empty line (keeps UI code simple).
    if len(s) == 0:
        return [""]

    # Split into words by spaces.
    words = s.split(" ")

    # Output lines and current line buffer.
    lines = []
    cur = ""

    for w in words:
        # Candidate line: add the new word to current line if possible.
        test = w if len(cur) == 0 else (cur + " " + w)

        # If candidate fits the width, keep building the current line.
        if font.size(test)[0] <= int(max_w):
            cur = test
            continue

        # If current line has content, finalize it and start a new line with this word.
        if len(cur) > 0:
            lines.append(cur)
            cur = w
            continue

        # If we get here, cur is empty and w alone does not fit.
        # So we chunk the word by characters.
        chunk = ""
        for ch in w:
            test2 = chunk + ch

            # If adding this character still fits, keep growing the chunk.
            if font.size(test2)[0] <= int(max_w):
                chunk = test2
            else:
                # If chunk has content, push it as a line and restart chunk.
                if len(chunk) > 0:
                    lines.append(chunk)
                chunk = ch

        # Whatever remains becomes the new current line buffer.
        cur = chunk

    # Append last line if it has content.
    if len(cur) > 0:
        lines.append(cur)

    return lines


def ellipsize(font, text, max_w):  # 51
    """
    Shorten a string to fit a pixel width by adding an ellipsis.

    Behavior:
    - If it already fits, return as is
    - Otherwise trim characters from the end until "prefix + ..." fits
    - If nothing fits, return "..." alone
    """
    # Convert to string so font.size works.
    s = str(text)

    # If it fits, do nothing.
    if font.size(s)[0] <= int(max_w):
        return s

    # Ellipsis suffix.
    suffix = "..."

    # Start by trying to keep as many characters as possible.
    keep = max(0, len(s))

    # Reduce keep until it fits.
    while keep > 0:
        cand = s[:keep] + suffix
        if font.size(cand)[0] <= int(max_w):
            return cand
        keep = int(op.sub(keep, 1))

    # Worst case: only the ellipsis fits.
    return suffix


class ScrollList:
    def __init__(self, rect):  # 52
        # The clickable area that represents the whole list region.
        # We store it as a pygame.Rect so collision checks are easy.
        self.rect = pygame.Rect(rect)

        # Current scroll offset in "rows" (not pixels).
        # Example: scroll = 0 means show items starting at index 0.
        self.scroll = 0

        # True while the user is dragging the scrollbar thumb.
        self.dragging = False

        # When dragging, we keep where inside the thumb the user grabbed it
        # so the thumb does not jump when the mouse moves.
        self.drag_grab_dy = 0

        # Cached thumb rect (small draggable box on the scrollbar track).
        # None means there is no scrollbar (everything fits).
        self.thumb_rect = None

        # Total number of items in the list.
        self.total = 0

        # Number of items visible in the viewport at once.
        # Must be at least 1 to avoid division by zero.
        self.visible = 1

    def set_counts(self, total, visible):  # 53
        # Update list length and how many items the viewport can show.
        self.total = int(total)
        self.visible = max(1, int(visible))

        # Clamp scroll so we never point outside valid bounds.
        # Example: if total shrinks, scroll might become too large.
        self.scroll = clamp_scroll(self.scroll, self.total, self.visible)

    def scroll_by(self, dy):  # 54
        # Scroll the list by dy "rows".
        # Note: subtracting dy means positive dy moves the viewport down (typical wheel behavior).
        self.scroll = clamp_scroll(op.sub(self.scroll, int(dy)), self.total, self.visible)

    def track_rect(self):  # 55
        # Build the scrollbar track rectangle.
        # It sits on the right side inside self.rect.
        return pygame.Rect(
            int(op.sub(self.rect.right, 12)),       # x: 12 px from the right edge
            int(op.add(self.rect.y, 2)),            # y: slight padding from top
            10,                                     # width of the track
            int(op.sub(self.rect.height, 4)),       # height with padding top + bottom
        )

    def compute_thumb(self):  # 56
        # Compute the draggable thumb size + position based on total/visible/scroll.
        track = self.track_rect()

        # If everything fits in the viewport, no scrollbar needed.
        if self.total <= self.visible:
            self.thumb_rect = None
            return None

        # Fraction of content visible: visible / total.
        # Bigger fraction means bigger thumb.
        frac = float(self.visible) / float(self.total)

        # Thumb height is proportional to visible fraction, with a minimum so it stays usable.
        thumb_h = max(18, int(op.mul(track.height, frac)))

        # Maximum scroll offset in rows.
        # Example: total=20, visible=5 -> max_off=15
        max_off = max(0, int(op.sub(self.total, self.visible)))

        # Convert current scroll offset into a 0..1 fraction along the track.
        pos_frac = 0.0 if max_off == 0 else float(self.scroll) / float(max_off)

        # Usable travel distance for the thumb (track height minus thumb height).
        usable = int(op.sub(track.height, thumb_h))

        # Thumb y position = track.y + pos_frac * usable
        thumb_y = int(op.add(track.y, op.mul(pos_frac, usable)))

        # Create the thumb rect with a 1 px inset so borders look nicer.
        self.thumb_rect = pygame.Rect(
            int(op.add(track.x, 1)),                 # inset x
            int(thumb_y),                            # computed y
            int(op.sub(track.width, 2)),             # inset width
            int(thumb_h),                            # computed height
        )
        return self.thumb_rect

    def handle_event(self, event):  # 57
        # Handle mouse interactions for the scrollbar:
        # - click thumb to drag
        # - click track to page up/down
        # - drag thumb to update scroll offset

        # If no scrollbar needed, ignore and make sure dragging is off.
        if self.total <= self.visible:
            self.dragging = False
            return False

        # Track is the full scrollbar lane; thumb is the draggable part.
        track = self.track_rect()
        thumb = self.compute_thumb()

        # Mouse pressed: start dragging or page jump.
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # left click
                mx, my = event.pos

                # If click is inside thumb: start dragging.
                if thumb is not None and thumb.collidepoint((mx, my)):
                    self.dragging = True

                    # Store where inside the thumb the click happened
                    # so the thumb stays "attached" at the same relative point.
                    self.drag_grab_dy = int(op.sub(my, thumb.y))
                    return True

                # If click is on track but not on thumb: page up/down by one viewport.
                if track.collidepoint((mx, my)) and thumb is not None:
                    if my < thumb.y:
                        # Click above thumb: page up.
                        self.scroll = clamp_scroll(
                            op.sub(self.scroll, self.visible),
                            self.total,
                            self.visible,
                        )
                    else:
                        # Click below thumb: page down.
                        self.scroll = clamp_scroll(
                            op.add(self.scroll, self.visible),
                            self.total,
                            self.visible,
                        )
                    return True

        # Mouse released: stop dragging if we were dragging.
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.dragging:
                self.dragging = False
                return True

        # Mouse moved while dragging: update thumb position -> update scroll offset.
        if event.type == pygame.MOUSEMOTION and self.dragging:
            mx, my = event.pos
            if thumb is None:
                return False

            # Proposed thumb y based on mouse position minus grab offset.
            new_thumb_y = int(op.sub(my, self.drag_grab_dy))

            # Track bounds.
            track_y0 = track.y
            track_y1 = int(op.add(track.y, track.height))

            # Thumb height for clamping.
            thumb_h = thumb.height

            # Clamp thumb so it stays inside the track.
            min_y = track_y0
            max_y = int(op.sub(track_y1, thumb_h))
            new_thumb_y = int(max(min_y, min(new_thumb_y, max_y)))

            # Convert thumb y back into scroll fraction.
            usable = int(op.sub(track.height, thumb_h))
            pos_frac = 0.0 if usable <= 0 else float(op.sub(new_thumb_y, track.y)) / float(usable)

            # Convert fraction into row scroll offset.
            max_off = max(0, int(op.sub(self.total, self.visible)))
            self.scroll = int(round(op.mul(pos_frac, max_off)))

            # Clamp again (safety net for rounding edge cases).
            self.scroll = clamp_scroll(self.scroll, self.total, self.visible)
            return True

        # Not handled.
        return False

    def draw(self, surface, colors):  # 58
        # Draw the scrollbar only if content does not fit.

        if self.total <= self.visible:
            return

        # Draw track background and border.
        track = self.track_rect()
        draw_rect(surface, track, colors["input_bg"])
        draw_rect(surface, track, colors["border"], 2)

        # Compute and draw thumb.
        thumb = self.compute_thumb()
        if thumb is None:
            return

        draw_rect(surface, thumb, colors["button"])
        draw_rect(surface, thumb, colors["border"], 2)

def main():  # 59
    """
    Entry point for the whole game.
    Responsibilities:
    1) Boot pygame + audio.
    2) Load AI model (lazy).
    3) Load sounds + music routing.
    4) Load data tables (pokemon, moves, type chart).
    5) Build UI objects + state variables.
    6) Run the main loop (select screen -> battle -> end).
    """

    # We mutate these globals inside main(), so we must declare them.
    global p1_attack_active, p1_attack_t, p2_attack_active, p2_attack_t
    global atk_fx_active, atk_fx_t, atk_fx_type, atk_fx_who
    global p1_faint_active, p1_faint_t, p2_faint_active, p2_faint_t

    # Pre configure mixer BEFORE pygame.init() so audio latency is lower.
    # 44100 Hz sample rate, 16 bit signed, stereo, 512 buffer size.
    pygame.mixer.pre_init(44100, -16, 2, 512)

    # Initialize pygame core modules (display, event system, etc.).
    pygame.init()

    # Initialize mixer explicitly (some systems need this after init()).
    pygame.mixer.init()

    # Pick computation device for torch model.
    # If CUDA is available, use GPU, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Agent model handle (lazy loaded so the menu loads instantly).
    agent = None

    # The observation vector size expected by the agent model.
    agent_obs_dim = None

    def ensure_agent_loaded():  # 59.1
        """
        Lazy load the trained agent from a checkpoint the first time we need it.
        This avoids loading torch stuff if you play random opponent mode only.
        """
        nonlocal agent, agent_obs_dim

        # If already loaded, do nothing.
        if agent is not None:
            return

        # Load checkpoint from disk onto the chosen device.
        payload = torch.load("world_runs/ckpt_ep_5000000.pt", map_location=device)

        # Meta contains settings saved during training (like obs_dim).
        meta = payload.get("meta", {})

        # Read obs_dim from meta and make sure it is valid.
        obs_dim = int(meta.get("obs_dim", 0))
        if obs_dim <= 0:
            raise RuntimeError("Checkpoint meta obs_dim is missing or invalid")

        # Build network with obs_dim inputs and 4 outputs (4 moves).
        q = QNet(obs_dim, 4).to(device)

        # Load trained weights.
        q.load_state_dict(payload["q_state"])

        # Switch to eval mode (no dropout, no training behavior).
        q.eval()

        # Store globally for reuse.
        agent = q
        agent_obs_dim = obs_dim

    @torch.no_grad()
    def agent_act(obs_np):
        """
        Run a forward pass of the agent network and choose the best action.
        obs_np must match the obs_dim expected by the model.
        """
        # Convert numpy array to torch tensor on correct device.
        st = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        # Forward pass -> Q values for actions, take argmax.
        return int(torch.argmax(agent(st), dim=1).item())

    # Directory where your sound files live.
    sound_dir = Path("sound_effects")

    # Background menu music.
    music_path = sound_dir / "game_music_loop.mp3"

    # Punch fallback sound when no specific move sound exists.
    punch_path = sound_dir / "punch_03.mp3"

    # Battle loop music.
    battle_path = sound_dir / "battle_music_2.mp3"

    # Load status sounds for conditions like burn/poison/paralysis etc.
    # PROC is your status dictionary from env/data code.
    status_sfx = build_status_sfx(PROC)

    # Map move types to sound file stems.
    # Example: fire -> burn.mp3 (because your file is named burn.mp3).
    MOVE_TYPE_TO_SFX = {
        "fire": "burn",
        "water": "water",
        "ice": "wind",
        "steel": "steel",
        "ghost": "ghost",
        "psychic": "psychic",
        "physics": "psychic",    # You treat "physics" as psychic visuals and sound.
        "electric": "paralysis", # Optional: only works if paralysis sound exists.
        "poison": "poison",
        "dragon": "dragon",
        "ground": "ground",
        "rock": "rock",
        "dark": "dark",
        "grass": "grass",
        "fairy": "fairy",
    }

    # Cache of pygame Sound objects for each move type.
    move_sfx = {}

    # Load each sound file now (once) so runtime playback is instant.
    for mv_type, stem in MOVE_TYPE_TO_SFX.items():
        # find_sfx_file returns the first existing extension (wav/mp3/ogg).
        # safe_sound returns None if file missing or load fails.
        move_sfx[mv_type] = safe_sound(find_sfx_file(stem))

    def play_move_sfx(move_type):  # 59.2
        """
        Play the "move start" sound for a move type if it exists.
        """
        # Normalize key.
        k = str(move_type).strip().lower()

        # Look up cached sound object.
        snd = move_sfx.get(k)

        # Play only if successfully loaded.
        if snd is not None:
            snd.play()

    # Leaf animation uses a "hit" sound on impact.
    # You reuse the grass sound for that.
    global leaf_hit_sfx
    leaf_hit_sfx = move_sfx.get("grass")

    # Steel animation has an extra snap sound during the snap phase.
    global steel_snap_sfx
    steel_snap_sfx = safe_sound(find_sfx_file("steel_snap"))

    # If it loaded fine, reduce volume so it is not deafening.
    if steel_snap_sfx is not None:
        steel_snap_sfx.set_volume(0.8)

    def play_status_sfx(key):
        """
        Play a status effect sound (burn, poison, etc) if available.
        """
        snd = status_sfx.get(key)
        if snd is not None:
            snd.play()

    # Punch sound is your fallback for missing move sounds.
    punch_sfx = pygame.mixer.Sound(str(punch_path))
    punch_sfx.set_volume(0.7)

    # Faint sound for pokemon fainting.
    faint_path = sound_dir / "pokemon_faint.wav"
    faint_sfx = pygame.mixer.Sound(str(faint_path))
    faint_sfx.set_volume(0.8)

    # Victory sound for winning.
    win_path = sound_dir / "fight_won.wav"
    win_sfx = pygame.mixer.Sound(str(win_path))
    win_sfx.set_volume(0.85)

    # Track whether menu music is currently supposed to be on.
    menu_music_on = False

    def ensure_menu_music():  # 59.3
        """
        Start menu background music if it is not already playing.
        """
        nonlocal menu_music_on

        # If we do not consider it on, or pygame says nothing is playing, restart it.
        if (not menu_music_on) or (not pygame.mixer.music.get_busy()):
            pygame.mixer.music.load(str(music_path))   # load menu track
            pygame.mixer.music.set_volume(0.35)        # lower than SFX
            pygame.mixer.music.play(loops=999999)      # loop forever
            menu_music_on = True                       # mark as on

    def stop_menu_music():  # 59.4
        """
        Fade out menu music if it is currently on.
        """
        nonlocal menu_music_on
        if menu_music_on:
            pygame.mixer.music.fadeout(400)  # fade smoothly, avoids hard cut
            menu_music_on = False

    # Track whether battle music is currently supposed to be on.
    battle_music_on = False

    def ensure_battle_music():  # 59.5
        """
        Start battle music if it is not already playing.
        Also fades out menu music if it was on.
        """
        nonlocal battle_music_on, menu_music_on

        # If menu music was running, fade it out first.
        if menu_music_on:
            pygame.mixer.music.fadeout(250)
            menu_music_on = False

        # Start battle track if not on or not playing.
        if (not battle_music_on) or (not pygame.mixer.music.get_busy()):
            pygame.mixer.music.load(str(battle_path))
            pygame.mixer.music.set_volume(0.35)
            pygame.mixer.music.play(loops=999999)
            battle_music_on = True

    def stop_battle_music():  # 59.6
        """
        Fade out battle music if it is currently on.
        """
        nonlocal battle_music_on
        if battle_music_on:
            pygame.mixer.music.fadeout(250)
            battle_music_on = False

    # Load all core data into your data module (csv and pickles).
    data.load_data(
        pokemon_csv_path="pokemon_kanto_johto_sinnoh.csv",
        type_chart_csv_path="type_chart.csv",
        moves_pickle_path="all_moves_by_name.pkl",
    )

    # Typing effect speed for the battle log.
    TYPE_MS_PER_CHAR = 28         # each new character appears every 28 ms
    TYPE_EXTRA_PAUSE_MS = 120     # pause after finishing a full line

    # Typing effect state variables.
    typing_active = False         # True while we are revealing characters
    typing_text = ""              # full message being typed
    typing_len = 0                # how many chars currently visible
    typing_next_tick = 0          # next pygame time tick when we reveal one more char

    # Build name pool from dataframe, normalize to lowercase, unique, sorted.
    name_pool = [str(x).strip().lower() for x in list(data.df_poke["Name"].astype(str).values)]
    name_pool = sorted(list(dict.fromkeys(name_pool)))  # dict trick removes duplicates

    # Window title.
    pygame.display.set_caption("Pokemon Battle")

    # Create the main window surface.
    screen = pygame.display.set_mode((1100, 720))

    # Clock controls FPS and gives dt timing.
    clock = pygame.time.Clock()

    # Pre build the selection screen animated background rows.
    select_bg_rows = build_select_background()

    # Fonts for UI.
    font = pygame.font.SysFont("arial", 18)     # small text
    font_big = pygame.font.SysFont("arial", 26) # headings and buttons

    # Theme colors for UI and battle area.
    colors = {
        "bg": (20, 20, 28),          # main background
        "panel": (35, 35, 48),       # panels
        "border": (90, 90, 120),     # outlines
        "text": (235, 235, 245),     # normal text
        "muted": (160, 160, 175),    # softer text (placeholder)
        "button": (50, 55, 80),      # active button fill
        "button_off": (35, 35, 48),  # disabled button fill
        "input_bg": (28, 28, 40),    # input box background
        "ok": (80, 170, 110),        # green highlight
        "warn": (190, 80, 80),       # red highlight
        "battle_bg": (168, 216, 255) # battle area sky color
    }

    # Type specific colors for move buttons.
    TYPE_COLORS = {
        "normal":   {"primary": (168, 168, 120), "dark": (120, 120, 85),  "text": (20, 20, 24)},
        "fire":     {"primary": (240, 128, 48),  "dark": (160, 75, 20),   "text": (20, 20, 24)},
        "water":    {"primary": (104, 144, 240), "dark": (55, 85, 170),   "text": (20, 20, 24)},
        "electric": {"primary": (248, 208, 48),  "dark": (170, 135, 15),  "text": (20, 20, 24)},
        "grass":    {"primary": (120, 200, 80),  "dark": (55, 120, 40),   "text": (20, 20, 24)},
        "ice":      {"primary": (152, 216, 216), "dark": (90, 150, 150),  "text": (20, 20, 24)},
        "fighting": {"primary": (192, 48, 40),   "dark": (120, 20, 18),   "text": (245, 245, 250)},
        "poison":   {"primary": (160, 64, 160),  "dark": (95, 25, 95),    "text": (245, 245, 250)},
        "ground":   {"primary": (224, 192, 104), "dark": (150, 120, 55),  "text": (20, 20, 24)},
        "flying":   {"primary": (168, 144, 240), "dark": (110, 90, 170),  "text": (20, 20, 24)},
        "psychic":  {"primary": (248, 88, 136),  "dark": (170, 40, 80),   "text": (245, 245, 250)},
        "bug":      {"primary": (168, 184, 32),  "dark": (110, 120, 10),  "text": (20, 20, 24)},
        "rock":     {"primary": (184, 160, 56),  "dark": (120, 100, 25),  "text": (20, 20, 24)},
        "ghost":    {"primary": (112, 88, 152),  "dark": (70, 50, 110),   "text": (245, 245, 250)},
        "dragon":   {"primary": (112, 56, 248),  "dark": (60, 20, 170),   "text": (245, 245, 250)},
        "dark":     {"primary": (112, 88, 72),   "dark": (65, 50, 40),    "text": (245, 245, 250)},
        "steel":    {"primary": (184, 184, 208), "dark": (120, 120, 150), "text": (20, 20, 24)},
        "fairy":    {"primary": (238, 153, 172), "dark": (170, 95, 115),  "text": (20, 20, 24)},
        "unknown":  {"primary": (100, 100, 110), "dark": (70, 70, 80),    "text": (245, 245, 250)},
    }

    # Intro animation speed for sliding pokemon into the battle area.
    INTRO_SPEED_PX_PER_S = 100.0

    # Current animated positions (used during intro slide).
    p1_anim_x = 0.0
    p2_anim_x = 0.0
    p1_anim_y = 0.0
    p2_anim_y = 0.0

    # Target positions where sprites should end up after intro.
    p1_target_x = 0.0
    p2_target_x = 0.0
    p1_target_y = 0.0
    p2_target_y = 0.0

    # Placeholder icon used when we do not have a real icon downloaded yet.
    placeholder_icon = make_placeholder_icon(font, colors)

    # Placeholder battle sprite used when we do not have the pokemon sprite.
    placeholder_battle = make_placeholder_battle_sprite(font_big, colors, BATTLE_PX)

    # Current sprite surfaces to draw for p1 and p2 in battle view.
    p1_battle_sprite = placeholder_battle
    p2_battle_sprite = placeholder_battle

    # Game environment with battle rules.
    env = BattleEnv()

    # Which top level screen we are in: "select", "battle", "end".
    state = "select"

    # End screen result: "win", "lose", or other.
    end_result = None

    # Search boxes for selection screen (left: you, right: opponent).
    left_search = TextInput((40, 90, 350, 40), font, "search your pokemon")
    right_search = TextInput((600, 90, 350, 40), font, "search opponent pokemon")

    # The currently selected pokemon names (strings) from the lists.
    left_selected = None
    right_selected = None

    # Remember last queries to reset scroll when user types.
    left_last_q = ""
    right_last_q = ""

    # Buttons in the selection screen and battle screen.
    start_btn = Button((420, 640, 260, 50), "Start battle", font_big)
    exit_btn = Button((40, 660, 200, 50), "Exit fight", font_big)
    again_btn = Button((860, 660, 200, 50), "Play again", font_big)
    mode_btn = Button((420, 570, 260, 50), "Opponent: Agent", font_big)

    # 4 move buttons in battle (one per action).
    move_btns = []
    for i in range(4):
        x = 40
        y = op.add(420, op.mul(i, 60))  # vertical spacing 60 px per button
        move_btns.append(Button((x, y, 520, 52), "", font))

    # Placeholder rects (these get resized later in the select loop).
    left_box = pygame.Rect(40, 150, 100, 100)
    right_box = pygame.Rect(640, 150, 100, 100)

    # Scrollbar logic for each selection list.
    left_list = ScrollList(left_box)
    right_list = ScrollList(right_box)

    # Names currently visible in list windows.
    shown_left = []
    shown_right = []

    # Smooth HP bar values (what you display) vs target HP (true HP after events).
    display_p1_hp = 0.0
    display_p2_hp = 0.0
    target_p1_hp = 0.0
    target_p2_hp = 0.0

    # Max HP used for scaling HP bars on screen.
    max_p1_hp_screen = 1.0
    max_p2_hp_screen = 1.0

    # Battle state machine:
    # "intro": slide in
    # "choose": waiting for player action
    # "anim": playing event animations
    battle_mode = "choose"

    # Event queue produced by env step transcript (attacks, hp changes, messages).
    pending_events = []
    event_i = 0                 # current event index being consumed
    event_wait_until = 0        # pygame tick time when next event is allowed
    turn_done_flag = False      # True when env says battle ended

    # Mode flags:
    use_agent_mode = True       # True: opponent is AI agent
    human_side = "p1"           # Which env side is controlled by the human
    last_obs = None             # Latest observation used by agent decision

    # Battle log UI state.
    log_lines = []              # raw log strings
    LOG_LINE_H = 24             # pixels per rendered log line
    log_scroll = ScrollList(pygame.Rect(600, 190, 460, 400))
    log_follow = True           # if True, auto scroll to bottom when new text appears

    # Log layout widths.
    LOG_TEXT_W = 436
    LOG_MAX_W = int(op.sub(LOG_TEXT_W, 8))

    # Cached wrapped lines (log_lines expanded into wrapped lines).
    log_render = []

    def rebuild_log_render():  # 59.7
        """
        Rebuild the wrapped log lines cache after log_lines changes.
        This makes scrolling and drawing cheap.
        """
        nonlocal log_render
        out = []
        for line in log_lines:
            out.extend(wrap_text(font, line, LOG_MAX_W))
        log_render = out

    # Remember last battle config so "Play again" can reset with same teams and movesets.
    remember_p1 = None
    remember_p2 = None
    remember_p1_moveset = None
    remember_p2_moveset = None

    def log_at_bottom(total=None):  # 59.8
        """
        Return True if the log scroll is currently at the bottom.
        Used so we only auto follow when the user has not scrolled up.
        """
        if total is None:
            total = len(log_render)
        visible = log_scroll.visible
        max_scroll = max(0, int(op.sub(int(total), int(visible))))
        return int(log_scroll.scroll) >= int(max_scroll)

    def push_log(msg):  # 59.9
        """
        Append a new message to the log and update scroll if follow mode is active.
        """
        nonlocal log_follow

        # If we are already at bottom, keep follow on.
        if log_at_bottom():
            log_follow = True

        # Add new line.
        log_lines.append(str(msg))

        # Re wrap text after adding new line.
        rebuild_log_render()

        # If follow is on, jump scroll to bottom.
        if log_follow:
            total = len(log_render)
            visible = log_scroll.visible
            max_scroll = max(0, int(op.sub(total, visible)))
            log_scroll.scroll = int(max_scroll)

    def begin_battle():  # 59.10
        """
        Transition from the selection screen into an actual battle.
    
        What this function sets up:
        - resets faint / animation state
        - resets the env with the chosen pokemon (and possibly the agent)
        - decides which env side is the human vs agent
        - loads correct sprites for screen p1 (human back) and screen p2 (opponent front)
        - initializes HP display targets and intro slide positions
        - fills the move buttons with the human moveset (names, power, accuracy, colors)
        - clears pending animation events and log
        - switches state to "battle"
        """
    
        # We are going to reassign these variables from the outer main() scope,
        # so we must mark them as nonlocal.
        nonlocal state
        nonlocal display_p1_hp, display_p2_hp, target_p1_hp, target_p2_hp
        nonlocal battle_mode, pending_events, event_i, event_wait_until, turn_done_flag
        nonlocal log_lines, log_follow
        nonlocal p1_battle_sprite, p2_battle_sprite
    
        # These are the animated intro positions and their final targets.
        nonlocal p1_anim_x, p2_anim_x, p1_anim_y, p2_anim_y
        nonlocal p1_target_x, p2_target_x, p1_target_y, p2_target_y
    
        # Faint flags live as globals, so declare them global before modifying.
        global p1_faint_active, p1_faint_t, p2_faint_active, p2_faint_t
    
        # Reset faint animation state for both sides (fresh battle).
        p1_faint_active = False
        p1_faint_t = 0.0
        p2_faint_active = False
        p2_faint_t = 0.0
    
        # Clear battle log and enable auto follow (so it scrolls to the latest).
        log_lines = []
        log_follow = True
    
        # Rebuild wrapped log cache so render matches the cleared log.
        rebuild_log_render()
    
        # Read player selections from the selection UI.
        # left_selected is "your pokemon", right_selected is opponent.
        p1 = str(left_selected).strip().lower()   # human pick from left side UI
        p2 = str(right_selected).strip().lower()  # opponent pick from right side UI
    
        # We may switch modes here, so we modify outer variables.
        nonlocal use_agent_mode, human_side, last_obs
    
        if use_agent_mode:
            # In agent mode, you decided:
            # - agent controls env.p1
            # - human controls env.p2
            human_side = "p2"
    
            # IMPORTANT: you call env.reset(opponent_first, human_second)
            # That is why you pass (p2, p1) here.
            # This makes env.p1 = p2 (agent) and env.p2 = p1 (human).
            last_obs = env.reset(
                p2,                       # env.p1 pokemon (agent controlled)
                p1,                       # env.p2 pokemon (human controlled)
                p1_moveset=None,          # let env choose moveset
                p2_moveset=None,          # let env choose moveset
                seed=random.randint(0, 10**9),  # random seed for reproducibility / variety
            )
    
            # Load the torch model now that we know we actually need it.
            ensure_agent_loaded()
    
        else:
            # In random mode, you decided:
            # - human controls env.p1
            # - opponent controls env.p2
            human_side = "p1"
    
            # Here you pass (p1, p2) so env.p1 = human pick, env.p2 = opponent pick.
            last_obs = env.reset(
                p1,                       # env.p1 pokemon (human controlled)
                p2,                       # env.p2 pokemon (opponent)
                p1_moveset=None,
                p2_moveset=None,
                seed=random.randint(0, 10**9),
            )
    
        # Store battle setup so "Play again" can recreate it with same movesets.
        nonlocal remember_p1, remember_p2, remember_p1_moveset, remember_p2_moveset
    
        remember_p1 = p1
        remember_p2 = p2
    
        # Deepcopy because env.p1_moveset / env.p2_moveset are mutable objects.
        # Without deepcopy, later changes could silently affect the stored version.
        remember_p1_moveset = copy.deepcopy(env.p1_moveset)
        remember_p2_moveset = copy.deepcopy(env.p2_moveset)
    
        # Screen rule:
        # - screen p1 slot is ALWAYS the human (drawn with back sprite)
        # - screen p2 slot is ALWAYS the opponent (drawn with front sprite)
        #
        # But env.p1 / env.p2 depends on mode.
        # So we map env slots to screen slots here.
        if human_side == "p1":
            # Human is env.p1, so screen p1 pulls from env.p1.
            you_name_lower = str(env.p1["Name"]).strip().lower()
            opp_name_lower = str(env.p2["Name"]).strip().lower()
    
            # HP values pulled from env state.
            you_hp = float(env.p1_hp)
            opp_hp = float(env.p2_hp)
    
            # The human move buttons should reflect the human moveset.
            you_moveset = env.p1_moveset
    
        else:
            # Human is env.p2 (agent mode), so screen p1 must pull from env.p2.
            you_name_lower = str(env.p2["Name"]).strip().lower()
            opp_name_lower = str(env.p1["Name"]).strip().lower()
    
            # Swap HP values accordingly so screen p1 is always the human HP.
            you_hp = float(env.p2_hp)
            opp_hp = float(env.p1_hp)
    
            # Human moveset is env.p2_moveset.
            you_moveset = env.p2_moveset
    
        # These max HP values are used to scale the HP bars on screen.
        nonlocal max_p1_hp_screen, max_p2_hp_screen
        max_p1_hp_screen = float(you_hp)  # screen p1 is human
        max_p2_hp_screen = float(opp_hp)  # screen p2 is opponent
    
        # Load battle sprites with caching and fallback placeholder.
        # Screen p1 is always human -> back sprite dir.
        p1_battle_sprite = get_battle_sprite(
            you_name_lower,
            SPRITE_BACK_DIR,
            BATTLE_PX,
            placeholder_battle,
        )
    
        # Screen p2 is always opponent -> front sprite dir.
        p2_battle_sprite = get_battle_sprite(
            opp_name_lower,
            SPRITE_FRONT_DIR,
            BATTLE_PX,
            placeholder_battle,
        )
    
        # Initialize displayed HP values (what the bars show).
        display_p1_hp = you_hp
        display_p2_hp = opp_hp
    
        # Initialize target HP values (what display will ease toward).
        target_p1_hp = you_hp
        target_p2_hp = opp_hp
    
        # Define the battle rectangle (where sprites are drawn).
        battle_area = pygame.Rect(40, 120, 520, 290)
    
        # Compute fixed sprite anchor points inside the battle area.
        # Opponent goes more to the right and higher.
        opp_x = int(op.add(battle_area.x, 300))
        opp_y = int(op.add(battle_area.y, 10))
    
        # Human goes more to the left and lower.
        you_x = int(op.add(battle_area.x, 10))
        you_y = int(op.add(battle_area.y, 80))
    
        # Save final resting positions for intro slide animation.
        p2_target_x = float(opp_x)
        p2_target_y = float(opp_y)
        p1_target_x = float(you_x)
        p1_target_y = float(you_y)
    
        # Start opponent sprite offscreen on the left for the intro slide in.
        p2_anim_x = float(op.sub(battle_area.x, BATTLE_PX))
        p2_anim_y = float(opp_y)
    
        # Start human sprite offscreen on the right for the intro slide in.
        # battle_area.right is the x coordinate at the right edge of the battle area.
        p1_anim_x = float(op.add(battle_area.right, 0))
        p1_anim_y = float(you_y)
    
        # Enter intro animation mode (sprites slide into target positions).
        battle_mode = "intro"
    
        # Re sync display + target HP directly from env to be extra safe.
        # This matters if env.reset returned values that differ from our earlier you_hp.
        if human_side == "p1":
            display_p1_hp = float(env.p1_hp)
            display_p2_hp = float(env.p2_hp)
            target_p1_hp = float(env.p1_hp)
            target_p2_hp = float(env.p2_hp)
        else:
            display_p1_hp = float(env.p2_hp)
            display_p2_hp = float(env.p1_hp)
            target_p1_hp = float(env.p2_hp)
            target_p2_hp = float(env.p1_hp)
    
        # Clear event queue and reset animation event pointers.
        pending_events = []
        event_i = 0
        event_wait_until = 0
        turn_done_flag = False
    
        # Seed the log with who is who (nice UX).
        push_log("You are " + title_name(you_name_lower))
        push_log("Opponent is " + title_name(opp_name_lower))
        push_log("")  # blank spacer line
    
        # Populate move buttons for the human moveset.
        for j in range(4):
            mv = you_moveset[j]  # moveset is list of 4 move dicts
    
            # Pretty names for UI.
            mv_name = title_name(mv.get("move", "move"))
            mv_type = title_name(mv.get("type", ""))
    
            # Lowercase type used as dict key into TYPE_COLORS.
            mv_type_lower = str(mv.get("type", "unknown")).strip().lower()
    
            # Power and accuracy can be missing; fall back to sensible defaults.
            pw = mv.get("power")
            ac = mv.get("accuracy")
            if pw is None:
                pw = 40
            if ac is None:
                ac = 100
    
            # Update the button label text.
            move_btns[j].text = f"{j + 1}. {mv_name}   type {mv_type}   power {pw}   acc {ac}"
    
            # Color the move button according to type; fallback to "unknown".
            tcol = TYPE_COLORS.get(mv_type_lower, TYPE_COLORS["unknown"])
            move_btns[j].bg = tcol["primary"]  # background fill color
            move_btns[j].fg = tcol["text"]     # text color for readability
    
        # Switch top level state: now we render battle UI and accept battle inputs.
        state = "battle"


    def exit_fight():  # 59.11
        """
        Return from battle (or end screen) back to the selection screen.
    
        What this function does:
        - switches the main state back to "select"
        - hard resets the battle animation pipeline (mode, events, timers)
        - cancels any typing animation that is mid message
        - clears the log and rebuilds the wrapped render cache
        """
    
        # These live in main() scope (not global), so we mark them nonlocal to reassign.
        nonlocal state
        nonlocal battle_mode, pending_events, event_i, event_wait_until, turn_done_flag
        nonlocal log_lines, log_follow
        nonlocal typing_active, typing_text, typing_len, typing_next_tick
    
        # Switch UI state: next frame, the program draws the selection screen.
        state = "select"
    
        # Reset battle mode so when you enter battle again you start clean.
        # "choose" is the idle state where you can click move buttons.
        battle_mode = "choose"
    
        # Clear any queued animation events (attacks, hp changes, messages, sfx, faint).
        pending_events = []
    
        # Reset the event pointer so we would start from the first event next time.
        event_i = 0
    
        # Reset event timing gate (used to delay between events in animation mode).
        event_wait_until = 0
    
        # Reset turn done flag so the next battle does not instantly jump to end.
        turn_done_flag = False
    
        # Cancel the typewriter effect (animated text printing into the log).
        typing_active = False
    
        # Clear whatever message was currently being typed.
        typing_text = ""
    
        # Reset how many characters have been revealed so far.
        typing_len = 0
    
        # Reset the next scheduled tick for typing.
        typing_next_tick = 0
    
        # Clear the log lines shown on the right panel.
        log_lines = []
    
        # Re enable auto follow so the log sticks to the bottom when new lines arrive.
        log_follow = True
    
        # Rebuild the pre wrapped render cache so the log panel matches the cleared log.
        rebuild_log_render()


    def restart_same_battle():  # 59.12
        """
        Restart the exact same matchup again, using the same two Pokémon and the same movesets.
    
        What this does differently vs begin_battle():
        - It does NOT read left_selected / right_selected again.
        - It reuses remember_p1, remember_p2, and their saved movesets so the rematch feels consistent.
        - It still picks a fresh random seed so damage rolls etc can vary.
    
        Big picture steps:
        1) Reset faint state and UI log.
        2) If we do not have a remembered matchup yet, fall back to begin_battle().
        3) Reset the BattleEnv with remembered Pokémon and remembered movesets.
        4) Rebuild sprites, HP bars, intro positions, and move buttons.
        5) Switch state back to "battle" and start intro animation.
        """
    
        # These are variables defined in main() scope, so we need nonlocal to reassign them here.
        nonlocal state
        nonlocal display_p1_hp, display_p2_hp, target_p1_hp, target_p2_hp
        nonlocal max_p1_hp_screen, max_p2_hp_screen
        nonlocal battle_mode, pending_events, event_i, event_wait_until, turn_done_flag
        nonlocal log_lines, log_follow
        nonlocal p1_battle_sprite, p2_battle_sprite
        nonlocal p1_anim_x, p2_anim_x, p1_anim_y, p2_anim_y
        nonlocal p1_target_x, p2_target_x, p1_target_y, p2_target_y
        nonlocal last_obs, human_side, use_agent_mode
        nonlocal remember_p1, remember_p2, remember_p1_moveset, remember_p2_moveset
    
        # Faint animation flags are globals (shared by battle drawing code), so we reset them globally.
        global p1_faint_active, p1_faint_t, p2_faint_active, p2_faint_t
        p1_faint_active = False     # p1 sprite is not fading out anymore
        p1_faint_t = 0.0            # reset faint timer for p1
        p2_faint_active = False     # p2 sprite is not fading out anymore
        p2_faint_t = 0.0            # reset faint timer for p2
    
        # If we do not have a saved matchup, we cannot "restart same" anything.
        # So we just start a normal battle using the current UI selections.
        if (
            remember_p1 is None
            or remember_p2 is None
            or remember_p1_moveset is None
            or remember_p2_moveset is None
        ):
            begin_battle()
            return
    
        # Clear the log so the rematch starts fresh.
        log_lines = []
        log_follow = True           # keep the log pinned to the bottom
        rebuild_log_render()        # rebuild wrapped text cache
    
        # Fresh seed for this rematch: same teams, but randomness can still vary.
        seed_val = random.randint(0, 10**9)
    
        # Reset the environment using the remembered Pokémon and deep copied movesets.
        # Deep copy matters: env.reset may mutate moveset dicts later (PP, etc), and we want "remember_*" pristine.
        if use_agent_mode:
            # In agent mode you intentionally swap: agent is env.p1, human is env.p2.
            human_side = "p2"
            ensure_agent_loaded()   # load the neural net once, if not already loaded
            last_obs = env.reset(
                remember_p2,                        # env.p1 (agent) gets opponent Pokémon
                remember_p1,                        # env.p2 (human) gets your Pokémon
                p1_moveset=copy.deepcopy(remember_p1_moveset),
                p2_moveset=copy.deepcopy(remember_p2_moveset),
                seed=seed_val,
            )
        else:
            # Normal mode: human is env.p1, opponent is env.p2.
            human_side = "p1"
            last_obs = env.reset(
                remember_p1,                        # env.p1 (human)
                remember_p2,                        # env.p2 (opponent)
                p1_moveset=copy.deepcopy(remember_p1_moveset),
                p2_moveset=copy.deepcopy(remember_p2_moveset),
                seed=seed_val,
            )
    
        # Decide which Pokémon is "you" and which is "opponent" based on human_side.
        # This is about the ENV mapping, not the screen mapping.
        if human_side == "p1":
            you_name_lower = str(env.p1["Name"]).strip().lower()
            opp_name_lower = str(env.p2["Name"]).strip().lower()
            you_hp = float(env.p1_hp)
            opp_hp = float(env.p2_hp)
            you_moveset = env.p1_moveset
        else:
            you_name_lower = str(env.p2["Name"]).strip().lower()
            opp_name_lower = str(env.p1["Name"]).strip().lower()
            you_hp = float(env.p2_hp)
            opp_hp = float(env.p1_hp)
            you_moveset = env.p2_moveset
    
        # Save the max HP values used for the bar scaling on screen.
        max_p1_hp_screen = float(you_hp)
        max_p2_hp_screen = float(opp_hp)
    
        # Build the sprite surfaces for the battle screen.
        # Screen rule: p1 slot is always the human (back sprite), p2 slot always the opponent (front sprite).
        p1_battle_sprite = get_battle_sprite(
            you_name_lower, SPRITE_BACK_DIR, BATTLE_PX, placeholder_battle
        )
        p2_battle_sprite = get_battle_sprite(
            opp_name_lower, SPRITE_FRONT_DIR, BATTLE_PX, placeholder_battle
        )
    
        # Initialize displayed HP and targets so the bars start at full values.
        display_p1_hp = you_hp
        display_p2_hp = opp_hp
        target_p1_hp = you_hp
        target_p2_hp = opp_hp
    
        # Define where the fight happens on screen.
        battle_area = pygame.Rect(40, 120, 520, 290)
    
        # Target positions where sprites should end up after intro slide in.
        opp_x = int(op.add(battle_area.x, 300))
        opp_y = int(op.add(battle_area.y, 10))
        you_x = int(op.add(battle_area.x, 10))
        you_y = int(op.add(battle_area.y, 80))
    
        # Store the final positions as floats for smooth approach animation.
        p2_target_x = float(opp_x)
        p2_target_y = float(opp_y)
        p1_target_x = float(you_x)
        p1_target_y = float(you_y)
    
        # Start positions for intro: opponent comes from the left, you come from the right.
        p2_anim_x = float(op.sub(battle_area.x, BATTLE_PX))
        p2_anim_y = float(opp_y)
    
        p1_anim_x = float(op.add(battle_area.right, 0))
        p1_anim_y = float(you_y)
    
        # Reset the animation queue state for this new match.
        pending_events = []         # no events queued yet
        event_i = 0                 # start at first event
        event_wait_until = 0        # no delay gate yet
        turn_done_flag = False      # battle not finished yet
    
        # Kick off the intro slide in.
        battle_mode = "intro"
    
        # Sync the HP values from the env again (paranoia, but keeps things consistent).
        # This mirrors begin_battle() behavior.
        if human_side == "p1":
            display_p1_hp = float(env.p1_hp)
            display_p2_hp = float(env.p2_hp)
            target_p1_hp = float(env.p1_hp)
            target_p2_hp = float(env.p2_hp)
        else:
            display_p1_hp = float(env.p2_hp)
            display_p2_hp = float(env.p1_hp)
            target_p1_hp = float(env.p2_hp)
            target_p2_hp = float(env.p1_hp)
    
        # Announce matchup in the log.
        push_log("You are " + title_name(you_name_lower))
        push_log("Opponent is " + title_name(opp_name_lower))
        push_log("")
    
        # Update the move buttons with names, types, power, accuracy and button colors.
        for j in range(4):
            mv = you_moveset[j]
    
            # Build pretty names for the UI.
            mv_name = title_name(mv.get("move", "move"))
            mv_type = title_name(mv.get("type", ""))
            mv_type_lower = str(mv.get("type", "unknown")).strip().lower()
    
            # Provide defaults if the move lacks metadata.
            pw = mv.get("power")
            ac = mv.get("accuracy")
            if pw is None:
                pw = 40
            if ac is None:
                ac = 100
    
            # Assign the button label.
            move_btns[j].text = f"{j + 1}. {mv_name}   type {mv_type}   power {pw}   acc {ac}"
    
            # Color code the button by move type.
            tcol = TYPE_COLORS.get(mv_type_lower, TYPE_COLORS["unknown"])
            move_btns[j].bg = tcol["primary"]   # background color
            move_btns[j].fg = tcol["text"]      # text color
    
        # Finally switch to battle state so the main loop renders the battle UI again.
        state = "battle"


    def start_turn(action_idx):  # 59.13
        """
        Run one full turn inside the BattleEnv, then convert the env output into a queue of UI events.
    
        Inputs:
        - action_idx: the move button the human clicked (0..3)
    
        Outputs (side effects):
        - Updates last_obs (so the agent can act next turn)
        - Sets turn_done_flag if the battle ended
        - Builds pending_events like:
            {"kind": "atk", "who": "p1"/"p2", "atype": "fire"/...}
            {"kind": "msg", "text": "..."}
            {"kind": "hp", "who": "p1"/"p2", "new_hp": 12.0}
            {"kind": "faint", "who": ...}
            {"kind": "sfx", "key": "burn"/...}
        - Switches battle_mode to "anim" so the main loop plays them out with delays.
        """
    
        # Debug: show what move index was clicked.
        push_log("DEBUG start_turn called with action_idx " + str(action_idx))
        print("DEBUG start_turn called with action_idx", action_idx)
    
        # We mutate these variables defined in main(), so we declare them nonlocal.
        nonlocal battle_mode, pending_events, event_i, event_wait_until, turn_done_flag
        nonlocal last_obs, use_agent_mode, human_side
    
        try:
            # Snapshot HP before the env step so we can detect changes later.
            # (snaps includes per damage step HP values, but we still track baselines here.)
            p1_before = float(env.p1_hp)
            p2_before = float(env.p2_hp)
    
            # Actually run the environment turn.
            if use_agent_mode:
                # In agent mode:
                # - agent controls env.p1
                # - human controls env.p2
                a_p1 = agent_act(last_obs)      # agent picks its move from last observation
                a_p2 = int(action_idx)          # human clicked move index becomes env.p2 action
    
                # step_detailed returns:
                # obs, reward, done, info, transcript (text lines), snaps (hp snapshots)
                obs, reward, done, info, transcript, snaps = env.step_detailed(
                    a_p1, opp_action_idx=a_p2
                )
            else:
                # In normal mode, env.p1 is human and env chooses opponent internally.
                obs, reward, done, info, transcript, snaps = env.step_detailed(int(action_idx))
    
            # Save the new observation for the next turn (needed by agent mode).
            last_obs = obs
    
            # done=True means the battle ended (someone fainted, etc).
            turn_done_flag = bool(done)
    
            # Fresh event queue for this turn's animation playback.
            pending_events = []
            last_p1 = p1_before         # last known p1 hp (used to detect changes)
            last_p2 = p2_before         # last known p2 hp
            snap_i = 0                  # index into the snaps list
    
            # Cache names for attacker detection in transcript parsing.
            p1_name = str(env.p1["Name"]).strip().lower()
            p2_name = str(env.p2["Name"]).strip().lower()
    
            def move_type_for(env_side, idx):  # 59.13.1
                """
                Given env side ("p1" or "p2") and a move index, return the move's type string.
                Used to pick the right animation + sound for ATTACK events.
    
                Returns "unknown" if anything is missing or out of range.
                """
                try:
                    if idx is None:
                        return "unknown"
    
                    # Pick the correct moveset based on who is acting in the environment.
                    if env_side == "p1":
                        mv = env.p1_moveset[int(idx)]
                    else:
                        mv = env.p2_moveset[int(idx)]
    
                    # Move dict is expected to contain {"type": "..."}.
                    return str(mv.get("type", "unknown")).strip().lower()
                except Exception:
                    return "unknown"
    
            # We want to know which move index was used by each env side,
            # because we use it to fetch move type for FX.
            idx_env_p1 = None
            idx_env_p2 = None
    
            if use_agent_mode:
                # We already computed them above.
                idx_env_p1 = a_p1
                idx_env_p2 = action_idx
            else:
                # Human action is env.p1.
                idx_env_p1 = action_idx
    
                # Opponent action index is sometimes reported in info as "opp_action_idx".
                idx_env_p2 = info.get("opp_action_idx", None) if isinstance(info, dict) else None
    
            def env_to_screen(who_env):
                """
                Convert env side ("p1"/"p2") to screen side ("p1"/"p2").
    
                Reason: when use_agent_mode is True, the human is env.p2,
                but on screen you still want the human drawn as screen p1.
                So we swap labels in that case.
                """
                # If human is env.p1, screen mapping is direct.
                if human_side == "p1":
                    return who_env
    
                # If human is env.p2, swap env labels to keep human on screen p1.
                return "p2" if who_env == "p1" else "p1"
    
            # Parse transcript lines into animation events.
            for line in transcript:
                marker = str(line)             # keep original casing for display
                lower = marker.strip().lower() # normalized for keyword checks
    
                # 1) Status sound effects: if transcript mentions a PROC key, enqueue its SFX.
                # Example: "burned", "poisoned", etc.
                for key in PROC.keys():
                    if key in lower:
                        pending_events.append({"kind": "sfx", "key": key})
                        break
    
                # 2) Attack event: when transcript starts with "attack:", trigger attack animation + type.
                if lower.startswith("attack:"):
                    # Default assume env.p1 attacked.
                    who_env = "p1"
    
                    # If the line mentions p2's name but not p1's name, assume p2 attacked.
                    # This is a heuristic based on how your transcript strings are formatted.
                    if (p2_name in lower) and (p1_name not in lower):
                        who_env = "p2"
    
                    # Find the move type for the attacker so visuals can match move type.
                    if who_env == "p1":
                        atype = move_type_for("p1", idx_env_p1)
                    else:
                        atype = move_type_for("p2", idx_env_p2)
    
                    # Queue the attack animation event, but translated to screen coordinates.
                    pending_events.append(
                        {"kind": "atk", "who": env_to_screen(who_env), "atype": atype}
                    )
    
                # 3) Always show the transcript line in the log (typed out).
                pending_events.append({"kind": "msg", "text": marker})
    
                # 4) HP / faint events:
                # We only consume a snap when the transcript line is "damage related"
                # so HP bar changes roughly sync to the right messages.
                is_damage_related = (
                    marker.startswith("ATTACK:")
                    or ("takes damage from" in marker)
                    or ("hurt itself in confusion" in marker)
                    or marker.startswith("END ROUND")
                )
    
                # If we have a snap ready for this damage related marker, apply it.
                if is_damage_related and snap_i < len(snaps):
                    snap = snaps[snap_i]
                    snap_i = int(op.add(snap_i, 1))  # advance snap index
    
                    # Pull HP values out of snap; fall back to last known if missing.
                    new_p1 = float(snap.get("p1_hp", last_p1))
                    new_p2 = float(snap.get("p2_hp", last_p2))
    
                    # If p1 HP changed, enqueue hp bar update and possible faint.
                    if abs(op.sub(new_p1, last_p1)) > 1e-6:
                        pending_events.append(
                            {"kind": "hp", "who": env_to_screen("p1"), "new_hp": new_p1}
                        )
                        if float(new_p1) <= 0.0:
                            pending_events.append(
                                {"kind": "faint", "who": env_to_screen("p1")}
                            )
                        last_p1 = new_p1
    
                    # If p2 HP changed, enqueue hp bar update and possible faint.
                    if abs(op.sub(new_p2, last_p2)) > 1e-6:
                        pending_events.append(
                            {"kind": "hp", "who": env_to_screen("p2"), "new_hp": new_p2}
                        )
                        if float(new_p2) <= 0.0:
                            pending_events.append(
                                {"kind": "faint", "who": env_to_screen("p2")}
                            )
                        last_p2 = new_p2
    
                    # Debug: show how big the queue is getting.
                    push_log("DEBUG pending_events count " + str(len(pending_events)))
                    print("DEBUG pending_events count", len(pending_events))
                    if len(pending_events) > 0:
                        print("DEBUG first event", pending_events[0])
    
            # Reset event playback pointers.
            event_i = 0                              # start consuming from the first event
            event_wait_until = pygame.time.get_ticks()  # allow immediate playback
            battle_mode = "anim"                     # tell main loop to start animation playback
    
        except Exception as e:
            # If anything explodes, log it and return to choose mode so the game doesn't lock up.
            push_log("Crash in start_turn:")
            push_log(repr(e))
            battle_mode = "choose"


    # ----------------------------
    # Main game loop
    # ----------------------------
    running = True
    while running:
        # dt = delta time in seconds since last frame.
        # clock.tick(60) caps the loop to ~60 FPS and returns elapsed milliseconds.
        dt = float(clock.tick(60)) / 1000.0
    
        # -----------------------------------------
        # 1) Per state background + music switching
        # -----------------------------------------
        if state == "select":
            # Selection screen: stop battle music, ensure menu music is playing,
            # and draw the animated scrolling sprite background.
            stop_battle_music()
            ensure_menu_music()
            draw_select_background(screen, select_bg_rows, 1100, 720, dt)
    
        elif state == "battle":
            # Battle screen: stop menu music, ensure battle music, clear to dark UI bg.
            stop_menu_music()
            ensure_battle_music()
            screen.fill(colors["bg"])
    
        elif state == "end":
            # End screen: menu music should be off.
            stop_menu_music()
    
            # If player won, stop battle music to let the win sound stand out.
            # Otherwise keep battle music running (lose/end vibe).
            if end_result == "win":
                stop_battle_music()
            else:
                ensure_battle_music()
    
            # Same dark background as battle UI.
            screen.fill(colors["bg"])
    
        else:
            # Fallback if state got corrupted or you added a new state and forgot to draw it.
            stop_menu_music()
            stop_battle_music()
            screen.fill(colors["bg"])
    
        # -----------------------------------------
        # 2) Async icon downloads: move finished ones into caches
        # -----------------------------------------
        # Worker threads fetch bytes into _icon_ready_bytes.
        # This pulls them into pygame Surfaces on the main thread (pygame is not thread safe).
        process_ready_icons(placeholder_icon)
    
        # -----------------------------------------
        # 3) Selection state: compute lists, scrolling, and icon prefetching
        # -----------------------------------------
        if state == "select":
            # Define the visible list rectangles each frame (so you can tweak layout easily).
            left_box = pygame.Rect(40, 150, 350, 350)
            right_box = pygame.Rect(600, 150, 350, 350)
    
            # Update the scroll list widgets with the current rectangles.
            left_list.rect = left_box
            right_list.rect = right_box
    
            # Filter pokemon names based on the current text inputs.
            matches_left_all = list_matches(name_pool, left_search.text)
            matches_right_all = list_matches(name_pool, right_search.text)
    
            # How many rows fit in each list (based on ROW_H).
            left_visible = max(1, int(op.truediv(left_box.height, ROW_H)))
            right_visible = max(1, int(op.truediv(right_box.height, ROW_H)))
    
            # Tell scroll lists how many items exist and how many can be shown.
            left_list.set_counts(len(matches_left_all), left_visible)
            right_list.set_counts(len(matches_right_all), right_visible)
    
            # Slice the current window of items based on scroll offset.
            shown_left = matches_left_all[left_list.scroll : op.add(left_list.scroll, left_visible)]
            shown_right = matches_right_all[right_list.scroll : op.add(right_list.scroll, right_visible)]
    
            # Prefetch icons slightly beyond the visible window so scrolling feels instant.
            prefetch_left = matches_left_all[
                left_list.scroll : op.add(left_list.scroll, op.add(left_visible, 12))
            ]
            prefetch_right = matches_right_all[
                right_list.scroll : op.add(right_list.scroll, op.add(right_visible, 12))
            ]
    
            # Queue icon downloads (or load from disk cache) for the prefetched names.
            for nm in prefetch_left:
                request_icon(nm, placeholder_icon)
            for nm in prefetch_right:
                request_icon(nm, placeholder_icon)
    
        # -----------------------------------------
        # 4) Battle state: advance animation timers and consume pending_events
        # -----------------------------------------
        if state == "battle":
            # Smooth HP bar animation: display value eases toward the target.
            hp_speed = 10.0
            display_p1_hp = approach_value(display_p1_hp, target_p1_hp, op.mul(hp_speed, dt))
            display_p2_hp = approach_value(display_p2_hp, target_p2_hp, op.mul(hp_speed, dt))
    
            # Attacker lunge timer for p1.
            if p1_attack_active:
                p1_attack_t = float(op.add(p1_attack_t, LUNGE_T_STEP))
                if p1_attack_t >= float(attack_duration):
                    p1_attack_active = False
    
            # Attacker lunge timer for p2.
            if p2_attack_active:
                p2_attack_t = float(op.add(p2_attack_t, LUNGE_T_STEP))
                if p2_attack_t >= float(attack_duration):
                    p2_attack_active = False
    
            # Attack FX timer (particles, beams, etc).
            if atk_fx_active:
                atk_fx_t = float(op.add(atk_fx_t, FX_T_STEP))
                if atk_fx_t >= float(attack_duration):
                    atk_fx_t = 0.0
                    atk_fx_active = False
    
            # Faint animation timers.
            if p1_faint_active:
                p1_faint_t = float(op.add(p1_faint_t, FAINT_T_STEP))
                if p1_faint_t >= float(faint_duration):
                    p1_faint_t = float(faint_duration)
    
            if p2_faint_active:
                p2_faint_t = float(op.add(p2_faint_t, FAINT_T_STEP))
                if p2_faint_t >= float(faint_duration):
                    p2_faint_t = float(faint_duration)
    
            # Intro mode: slide sprites into position at a constant speed.
            if battle_mode == "intro":
                step = float(op.mul(INTRO_SPEED_PX_PER_S, dt))
    
                # Ease both sprites toward their final positions.
                p1_anim_x = approach_value(p1_anim_x, p1_target_x, step)
                p2_anim_x = approach_value(p2_anim_x, p2_target_x, step)
    
                # If both are close enough, switch to "choose".
                close1 = abs(op.sub(p1_anim_x, p1_target_x)) < 0.5
                close2 = abs(op.sub(p2_anim_x, p2_target_x)) < 0.5
                if close1 and close2:
                    battle_mode = "choose"
    
            # Anim mode: consume the pending_events queue with delays and typewriter effect.
            if battle_mode == "anim":
                now = pygame.time.get_ticks()
    
                # If we're currently typing a message, advance one char per tick.
                if typing_active:
                    if now >= typing_next_tick:
                        typing_len = int(op.add(typing_len, 1))
                        typing_next_tick = int(op.add(now, TYPE_MS_PER_CHAR))
    
                        # When finished typing, push the full line and move to next event.
                        if typing_len >= len(typing_text):
                            push_log(typing_text)
                            typing_active = False
                            typing_text = ""
                            typing_len = 0
    
                            event_i = int(op.add(event_i, 1))
                            event_wait_until = int(op.add(now, TYPE_EXTRA_PAUSE_MS))
                    else:
                        # Still waiting for next char tick; do nothing.
                        pass
    
                else:
                    # Not typing: we can consume the next event if its delay has passed.
                    if event_i < len(pending_events) and now >= event_wait_until:
                        ev = pending_events[event_i]
                        print(
                            "DEBUG anim consuming",
                            event_i,
                            "of",
                            len(pending_events),
                            "kind",
                            ev.get("kind"),
                        )
    
                        kind = ev.get("kind")
    
                        # Attack event: start the animation and play the correct move sound.
                        if kind == "atk":
                            who = ev.get("who", "p1")
                            atype = ev.get("atype", "unknown")
                            push_log("DEBUG atype " + str(atype))
                            print("DEBUG atype", atype)
    
                            # Starts lunge + sets atk_fx_type and related effect state.
                            start_attack(who, atype)
    
                            # Plays the move type sound (fire/water/steel/etc).
                            play_move_sfx(atype)
    
                            # If there was no type sound loaded, use punch as fallback.
                            if move_sfx.get(str(atype).strip().lower()) is None:
                                punch_sfx.play()
    
                            # Move to next event after a fixed animation delay.
                            event_i = int(op.add(event_i, 1))
                            event_wait_until = int(op.add(now, ANIM_DELAY_MS))
                            continue
    
                        # Faint event: play faint sound and start faint animation.
                        if kind == "faint":
                            who = ev.get("who", "p2")
                            faint_sfx.play()
                            start_faint(who)
                            event_i = int(op.add(event_i, 1))
                            event_wait_until = int(op.add(now, ANIM_DELAY_MS))
                            continue
    
                        # Status SFX event: play a status sound immediately (no delay).
                        if kind == "sfx":
                            play_status_sfx(ev.get("key", ""))
                            event_i = int(op.add(event_i, 1))
                            event_wait_until = now
                            continue
    
                        # Log message event: start typewriter effect for this line.
                        if kind == "msg":
                            typing_text = str(ev.get("text", ""))
                            typing_len = 0
                            typing_active = True
                            typing_next_tick = int(op.add(now, TYPE_MS_PER_CHAR))
    
                            # Special case: empty string means blank line; push instantly.
                            if len(typing_text) == 0:
                                push_log("")
                                typing_active = False
                                event_i = int(op.add(event_i, 1))
                                event_wait_until = int(op.add(now, TYPE_EXTRA_PAUSE_MS))
                            continue
    
                        # HP event: update target hp, and let display hp ease toward it.
                        if kind == "hp":
                            who = ev.get("who")
                            new_hp = float(ev.get("new_hp", 0.0))
                            if who == "p1":
                                target_p1_hp = new_hp
                            elif who == "p2":
                                target_p2_hp = new_hp
    
                            event_i = int(op.add(event_i, 1))
                            event_wait_until = int(op.add(now, ANIM_DELAY_MS))
                            continue
    
                        # Unknown kind: just skip it with the default delay.
                        event_i = int(op.add(event_i, 1))
                        event_wait_until = int(op.add(now, ANIM_DELAY_MS))
    
                    # If we've consumed all events, check if HP bars have finished easing.
                    if event_i >= len(pending_events):
                        close1 = abs(op.sub(display_p1_hp, target_p1_hp)) < 0.5
                        close2 = abs(op.sub(display_p2_hp, target_p2_hp)) < 0.5
    
                        # Only return to choose mode when hp bars are settled and not typing.
                        if close1 and close2 and (not typing_active):
                            battle_mode = "choose"
    
                            # If env said done, compute who won and move to end state.
                            if turn_done_flag:
                                if human_side == "p1":
                                    you_hp = float(env.p1_hp)
                                    opp_hp = float(env.p2_hp)
                                else:
                                    you_hp = float(env.p2_hp)
                                    opp_hp = float(env.p1_hp)
    
                                if opp_hp <= 0.0 and you_hp > 0.0:
                                    push_log("YOU WON!!!!")
                                    end_result = "win"
                                    stop_battle_music()
                                    win_sfx.play()
                                elif you_hp <= 0.0 and opp_hp > 0.0:
                                    push_log("YOU LOST!!!!")
                                    end_result = "lose"
                                else:
                                    push_log("Battle ended")
                                    end_result = "end"
    
                                state = "end"
    
        # -----------------------------------------
        # 5) Input handling: keyboard, mouse, scroll, and quit
        # -----------------------------------------
        for event in pygame.event.get():
            # Closing the window ends the loop.
            if event.type == pygame.QUIT:
                running = False
                break
    
            # -----------------------------
            # Selection screen input
            # -----------------------------
            if state == "select":
                # Let the text inputs handle clicks and typing.
                left_search.handle_event(event)
                right_search.handle_event(event)
    
                # If the query changed, reset scroll so results start at the top.
                if left_search.text != left_last_q:
                    left_last_q = left_search.text
                    left_list.scroll = 0
                    left_list.dragging = False
    
                if right_search.text != right_last_q:
                    right_last_q = right_search.text
                    right_list.scroll = 0
                    right_list.dragging = False
    
                # Let scrollbars handle drag/click. If they consumed the event, skip the rest.
                if left_list.handle_event(event):
                    continue
                if right_list.handle_event(event):
                    continue
    
                # Mouse wheel scrolling over the correct list.
                if event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if left_box.collidepoint((mx, my)):
                        left_list.scroll_by(event.y)
                    if right_box.collidepoint((mx, my)):
                        right_list.scroll_by(event.y)
    
                # Mouse clicks for selecting pokemon and pressing buttons.
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
    
                    if event.button == 1:
                        # Click selection on left list.
                        for idx, nm in enumerate(shown_left):
                            row_rect = pygame.Rect(
                                left_box.x,
                                int(op.add(left_box.y, op.mul(idx, ROW_H))),
                                left_box.width,
                                ROW_H,
                            )
                            if row_rect.collidepoint((mx, my)):
                                left_selected = nm
                                break
    
                        # Click selection on right list.
                        for idx, nm in enumerate(shown_right):
                            row_rect = pygame.Rect(
                                right_box.x,
                                int(op.add(right_box.y, op.mul(idx, ROW_H))),
                                right_box.width,
                                ROW_H,
                            )
                            if row_rect.collidepoint((mx, my)):
                                right_selected = nm
                                break
    
                        # Prevent picking the same pokemon on both sides.
                        if (
                            left_selected is not None
                            and right_selected is not None
                            and left_selected == right_selected
                        ):
                            right_selected = None
    
                        # Start button only enabled when both picks exist.
                        start_btn.enabled = bool(left_selected is not None and right_selected is not None)
    
                        # Toggle opponent mode when mode button clicked.
                        if mode_btn.hit((mx, my)):
                            use_agent_mode = (not use_agent_mode)
                            mode_btn.text = "Opponent: Agent" if use_agent_mode else "Opponent: Random"
    
                        # Start battle when start button clicked.
                        if start_btn.hit((mx, my)):
                            begin_battle()
    
            # -----------------------------
            # Battle screen input
            # -----------------------------
            elif state == "battle":
                mx, my = pygame.mouse.get_pos()
    
                # Debug hotkey: SPACE forces a fire attack animation.
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        start_attack("p1", "fire")
    
                # Mouse wheel scrolls the log.
                if event.type == pygame.MOUSEWHEEL:
                    if log_scroll.rect.collidepoint((mx, my)):
                        log_scroll.scroll_by(event.y)
                        log_follow = log_at_bottom()
                        continue
    
                # Scrollbar drag in log.
                if log_scroll.handle_event(event):
                    log_follow = log_at_bottom()
                    continue
    
                # Clicks: exit or choose moves.
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx2, my2 = event.pos
    
                    # Exit fight button always works.
                    if exit_btn.hit((mx2, my2)):
                        exit_fight()
                        continue
    
                    # Only allow move selection when we are in choose mode.
                    if battle_mode == "choose":
                        for i in range(4):
                            if move_btns[i].hit((mx2, my2)):
                                start_turn(i)
                                break
    
            # -----------------------------
            # End screen input
            # -----------------------------
            elif state == "end":
                # Mouse wheel scrolls the log.
                if event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if log_scroll.rect.collidepoint((mx, my)):
                        log_scroll.scroll_by(event.y)
                        log_follow = log_at_bottom()
                        continue
    
                # Scrollbar drag.
                if log_scroll.handle_event(event):
                    log_follow = log_at_bottom()
                    continue
    
                # Mouse clicks: exit or play again.
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
    
                    if exit_btn.hit((mx, my)):
                        exit_fight()
                        continue
    
                    if again_btn.hit((mx, my)):
                        restart_same_battle()
                        continue
    
        # If the user closed the window, stop before drawing UI.
        if not running:
            break
    
        # -----------------------------------------
        # 6) Drawing UI: selection or battle/end
        # -----------------------------------------
        if state == "select":
            # Titles.
            draw_text(screen, "Pick your pokemon", font_big, 40, 30, colors["bg"])
            draw_text(screen, "Pick opponent pokemon", font_big, 600, 30, colors["bg"])
    
            # Search inputs.
            left_search.draw(screen, colors)
            right_search.draw(screen, colors)
    
            # List panel backgrounds.
            draw_rect(screen, left_box, colors["panel"])
            draw_rect(screen, left_box, colors["border"], 2)
            draw_rect(screen, right_box, colors["panel"])
            draw_rect(screen, right_box, colors["border"], 2)
    
            # Draw left list rows.
            for idx, nm in enumerate(shown_left):
                y = int(op.add(left_box.y, op.mul(idx, ROW_H)))
    
                # Use downloaded icon if ready, else placeholder.
                icon = _icon_surfaces.get(nm, placeholder_icon)
                screen.blit(icon, (int(op.add(left_box.x, 10)), int(op.add(y, 10))))
    
                txt = title_name(nm)
                col = colors["ok"] if nm == left_selected else colors["text"]
    
                # Text goes to the right of the icon.
                draw_text(
                    screen,
                    txt,
                    font_big,
                    int(op.add(left_box.x, op.add(10, op.add(ICON_PX, 16)))),
                    int(op.add(y, 20)),
                    col,
                )
    
            # Draw right list rows.
            for idx, nm in enumerate(shown_right):
                y = int(op.add(right_box.y, op.mul(idx, ROW_H)))
    
                icon = _icon_surfaces.get(nm, placeholder_icon)
                screen.blit(icon, (int(op.add(right_box.x, 10)), int(op.add(y, 10))))
    
                txt = title_name(nm)
                col = colors["warn"] if nm == right_selected else colors["text"]
    
                draw_text(
                    screen,
                    txt,
                    font_big,
                    int(op.add(right_box.x, op.add(10, op.add(ICON_PX, 16)))),
                    int(op.add(y, 20)),
                    col,
                )
    
            # Scrollbars on both lists.
            left_list.draw(screen, colors)
            right_list.draw(screen, colors)
    
            # Selected labels at bottom.
            if left_selected is not None:
                draw_text(screen, "You: " + title_name(left_selected), font_big, 40, 600, colors["bg"])
            else:
                draw_text(screen, "You: none", font_big, 40, 600, colors["bg"])
    
            if right_selected is not None:
                draw_text(screen, "Opponent: " + title_name(right_selected), font_big, 800, 600, colors["bg"])
            else:
                draw_text(screen, "Opponent: none", font_big, 800, 600, colors["bg"])
    
            # Buttons: mode toggle + start.
            start_btn.enabled = bool(left_selected is not None and right_selected is not None)
            mode_btn.draw(screen, colors)
            start_btn.draw(screen, colors)
    
        elif state == "battle" or state == "end":
            # Decide which env pokemon names appear on screen (human is always screen p1).
            if human_side == "p1":
                p1_label = title_name(env.p1["Name"])
                p2_label = title_name(env.p2["Name"])
            else:
                p1_label = title_name(env.p2["Name"])  # screen p1 is human
                p2_label = title_name(env.p1["Name"])
    
            # Header.
            draw_text(screen, "Battle", font_big, 40, 20, colors["text"])
            draw_text(
                screen,
                "Mode: " + ("Agent" if use_agent_mode else "Random"),
                font,
                40,
                50,
                colors["muted"],
            )
    
            # HP bars.
            draw_hp_bar(
                screen, 40, 80, 520, 28,
                display_p1_hp, max_p1_hp_screen,
                "You: " + p1_label, font, colors
            )
            draw_hp_bar(
                screen, 600, 80, 460, 28,
                display_p2_hp, max_p2_hp_screen,
                "Opponent: " + p2_label, font, colors
            )
    
            # Battle area panel.
            battle_area = pygame.Rect(40, 120, 520, 290)
            draw_rect(screen, battle_area, colors["battle_bg"])
            draw_rect(screen, battle_area, colors["border"], 2)
    
            # Base positions for sprites.
            opp_x = int(op.add(battle_area.x, 300))
            opp_y = int(op.add(battle_area.y, 10))
            you_x = int(op.add(battle_area.x, 10))
            you_y = int(op.add(battle_area.y, 80))
    
            draw_p2_x = opp_x
            draw_p2_y = opp_y
            draw_p1_x = you_x
            draw_p1_y = you_y
    
            # Intro animation overrides base positions with animated ones.
            if battle_mode == "intro":
                draw_p2_x = int(p2_anim_x)
                draw_p2_y = int(p2_anim_y)
                draw_p1_x = int(p1_anim_x)
                draw_p1_y = int(p1_anim_y)
    
            # Lunge: attacker shifts position briefly.
            if p1_attack_active:
                dx = lunge_offset(p1_attack_t)
                draw_p1_x = int(op.add(draw_p1_x, dx))
    
            if p2_attack_active:
                dx2 = lunge_offset(p2_attack_t)
                draw_p2_x = int(op.sub(draw_p2_x, dx2))
    
            # Visibility flags: once faint animation completes, sprite disappears.
            p1_visible = True
            p2_visible = True
            if p1_faint_active and p1_faint_t >= faint_duration:
                p1_visible = False
            if p2_faint_active and p2_faint_t >= faint_duration:
                p2_visible = False
    
            # Start from base sprites.
            p2_to_draw = p2_battle_sprite
            p1_to_draw = p1_battle_sprite
    
            # Flash effect: target flickers when being attacked.
            if p1_attack_active:
                temp2 = p2_battle_sprite.copy()
                temp2.set_alpha(target_flash_alpha(p1_attack_t))
                p2_to_draw = temp2
    
            if p2_attack_active:
                temp1 = p1_battle_sprite.copy()
                temp1.set_alpha(target_flash_alpha(p2_attack_t))
                p1_to_draw = temp1
    
            # Faint effect: fade out + drop down.
            if p1_faint_active and p1_visible:
                a = faint_alpha(p1_faint_t)
                dy = faint_drop(p1_faint_t)
                temp1f = p1_to_draw.copy()
                temp1f.set_alpha(a)
                p1_to_draw = temp1f
                draw_p1_y = int(op.add(draw_p1_y, dy))
    
            if p2_faint_active and p2_visible:
                a = faint_alpha(p2_faint_t)
                dy = faint_drop(p2_faint_t)
                temp2f = p2_to_draw.copy()
                temp2f.set_alpha(a)
                p2_to_draw = temp2f
                draw_p2_y = int(op.add(draw_p2_y, dy))
    
            # Dark move: shake the target sprite.
            if atk_fx_active and str(atk_fx_type).strip().lower() == "dark":
                sx, sy = dark_shake_offset(atk_fx_t, attack_duration)
                # atk_fx_who is attacker, so target is the other sprite.
                if atk_fx_who == "p1":
                    draw_p2_x = int(op.add(draw_p2_x, sx))
                    draw_p2_y = int(op.add(draw_p2_y, sy))
                else:
                    draw_p1_x = int(op.add(draw_p1_x, sx))
                    draw_p1_y = int(op.add(draw_p1_y, sy))
    
            # Ground move: shake the target sprite.
            if atk_fx_active and str(atk_fx_type).strip().lower() == "ground":
                sx, sy = ground_shake_offset(atk_fx_t, attack_duration)
                if atk_fx_who == "p1":
                    draw_p2_x = int(op.add(draw_p2_x, sx))
                    draw_p2_y = int(op.add(draw_p2_y, sy))
                else:
                    draw_p1_x = int(op.add(draw_p1_x, sx))
                    draw_p1_y = int(op.add(draw_p1_y, sy))
    
            # Blit sprites if they are still visible.
            if p2_visible:
                screen.blit(p2_to_draw, (draw_p2_x, draw_p2_y))
            if p1_visible:
                screen.blit(p1_to_draw, (draw_p1_x, draw_p1_y))
    
            # Attack FX drawing: compute centers and call the dispatcher.
            if atk_fx_active:
                p1_center = (
                    int(op.add(draw_p1_x, op.truediv(p1_to_draw.get_width(), 2))),
                    int(op.add(draw_p1_y, op.truediv(p1_to_draw.get_height(), 2))),
                )
                p2_center = (
                    int(op.add(draw_p2_x, op.truediv(p2_to_draw.get_width(), 2))),
                    int(op.add(draw_p2_y, op.truediv(p2_to_draw.get_height(), 2))),
                )
    
                # src = attacker center, tgt = target center.
                if atk_fx_who == "p1":
                    src = p1_center
                    tgt = p2_center
                else:
                    src = p2_center
                    tgt = p1_center
    
                # Target sprite is used to scale certain effects.
                tgt_sprite = p2_to_draw if atk_fx_who == "p1" else p1_to_draw
                tgt_size = int(max(tgt_sprite.get_width(), tgt_sprite.get_height()))
    
                draw_attack_fx(
                    screen,
                    tgt,
                    atk_fx_type,
                    atk_fx_t,
                    src_center=src,
                    target_size=tgt_size,
                )
    
            # -----------------------------
            # Log panel + scroll rendering
            # -----------------------------
            panel = pygame.Rect(600, 130, 480, 500)
            draw_rect(screen, panel, colors["panel"])
            draw_rect(screen, panel, colors["border"], 2)
    
            draw_text(screen, "Log", font_big, 612, 146, colors["text"])
    
            # Outer includes scrollbar track, inner is the clipping area for text.
            log_outer = pygame.Rect(612, 182, 456, 436)
            log_text = pygame.Rect(612, 182, 436, 436)
            log_scroll.rect = log_outer
    
            # How many lines fit vertically in the text area.
            visible_lines = max(1, int(op.truediv(log_text.height, LOG_LINE_H)))
    
            # Base render list is the fully wrapped log.
            temp_render = log_render
    
            # If typewriter is active and user is following bottom, show a preview of the current line.
            preview_allowed = typing_active and (log_follow or log_at_bottom(len(temp_render)))
            if preview_allowed:
                preview = typing_text[:typing_len]
                temp_render = list(log_render)
                temp_render.extend(wrap_text(font, preview, LOG_MAX_W))
    
            # Update scroll limits based on the temp list.
            log_scroll.set_counts(len(temp_render), visible_lines)
    
            # If following, force scroll to bottom.
            if log_follow:
                log_scroll.scroll = max(0, int(op.sub(len(temp_render), visible_lines)))
    
            # Clip drawing to log_text rectangle.
            old_clip = screen.get_clip()
            screen.set_clip(log_text)
    
            # Render only the visible slice of lines.
            start = int(log_scroll.scroll)
            end = int(op.add(start, visible_lines))
            shown = temp_render[start:end]
    
            for i, line in enumerate(shown):
                y = int(op.add(log_text.y, op.mul(i, LOG_LINE_H)))
                draw_text(screen, line, font, log_text.x, y, colors["text"])
    
            # Restore clipping so the rest of the UI draws normally.
            screen.set_clip(old_clip)
    
            # Draw scrollbar for the log.
            log_scroll.draw(screen, colors)
    
            # -----------------------------
            # Move buttons + bottom buttons
            # -----------------------------
            for b in move_btns:
                # Only clickable during live battle when choosing a move.
                b.enabled = bool(state == "battle" and battle_mode == "choose")
                b.draw(screen, colors)
    
            exit_btn.enabled = True
            again_btn.enabled = bool(state == "end")
    
            exit_btn.draw(screen, colors)
            if state == "end":
                again_btn.draw(screen, colors)
    
        # Present the final frame to the screen.
        pygame.display.flip()
    
    # Loop ended: close pygame cleanly.
    pygame.quit()


if __name__ == "__main__":
    main()
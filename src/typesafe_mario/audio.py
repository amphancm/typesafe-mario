from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

from .state import MarioSnapshot

SAMPLE_RATE = 44_100
_STATUS_RANK: dict[str, int] = {"small": 0, "tall": 1, "fireball": 2, "fire": 2}

# Standard A4 = 440 Hz semitone offsets from C4.
_NOTE_OFFSETS: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

NoteEvent = tuple[str | None, float]


def note_frequency(note: str | None) -> float:
    """Convert a note name like 'E5' or 'Bb4' to Hz (0.0 for rests)."""
    if note is None or note == "-":
        return 0.0
    pitch = note[:-1]
    octave = int(note[-1])
    semitone = _NOTE_OFFSETS[pitch] + (octave - 4) * 12
    return 440.0 * (2.0 ** ((semitone - 9) / 12.0))


def _apply_envelope(
    wave: np.ndarray,
    *,
    attack_s: float = 0.003,
    release_s: float = 0.012,
    decay_rate: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    n = wave.shape[0]
    if n == 0:
        return wave
    env = np.ones(n, dtype=np.float32)
    if decay_rate > 0.0:
        t = np.arange(n, dtype=np.float32) / float(sample_rate)
        env *= np.exp(-decay_rate * t)
    attack_n = min(n, max(1, int(attack_s * sample_rate)))
    release_n = min(n, max(1, int(release_s * sample_rate)))
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n, dtype=np.float32)
    env[-release_n:] *= np.linspace(1.0, 0.0, release_n, dtype=np.float32)
    return wave * env


def _pulse_wave(
    freq: float,
    duration_s: float,
    *,
    duty: float = 0.5,
    gain: float = 0.25,
    decay_rate: float = 2.5,
    gate: float = 0.88,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    n = max(1, round(duration_s * sample_rate))
    if freq <= 0.0:
        return np.zeros(n, dtype=np.float32)
    active = max(1, round(n * gate))
    t = np.arange(active, dtype=np.float32) / float(sample_rate)
    phase = (freq * t) % 1.0
    wave = np.where(phase < duty, 1.0, -1.0).astype(np.float32)
    wave = _apply_envelope(wave, decay_rate=decay_rate, sample_rate=sample_rate) * gain
    if active < n:
        out = np.zeros(n, dtype=np.float32)
        out[:active] = wave
        return out
    return wave


def _triangle_wave(
    freq: float,
    duration_s: float,
    *,
    gain: float = 0.30,
    gate: float = 0.94,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    n = max(1, round(duration_s * sample_rate))
    if freq <= 0.0:
        return np.zeros(n, dtype=np.float32)
    active = max(1, round(n * gate))
    t = np.arange(active, dtype=np.float32) / float(sample_rate)
    phase = (freq * t) % 1.0
    wave = (4.0 * np.abs(phase - 0.5) - 1.0).astype(np.float32)
    wave = _apply_envelope(
        wave, attack_s=0.002, release_s=0.006, decay_rate=0.0, sample_rate=sample_rate
    )
    wave *= gain
    if active < n:
        out = np.zeros(n, dtype=np.float32)
        out[:active] = wave
        return out
    return wave


def _pulse_sweep(
    start_freq: float,
    end_freq: float,
    duration_s: float,
    *,
    duty: float = 0.5,
    gain: float = 0.30,
    decay_rate: float = 4.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    n = max(1, round(duration_s * sample_rate))
    freqs = np.geomspace(max(20.0, start_freq), max(20.0, end_freq), n).astype(np.float64)
    phase = np.cumsum(freqs / float(sample_rate)) % 1.0
    wave = np.where(phase < duty, 1.0, -1.0).astype(np.float32)
    return _apply_envelope(wave, decay_rate=decay_rate, sample_rate=sample_rate) * gain


def _noise_burst(
    duration_s: float,
    *,
    gain: float = 0.22,
    decay_rate: float = 28.0,
    downsample: int = 8,
    seed: int = 7,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    n = max(1, round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    step = max(1, downsample)
    raw = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n + step - 1) // step)
    wave = np.repeat(raw, step)[:n]
    return _apply_envelope(
        wave, attack_s=0.001, release_s=0.005, decay_rate=decay_rate, sample_rate=sample_rate
    ) * gain


def _render_sequence(
    events: Sequence[NoteEvent],
    *,
    step_s: float,
    waveform: str = "pulse",
    duty: float = 0.25,
    gain: float = 0.22,
    decay_rate: float = 2.4,
    gate: float = 0.86,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    segments: list[np.ndarray] = []
    for note, beats in events:
        dur = beats * step_s
        freq = note_frequency(note)
        if waveform == "triangle":
            seg = _triangle_wave(freq, dur, gain=gain, gate=gate, sample_rate=sample_rate)
        else:
            seg = _pulse_wave(
                freq,
                dur,
                duty=duty,
                gain=gain,
                decay_rate=decay_rate,
                gate=gate,
                sample_rate=sample_rate,
            )
        segments.append(seg)
    if not segments:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(segments)


def _mix_tracks(*tracks: np.ndarray) -> np.ndarray:
    if not tracks:
        return np.zeros(1, dtype=np.float32)
    length = max(track.shape[0] for track in tracks)
    out = np.zeros(length, dtype=np.float32)
    for track in tracks:
        out[: track.shape[0]] += track
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.92:
        out *= 0.92 / peak
    return out


def _to_stereo_int16(mono: np.ndarray) -> np.ndarray:
    clipped = np.clip(mono, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return np.column_stack((pcm, pcm))


# ---------------------------------------------------------------------------
# Super Mario Bros. inspired 8-bit chiptune tracks
# ---------------------------------------------------------------------------

_OVERWORLD_INTRO_MEL: tuple[NoteEvent, ...] = (
    ("E5", 1), ("E5", 1), (None, 1), ("E5", 1), (None, 1), ("C5", 1), ("E5", 2),
    ("G5", 2), (None, 2), ("G4", 2), (None, 2),
)
_OVERWORLD_INTRO_BASS: tuple[NoteEvent, ...] = (
    ("D3", 1), ("D3", 1), (None, 1), ("D3", 1), (None, 1), ("D3", 1), ("D3", 2),
    ("G3", 2), (None, 2), ("G2", 2), (None, 2),
)

_OVERWORLD_A_MEL: tuple[NoteEvent, ...] = (
    ("C5", 2), (None, 1), ("G4", 2), (None, 1), ("E4", 2),
    (None, 1), ("A4", 2), ("B4", 2), ("Bb4", 1), ("A4", 2),
    ("G4", 4 / 3), ("E5", 4 / 3), ("G5", 4 / 3), ("A5", 2), ("F5", 1), ("G5", 1),
    (None, 1), ("E5", 2), ("C5", 1), ("D5", 1), ("B4", 2), (None, 1),
)
_OVERWORLD_A_HARM: tuple[NoteEvent, ...] = (
    ("E4", 2), (None, 1), ("C4", 2), (None, 1), ("G3", 2),
    (None, 1), ("C4", 2), ("D4", 2), ("Db4", 1), ("C4", 2),
    ("C4", 4 / 3), ("G4", 4 / 3), ("B4", 4 / 3), ("C5", 2), ("A4", 1), ("B4", 1),
    (None, 1), ("G4", 2), ("E4", 1), ("F4", 1), ("D4", 2), (None, 1),
)
_OVERWORLD_A_BASS: tuple[NoteEvent, ...] = (
    ("G3", 2), (None, 1), ("E3", 2), (None, 1), ("C3", 2),
    (None, 1), ("F3", 2), ("G3", 2), ("Gb3", 1), ("F3", 2),
    ("E3", 4 / 3), ("C4", 4 / 3), ("E4", 4 / 3), ("F4", 2), ("D4", 1), ("E4", 1),
    (None, 1), ("C4", 2), ("A3", 1), ("B3", 1), ("G3", 2), (None, 1),
)

_OVERWORLD_B_MEL: tuple[NoteEvent, ...] = (
    (None, 2), ("G5", 1), ("F#5", 1), ("F5", 1), ("D#5", 2), ("E5", 1),
    (None, 1), ("G#4", 1), ("A4", 1), ("C5", 1), (None, 1), ("A4", 1), ("C5", 1), ("D5", 1),
    (None, 2), ("G5", 1), ("F#5", 1), ("F5", 1), ("D#5", 2), ("E5", 1),
    (None, 1), ("C6", 2), ("C6", 1), ("C6", 2), (None, 2),
    (None, 2), ("G5", 1), ("F#5", 1), ("F5", 1), ("D#5", 2), ("E5", 1),
    (None, 1), ("G#4", 1), ("A4", 1), ("C5", 1), (None, 1), ("A4", 1), ("C5", 1), ("D5", 1),
    (None, 2), ("Eb5", 2), (None, 1), ("D5", 2), (None, 1),
    ("C5", 2), (None, 6),
)
_OVERWORLD_B_BASS: tuple[NoteEvent, ...] = (
    ("C3", 2), (None, 1), ("G3", 2), (None, 1), ("C4", 2),
    ("F3", 2), (None, 1), ("C4", 2), ("C4", 1), ("F3", 2),
    ("C3", 2), (None, 1), ("G3", 2), (None, 1), ("C4", 2),
    ("G3", 2), (None, 1), ("G3", 2), (None, 1), ("G3", 2),
    ("C3", 2), (None, 1), ("G3", 2), (None, 1), ("C4", 2),
    ("F3", 2), (None, 1), ("C4", 2), ("C4", 1), ("F3", 2),
    ("Ab3", 2), (None, 1), ("Bb3", 2), (None, 1), ("C4", 2),
    ("C3", 2), (None, 1), ("G2", 2), (None, 1), ("C3", 2),
)

_UNDERGROUND_MEL: tuple[NoteEvent, ...] = (
    ("C4", 1), ("C5", 1), ("A3", 1), ("A4", 1), ("Bb3", 1), ("Bb4", 1), (None, 6),
    ("C4", 1), ("C5", 1), ("A3", 1), ("A4", 1), ("Bb3", 1), ("Bb4", 1), (None, 6),
    ("F3", 1), ("F4", 1), ("D3", 1), ("D4", 1), ("Eb3", 1), ("Eb4", 1), (None, 6),
    ("F3", 1), ("F4", 1), ("D3", 1), ("D4", 1), ("Eb3", 1), ("Eb4", 1), (None, 4),
    ("Eb4", 2 / 3), ("D4", 2 / 3), ("C#4", 2 / 3),
    ("C4", 2), ("Eb4", 2), ("D4", 2), ("G#3", 2), ("G3", 2), ("C#4", 2),
    ("C4", 2 / 3), ("F#4", 2 / 3), ("F4", 2 / 3), ("E4", 2 / 3), ("Bb4", 2 / 3), ("A4", 2 / 3),
    ("Ab4", 1.5), ("Eb4", 1.5), ("B3", 1.5), ("Bb3", 1.5), ("A3", 1.5), ("Ab3", 1.5),
    (None, 5),
)

_CASTLE_MEL: tuple[NoteEvent, ...] = (
    ("D4", 0.5), ("G#4", 0.5), ("C#5", 0.5), ("G#4", 0.5),
    ("D4", 0.5), ("G#4", 0.5), ("C#5", 0.5), ("G#4", 0.5),
    ("D4", 0.5), ("G#4", 0.5), ("C#5", 0.5), ("G#4", 0.5),
    ("D4", 0.5), ("G#4", 0.5), ("C#5", 0.5), ("G#4", 0.5),
    ("C#4", 0.5), ("G4", 0.5), ("C5", 0.5), ("G4", 0.5),
    ("C#4", 0.5), ("G4", 0.5), ("C5", 0.5), ("G4", 0.5),
    ("C4", 0.5), ("F#4", 0.5), ("B4", 0.5), ("F#4", 0.5),
    ("C4", 0.5), ("F#4", 0.5), ("B4", 0.5), ("F#4", 0.5),
)
_CASTLE_BASS: tuple[NoteEvent, ...] = (
    ("D3", 4), ("D3", 4), ("C#3", 4), ("C3", 4),
)

_WATER_MEL: tuple[NoteEvent, ...] = (
    ("D4", 1), ("E4", 1), ("F#4", 1), ("G4", 1), ("A4", 1), ("A#4", 1),
    ("B4", 1), ("B4", 1), ("B4", 1), ("B4", 2), ("G4", 1),
    ("E5", 3), ("D#5", 3), ("E5", 3), (None, 1), ("G4", 1), ("A4", 1),
    ("B4", 1), ("C5", 1), ("D5", 1), ("E5", 3), ("D#5", 2), ("F5", 1), ("E5", 3),
    (None, 3),
)
_WATER_BASS: tuple[NoteEvent, ...] = (
    ("G3", 3), ("D3", 3), ("G3", 3), ("G3", 3),
    ("C3", 3), ("G3", 3), ("C3", 3), ("G3", 3),
    ("G3", 3), ("C3", 3), ("B2", 3), ("C3", 3), ("G3", 3),
)


def synthesize_bgm(theme: str, *, fast: bool = False, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize a seamless looping 8-bit BGM track as stereo int16 PCM."""
    tempo_scale = 0.76 if fast else 1.0
    if theme == "underground":
        step = 0.145 * tempo_scale
        mel = _render_sequence(
            _UNDERGROUND_MEL, step_s=step, duty=0.25, gain=0.22, decay_rate=3.0, sample_rate=sample_rate
        )
        bass_notes = tuple((n[:-1] + str(max(2, int(n[-1]) - 1)) if n else None, b) for n, b in _UNDERGROUND_MEL)
        bass = _render_sequence(
            bass_notes, step_s=step, waveform="triangle", gain=0.24, sample_rate=sample_rate
        )
        return _to_stereo_int16(_mix_tracks(mel, bass))

    if theme == "castle":
        step = 0.16 * tempo_scale
        mel = _render_sequence(
            _CASTLE_MEL, step_s=step, duty=0.125, gain=0.20, decay_rate=5.0, gate=0.80, sample_rate=sample_rate
        )
        bass = _render_sequence(
            _CASTLE_BASS, step_s=step, waveform="triangle", gain=0.26, sample_rate=sample_rate
        )
        return _to_stereo_int16(_mix_tracks(mel, bass))

    if theme == "water":
        step = 0.22 * tempo_scale
        mel = _render_sequence(
            _WATER_MEL, step_s=step, duty=0.5, gain=0.20, decay_rate=1.8, gate=0.92, sample_rate=sample_rate
        )
        bass = _render_sequence(
            _WATER_BASS, step_s=step, waveform="triangle", gain=0.24, sample_rate=sample_rate
        )
        return _to_stereo_int16(_mix_tracks(mel, bass))

    # Default: iconic Overworld theme (Intro + A + A + B).
    step = 0.148 * tempo_scale
    mel_seq = _OVERWORLD_INTRO_MEL + _OVERWORLD_A_MEL + _OVERWORLD_A_MEL + _OVERWORLD_B_MEL
    harm_seq = (
        tuple((None, b) for _, b in _OVERWORLD_INTRO_MEL)
        + _OVERWORLD_A_HARM
        + _OVERWORLD_A_HARM
        + tuple((None, b) for _, b in _OVERWORLD_B_MEL)
    )
    bass_seq = _OVERWORLD_INTRO_BASS + _OVERWORLD_A_BASS + _OVERWORLD_A_BASS + _OVERWORLD_B_BASS

    mel = _render_sequence(
        mel_seq, step_s=step, duty=0.25, gain=0.23, decay_rate=2.4, gate=0.84, sample_rate=sample_rate
    )
    harm = _render_sequence(
        harm_seq, step_s=step, duty=0.5, gain=0.11, decay_rate=2.8, gate=0.80, sample_rate=sample_rate
    )
    bass = _render_sequence(
        bass_seq, step_s=step, waveform="triangle", gain=0.25, gate=0.90, sample_rate=sample_rate
    )
    return _to_stereo_int16(_mix_tracks(mel, harm, bass))


def synthesize_sfx(name: str, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize an iconic 8-bit Mario sound effect as stereo int16 PCM."""
    if name == "jump_small":
        return _to_stereo_int16(
            _pulse_sweep(210.0, 620.0, 0.14, duty=0.25, gain=0.30, decay_rate=5.0, sample_rate=sample_rate)
        )
    if name == "jump_big":
        return _to_stereo_int16(
            _pulse_sweep(135.0, 410.0, 0.16, duty=0.5, gain=0.32, decay_rate=4.5, sample_rate=sample_rate)
        )
    if name == "coin":
        part1 = _pulse_wave(note_frequency("B5"), 0.072, duty=0.5, gain=0.32, decay_rate=2.0, gate=1.0, sample_rate=sample_rate)
        part2 = _pulse_wave(note_frequency("E6"), 0.28, duty=0.5, gain=0.30, decay_rate=6.5, gate=1.0, sample_rate=sample_rate)
        return _to_stereo_int16(np.concatenate((part1, part2)))
    if name == "stomp":
        pop = _pulse_sweep(520.0, 140.0, 0.095, duty=0.25, gain=0.34, decay_rate=10.0, sample_rate=sample_rate)
        hit = _noise_burst(0.055, gain=0.24, decay_rate=35.0, downsample=10, seed=11, sample_rate=sample_rate)
        return _to_stereo_int16(_mix_tracks(pop, hit))
    if name == "bump":
        thud = _pulse_sweep(165.0, 75.0, 0.085, duty=0.5, gain=0.30, decay_rate=14.0, sample_rate=sample_rate)
        return _to_stereo_int16(thud)
    if name == "powerup":
        notes: tuple[NoteEvent, ...] = (
            ("C5", 1), ("G4", 1), ("C5", 1), ("E5", 1), ("G5", 1), ("C6", 1), ("G5", 1),
            ("Ab4", 1), ("C5", 1), ("Eb5", 1), ("Ab5", 1), ("C6", 1), ("Eb6", 1), ("Ab5", 1),
            ("Bb4", 1), ("D5", 1), ("F5", 1), ("Bb5", 1), ("D6", 1), ("F6", 2),
        )
        return _to_stereo_int16(
            _render_sequence(notes, step_s=0.042, duty=0.25, gain=0.28, decay_rate=2.0, gate=0.95, sample_rate=sample_rate)
        )
    if name == "powerdown":
        seg = _pulse_sweep(480.0, 140.0, 0.14, duty=0.5, gain=0.30, decay_rate=4.0, sample_rate=sample_rate)
        gap = np.zeros(int(0.03 * sample_rate), dtype=np.float32)
        return _to_stereo_int16(np.concatenate((seg, gap, seg * 0.9, gap, seg * 0.8)))
    if name == "one_up":
        notes = (("E6", 1), ("G6", 1), ("E7", 1), ("C7", 1), ("D7", 1), ("G7", 1.5))
        return _to_stereo_int16(
            _render_sequence(notes, step_s=0.095, duty=0.5, gain=0.30, decay_rate=3.0, gate=0.90, sample_rate=sample_rate)
        )
    if name == "hurry_up":
        notes = (
            ("F#4", 1), ("B4", 1), ("F#5", 1), (None, 0.5),
            ("G4", 1), ("C5", 1), ("G5", 1), (None, 0.5),
            ("G#4", 1), ("C#5", 1), ("G#5", 1.5),
        )
        return _to_stereo_int16(
            _render_sequence(notes, step_s=0.065, duty=0.25, gain=0.28, decay_rate=3.5, gate=0.88, sample_rate=sample_rate)
        )
    if name == "stage_clear":
        slide = _pulse_sweep(980.0, 220.0, 0.38, duty=0.25, gain=0.24, decay_rate=2.5, sample_rate=sample_rate)
        fanfare_notes: tuple[NoteEvent, ...] = (
            ("G4", 2 / 3), ("C5", 2 / 3), ("E5", 2 / 3), ("G5", 2 / 3), ("C6", 2 / 3), ("E6", 2 / 3),
            ("G6", 2), ("E6", 2),
            ("Ab4", 2 / 3), ("C5", 2 / 3), ("Eb5", 2 / 3), ("Ab5", 2 / 3), ("C6", 2 / 3), ("Eb6", 2 / 3),
            ("Ab6", 2), ("Eb6", 2),
            ("Bb4", 2 / 3), ("D5", 2 / 3), ("F5", 2 / 3), ("Bb5", 2 / 3), ("D6", 2 / 3), ("F6", 2 / 3),
            ("Bb6", 2), ("Bb6", 2 / 3), ("Bb6", 2 / 3), ("Bb6", 2 / 3), ("C7", 4.5),
        )
        fanfare = _render_sequence(
            fanfare_notes, step_s=0.14, duty=0.25, gain=0.28, decay_rate=2.0, gate=0.88, sample_rate=sample_rate
        )
        return _to_stereo_int16(np.concatenate((slide, fanfare)))
    if name == "death":
        hit = _pulse_sweep(340.0, 180.0, 0.08, duty=0.5, gain=0.30, decay_rate=8.0, sample_rate=sample_rate)
        pause = np.zeros(int(0.18 * sample_rate), dtype=np.float32)
        notes = (
            ("B4", 1), ("F5", 1), (None, 1), ("F5", 1),
            ("F5", 4 / 3), ("E5", 4 / 3), ("D5", 4 / 3),
            ("C5", 1), ("E4", 1), (None, 1), ("E4", 1), ("C4", 2.5),
        )
        jingle = _render_sequence(
            notes, step_s=0.135, duty=0.25, gain=0.28, decay_rate=2.4, gate=0.86, sample_rate=sample_rate
        )
        return _to_stereo_int16(np.concatenate((hit, pause, jingle)))
    raise ValueError(f"Unknown SFX name: {name!r}")


def infer_music_theme(snapshot: MarioSnapshot, *, area_type: int | None = None) -> str:
    """Resolve the BGM theme from NES RAM area_type (0x074E) or world/stage fallback."""
    if area_type is not None:
        if area_type == 0:
            return "water"
        if area_type == 2:
            return "underground"
        if area_type == 3:
            return "castle"
        return "overworld"
    if snapshot.stage == 4:
        return "castle"
    if snapshot.stage == 2 and snapshot.world in {1, 4}:
        return "underground"
    return "overworld"


def detect_audio_events(
    prev: MarioSnapshot | None,
    curr: MarioSnapshot,
    *,
    finishing: bool = False,
) -> list[str]:
    """Return ordered SFX event names triggered by the transition from prev -> curr."""
    if prev is None:
        return []

    events: list[str] = []
    if curr.dead and not prev.dead:
        return ["death"]
    if curr.clear and not prev.clear:
        return ["stage_clear"]

    if curr.coins > prev.coins:
        events.append("coin")
    if curr.lives > prev.lives:
        events.append("one_up")

    prev_rank = _STATUS_RANK.get(prev.status, 0)
    curr_rank = _STATUS_RANK.get(curr.status, 0)
    if curr_rank > prev_rank:
        events.append("powerup")
    elif curr_rank < prev_rank and not curr.dead:
        events.append("powerdown")

    # Jump takeoff: grounded -> airborne with upward velocity.
    if prev.grounded and not curr.grounded and curr.dy > 1:
        events.append("jump_big" if curr_rank >= 1 else "jump_small")

    # Enemy stomp / defeat: score jumps by >= 100 without coin pickup, powerup, or flagpole tally.
    score_delta = curr.score - prev.score
    if (
        score_delta >= 100
        and curr.coins == prev.coins
        and curr_rank == prev_rank
        and not curr.clear
        and not finishing
    ):
        events.append("stomp")
    elif (
        not prev.grounded
        and not curr.grounded
        and prev.dy > 3
        and curr.dy <= 0
        and curr.coins == prev.coins
        and score_delta == 0
    ):
        events.append("bump")

    if prev.time_left >= 100 and 0 < curr.time_left < 100 and not curr.clear and not curr.dead:
        events.append("hurry_up")

    return events


@dataclass
class AudioConfig:
    enabled: bool = True
    volume: float = 0.5
    bgm_volume_ratio: float = 0.55
    sfx_volume_ratio: float = 0.95


class MarioAudioEngine:
    """Real-time 8-bit chiptune BGM and SFX engine driven by Mario game state."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()
        if not 0.0 <= self.config.volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")

        self._muted: bool = not self.config.enabled
        self._available: bool = False
        self._mixer: Any = None
        self._bgm_channel: Any = None
        self._sfx_sounds: dict[str, Any] = {}
        self._bgm_sounds: dict[tuple[str, bool], Any] = {}
        self._current_bgm_key: tuple[str, bool] | None = None
        self._bgm_stopped_for_event: bool = False
        self._hurry_active: bool = False
        self._prev_snapshot: MarioSnapshot | None = None
        self._bump_cooldown: int = 0

        if self.config.enabled:
            self._init_mixer()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def sound_on(self) -> bool:
        return self._available and not self._muted

    def _init_mixer(self) -> None:
        try:
            import pygame
        except ImportError:
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
            init_info = pygame.mixer.get_init()
            if not init_info:
                return
            sample_rate = int(init_info[0])
            pygame.mixer.set_num_channels(8)
            self._mixer = pygame.mixer
            self._bgm_channel = pygame.mixer.Channel(0)
            for sfx_name in (
                "jump_small",
                "jump_big",
                "coin",
                "stomp",
                "bump",
                "powerup",
                "powerdown",
                "one_up",
                "hurry_up",
                "stage_clear",
                "death",
            ):
                pcm = synthesize_sfx(sfx_name, sample_rate=sample_rate)
                self._sfx_sounds[sfx_name] = pygame.sndarray.make_sound(pcm)
            self._available = True
            self._apply_volumes()
        except (pygame.error, RuntimeError, OSError, ValueError):
            # Graceful fallback in headless/CI environments without an audio device.
            self._available = False
            self._mixer = None
            self._bgm_channel = None

    def _get_bgm_sound(self, theme: str, *, fast: bool) -> Any | None:
        if not self._available or self._mixer is None:
            return None
        key = (theme, fast)
        if key not in self._bgm_sounds:
            try:
                import pygame

                init_info = self._mixer.get_init()
                sample_rate = int(init_info[0]) if init_info else SAMPLE_RATE
                pcm = synthesize_bgm(theme, fast=fast, sample_rate=sample_rate)
                self._bgm_sounds[key] = pygame.sndarray.make_sound(pcm)
            except (ImportError, pygame.error, RuntimeError, OSError, ValueError):
                return None
        return self._bgm_sounds[key]

    def _apply_volumes(self) -> None:
        if not self._available or self._bgm_channel is None:
            return
        if self._muted:
            self._bgm_channel.set_volume(0.0)
            for sound in self._sfx_sounds.values():
                sound.set_volume(0.0)
            return
        self._bgm_channel.set_volume(self.config.volume * self.config.bgm_volume_ratio)
        sfx_vol = self.config.volume * self.config.sfx_volume_ratio
        for sound in self._sfx_sounds.values():
            sound.set_volume(sfx_vol)

    def set_muted(self, muted: bool) -> bool:
        self._muted = bool(muted)
        self._apply_volumes()
        return self._muted

    def toggle_mute(self) -> bool:
        return self.set_muted(not self._muted)

    def _play_bgm(self, theme: str, *, fast: bool = False) -> None:
        if not self._available or self._bgm_channel is None:
            return
        key = (theme, fast)
        if self._current_bgm_key == key and self._bgm_channel.get_busy():
            return
        sound = self._get_bgm_sound(theme, fast=fast)
        if sound is None:
            return
        self._current_bgm_key = key
        self._bgm_channel.play(sound, loops=-1)
        self._apply_volumes()

    def _stop_bgm(self) -> None:
        if self._available and self._bgm_channel is not None:
            self._bgm_channel.stop()
        self._current_bgm_key = None

    def _play_sfx(self, name: str) -> None:
        if not self._available or self._muted:
            return
        sound = self._sfx_sounds.get(name)
        if sound is None:
            return
        sound.set_volume(self.config.volume * self.config.sfx_volume_ratio)
        # Reserve channel 0 for BGM; play SFX on channels 1..7.
        if self._mixer is not None:
            for ch_idx in range(1, self._mixer.get_num_channels()):
                channel = self._mixer.Channel(ch_idx)
                if not channel.get_busy():
                    channel.play(sound)
                    return
            self._mixer.Channel(1).play(sound)

    def reset(self) -> None:
        """Reset episode audio state (e.g. on game start or restart)."""
        self._stop_bgm()
        if self._available and self._mixer is not None:
            for ch_idx in range(1, self._mixer.get_num_channels()):
                self._mixer.Channel(ch_idx).stop()
        self._bgm_stopped_for_event = False
        self._hurry_active = False
        self._prev_snapshot = None
        self._bump_cooldown = 0

    def update(
        self,
        snapshot: MarioSnapshot | None,
        *,
        area_type: int | None = None,
        finishing: bool = False,
    ) -> list[str]:
        """Advance audio state from the latest snapshot and return triggered event names."""
        if not isinstance(snapshot, MarioSnapshot):
            return []

        if self._bump_cooldown > 0:
            self._bump_cooldown -= 1

        events = detect_audio_events(self._prev_snapshot, snapshot, finishing=finishing)
        filtered: list[str] = []
        for event in events:
            if event == "bump":
                if self._bump_cooldown > 0:
                    continue
                self._bump_cooldown = 6
            filtered.append(event)

        if "death" in filtered or "stage_clear" in filtered:
            self._bgm_stopped_for_event = True
            self._stop_bgm()

        if "hurry_up" in filtered:
            self._hurry_active = True

        for event in filtered:
            self._play_sfx(event)

        if not self._bgm_stopped_for_event and not snapshot.dead and not snapshot.clear:
            theme = infer_music_theme(snapshot, area_type=area_type)
            self._play_bgm(theme, fast=self._hurry_active)

        self._prev_snapshot = snapshot
        return filtered

    def close(self) -> None:
        self.reset()
        if self._available and self._mixer is not None:
            with suppress(RuntimeError, OSError):
                self._mixer.quit()
        self._available = False
        self._mixer = None
        self._bgm_channel = None
        self._sfx_sounds.clear()
        self._bgm_sounds.clear()

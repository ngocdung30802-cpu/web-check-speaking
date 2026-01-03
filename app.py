# app.py
import os
import re
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from difflib import SequenceMatcher

import streamlit as st
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import detect_silence
import azure.cognitiveservices.speech as speechsdk

try:
    import pandas as pd
except Exception:
    pd = None


# =========================
# INIT
# =========================
load_dotenv()
st.set_page_config(page_title="Speaking Checker — Level 1", layout="wide")
st.title("Speaking Checker — Level 1")
st.caption("Tab 1: Chấm theo vocab/phrase (Table A) | Tab 2: Chấm đọc đoạn văn (Reading Passage)")


# =========================
# BASIC CHECKS
# =========================
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# =========================
# TOKEN HELPERS
# =========================
def _norm_token(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch == "'":
            out.append(ch)
    return "".join(out)


def _item_tokens(item: str) -> List[str]:
    return [_norm_token(x) for x in (item or "").split() if _norm_token(x)]


def _split_to_tokens(text: str) -> List[str]:
    raw = (text or "").replace("\n", " ").strip()
    toks = []
    for p in raw.split():
        t = _norm_token(p)
        if t:
            toks.append(t)
    return toks


def guess_student_name(filename: str) -> str:
    base = Path(filename).stem
    base = base.replace("_", " ").strip()
    return base


# =========================
# AUDIO HELPERS
# =========================
def to_wav_16k_mono(src_path: str, out_path: str) -> Tuple[int, int]:
    """
    Convert audio -> 16kHz mono WAV (PCM 16-bit).
    Requires ffmpeg for mp3/m4a/webm/ogg.
    Returns (duration_ms, 16000)
    """
    audio = AudioSegment.from_file(src_path)
    duration_ms = len(audio)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    h = audio.export(out_path, format="wav")
    try:
        h.close()
    except Exception:
        pass
    return duration_ms, 16000


def analyze_audio_quality(
    wav_path: str,
    min_duration_s: float = 5.0,
    min_dbfs: float = -35.0,
    max_silence_ratio: float = 0.65,
    min_silence_len_ms: int = 500,
    silence_rel_db: float = 16.0,
    long_pause_ms: int = 1000,
) -> Tuple[Dict[str, float], List[str]]:
    """
    QC gate:
      - too short
      - too quiet (dBFS too low)
      - too much silence ratio
    """
    audio = AudioSegment.from_file(wav_path)
    dur_ms = max(1, len(audio))
    duration_s = dur_ms / 1000.0

    dbfs = audio.dBFS
    if dbfs == float("-inf"):
        dbfs = -100.0

    rms = float(audio.rms or 0)

    silence_thresh = max(dbfs - float(silence_rel_db), -60.0)
    sil = detect_silence(
        audio,
        min_silence_len=int(min_silence_len_ms),
        silence_thresh=float(silence_thresh),
    )

    silence_ms = 0
    long_pause_count = 0
    for s, e in sil:
        seg = max(0, int(e) - int(s))
        silence_ms += seg
        if seg >= int(long_pause_ms):
            long_pause_count += 1

    silence_ratio = silence_ms / float(dur_ms)
    silence_total_s = silence_ms / 1000.0

    issues = []
    if duration_s < float(min_duration_s):
        issues.append(f"Audio quá ngắn ({duration_s:.1f}s < {min_duration_s:.1f}s)")
    if dbfs < float(min_dbfs):
        issues.append(f"Nói nhỏ/volume thấp ({dbfs:.1f} dBFS < {min_dbfs:.1f} dBFS)")
    if silence_ratio > float(max_silence_ratio):
        issues.append(f"Ngắt nghỉ quá nhiều (silence {silence_ratio:.0%} > {max_silence_ratio:.0%})")

    metrics = {
        "duration_s": float(duration_s),
        "dbfs": float(dbfs),
        "rms": float(rms),
        "silence_ratio": float(silence_ratio),
        "silence_total_s": float(silence_total_s),
        "silence_thresh_dbfs": float(silence_thresh),
        "long_pause_count": float(long_pause_count),
    }
    return metrics, issues


# =========================
# AZURE HELPERS
# =========================
def _safe_get_jsonresult_from_properties(props, prop_id) -> Optional[str]:
    raw_json = None
    if hasattr(props, "get_property"):
        try:
            raw_json = props.get_property(prop_id)
        except Exception:
            raw_json = None
    if not raw_json and hasattr(props, "get"):
        try:
            raw_json = props.get(prop_id)
        except Exception:
            raw_json = None
    if not raw_json and hasattr(props, "items"):
        try:
            for k, v in props.items():
                if "jsonresult" in str(k).lower():
                    raw_json = v
                    break
        except Exception:
            pass
    return raw_json


def _set_timeout_safe(
    speech_config: speechsdk.SpeechConfig,
    prop_id_attr: str,
    prop_name_fallback: str,
    value_ms: int,
) -> None:
    v = str(int(value_ms))
    pid = getattr(speechsdk.PropertyId, prop_id_attr, None)
    if pid is not None:
        try:
            speech_config.set_property(pid, v)
            return
        except Exception:
            pass
    if hasattr(speech_config, "set_property_by_name"):
        try:
            speech_config.set_property_by_name(prop_name_fallback, v)
            return
        except Exception:
            pass


def _enable_prosody_best_effort(pa_config, enable: bool) -> None:
    if not enable:
        return
    try:
        pa_config.enable_prosody_assessment()  # Method chuẩn theo docs
        return
    except Exception:
        # Nếu version SDK cũ quá (<1.35), sẽ raise → có thể log warning
        pass


def _is_end_of_stream_cancel(reason_str: str, details_str: str) -> bool:
    rs = (reason_str or "").lower()
    ds = (details_str or "").lower()
    return ("endofstream" in rs) or ("endofstream" in ds)


def run_pron_assessment_continuous(
    wav_path: str,
    reference_text: str,
    locale: str,
    speech_key: str,
    speech_region: str,
    end_silence_timeout_ms: int,
    seg_silence_timeout_ms: int,
    initial_silence_timeout_ms: int,
    max_wait_seconds: int = 120,
    enable_prosody: bool = True,
) -> List[dict]:
    """
    Continuous recognition over file (Pronunciation Assessment).
    """
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = locale

    _set_timeout_safe(
        speech_config,
        "SpeechServiceConnection_EndSilenceTimeoutMs",
        "SpeechServiceConnection_EndSilenceTimeoutMs",
        end_silence_timeout_ms,
    )
    _set_timeout_safe(
        speech_config,
        "SpeechServiceConnection_InitialSilenceTimeoutMs",
        "SpeechServiceConnection_InitialSilenceTimeoutMs",
        initial_silence_timeout_ms,
    )
    _set_timeout_safe(
        speech_config,
        "SpeechServiceConnection_SegmentationSilenceTimeoutMs",
        "SpeechServiceConnection_SegmentationSilenceTimeoutMs",
        seg_silence_timeout_ms,
    )

    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    pa_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=False,  # Không hỗ trợ trong continuous mode → tắt luôn
    )

    # Chỉ bật prosody nếu locale là en-US và user bật
    if enable_prosody and locale == "en-US":
        try:
            pa_config.enable_prosody_assessment()
        except Exception:
            pass  # SDK cũ sẽ bỏ qua

    pa_config.apply_to(recognizer)

    results_json: List[dict] = []
    done = {"flag": False}
    cancel_info = {"reason": None, "details": None, "is_fatal": False}

    def on_recognized(evt):
        r = evt.result
        if r.reason == speechsdk.ResultReason.RecognizedSpeech:
            raw = _safe_get_jsonresult_from_properties(
                r.properties, speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if raw:
                try:
                    results_json.append(json.loads(raw))
                except Exception:
                    pass

    def on_canceled(evt):
        reason_str, details_str = "", ""
        try:
            details = speechsdk.CancellationDetails.from_result(evt.result)
            reason_str = str(details.reason)
            details_str = str(details.error_details or "")
        except Exception:
            try:
                cd = evt.result.cancellation_details
                reason_str = str(cd.reason)
                details_str = str(cd.error_details or "")
            except Exception:
                reason_str, details_str = "Canceled", ""

        cancel_info["reason"] = reason_str
        cancel_info["details"] = details_str

        if _is_end_of_stream_cancel(reason_str, details_str):
            cancel_info["is_fatal"] = False
            done["flag"] = True
            return

        cancel_info["is_fatal"] = True
        done["flag"] = True

    def on_session_stopped(evt):
        done["flag"] = True

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_session_stopped)

    started = False
    try:
        recognizer.start_continuous_recognition()
        started = True
        start_t = time.time()
        while not done["flag"]:
            time.sleep(0.1)
            if (time.time() - start_t) > max_wait_seconds:
                cancel_info["reason"] = "Timeout"
                cancel_info["details"] = "max_wait_seconds exceeded"
                cancel_info["is_fatal"] = False
                done["flag"] = True
    finally:
        if started:
            try:
                recognizer.stop_continuous_recognition()
            except Exception:
                pass

    if cancel_info["is_fatal"] and not results_json:
        raise RuntimeError(f"Azure canceled: {cancel_info['reason']} | {cancel_info['details']}")

    return results_json

def call_with_retry(fn, max_retries: int = 3, base_sleep: float = 0.9):
    last_err = None
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            last_err = e
            if "endofstream" in msg:
                return []
            if (
                ("1007" in msg)
                or ("connection was closed" in msg)
                or ("remote host" in msg)
                or ("validate speech context" in msg)
                or ("wsarecv" in msg)
            ):
                time.sleep(base_sleep * (1.6 ** i))
                continue
            raise
    raise last_err


# =========================
# JSON PARSE HELPERS
# =========================
def _tick_to_ms(v) -> Optional[float]:
    try:
        if v is None:
            return None
        vv = float(v)
        if vv <= 0:
            return 0.0
        return vv / 10000.0
    except Exception:
        return None


def extract_word_rows(result_json: dict) -> List[Dict]:
    """
    Flatten Azure JSON to rows per recognized word.
    Includes phonemes_detail and word timestamps (best-effort).
    """
    rows: List[Dict] = []
    nbest = result_json.get("NBest", [])
    if not nbest:
        return rows

    prosody = None
    try:
        pa0 = (nbest[0].get("PronunciationAssessment", {}) or {})
        prosody = pa0.get("ProsodyScore", None)
    except Exception:
        prosody = None

    words = nbest[0].get("Words", []) or []
    for w in words:
        word = w.get("Word", "") or ""
        pa = w.get("PronunciationAssessment", {}) or {}
        accuracy = pa.get("AccuracyScore", None)
        error_type = (pa.get("ErrorType", "None") or "None").strip()

        offset_ms = _tick_to_ms(w.get("Offset"))
        dur_ms = _tick_to_ms(w.get("Duration"))
        end_ms = None
        if isinstance(offset_ms, (int, float)) and isinstance(dur_ms, (int, float)):
            end_ms = float(offset_ms) + float(dur_ms)

        phonemes_detail = []
        for ph in (w.get("Phonemes", []) or []):
            ph_pa = (ph.get("PronunciationAssessment", {}) or {})
            ph_acc = ph_pa.get("AccuracyScore", None)
            ph_txt = ph.get("Phoneme") or ph.get("PhonemeText") or ""
            phonemes_detail.append({"ph": ph_txt, "acc": ph_acc})

        rows.append(
            {
                "word": word,
                "tok": _norm_token(word),
                "accuracy": accuracy,
                "error_type": error_type,
                "phonemes_detail": phonemes_detail,
                "prosody": prosody,
                "offset_ms": offset_ms,
                "dur_ms": dur_ms,
                "end_ms": end_ms,
            }
        )
    return rows


# =========================
# ALIGNMENT (monotonic)
# =========================
def _token_match(expected: str, heard: str) -> bool:
    if expected == heard:
        return True
    # plural tolerance
    if expected.endswith("s") and expected[:-1] == heard:
        return True
    if heard.endswith("s") and heard[:-1] == expected:
        return True
    variants = {"realisation": "realization", "realization": "realisation"}
    if variants.get(expected) == heard:
        return True
    return False


def map_expected_to_recognized(expected_tokens: List[str], rec_tokens: List[str], lookahead: int = 4) -> List[Optional[int]]:
    j = 0
    mapping: List[Optional[int]] = []
    n = len(rec_tokens)
    for et in expected_tokens:
        found = None
        for jj in range(j, min(n, j + lookahead + 1)):
            if _token_match(et, rec_tokens[jj]):
                found = jj
                break
        mapping.append(found)
        if found is not None:
            j = found + 1
    return mapping


# =========================
# DETECTORS
# =========================
BAD_TYPES = {"Mispronunciation", "Omission", "Insertion"}
S_PHONEMES = {"s", "z", "S", "Z"}
# Phoneme sets (best-effort). We only flag "missing final sound" if tail phoneme is consonant.
VOWELISH_PHONEMES = {
    "aa","ae","ah","ao","aw","ax","ay",
    "eh","el","em","en","er","ey",
    "ih","iy",
    "ow","oy",
    "uh","uw",
    "y","w",
}

CONSONANTISH_PHONEMES = {
    "b","d","f","g","hh","jh","k","l","m","n","ng","p","r","s","t","th","v","w","y","z",
    "ch","sh","zh","dh",
    "c","q","x",
}

def detect_missing_plural_s(expected_tok: str, heard_tok: str) -> bool:
    return expected_tok.endswith("s") and expected_tok[:-1] == heard_tok


def detect_missing_s_sound(row: Dict, s_acc_threshold: float) -> Tuple[bool, bool, Optional[str]]:
    """
    Returns (has_issue, is_final, phoneme_text)
    """
    phs = row.get("phonemes_detail") or []
    if not phs:
        return False, False, None

    exp_seq = [(p.get("ph") or "").strip() for p in phs]
    acc_seq = [p.get("acc", None) for p in phs]

    if exp_seq:
        last_ph = exp_seq[-1]
        last_acc = acc_seq[-1]
        if last_ph in S_PHONEMES and isinstance(last_acc, (int, float)) and last_acc < s_acc_threshold:
            return True, True, last_ph

    for ph, acc in zip(exp_seq, acc_seq):
        if ph in S_PHONEMES and isinstance(acc, (int, float)) and acc < s_acc_threshold:
            return True, False, ph

    return False, False, None


def detect_missing_final_sound(row: Dict, final_acc_threshold: float) -> Optional[str]:
    """
    Flag 'missing final sound' ONLY if the weak tail phoneme is consonant-ish.
    Avoid vowel tails like /iy/ /ay/ /ow/ etc.
    Return tail phoneme label (best effort) or None.
    """
    phs = row.get("phonemes_detail") or []
    if not phs:
        return None

    tail = phs[-2:] if len(phs) >= 2 else phs[-1:]

    for p in tail[::-1]:
        ph = (p.get("ph") or "").strip()
        acc = p.get("acc", None)
        if not ph or not isinstance(acc, (int, float)):
            continue

        ph_norm = ph.lower()
        ph_norm = re.sub(r"[^a-z]", "", ph_norm)  # ay1 -> ay, iy0 -> iy, sh2 -> sh

        # bỏ qua vowel tails
        if ph_norm in VOWELISH_PHONEMES:
            continue

        is_consonantish = (
            ph_norm in CONSONANTISH_PHONEMES
            or ph_norm in {"sh", "ch", "th", "ng", "jh", "zh", "dh"}
            or (len(ph_norm) <= 3 and ph_norm.isalpha() and ph_norm not in VOWELISH_PHONEMES)
        )

        if is_consonantish and float(acc) < float(final_acc_threshold):
            return ph

    return None

# =========================
# SCORING (word-based)
# =========================
def compute_word_based_score(
    expected_tokens: List[str],
    rec_rows: List[Dict],
    lookahead: int,
    missing_token_score: float = 0.0,
) -> Dict[str, object]:
    rec_tokens = [r.get("tok") or _norm_token(r.get("word", "")) for r in rec_rows]
    mapping = map_expected_to_recognized(expected_tokens, rec_tokens, lookahead=lookahead)

    total = len(expected_tokens)
    if total == 0:
        return {"score_pct": None, "missing": 0, "total": 0, "avg_found": None, "mapping": mapping}

    scores = []
    found_scores = []
    missing = 0

    for et, mi in zip(expected_tokens, mapping):
        if mi is None:
            scores.append(float(missing_token_score))
            missing += 1
        else:
            acc = rec_rows[mi].get("accuracy", None)
            s = float(acc) if isinstance(acc, (int, float)) else 0.0
            scores.append(s)
            found_scores.append(s)

    score_pct = sum(scores) / float(total)
    avg_found = (sum(found_scores) / len(found_scores)) if found_scores else None
    return {
        "score_pct": float(score_pct),
        "missing": int(missing),
        "total": int(total),
        "avg_found": float(avg_found) if avg_found is not None else None,
        "mapping": mapping,
    }


# =========================
# TAB 1 HELPERS (Table A)
# =========================
def fmt_entry(phrase: str, word: str, extra: str = "") -> str:
    extra = (extra or "").strip()
    if extra and not extra.startswith("("):
        extra = f"({extra})"
    if phrase.strip().lower() == word.strip().lower():
        return f"{word}{extra}"
    return f"{phrase}: {word}{extra}"


def uniq_keep_order(items: List[str], limit: int = 80) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) >= limit:
            break
    return out


SEVERITY = {
    "omission": 6,
    "mispron": 5,
    "missing_final": 4,
    "missing_s": 3,
    "missing_plural_s": 2,
    "low_acc": 1,
    "prosody": 0,
}


def build_student_error_buckets_and_issues(
    vocab_list: List[str],
    all_rows: List[Dict],
    lookahead: int,
    accuracy_threshold: float,
    s_acc_threshold: float,
    final_acc_threshold: float,
    enable_prosody: bool,
    prosody_threshold: float,
) -> Tuple[Dict[str, List[str]], List[Dict]]:
    rec_rows = [r for r in all_rows if (r.get("word") or "").strip()]
    rec_tokens = [_norm_token(r["word"]) for r in rec_rows]

    expected_tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for item in vocab_list:
        toks = _item_tokens(item)
        start = cursor
        expected_tokens.extend(toks)
        cursor += len(toks)
        spans.append((start, cursor))

    mapping = map_expected_to_recognized(expected_tokens, rec_tokens, lookahead=lookahead)

    buckets = {
        "mispron_low": [],
        "missing_final": [],
        "missing_s": [],
        "missing_plural_s": [],
        "prosody": [],
        "omission": [],
    }
    issues_struct: List[Dict] = []

    def add_issue(cat: str, phrase: str, heard_word: str, extra: str, label: str):
        issues_struct.append(
            {
                "cat": cat,
                "severity": SEVERITY.get(cat, 0),
                "phrase": phrase,
                "word": heard_word,
                "extra": extra,
                "label": label,
                "entry": fmt_entry(phrase, heard_word, extra),
            }
        )

    for item_idx, item in enumerate(vocab_list):
        start, end = spans[item_idx]
        toks = expected_tokens[start:end]
        mapped = mapping[start:end]

        if not toks:
            continue

        if any(x is None for x in mapped):
            buckets["omission"].append(item)
            add_issue("omission", item, item, "", "(thiếu từ)")
            continue

        for expected_tok, rec_i in zip(toks, mapped):
            row = rec_rows[rec_i]
            heard_word = (row.get("word") or expected_tok).strip()
            heard_tok = _norm_token(heard_word)

            et = (row.get("error_type") or "None").strip()
            acc = row.get("accuracy", None)

            if et in BAD_TYPES:
                buckets["mispron_low"].append(fmt_entry(item, heard_word, ""))
                add_issue("mispron", item, heard_word, "", "(phát âm sai)")
            elif isinstance(acc, (int, float)) and acc < accuracy_threshold:
                buckets["mispron_low"].append(fmt_entry(item, heard_word, ""))
                add_issue("low_acc", item, heard_word, "", "(chưa chuẩn)")

            if detect_missing_plural_s(expected_tok, heard_tok):
                buckets["missing_plural_s"].append(fmt_entry(item, heard_word, ""))
                add_issue("missing_plural_s", item, heard_word, "", "(thiếu -s số nhiều)")

            has_s, _, s_ph = detect_missing_s_sound(row, s_acc_threshold=s_acc_threshold)
            if has_s:
                ph_txt = (s_ph or "").lower()
                extra = f"/{ph_txt}/" if ph_txt else ""
                buckets["missing_s"].append(fmt_entry(item, heard_word, extra))
                if ph_txt == "s":
                    add_issue("missing_s", item, heard_word, "/s/", "(thiếu /s/)")
                elif ph_txt == "z":
                    add_issue("missing_s", item, heard_word, "/z/", "(thiếu /z/)")
                else:
                    add_issue("missing_s", item, heard_word, extra, "(thiếu âm)")

            final_ph = detect_missing_final_sound(row, final_acc_threshold=final_acc_threshold)
            if final_ph:
                extra = f"/{final_ph}/"
                buckets["missing_final"].append(fmt_entry(item, heard_word, extra))
                add_issue("missing_final", item, heard_word, extra,"")

            if enable_prosody:
                p = row.get("prosody", None)
                if isinstance(p, (int, float)) and p < prosody_threshold:
                    buckets["prosody"].append(fmt_entry(item, heard_word, ""))
                    add_issue("prosody", item, heard_word, "", "(prosody)")

    for k in list(buckets.keys()):
        buckets[k] = uniq_keep_order(buckets[k], limit=80)

    return buckets, issues_struct


def summarize_worst_per_word(issues_struct: List[Dict], limit: int = 60) -> List[str]:
    best: Dict[str, Dict] = {}
    for it in issues_struct:
        phrase = (it.get("phrase") or "").strip()
        word = (it.get("word") or "").strip()
        key = f"{phrase}|||{word}"
        if key not in best:
            best[key] = it
        else:
            if int(it.get("severity", 0)) > int(best[key].get("severity", 0)):
                best[key] = it

    items = list(best.values())
    items.sort(key=lambda x: (-int(x.get("severity", 0)), x.get("entry", "")))

    out = []
    for it in items:
        entry = (it.get("entry") or "").strip()
        label = (it.get("label") or "").strip()
        if entry and label:
            out.append(f"{entry} {label}".strip())
        elif entry:
            out.append(entry)
        if len(out) >= int(limit):
            break

    return out


# =========================
# TAB 2 HELPERS (Passage)
# =========================
def split_passage_sentences(passage_text: str) -> List[str]:
    txt = (passage_text or "").strip()
    if not txt:
        return []
    txt = re.sub(r"\s+", " ", txt)
    sents = re.split(r"(?<=[.!?])\s+", txt)
    out = []
    for s in sents:
        s = (s or "").strip()
        if s:
            out.append(s)
    return out


def build_sentence_spans(passage_text: str) -> Tuple[List[str], List[Dict]]:
    sents = split_passage_sentences(passage_text)
    spans = []
    expected_tokens = []
    cursor = 0
    for s in sents:
        toks = _split_to_tokens(s)
        if not toks:
            continue
        start = cursor
        expected_tokens.extend(toks)
        cursor += len(toks)
        spans.append({"text": s, "tokens": toks, "start": start, "end": cursor})
    return expected_tokens, spans


def gap_ms_between(rec_rows: List[Dict], i: int, j: int) -> Optional[float]:
    try:
        a = rec_rows[i]
        b = rec_rows[j]
        end_a = a.get("end_ms")
        start_b = b.get("offset_ms")
        if isinstance(end_a, (int, float)) and isinstance(start_b, (int, float)):
            g = float(start_b) - float(end_a)
            return max(0.0, g)
    except Exception:
        pass
    return None


def detect_sentence_issues(
    sentence_spans: List[Dict],
    expected_tokens: List[str],
    rec_rows: List[Dict],
    mapping: List[Optional[int]],
    sentence_acc_threshold: float,
    pause_inside_sentence_ms: int,
    min_tokens_for_sentence_check: int,
    max_notes: int = 5,
) -> List[str]:
    notes = []

    for sp in sentence_spans:
        start, end = sp["start"], sp["end"]
        toks = expected_tokens[start:end]
        mapped = mapping[start:end]
        if len(toks) < int(min_tokens_for_sentence_check):
            continue

        missing = sum(1 for x in mapped if x is None)
        found_indices = [x for x in mapped if isinstance(x, int)]
        found_indices_sorted = sorted(found_indices)

        accs = []
        for mi in found_indices_sorted:
            acc = rec_rows[mi].get("accuracy")
            if isinstance(acc, (int, float)):
                accs.append(float(acc))
        avg_acc = (sum(accs) / len(accs)) if accs else None

        long_pause_count = 0
        if len(found_indices_sorted) >= 2:
            for a, b in zip(found_indices_sorted[:-1], found_indices_sorted[1:]):
                g = gap_ms_between(rec_rows, a, b)
                if isinstance(g, (int, float)) and g >= float(pause_inside_sentence_ms):
                    long_pause_count += 1

        flag_pause = long_pause_count > 0
        flag_missing = missing > 0
        flag_acc = isinstance(avg_acc, (int, float)) and avg_acc < float(sentence_acc_threshold)

        if flag_pause or flag_missing or flag_acc:
            reasons = []
            if flag_missing:
                reasons.append(f"thiếu {missing} từ")
            if flag_pause:
                reasons.append(f"ngắt nghỉ trong câu {long_pause_count} lần")
            if flag_acc:
                reasons.append("độ chuẩn cả câu thấp")

            sent_preview = sp["text"]
            if len(sent_preview) > 95:
                sent_preview = sent_preview[:92] + "..."
            notes.append(f'“{sent_preview}” — ' + ", ".join(reasons))

        if len(notes) >= int(max_notes):
            break

    return notes


def _similarity(a: str, b: str) -> float:
    try:
        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def _label_from_word_issue(
    expected_tok: str,
    heard_tok: str,
    row: Optional[Dict],
    accuracy_threshold: float,
    s_acc_threshold: float,
    final_acc_threshold: float,
) -> str:
    if row is None:
        return "(thiếu từ)"

    if detect_missing_plural_s(expected_tok, heard_tok):
        return "(thiếu -s số nhiều)"

    has_s, _, s_ph = detect_missing_s_sound(row, s_acc_threshold=float(s_acc_threshold))
    if has_s and s_ph:
        s_ph_norm = s_ph.lower()
        if s_ph_norm == "s":
            return "(thiếu /s/)"
        if s_ph_norm == "z":
            return "(thiếu /z/)"
        return f"(thiếu /{s_ph_norm}/)"

    final_ph = detect_missing_final_sound(row, final_acc_threshold=float(final_acc_threshold))
    if final_ph:
        return f"(/{final_ph}/)"

    etype = (row.get("error_type") or "None").strip()
    acc = row.get("accuracy", None)

    if etype in BAD_TYPES:
        return "(phát âm sai)"

    if isinstance(acc, (int, float)) and float(acc) < float(accuracy_threshold):
        return "(chưa chuẩn)"

    return ""


def build_passage_word_issues_concise(
    expected_tokens: List[str],
    rec_rows: List[Dict],
    mapping: List[Optional[int]],
    accuracy_threshold: float,
    s_acc_threshold: float,
    final_acc_threshold: float,
    ignore_stopwords: bool,
    lookahead: int,
    max_items: int = 30,
    suggest_min_ratio: float = 0.82,
) -> List[str]:
    """
    Concise notable issues for passage:
    - Avoid duplicate noise
    - IMPORTANT FIX: If a token appears anywhere as a recognized pronunciation issue,
      do NOT also list it as "(thiếu từ)" later due to mapping drift / pause.
    - IMPORTANT FIX: 'missing final sound' relies on detect_missing_final_sound which now ignores vowel tails.
    """
    stopwords = {
        "a","an","the","to","of","in","on","at","for","with","and","or","but",
        "is","am","are","was","were","be","been","being","do","does","did",
        "i","you","he","she","it","we","they","me","him","her","us","them",
        "my","your","his","her","our","their","this","that","these","those",
        "as","from","by","not","so","if","then","than","too","very"
    }

    rec_tokens = [r.get("tok") or _norm_token(r.get("word", "")) for r in rec_rows]

    # --- PASS 1: collect tokens that are actually recognized with any pronunciation issue
    # This helps us suppress fake "(thiếu từ)" for tokens that were read (but mapping drifted).
    tokens_with_pron_issue = set()

    for r in rec_rows:
        tok = r.get("tok") or _norm_token(r.get("word", ""))
        if not tok:
            continue

        etype = (r.get("error_type") or "None").strip()
        acc = r.get("accuracy", None)

        # Basic pron/low accuracy
        if etype in BAD_TYPES:
            tokens_with_pron_issue.add(tok)
            continue
        if isinstance(acc, (int, float)) and float(acc) < float(accuracy_threshold):
            tokens_with_pron_issue.add(tok)
            continue

        # Missing s/z phoneme
        has_s, _, _ = detect_missing_s_sound(r, s_acc_threshold=float(s_acc_threshold))
        if has_s:
            tokens_with_pron_issue.add(tok)
            continue

        # Missing final sound (now consonant-only)
        final_ph = detect_missing_final_sound(r, final_acc_threshold=float(final_acc_threshold))
        if final_ph:
            tokens_with_pron_issue.add(tok)
            continue

    out: List[str] = []
    seen = set()

    jptr = 0
    n = len(rec_tokens)

    for et, mi in zip(expected_tokens, mapping):
        # stopwords filter
        if ignore_stopwords and et in stopwords:
            if isinstance(mi, int):
                jptr = max(jptr, mi + 1)
            continue

        # CASE A: missing mapping
        if mi is None:
            # FIX: suppress missing if we already have this token as a real pron issue somewhere
            # (e.g., 'cream' has (nuốt âm cuối /m/) so don't also show 'cream (thiếu từ)')
            if et in tokens_with_pron_issue:
                continue

            key = ("missing", et)
            if key not in seen:
                out.append(f"{et} (thiếu từ)")
                seen.add(key)

        # CASE B: mapped to recognized word
        else:
            row = rec_rows[mi]
            heard = (row.get("tok") or _norm_token(row.get("word", "")))
            label = _label_from_word_issue(
                expected_tok=et,
                heard_tok=heard,
                row=row,
                accuracy_threshold=float(accuracy_threshold),
                s_acc_threshold=float(s_acc_threshold),
                final_acc_threshold=float(final_acc_threshold),
            )

            # Determine if we should include this token as a notable issue
            etype = (row.get("error_type") or "None").strip()
            acc = row.get("accuracy", None)

            is_issue = False
            if label:
                is_issue = True
            else:
                if etype in BAD_TYPES:
                    is_issue = True
                if isinstance(acc, (int, float)) and float(acc) < float(accuracy_threshold):
                    is_issue = True

            if is_issue:
                # Keep only one entry per expected token to avoid spam
                key = ("bad", et)
                if key not in seen:
                    out.append(f"{et} {label}".strip())
                    seen.add(key)

            jptr = max(jptr, mi + 1)

        if len(out) >= int(max_items):
            break

    return out

# =========================
# SNAPSHOT SAVE (NEW)
# =========================
def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


def save_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data or b"")


def append_log_json(session_dir: str, payload: Dict[str, Any]) -> None:
    os.makedirs(session_dir, exist_ok=True)
    log_path = os.path.join(session_dir, "logs.json")
    try:
        old = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                old = json.load(f) or []
        if not isinstance(old, list):
            old = []
        old.append(payload)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(old, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# =========================
# SIDEBAR (shared config)
# =========================
st.sidebar.header("Azure + QC (dùng chung cho 2 tab)")

# ---- Class Session (Lớp -> Buổi học) ----
st.sidebar.divider()
st.sidebar.header("Class Session (Lớp → Buổi học)")

def _slugify(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^0-9A-Za-z _\-\(\)\.]", "", s)
    s = s.strip().replace(" ", "_")
    return s or "untitled"


SESSIONS_ROOT = os.getenv("SESSIONS_ROOT", "sessions")


def _list_dirs(path: str) -> List[str]:
    try:
        if not os.path.isdir(path):
            return []
        items = []
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isdir(p):
                items.append(name)
        items.sort()
        return items
    except Exception:
        return []


existing_classes = _list_dirs(SESSIONS_ROOT)

class_mode = st.sidebar.radio("Class mode", ["Chọn có sẵn", "Tạo mới"], horizontal=True, key="class_mode")

if class_mode == "Chọn có sẵn" and existing_classes:
    chosen_class_raw = st.sidebar.selectbox("Chọn lớp", existing_classes, index=0, key="class_pick")
else:
    chosen_class_raw = st.sidebar.text_input("Nhập tên lớp (vd: G7)", value="", placeholder="G7", key="class_new")

chosen_class = _slugify(chosen_class_raw)
class_path = os.path.join(SESSIONS_ROOT, chosen_class)
existing_sessions = _list_dirs(class_path)

session_mode = st.sidebar.radio("Session mode", ["Chọn có sẵn", "Tạo mới"], horizontal=True, key="sess_mode")

if session_mode == "Chọn có sẵn" and existing_sessions:
    chosen_session_raw = st.sidebar.selectbox("Chọn buổi học", existing_sessions, index=len(existing_sessions) - 1, key="sess_pick")
else:
    chosen_session_raw = st.sidebar.text_input("Nhập buổi học (vd: Week_3)", value="", placeholder="Week_3", key="sess_new")

chosen_session = _slugify(chosen_session_raw)
SESSION_DIR = os.path.join(SESSIONS_ROOT, chosen_class, chosen_session)

st.sidebar.caption(f"📁 Session folder: `{SESSION_DIR}`")

if st.sidebar.button("📌 Tạo / Load session", use_container_width=True, key="btn_make_session"):
    os.makedirs(SESSION_DIR, exist_ok=True)
    st.sidebar.success("Session ready ✅")

# Auto ensure folder exists when user already typed class/session (avoid save errors)
if chosen_class and chosen_session:
    try:
        os.makedirs(SESSION_DIR, exist_ok=True)
    except Exception:
        pass

# ---- Azure key/region ----
speech_key_ui = st.sidebar.text_input("SPEECH_KEY", type="password", value="")
speech_region_ui = st.sidebar.text_input("SPEECH_REGION", value=os.getenv("SPEECH_REGION", "southeastasia"))

SPEECH_KEY = (speech_key_ui.strip() or os.getenv("SPEECH_KEY", "").strip())
SPEECH_REGION = (speech_region_ui.strip() or os.getenv("SPEECH_REGION", "").strip())

st.sidebar.divider()
st.sidebar.subheader("🎧 Kiểm tra chất lượng audio")
st.sidebar.caption("Loại audio quá ngắn, quá nhỏ hoặc ngắt nghỉ quá nhiều (khuyến nghị bật).")

enable_audio_qc = st.sidebar.checkbox("Bật kiểm tra audio", value=True)

min_duration_s = st.sidebar.slider(
    "Thời lượng tối thiểu (giây)",
    1.0, 25.0, 6.0, step=0.5,
    disabled=not enable_audio_qc,
    help="Nếu audio ngắn hơn mức này → sẽ không chấm và yêu cầu thu lại."
)

min_dbfs = st.sidebar.slider(
    "Âm lượng tối thiểu",
    -60.0, -10.0, -35.0, step=1.0,
    disabled=not enable_audio_qc,
    help="Nếu học sinh nói quá nhỏ → hệ thống dễ nhận sai. Dưới ngưỡng sẽ bị chặn."
)

max_silence_ratio = st.sidebar.slider(
    "Tỷ lệ im lặng tối đa",
    0.10, 0.95, 0.65, step=0.05,
    disabled=not enable_audio_qc,
    help="Nếu tỷ lệ im lặng quá cao (ngắt nghỉ quá nhiều) → chặn để tránh kết quả sai lệch."
)

min_silence_len_ms = st.sidebar.slider(
    "Khoảng im lặng tối thiểu (ms)",
    200, 1500, 500, step=50,
    disabled=not enable_audio_qc,
    help="Đoạn im lặng ngắn hơn mức này sẽ không tính là 'im lặng'."
)

silence_rel_db = st.sidebar.slider(
    "Độ nhạy phát hiện im lặng",
    8.0, 25.0, 16.0, step=1.0,
    disabled=not enable_audio_qc,
    help="Số càng nhỏ → bắt im lặng 'nhạy' hơn (dễ tính là im lặng). Số càng lớn → ít nhạy hơn."
)

long_pause_ms = st.sidebar.slider(
    "Ngắt nghỉ dài (ms)",
    600, 2000, 1000, step=100,
    disabled=not enable_audio_qc,
    help="Đếm số lần học sinh ngừng quá lâu. Dùng để tham khảo khi đánh giá tốc độ/nhịp đọc."
)

st.sidebar.divider()
st.sidebar.subheader("Timeouts (Azure)")
auto_tune_pause = st.sidebar.checkbox("Tự chỉnh timeout theo độ dài", value=True)
pause_profile = st.sidebar.selectbox("Mức ngắt nghỉ", ["Nhẹ", "Vừa", "Nhiều"], index=1, disabled=not auto_tune_pause)
end_silence_timeout_ms = st.sidebar.slider("EndSilenceTimeoutMs", 1000, 20000, 16000, step=500)
seg_silence_timeout_ms = st.sidebar.slider("SegmentationSilenceTimeoutMs", 300, 5000, 2800, step=100)
initial_silence_timeout_ms = st.sidebar.slider("InitialSilenceTimeoutMs", 2000, 30000, 15000, step=500)
locale = st.sidebar.selectbox("Locale", ["en-US", "en-GB", "en-AU"], index=0)

if locale != "en-US":
    st.sidebar.caption("⚠️ Prosody assessment chỉ hỗ trợ en-US (American English). Tự động tắt.")
    enable_prosody = False
else:
    enable_prosody = st.sidebar.checkbox("Enable Prosody", value=True)

# ---- History download ----
st.sidebar.divider()
st.sidebar.subheader("History (tải snapshot)")


def _dl_button_if_exists(label: str, file_path: str, mime: str):
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                st.sidebar.download_button(
                    label,
                    f.read(),
                    file_name=os.path.basename(file_path),
                    mime=mime,
                    use_container_width=True,
                )
        except Exception:
            pass


if chosen_class and chosen_session and os.path.isdir(SESSION_DIR):
    _dl_button_if_exists("⬇️ vocab_tableA_summary.csv", os.path.join(SESSION_DIR, "vocab_tableA_summary.csv"), "text/csv")
    _dl_button_if_exists("⬇️ vocab_summary.txt", os.path.join(SESSION_DIR, "vocab_summary.txt"), "text/plain")
    _dl_button_if_exists("⬇️ passage_level1_summary.csv", os.path.join(SESSION_DIR, "passage_level1_summary.csv"), "text/csv")
    _dl_button_if_exists("⬇️ passage_summary.txt", os.path.join(SESSION_DIR, "passage_summary.txt"), "text/plain")
    _dl_button_if_exists("⬇️ logs.json", os.path.join(SESSION_DIR, "logs.json"), "application/json")
else:
    st.sidebar.caption("Chọn lớp + buổi học để hiện file đã lưu.")

# === MỚI: Tách riêng aggregator cho từng tab ===
if "top_issues_vocab" not in st.session_state:
    st.session_state["top_issues_vocab"] = {}       # Dành riêng cho Tab 1 (Vocab/Phrase)

if "top_issues_passage" not in st.session_state:
    st.session_state["top_issues_passage"] = {}     # Dành riêng cho Tab 2 (Reading Passage)

# =========================
# CLASS-WIDE TOP 10 ISSUES (Tab1 + Tab2)
# =========================
def _extract_issue_key(issue_text: str) -> Optional[str]:
    """
    Chỉ trích xuất từ gốc (word) để gom nhóm top từ bị sai nhiều nhất.
    Bỏ hoàn toàn loại lỗi và prosody.
    Ví dụ:
    - "look (nuốt âm cuối /k/)" → "look"
    - "rice (thiếu /s/)" → "rice"
    - "fish (phát âm sai)" → "fish"
    - "im (phát âm sai)" → "im"
    """
    s = (issue_text or "").strip()
    
    # Bỏ các dòng QC fail hoặc nhiễu
    if s.startswith("⚠️") or s.startswith("❌"):
        return None
    s_low = s.lower()
    if "convert error" in s_low or "qc error" in s_low or ("audio" in s_low and "qc" in s_low):
        return None
    if "prosody" in s_low:
        return None  # Bỏ prosody ra khỏi top
    if not s:
        return None

    # Lấy từ gốc: phần đầu tiên trước khoảng trắng hoặc ngoặc
    # Ví dụ: "look (nuốt âm cuối /k/)" → "look"
    #       "im (phát âm sai)" → "im"
    match = re.match(r"^(\S+)", s)
    if match:
        word = match.group(1).strip()
        # Chuẩn hóa về lowercase để gom chính xác (look và Look là một)
        return word.lower()
    
        return None
    
def _agg_add(agg: Dict[str, set], student: str, issues: List[str]) -> None:
    """
    agg: dict key -> set(students)
    Each key counted at most once per student.
    """
    st_name = (student or "").strip()
    if not st_name:
        return

    for it in issues or []:
        k = _extract_issue_key(it)
        if not k:
            continue
        if k not in agg:
            agg[k] = set()
        agg[k].add(st_name)

def _render_top10_issues(agg: Dict[str, set], title: str = "10 từ bị sai phổ biến nhất trong lớp", top_n: int = 10) -> None:
    if not agg:
        st.info("Chưa có dữ liệu để tổng hợp (hãy chấm ít nhất 1 loạt bài).")
        return

    # Chuyển sang list và sort theo số học sinh giảm dần, rồi alphabet
    items = [(k.capitalize(), sorted(list(v))) for k, v in agg.items() if v]
    items.sort(key=lambda x: (-len(x[1]), x[0].lower()))

    st.subheader(title)
    for rank, (word, students) in enumerate(items[:top_n], start=1):
        st.markdown(f"**{rank}. {word}** — {', '.join(students)}")

# =========================
# TABS
# =========================
tab_vocab, tab_passage = st.tabs(["1) Chấm vocab/phrase (Table A)", "2) Chấm đọc đoạn văn (Passage Level 1)"])

# =============================================================================
# TAB 1 — VOCAB / TABLE A
# =============================================================================
with tab_vocab:
    st.subheader("Tab 1 — Chấm theo danh sách từ/cụm từ (Table A + Summary)")

    if not has_ffmpeg():
        st.warning("Server chưa có ffmpeg. Upload mp3/m4a/webm/ogg có thể convert fail. Khuyến nghị upload .wav hoặc cài ffmpeg.")

    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.markdown("### 1) Danh sách từ / cụm từ")
        vocab_text = st.text_area(
            "Mỗi dòng 1 item (word hoặc phrase)",
            height=220,
            placeholder="confident\ndevelop country\nset off\ntake something for granted",
            key="vocab_text_tab1",
        )

        st.markdown("### 2) Upload audio (nhiều file để chấm cả lớp)")
        uploaded_files = st.file_uploader(
            "Tên file nên là tên học sinh (vd: AnhNguyet.m4a)",
            type=["mp3", "m4a", "wav", "webm", "ogg"],
            accept_multiple_files=True,
            key="uploader_tab1",
        )

        run_btn = st.button("✅ CHẤM VOCAB (Batch)", type="primary", use_container_width=True, key="run_tab1")

    with colB:
        st.markdown("### Ngưỡng chấm (Tab 1)")

    strictness_t1 = st.slider(
        "Strictness (Dễ → Khắt khe)",
        0, 100, 55,
        key="t1_strictness",
        help="Kéo sang phải = chấm khắt khe hơn (dễ bắt lỗi hơn)."
    )

    def _lerp(a, b, t):
        return a + (b - a) * t

    t1 = strictness_t1 / 100.0
    t1_default_acc_thr   = int(round(_lerp(85, 95, t1)))
    t1_default_s_thr     = int(round(_lerp(80, 92, t1)))
    t1_default_final_thr = int(round(_lerp(80, 92, t1)))
    t1_default_pros_thr  = int(round(_lerp(60, 80, t1)))
    t1_default_lookahead = int(round(_lerp(6, 3, t1)))

    with st.expander("Advanced (tuỳ chỉnh chi tiết)", expanded=False):
        st.caption("Các giá trị ở đây sẽ override Strictness.")
        t1_acc_thr_adv = st.slider("Low accuracy (Accuracy <)", 0, 100, t1_default_acc_thr, key="t1_acc_thr_adv")
        t1_s_thr_adv = st.slider("Bắt thiếu /s/~/z/ (ngưỡng)", 0, 100, t1_default_s_thr, key="t1_s_thr_adv")
        t1_final_thr_adv = st.slider("Bắt nuốt âm cuối (ngưỡng)", 0, 100, t1_default_final_thr, key="t1_final_thr_adv")
        t1_pros_thr_adv = st.slider("Ngưỡng Prosody warning", 0, 100, t1_default_pros_thr, key="t1_pros_thr_adv")
        t1_lookahead_adv = st.slider("Lookahead (chịu token nhiễu)", 1, 8, t1_default_lookahead, key="t1_lookahead_adv")

    # Final values used by Tab 1
    accuracy_threshold = t1_acc_thr_adv
    s_acc_threshold = t1_s_thr_adv
    final_acc_threshold = t1_final_thr_adv
    prosody_threshold = t1_pros_thr_adv
    lookahead = t1_lookahead_adv

    st.divider()
    st.markdown("### Score (%)")
    missing_token_score = st.slider("Điểm cho từ bị thiếu", 0.0, 50.0, 0.0, step=1.0, key="t1_missing_score")

    st.divider()
    st.markdown("### Worst-per-word summary")
    worst_limit = st.slider("Giới hạn số lỗi trong summary", 10, 120, 60, step=5, key="t1_worst_limit")

    st.divider()
    st.markdown("### Ghi chú")
    st.caption(
            "Table A là để phân tích chi tiết theo loại lỗi. "
            "Summary bên dưới sẽ tự gom và chỉ giữ lỗi nặng nhất cho mỗi từ/phrase (tránh lặp)."
        )

    st.markdown(
        """
#### Quy ước Table A
- **Phát âm sai/điểm thấp**: Azure báo Mispronunciation/Omission/Insertion hoặc Accuracy dưới ngưỡng.
- **Nuốt âm cuối**: suy luận từ phoneme tail yếu (best-effort).
- **Thiếu/ yếu /s/~/z/**: bắt theo phoneme /s/ hoặc /z/ yếu (best-effort).
- **Thiếu -s số nhiều**: expected có 's' nhưng Azure nghe ra dạng không 's' (heuristic).
- **Prosody**: chỉ là cảnh báo (best-effort).
"""
    )

    if run_btn:
        if not vocab_text.strip():
            st.error("Bạn chưa dán danh sách từ.")
            st.stop()
        if not uploaded_files:
            st.error("Bạn chưa upload audio.")
            st.stop()
        if not SPEECH_KEY or not SPEECH_REGION:
            st.error("Thiếu SPEECH_KEY hoặc SPEECH_REGION (nhập ở sidebar).")
            st.stop()

        vocab_list = [line.strip() for line in vocab_text.splitlines() if line.strip()]
        if not vocab_list:
            st.error("Danh sách rỗng.")
            st.stop()

        reference_text = " ".join(vocab_list)

        expected_tokens_vocab = []
        for item in vocab_list:
            expected_tokens_vocab.extend(_item_tokens(item))

        st.divider()
        st.subheader("📊 Table A — Bảng tổng kết theo học sinh")

        summary_placeholder = st.empty()
        download_placeholder = st.empty()

        progress = st.progress(0)
        status_line = st.empty()

        table_rows: List[Dict] = []
        summary_lines: List[str] = []

        for idx, up in enumerate(uploaded_files):
            status_line.write(f"Đang chấm: **{up.name}** ({idx+1}/{len(uploaded_files)})")
            progress.progress(int(((idx + 1) / max(1, len(uploaded_files))) * 100))

            student = guess_student_name(up.name)

            with tempfile.TemporaryDirectory() as tmp_dir:
                suffix = Path(up.name).suffix.lower() or ".wav"
                src_path = os.path.join(tmp_dir, f"input{suffix}")
                wav_path = os.path.join(tmp_dir, "input_16k_mono.wav")

                with open(src_path, "wb") as f:
                    f.write(up.getbuffer())

                # Convert
                try:
                    duration_ms, _ = to_wav_16k_mono(src_path, wav_path)
                except Exception as e:
                    table_rows.append(
                        {
                            "Học sinh": student,
                            "Audio QC": f"❌ Convert error: {e}",
                            "Score (%)": "",
                            "Missing words": "",
                            "Phát âm sai/điểm thấp": "",
                            "Nuốt âm cuối": "",
                            "Thiếu/ yếu /s/~/z/": "",
                            "Thiếu -s số nhiều": "",
                            "Prosody": "",
                            "Omission (không match)": "",
                        }
                    )
                    continue

                # QC gate
                qc_msg = ""
                metrics = {}
                if enable_audio_qc:
                    try:
                        metrics, issues = analyze_audio_quality(
                            wav_path=wav_path,
                            min_duration_s=float(min_duration_s),
                            min_dbfs=float(min_dbfs),
                            max_silence_ratio=float(max_silence_ratio),
                            min_silence_len_ms=int(min_silence_len_ms),
                            silence_rel_db=float(silence_rel_db),
                            long_pause_ms=int(long_pause_ms),
                        )
                        if issues:
                            qc_msg = "⚠️ " + " | ".join(issues)
                            table_rows.append(
                                {
                                    "Học sinh": student,
                                    "Audio QC": qc_msg,
                                    "Score (%)": "",
                                    "Missing words": "",
                                    "Phát âm sai/điểm thấp": "",
                                    "Nuốt âm cuối": "",
                                    "Thiếu/ yếu /s/~/z/": "",
                                    "Thiếu -s số nhiều": "",
                                    "Prosody": "",
                                    "Omission (không match)": "",
                                }
                            )
                            continue
                        qc_msg = f"✅ OK (dur {metrics['duration_s']:.1f}s, {metrics['dbfs']:.1f} dBFS, silence {metrics['silence_ratio']:.0%})"
                    except Exception as e:
                        qc_msg = f"⚠️ QC error: {e}"

                dur_s = max(1, int(duration_ms / 1000))

                # Auto-tune timeouts
                _end = int(end_silence_timeout_ms)
                _seg = int(seg_silence_timeout_ms)
                _init = int(initial_silence_timeout_ms)

                if auto_tune_pause:
                    if pause_profile == "Nhẹ":
                        seg, end, buffer = 1800, 12000, 35
                    elif pause_profile == "Vừa":
                        seg, end, buffer = 2800, 16000, 50
                    else:
                        seg, end, buffer = 4200, 20000, 70

                    if dur_s >= 90:
                        seg = min(5000, seg + 700)
                        end = min(20000, end + 2000)
                        buffer += 20
                    elif dur_s <= 20:
                        seg = max(1200, seg - 500)
                        end = max(10000, end - 2000)

                    _seg, _end = int(seg), int(end)
                    _init = max(int(_init), 15000)
                    max_wait_seconds = min(600, max(60, dur_s + buffer))
                else:
                    max_wait_seconds = max(60, min(dur_s + 30, 600))

                # Azure call
                try:
                    results_list = call_with_retry(
                        lambda: run_pron_assessment_continuous(
                            wav_path=wav_path,
                            reference_text=reference_text,
                            locale=locale,
                            speech_key=SPEECH_KEY,
                            speech_region=SPEECH_REGION,
                            end_silence_timeout_ms=_end,
                            seg_silence_timeout_ms=_seg,
                            initial_silence_timeout_ms=_init,
                            max_wait_seconds=max_wait_seconds,
                            enable_prosody=bool(enable_prosody),
                        ),
                        max_retries=3,
                        base_sleep=1.0,
                    )
                except Exception as e:
                    table_rows.append(
                        {
                            "Học sinh": student,
                            "Audio QC": qc_msg,
                            "Score (%)": "",
                            "Missing words": "",
                            "Phát âm sai/điểm thấp": f"❌ Azure error: {e}",
                            "Nuốt âm cuối": "",
                            "Thiếu/ yếu /s/~/z/": "",
                            "Thiếu -s số nhiều": "",
                            "Prosody": "",
                            "Omission (không match)": "",
                        }
                    )
                    continue

                if not results_list:
                    table_rows.append(
                        {
                            "Học sinh": student,
                            "Audio QC": qc_msg,
                            "Score (%)": "",
                            "Missing words": "",
                            "Phát âm sai/điểm thấp": "⚠️ Không nhận diện được segment",
                            "Nuốt âm cuối": "",
                            "Thiếu/ yếu /s/~/z/": "",
                            "Thiếu -s số nhiều": "",
                            "Prosody": "",
                            "Omission (không match)": "",
                        }
                    )
                    continue

                all_rows: List[Dict] = []
                for rj in results_list:
                    all_rows.extend(extract_word_rows(rj))

                rec_rows = [r for r in all_rows if (r.get("tok") or "").strip()]
                if not rec_rows:
                    table_rows.append(
                        {
                            "Học sinh": student,
                            "Audio QC": qc_msg,
                            "Score (%)": "",
                            "Missing words": "",
                            "Phát âm sai/điểm thấp": "⚠️ Không có Words trong JSON",
                            "Nuốt âm cuối": "",
                            "Thiếu/ yếu /s/~/z/": "",
                            "Thiếu -s số nhiều": "",
                            "Prosody": "",
                            "Omission (không match)": "",
                        }
                    )
                    continue

                # Score (%)
                score_pack = compute_word_based_score(
                    expected_tokens=expected_tokens_vocab,
                    rec_rows=rec_rows,
                    lookahead=int(lookahead),
                    missing_token_score=float(missing_token_score),
                )

                # Table A buckets + structured issues
                buckets, issues_struct = build_student_error_buckets_and_issues(
                    vocab_list=vocab_list,
                    all_rows=all_rows,
                    lookahead=int(lookahead),
                    accuracy_threshold=float(accuracy_threshold),
                    s_acc_threshold=float(s_acc_threshold),
                    final_acc_threshold=float(final_acc_threshold),
                    enable_prosody=bool(enable_prosody),
                    prosody_threshold=float(prosody_threshold),
                )

                def join_cell(xs: List[str]) -> str:
                    if not xs:
                        return ""
                    return " | ".join(xs)

                table_rows.append(
                    {
                        "Học sinh": student,
                        "Audio QC": qc_msg,
                        "Score (%)": f"{score_pack['score_pct']:.1f}" if isinstance(score_pack.get("score_pct"), (int, float)) else "",
                        "Missing words": f"{score_pack['missing']}/{score_pack['total']}" if score_pack.get("total") else "",
                        "Phát âm sai/điểm thấp": join_cell(buckets["mispron_low"]),
                        "Nuốt âm cuối": join_cell(buckets["missing_final"]),
                        "Thiếu/ yếu /s/~/z/": join_cell(buckets["missing_s"]),
                        "Thiếu -s số nhiều": join_cell(buckets["missing_plural_s"]),
                        "Prosody": join_cell(buckets["prosody"]),
                        "Omission (không match)": join_cell(buckets["omission"]),
                    }
                )

                worst_list = summarize_worst_per_word(issues_struct, limit=int(worst_limit))

                score_txt = ""
                if isinstance(score_pack.get("score_pct"), (int, float)):
                    score_txt = f" (Score {score_pack['score_pct']:.1f}%)"

                if worst_list:
                 summary_lines.append(f"{len(summary_lines)+1}. {student}{score_txt}: " + " | ".join(worst_list))
                else:
                    summary_lines.append(f"{len(summary_lines)+1}. {student}{score_txt}: (Không có lỗi đáng chú ý)")

                # Update global Top-issues aggregator (Tab 1)
                if worst_list:
                    _agg_add(st.session_state["top_issues_agg"], student, worst_list)

            # live update
            if pd is not None:
                summary_placeholder.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        status_line.write("✅ Xong!")

        # Render final table + export
        csv_bytes = b""
        if pd is not None:
            df = pd.DataFrame(table_rows)
            summary_placeholder.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            download_placeholder.download_button(
                "⬇️ Tải CSV Table A (Vocab)",
                data=csv_bytes,
                file_name="vocab_tableA_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Text summary under table
        st.divider()
        st.subheader("📌 Tóm tắt lỗi theo từng học sinh (worst-per-word) — để copy/paste gửi học sinh")
        full_text = "\n".join(summary_lines)

        # --------- SAVE SNAPSHOT (Tab 1) ----------
        try:
            os.makedirs(SESSION_DIR, exist_ok=True)
            if csv_bytes:
                save_bytes(os.path.join(SESSION_DIR, "vocab_tableA_summary.csv"), csv_bytes)
            save_text(os.path.join(SESSION_DIR, "vocab_summary.txt"), full_text)

            append_log_json(SESSION_DIR, {
                "type": "vocab",
                "class": chosen_class,
                "session": chosen_session,
                "locale": locale,
                "files_count": len(uploaded_files) if uploaded_files else 0,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            st.warning(f"Snapshot save warning (Tab 1): {e}")

        st.text_area("Copy nhanh (Ctrl+A → Ctrl+C)", value=full_text, height=240, key="t1_summary_text")

        st.markdown("#### Preview")
        for line in summary_lines:
            st.markdown(line)
        st.divider()
        st.subheader("🔤 Top 10 lỗi phổ biến trong buổi học này — Vocab/Phrase (Table A)")

        if st.session_state["top_issues_vocab"]:
            _render_top10_issues(
                st.session_state["top_issues_vocab"],
                title="",  # Không cần title nữa vì đã có subheader
                top_n=10
            )
        else:
            st.info("Chưa có dữ liệu lỗi từ Tab Vocab trong session này.")

# =============================================================================
# TAB 2 — PASSAGE LEVEL 1
# =============================================================================
with tab_passage:
    st.subheader("Tab 2 — Chấm đọc đoạn văn (Level 1)")

    if not has_ffmpeg():
        st.warning("Server chưa có ffmpeg. Upload mp3/m4a/webm/ogg có thể convert fail. Khuyến nghị upload .wav hoặc cài ffmpeg.")

    colL, colR = st.columns([2, 1], gap="large")
    with colL:
        passage_text = st.text_area(
            "Dán đoạn văn mẫu (reference text)",
            height=220,
            placeholder="Paste your reading passage here...",
            key="passage_text_tab2",
        )
        uploaded_files_passage = st.file_uploader(
            "Upload audio (nhiều file) — tên file là tên học sinh",
            type=["mp3", "m4a", "wav", "webm", "ogg"],
            accept_multiple_files=True,
            key="uploader_tab2",
        )
        run_passage = st.button("✅ CHẤM PASSAGE (Batch)", type="primary", use_container_width=True, key="run_tab2")

    with colR:
        st.markdown("### Chấm điểm & Notable issues (Tab 2)")

    strictness_t2 = st.slider(
        "Strictness (Dễ → Khắt khe)",
        0, 100, 55,
        key="t2_strictness",
        help="Kéo sang phải = chấm khắt khe hơn (dễ bắt lỗi hơn)."
    )

    def _lerp(a, b, t):
        return a + (b - a) * t

    t2 = strictness_t2 / 100.0
    t2_default_acc_thr   = int(round(_lerp(80, 92, t2)))
    t2_default_s_thr     = int(round(_lerp(78, 90, t2)))
    t2_default_final_thr = int(round(_lerp(78, 90, t2)))
    t2_default_lookahead = int(round(_lerp(6, 3, t2)))
    t2_default_max_word  = int(round(_lerp(35, 20, t2)))

    with st.expander("Advanced (tuỳ chỉnh chi tiết)", expanded=False):
        st.caption("Các giá trị ở đây sẽ override Strictness.")

        lookahead_p = st.slider("Lookahead (chịu nhiễu)", 1, 8, t2_default_lookahead, key="t2_lookahead_adv")
        missing_token_score_p = st.slider("Điểm cho từ bị thiếu", 0.0, 50.0, 0.0, step=1.0, key="t2_missing_score_adv")

        accuracy_threshold_p = st.slider("Ngưỡng low accuracy", 0, 100, t2_default_acc_thr, key="t2_acc_thr_adv")
        s_acc_threshold_p = st.slider("Ngưỡng bắt thiếu /s/~/z/", 0, 100, t2_default_s_thr, key="t2_s_thr_adv")
        final_acc_threshold_p = st.slider("Ngưỡng bắt nuốt âm cuối", 0, 100, t2_default_final_thr, key="t2_final_thr_adv")

        ignore_stopwords = st.checkbox("Giảm nhiễu: bỏ stopwords", value=True, key="t2_stop_adv")
        max_word_issues = st.slider("Max notable word issues", 5, 80, t2_default_max_word, step=1, key="t2_max_word_adv")

        st.divider()
        st.markdown("### Sentence-level notes (câu dài sai)")
        min_tokens_for_sentence_check = st.slider("Chỉ check câu >= (tokens)", 5, 30, 10, step=1, key="t2_min_tok_sent")
        pause_inside_sentence_ms = st.slider("Ngắt nghỉ trong câu > (ms)", 300, 2000, 800, step=50, key="t2_pause_in_sent")
        sentence_acc_threshold = st.slider("Ngưỡng avg accuracy theo câu", 0, 100, 80, key="t2_sent_acc")
        max_sentence_notes = st.slider("Max sentence notes", 1, 12, 5, step=1, key="t2_max_sent_notes")

    if run_passage:
        if not passage_text.strip():
            st.error("Bạn chưa dán đoạn văn mẫu.")
            st.stop()
        if not uploaded_files_passage:
            st.error("Bạn chưa upload audio.")
            st.stop()
        if not SPEECH_KEY or not SPEECH_REGION:
            st.error("Thiếu SPEECH_KEY hoặc SPEECH_REGION (nhập ở sidebar).")
            st.stop()

        expected_tokens, sentence_spans = build_sentence_spans(passage_text)
        if len(expected_tokens) < 5:
            st.error("Đoạn văn quá ngắn hoặc không tách được token.")
            st.stop()

        reference_text = passage_text.strip()

        st.divider()
        st.subheader("📊 Bảng kết quả — Reading Passage (Level 1)")

        table_rows: List[Dict] = []
        sentence_issues_by_student: Dict[str, List[str]] = {}
        word_issues_by_student: Dict[str, List[str]] = {}

        progress = st.progress(0)
        status = st.empty()

        for idx, up in enumerate(uploaded_files_passage):
            student = guess_student_name(up.name)
            status.write(f"Đang chấm: **{up.name}** ({idx+1}/{len(uploaded_files_passage)})")
            progress.progress(int(((idx + 1) / max(1, len(uploaded_files_passage))) * 100))

            with tempfile.TemporaryDirectory() as tmp_dir:
                suffix = Path(up.name).suffix.lower() or ".wav"
                src_path = os.path.join(tmp_dir, f"input{suffix}")
                wav_path = os.path.join(tmp_dir, "input_16k_mono.wav")

                with open(src_path, "wb") as f:
                    f.write(up.getbuffer())

                try:
                    duration_ms, _ = to_wav_16k_mono(src_path, wav_path)
                except Exception as e:
                    table_rows.append(
                        {
                            "Học sinh": student,
                            "Audio QC": f"❌ Convert error: {e}",
                            "Score (%)": "",
                            "Missing words": "",
                            "WPM": "",
                            "Pause ratio": "",
                            "Long pauses": "",
                            "Sentence issues": "",
                            "Notable issues": "",
                        }
                    )
                    continue

                qc_msg = ""
                metrics = {}
                if enable_audio_qc:
                    try:
                        metrics, issues = analyze_audio_quality(
                            wav_path=wav_path,
                            min_duration_s=float(min_duration_s),
                            min_dbfs=float(min_dbfs),
                            max_silence_ratio=float(max_silence_ratio),
                            min_silence_len_ms=int(min_silence_len_ms),
                            silence_rel_db=float(silence_rel_db),
                            long_pause_ms=int(long_pause_ms),
                        )
                        if issues:
                            qc_msg = "⚠️ " + " | ".join(issues)
                            table_rows.append(
                                {
                                    "Học sinh": student,
                                    "Audio QC": qc_msg,
                                    "Score (%)": "",
                                    "Missing words": "",
                                    "WPM": "",
                                    "Pause ratio": "",
                                    "Long pauses": "",
                                    "Sentence issues": "",
                                    "Notable issues": "",
                                }
                            )
                            continue
                        qc_msg = f"✅ OK (dur {metrics['duration_s']:.1f}s, {metrics['dbfs']:.1f} dBFS, silence {metrics['silence_ratio']:.0%})"
                    except Exception as e:
                        qc_msg = f"⚠️ QC error: {e}"

                dur_s = max(1, int(duration_ms / 1000))

                _end = int(end_silence_timeout_ms)
                _seg = int(seg_silence_timeout_ms)
                _init = int(initial_silence_timeout_ms)

                if auto_tune_pause:
                    if pause_profile == "Nhẹ":
                        seg, end, buffer = 1800, 12000, 35
                    elif pause_profile == "Vừa":
                        seg, end, buffer = 2800, 16000, 50
                    else:
                        seg, end, buffer = 4200, 20000, 70

                    if dur_s >= 90:
                        seg = min(5000, seg + 700)
                        end = min(20000, end + 2000)
                        buffer += 20
                    elif dur_s <= 20:
                        seg = max(1200, seg - 500)
                        end = max(10000, end - 2000)

                    _seg, _end = int(seg), int(end)
                    _init = max(int(_init), 15000)
                    max_wait_seconds = min(600, max(60, dur_s + buffer))
                else:
                    max_wait_seconds = max(60, min(dur_s + 30, 600))

                try:
                    results_list = call_with_retry(
                        lambda: run_pron_assessment_continuous(
                            wav_path=wav_path,
                            reference_text=reference_text,
                            locale=locale,
                            speech_key=SPEECH_KEY,
                            speech_region=SPEECH_REGION,
                            end_silence_timeout_ms=_end,
                            seg_silence_timeout_ms=_seg,
                            initial_silence_timeout_ms=_init,
                            max_wait_seconds=max_wait_seconds,
                            enable_prosody=bool(enable_prosody),
                        ),
                        max_retries=3,
                        base_sleep=1.0,
                    )
                except Exception as e:
                    table_rows.append(
                        {
                            "Học sinh": student,
                            "Audio QC": qc_msg,
                            "Score (%)": "",
                            "Missing words": "",
                            "WPM": "",
                            "Pause ratio": "",
                            "Long pauses": "",
                            "Sentence issues": "",
                            "Notable issues": f"❌ Azure error: {e}",
                        }
                    )
                    continue

                if not results_list:
                    table_rows.append(
                        {
                            "Học sinh": student,
                            "Audio QC": qc_msg,
                            "Score (%)": "",
                            "Missing words": "",
                            "WPM": "",
                            "Pause ratio": "",
                            "Long pauses": "",
                            "Sentence issues": "",
                            "Notable issues": "⚠️ Không nhận diện được segment",
                        }
                    )
                    continue

                all_rows: List[Dict] = []
                for rj in results_list:
                    all_rows.extend(extract_word_rows(rj))

                rec_rows = [r for r in all_rows if (r.get("tok") or "").strip()]
                if not rec_rows:
                    table_rows.append(
                        {
                            "Học sinh": student,
                            "Audio QC": qc_msg,
                            "Score (%)": "",
                            "Missing words": "",
                            "WPM": "",
                            "Pause ratio": "",
                            "Long pauses": "",
                            "Sentence issues": "",
                            "Notable issues": "⚠️ Không có Words trong JSON",
                        }
                    )
                    continue

                score_pack = compute_word_based_score(
                    expected_tokens=expected_tokens,
                    rec_rows=rec_rows,
                    lookahead=int(lookahead_p),
                    missing_token_score=float(missing_token_score_p),
                )
                mapping = score_pack["mapping"]

                duration_s = float(metrics.get("duration_s", dur_s)) if metrics else float(dur_s)
                minutes = max(1e-6, duration_s / 60.0)
                wpm = len(rec_rows) / minutes
                pause_ratio = float(metrics.get("silence_ratio", 0.0)) if metrics else 0.0
                long_pauses = int(metrics.get("long_pause_count", 0)) if metrics else 0

                word_issues = build_passage_word_issues_concise(
                    expected_tokens=expected_tokens,
                    rec_rows=rec_rows,
                    mapping=mapping,
                    accuracy_threshold=float(accuracy_threshold_p),
                    s_acc_threshold=float(s_acc_threshold_p),
                    final_acc_threshold=float(final_acc_threshold_p),
                    ignore_stopwords=bool(ignore_stopwords),
                    lookahead=int(lookahead_p),
                    max_items=int(max_word_issues),
                )
                word_issues_by_student[student] = word_issues

                # Update global Top-issues aggregator (Tab 2)
                if word_issues:
                    _agg_add(st.session_state["top_issues_agg"], student, word_issues)

                sent_notes = detect_sentence_issues(
                    sentence_spans=sentence_spans,
                    expected_tokens=expected_tokens,
                    rec_rows=rec_rows,
                    mapping=mapping,
                    sentence_acc_threshold=float(sentence_acc_threshold),
                    pause_inside_sentence_ms=int(pause_inside_sentence_ms),
                    min_tokens_for_sentence_check=int(min_tokens_for_sentence_check),
                    max_notes=int(max_sentence_notes),
                )
                sentence_issues_by_student[student] = sent_notes

                table_rows.append(
                    {
                        "Học sinh": student,
                        "Audio QC": qc_msg,
                        "Score (%)": f"{score_pack['score_pct']:.1f}" if isinstance(score_pack.get("score_pct"), (int, float)) else "",
                        "Missing words": f"{score_pack['missing']}/{score_pack['total']}" if score_pack.get("total") else "",
                        "WPM": f"{wpm:.0f}",
                        "Pause ratio": f"{pause_ratio:.0%}",
                        "Long pauses": str(long_pauses),
                        "Sentence issues": " | ".join(sent_notes) if sent_notes else "",
                        "Notable issues": " | ".join(word_issues) if word_issues else "",
                    }
                )

        status.write("✅ Xong!")
        progress.progress(100)

        csv_bytes = b""
        if pd is not None:
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Tải CSV (Passage Level 1)",
                data=csv_bytes,
                file_name="passage_level1_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        st.subheader("📌 Tóm tắt để copy/paste gửi học sinh")

        summary_lines = []
        for i, row in enumerate(table_rows, start=1):
            student = (row.get("Học sinh") or "").strip()
            qc = (row.get("Audio QC") or "").strip()

            if qc.startswith("⚠️") or qc.startswith("❌"):
                summary_lines.append(f"{i}. {student}: {qc}")
                continue

            score_txt = (row.get("Score (%)") or "").strip()
            miss_txt = (row.get("Missing words") or "").strip()
            wpm_txt = (row.get("WPM") or "").strip()
            pause_txt = (row.get("Pause ratio") or "").strip()

            sent_notes = sentence_issues_by_student.get(student, [])
            word_issues = word_issues_by_student.get(student, [])

            sent_text = (" | ".join(sent_notes)) if sent_notes else "(No sentence-level issue)"
            word_text = (" | ".join(word_issues)) if word_issues else "(No notable word issue)"

            summary_lines.append(
                f"{i}. {student} (Score {score_txt}%,)\n"
                f"   - Sentence: {sent_text}\n"
                f"   - Words: {word_text}"
            )

        full_text = "\n".join(summary_lines)

        # --------- SAVE SNAPSHOT (Tab 2) ----------
        try:
            os.makedirs(SESSION_DIR, exist_ok=True)
            if csv_bytes:
                save_bytes(os.path.join(SESSION_DIR, "passage_level1_summary.csv"), csv_bytes)
            save_text(os.path.join(SESSION_DIR, "passage_summary.txt"), full_text)

            append_log_json(SESSION_DIR, {
                "type": "passage",
                "class": chosen_class,
                "session": chosen_session,
                "locale": locale,
                "files_count": len(uploaded_files_passage) if uploaded_files_passage else 0,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            st.warning(f"Snapshot save warning (Tab 2): {e}")

        st.text_area("Copy nhanh (Ctrl+A → Ctrl+C)", value=full_text, height=320, key="t2_summary_text")
        st.markdown("#### Preview")
        for line in summary_lines:
            st.markdown(line.replace("\n", "  \n"))
        st.divider()
        st.subheader("📖 Top 10 lỗi phổ biến trong buổi học này — Reading Passage")

        if st.session_state["top_issues_passage"]:
            _render_top10_issues(
                st.session_state["top_issues_passage"],
                title="",  # Không cần title vì đã có subheader
                top_n=10
            )
        else:
            st.info("Chưa có dữ liệu lỗi từ Tab Passage trong session này.")

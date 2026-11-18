"""Lia server. Same rules and question flow as the Kaggle notebook.

Pick where the model's words come from with the LIA_BACKEND setting:
  lmstudio  LM Studio's local server (the default; address http://127.0.0.1:1234/v1)
  fake      canned replies, only when you ask for them (test the app with no model)
  direct    load MedGemma with transformers (Kaggle or a big GPU)

Run:  python lia_app.py        (or double-click start_lia.bat, or Lia.exe)
Old way still works:  python -m uvicorn server:app --host 127.0.0.1 --port 8000
"""
import os, re, sys, json, time, uuid, threading, urllib.request, urllib.error
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BACKEND = os.environ.get("LIA_BACKEND", "lmstudio").lower()   # demo mode must be asked for: LIA_BACKEND=fake
LM_URL = os.environ.get("LIA_URL", "http://127.0.0.1:1234/v1")
LM_MODEL = os.environ.get("LIA_MODEL", "")          # leave empty to auto-detect
# HERE holds the page and static files (inside the .exe when packaged).
# APP_DIR is the folder you can see: logs, models and crew faces live there, next to server.py or Lia.exe.
# (If there is no models folder next to Lia.exe, the copy packed inside the .exe is used, if one was packed.)
FROZEN = getattr(sys, "frozen", False)
HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
APP_DIR = Path(sys.executable).parent if FROZEN else Path(__file__).parent
MODELS = Path(os.environ.get("LIA_MODELS") or (APP_DIR / "models" if (APP_DIR / "models").exists() else HERE / "models"))
LOG_FILE = APP_DIR / "logs" / "lia_log.jsonl"
RANK = {"low": 0, "medium": 1, "high": 2}

# ---------------------------------------------------------------- safety rules (DRAFT: have a reviewer check all of these)
FLOORS = {"pulmonary embol": "high", "thrombo": "high", "dvt": "high", "blood clot": "high", "embol": "high",
          "myocardial": "high", "heart attack": "high", "sepsis": "high", "radiation sickness": "high",
          "acute radiation": "high", "obstruction": "high", "ileus": "high", "volvulus": "high",
          "appendicitis": "high", "stroke": "high",
          "anaphyla": "high", "angioedema": "high", "laryngeal": "high", "subarachnoid": "high",
          "intracranial hemorrhage": "high", "brain bleed": "high", "aneurysm": "high", "meningitis": "high",
          "encephalitis": "high", "cardiac arrest": "high", "overdose": "high",
          "allergic reaction": "medium",
          "hemolytic": "medium", "kidney stone": "medium", "renal colic": "medium", "arrhythmia": "medium",
          "atrial fibrillation": "medium", "ventricular": "medium", "intracranial": "medium", "papilledema": "medium"}
TEXT_FLAGS = re.compile(r"hopeless|suicid|want to die|end my life|self-harm|can'?t breathe|cannot breathe|"
                        r"unconscious|seizure|passed out|not breathing|heavy bleeding|crushing", re.I)
SELF_HARM = re.compile(r"hopeless|suicid|want to die|end my life|self-harm", re.I)
CARE_NOTE = ("If you are having thoughts of harming yourself, tell the crew commander or flight surgeon "
             "right now. You do not have to handle this alone.")   # DRAFT wording: a professional should set this

# Warning-sign patterns in what the patient said, checked in code no matter what the model answers.
# Each rule fires only if ALL of its patterns match. DRAFT: a reviewer should check and extend this list.
def _all(*pats):
    return [re.compile(p, re.I) for p in pats]

NEGATION = re.compile(r"\b(no|not|never|none|nothing|nor|neither|without|denies|denied|deny)\b|n't", re.I)

def _present(pattern, text):
    """True if the pattern appears in the text and is not negated ('denies shortness of breath')."""
    for m in pattern.finditer(text):
        before = re.split(r"[.,;:!?]", text[max(0, m.start() - 30):m.start()])[-1]
        if not NEGATION.search(before):
            return True
    return False

SYMPTOM_RULES = [
    ("sudden or worst-ever headache", "high", _all(
        r"worst headache|thunderclap|headache[^.]{0,60}(all at once|out of nowhere)|"
        r"(sudden|suddenly)[^.]{0,40}(severe|terrible|insane|unbearable|intense)[^.]{0,20}headache")),
    ("headache or fever with stiff neck", "high", _all(
        r"stiff neck|neck (is |feels )?stiff", r"headache|fever|\btemp|light hurts|sensitive to light")),
    ("throat or breathing symptoms with swelling, hives or itching", "high", _all(
        r"throat[^.]{0,30}(tight|closing|swell|swollen)|trouble swallowing|difficulty swallowing|wheez|"
        r"voice[^.]{0,25}(weird|hoarse|different|changed)",
        r"hives|rash|itch|swell|swollen|puffy|allerg|sting|new (food|medic)")),
    ("stroke warning signs", "high", _all(
        r"face[^.]{0,20}droop|droop[^.]{0,20}face|slurred|can'?t (speak|talk)|weakness (on|in) one side|"
        r"one side of (my|his|her|the) (body|face)|sudden(ly)?[^.]{0,20}(numb|weak)")),
    ("chest pressure spreading to arm, jaw or shoulder", "high", _all(
        r"(pressure|heavy|tight|squeez|crush)[^.]{0,50}chest|chest[^.]{0,50}(pressure|heavy|tight|squeez|crush)",
        r"\b(arm|jaw|shoulder)\b")),
    ("trouble breathing", "high", _all(
        r"trouble breathing|difficulty breathing|hard to breathe|can'?t catch my breath|gasping|"
        r"struggling to breathe|blue lips")),
    ("possible poisoning or overdose", "high", _all(r"overdos|poison|swallowed|took too many")),
    ("shortness of breath", "medium", _all(r"short(ness)? of breath|out of breath")),
]

# Red-flag rules taken from the NASA HUNCH project document (section "Red Flag/Emergency Rules").
SYMPTOM_RULES += [
    ("chest pain + shortness of breath + sweating (project doc)", "high", _all(
        r"chest (pain|pressure|tightness|discomfort)|pain in (my |the )?chest",
        r"short(ness)? of breath|trouble breathing|hard to breathe|out of breath", r"sweat|clammy")),
    ("sudden severe headache (project doc)", "high", _all(
        r"headache", r"sudden|all at once|out of nowhere|abrupt",
        r"severe|worst|terrible|unbearable|intense|excruciating|insane")),
    ("severe abdominal pain + vomiting (project doc)", "high", _all(
        r"(severe|intense|unbearable|terrible|excruciating)[^.]{0,40}(abdom|stomach|belly)|"
        r"(abdom|stomach|belly)[^.]{0,40}(severe|intense|unbearable|terrible|excruciating)",
        r"vomit|threw up|throwing up")),
    ("confusion + fever (project doc)", "high", _all(
        r"confus|disorient|delirious|can'?t think straight", r"fever|chills|\btemp")),
    ("blood in vomit or stool (project doc)", "high", _all(
        r"vomiting blood|blood[^.]{0,20}(vomit|stool|poop|bowel)|(vomit|stool|poop|bowel)[^.]{0,20}blood|"
        r"bloody stool|tarry|coffee[- ]ground")),
    ("rash that does not fade + fever (project doc)", "high", _all(
        r"non-?blanching|(doesn'?t|does not|won'?t) (fade|blanch)|petechia", r"fever|chills|\btemp")),
]

def normalize_units(text):
    """Turns spelled-out units into the symbols the temperature check reads: 120 fahrenheit -> 120 F."""
    text = re.sub(r"\bfahrenheit\b", "F", text, flags=re.I)
    text = re.sub(r"\bcelsius\b|\bcentigrade\b", "C", text, flags=re.I)
    text = re.sub(r"\s*(?:degrees?|deg\.?)\b", " °", text, flags=re.I)
    return text

FEVERISH = re.compile(r"feverish|fever|chills|burning up|hot to the touch", re.I)

def vitals_rules(text):
    """Reads temperature, pulse and oxygen readings from what the patient typed or said.
    Returns (rules, warnings): rules are (minimum urgency, reason) and can only raise urgency; warnings say 'recheck this'.
    DRAFT cut-offs: a reviewer should check them."""
    text = normalize_units(text)
    rules, warn, temps, seen = [], [], [], set()
    for m in re.finditer(r"(\d{2,4}(?:\.\d)?)\s*°?\s*([FfCc])?\b", text):
        raw, v, unit = m.group(0).strip(), float(m.group(1)), (m.group(2) or "").upper()
        before, after = text[max(0, m.start() - 25):m.start()].lower(), text[m.end():m.end() + 3]
        if not unit and not re.search(r"temp|fever|thermometer|°", before + after):
            continue
        if re.search(r"pulse|heart rate|\bhr\b|spo2|oxygen|o2", before[-14:]):
            continue
        key = ("t", v)
        if key in seen:
            continue
        seen.add(key)
        if not unit and v >= 1000:
            warn.append(f"The temperature '{raw}' looks like a typing mistake. Did you mean {v / 10:.1f}? Recheck it.")
            rules.append(("medium", "temperature reading looks mistyped"))
            continue
        if unit == "C" or (not unit and 30 <= v <= 43):
            c = v
        elif unit == "F" or (not unit and 60 <= v <= 130):
            c = (v - 32) * 5 / 9
        else:
            warn.append(f"The temperature reading ({raw}) is not a possible body temperature. Recheck it with another thermometer right away.")
            rules.append(("medium", "temperature reading is not possible"))
            continue
        if c < 25 or c > 45:
            hint = f" Did you mean {v:g} °C?" if unit == "F" and 30 <= v <= 43 else ""
            warn.append(f"The temperature reading ({raw}) is not a possible body temperature.{hint} Recheck it with another thermometer right away.")
            rules.append(("medium", "temperature reading is not possible"))
        else:
            temps.append(c)
            if c < 32: rules.append(("high", "very low body temperature"))
            elif c < 35: rules.append(("medium", "low body temperature"))
    for m in re.finditer(r"(?:pulse|heart rate|\bhr\b)\D{0,15}?(\d{2,3})\b|\b(\d{2,3})\s*(?:bpm|beats)", text, re.I):
        v = float(m.group(1) or m.group(2))
        if ("p", v) in seen: continue
        seen.add(("p", v))
        if v < 20 or v > 250:
            warn.append(f"The pulse reading ({v:g} bpm) looks wrong. Recheck it now.")
            rules.append(("medium", "pulse reading is not possible"))
        elif v <= 40 or v >= 150: rules.append(("high", f"pulse {v:g} bpm is very abnormal"))
        elif v <= 50 or v >= 120: rules.append(("medium", f"pulse {v:g} bpm is abnormal"))
    for m in re.finditer(r"(?:spo2|sp02|o2 sat\w*|oxygen(?: saturation| level| sat\w*)?)\D{0,15}?(\d{2,3})\b|\b(\d{2,3})\s*(?:%|percent)\s*(?:spo2|oxygen|o2)", text, re.I):
        v = float(m.group(1) or m.group(2))
        if ("o", v) in seen: continue
        seen.add(("o", v))
        if v > 100 or v < 50:
            warn.append(f"The oxygen reading ({v:g}%) looks wrong. Recheck it now.")
            rules.append(("medium", "oxygen reading is not possible"))
        elif v <= 90: rules.append(("high", f"oxygen level {v:g}% is low"))
        elif v <= 94: rules.append(("medium", f"oxygen level {v:g}% is a little low"))
    if temps and max(temps) < 37.5 and _present(FEVERISH, text):
        warn.append("You mention a fever, but the reading looks normal. Recheck it, and trust how you feel.")
    return rules, warn

def temp_floor(text):
    text = normalize_units(text)
    for m in re.finditer(r"(\d{2,3}(?:\.\d)?)\s*°?\s*([FfCc])?\b", text):
        v = float(m.group(1)); unit = (m.group(2) or "").upper()
        before = text[max(0, m.start() - 25):m.start()].lower()
        after = text[m.end():m.end() + 3]
        if not unit and not re.search(r"temp|fever|°", before + after):
            continue
        if unit == "C" or (not unit and 30 <= v <= 43):
            c = v
        elif unit == "F" or (not unit and 60 <= v <= 130):      # no unit: above 60 can only be Fahrenheit
            c = (v - 32) * 5 / 9
        else:
            continue
        if c >= 39.4: return "high"
        if c >= 38.0: return "medium"
    return None

def floor_level(preds, patient_text, full_text):
    """Returns (minimum urgency, list of rules that fired). Rules can only raise urgency."""
    level, why = "low", []
    def bump(lvl, reason):
        nonlocal level
        if RANK[lvl] > RANK[level]:
            level = lvl
        if reason not in why:
            why.append(reason)
    for p in preds:
        for key, lvl in FLOORS.items():
            if key in p:
                bump(lvl, f"dangerous possibility: {key}")
    if _present(TEXT_FLAGS, patient_text):
        bump("high", "emergency words")
    for reason, lvl, pats in SYMPTOM_RULES:
        if all(_present(p, patient_text) for p in pats):
            bump(lvl, "warning sign: " + reason)
    for lvl, reason in vitals_rules(patient_text)[0]:
        bump(lvl, reason)
    t = temp_floor(full_text)
    if t:
        bump(t, "high temperature")
        if re.search(r"headache|head hurts|head pain", patient_text, re.I) and \
           re.search(r"blurr|vision|double|can'?t see", patient_text, re.I):
            bump("high", "fever with headache and vision changes")
    if re.search(r"bleed", patient_text, re.I) and (
            re.search(r"\b(2\d|[3-9]\d|\d{3})\s*(min|minute)", patient_text, re.I) or
            re.search(r"\bhours?\b|\bhrs?\b", patient_text, re.I) or
            re.search(r"weak|dizz|faint|lighthead", patient_text, re.I)):
        bump("medium", "bleeding that is long-lasting or with weakness")
    return level, why

# ---------------------------------------------------------------- model backends
def msg(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}

_gen_lock = threading.Lock()
_lm_model_name = ""
_pipe = None

def lm_request(path, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(LM_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def lm_model(check=False):
    """Name of the model to use. check=True asks LM Studio again, so a stopped server is noticed."""
    global _lm_model_name
    if check or not (LM_MODEL or _lm_model_name):
        ids = [m["id"] for m in lm_request("/models", timeout=4 if check else 300)["data"]]
        if not ids:
            raise RuntimeError("LM Studio is running but has no model. Download and load MedGemma in LM Studio.")
        _lm_model_name = next((i for i in ids if "gemma" in i.lower()), ids[0])
    return LM_MODEL or _lm_model_name

def img_msg(role, text, url):
    """A message with a picture attached (only used with LM Studio and a model that can see images)."""
    return {"role": role, "content": [{"type": "text", "text": text}, {"type": "image", "url": url}]}

def to_plain(msgs):
    """Gemma's chat template in LM Studio only accepts user/assistant/user/assistant... with no system role,
    so the system text is folded into the first user message and same-role neighbours are merged."""
    plain, sys_text = [], ""
    for m in msgs:
        t = m["content"][0]["text"]
        imgs = [p["url"] for p in m["content"] if p.get("type") == "image"]
        if m["role"] == "system":
            sys_text += t + "\n\n"
        elif plain and plain[-1]["role"] == m["role"] and not imgs and isinstance(plain[-1]["content"], str):
            plain[-1]["content"] += "\n\n" + t
        else:
            c = t if not imgs else [{"type": "text", "text": t}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
            plain.append({"role": m["role"], "content": c})
    if sys_text:
        if plain and plain[0]["role"] == "user":
            c = plain[0]["content"]
            if isinstance(c, str): plain[0]["content"] = sys_text + c
            else: c[0]["text"] = sys_text + c[0]["text"]
        else:
            plain.insert(0, {"role": "user", "content": sys_text.strip()})
    if plain and plain[0]["role"] != "user":
        plain.insert(0, {"role": "user", "content": "Hello."})
    return plain

def lm_gen(msgs, max_new_tokens):
    plain = to_plain(msgs)
    data = lm_request("/chat/completions", {"model": lm_model(), "messages": plain,
                                            "temperature": 0, "max_tokens": max_new_tokens})
    return data["choices"][0]["message"]["content"].strip()

def direct_gen(msgs, max_new_tokens):
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        _pipe = pipeline("image-text-to-text", model=os.environ.get("LIA_HF_MODEL", "google/medgemma-4b-it"),
                         dtype=torch.bfloat16, device_map="auto")
        _pipe.model.generation_config.max_length = None
    out = _pipe(text=msgs, generate_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False})
    return out[0]["generated_text"][-1]["content"].strip()

FAKE_CASES = [
    (r"chest|heart", ["musculoskeletal chest pain", "acid reflux", "cardiac arrhythmia"], "medium"),
    (r"head|dizz", ["tension headache", "dehydration", "space adaptation syndrome"], "low"),
    (r"stomach|abdom|belly|nausea", ["gastroenteritis", "constipation", "appendicitis"], "medium"),
    (r"fever|temp|chill", ["viral infection", "bacterial infection", "dehydration"], "medium"),
]
FAKE_QUESTIONS = [
    ("exactly", "Where exactly is the pain, and has it moved?"),
    ("vomiting", "Have you had any vomiting or diarrhea?"),
    ("happened", "Did anything happen just before this started?"),
    ("how bad", "On a scale of 1 to 10, how bad is it right now?"),
    ("how long", "How long has this been going on, and is it getting better or worse?"),
    ("temperature", "Have you measured your temperature or pulse?"),
    ("other symptoms", "Are there any other symptoms?"),
    ("anything else", "Is there anything else you think I should know?"),
]

def fake_gen(msgs, max_new_tokens):
    system = msgs[0]["content"][0]["text"]
    last = msgs[-1]["content"][0]["text"]
    if system.startswith("You are Lia, a medical assistant"):      # the final-answer step
        low = last.lower()
        conds, urg = ["fatigue or sleep disruption", "dehydration", "space motion sickness"], "low"
        for pat, c, u in FAKE_CASES:
            if re.search(pat, low):
                conds, urg = c, u
                break
        return json.dumps({"title": (conds[0].capitalize() if conds else "Health check"), "conditions": conds, "urgency": urg,
                           "next_steps": ["Rest and drink water.", "Write down how you feel every hour.",
                                          "Tell the crew commander."],
                           "call_commander_if": ["You feel worse.", "New symptoms appear."]})
    m = re.search(r"question about: (.*?)\. If the patient", last, re.S)
    topic = (m.group(1) if m else "").lower()
    for key, q in FAKE_QUESTIONS:
        if key in topic:
            return q
    return "Can you tell me a bit more?"

def gen(msgs, max_new_tokens=200):
    with _gen_lock:
        try:
            if BACKEND == "lmstudio":
                return lm_gen(msgs, max_new_tokens)
            if BACKEND == "direct":
                return direct_gen(msgs, max_new_tokens)
            return fake_gen(msgs, max_new_tokens)
        except urllib.error.HTTPError as e:      # the server answered, but refused the request
            try:
                body = e.read().decode("utf-8", "replace")[:600]
            except Exception:
                body = ""
            raise HTTPException(502, f"The model server answered with an error (HTTP {e.code}): {body}")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise HTTPException(503, f"Cannot reach the model server at {LM_URL}. Is LM Studio's server on? ({e})")
        except RuntimeError as e:
            raise HTTPException(503, str(e))

# ---------------------------------------------------------------- Lia's conversation
CHAT_SYSTEM = """You are Lia, a calm, friendly medical assistant for astronauts on a long mission.
You are not a doctor. Keep replies short and in simple words. Never start with "Okay."
Ask exactly ONE short, natural question. Do not give a diagnosis or advice yet.
Do not repeat a question you already asked."""

FINAL_SYSTEM = """You are Lia, a medical assistant for astronauts. You are not a doctor.
Read the conversation and reply with ONLY a JSON object with these keys:
"conditions": list of the 3 most likely specific medical diagnoses, most likely first
"urgency": exactly one of low, medium, high
"next_steps": list of 3 to 4 short, simple actions to take now
"call_commander_if": list of 2 to 3 short warning signs that mean it is getting worse
Urgency guide: low = can be monitored; medium = tell the commander and get a medical check soon; high = possibly life-threatening, act immediately.
Example for a different case (an ankle injury): {"conditions": ["ankle sprain", "ankle fracture", "torn ligament"], "urgency": "medium", "next_steps": ["Rest the ankle", "Keep it raised", "Wrap it gently"], "call_commander_if": ["Numbness or colour change in the foot", "Pain gets much worse"]}
"title": a 2 to 5 word plain summary of the main problem, for example "Fever with cough" or "Lower right belly pain"
Name real diagnoses. Never write placeholder words like "most likely" or "second".
If the crew member has a known health profile (conditions, allergies, medications), take it into account: name any allergy or medicine that could cause or worsen the symptoms in next_steps, and mention a known heart or lung condition in call_commander_if when it is relevant. Earlier checks are background only: never diagnose from them alone, and say so if they change the picture."""

TOPICS = [
    "what happened just before this started (an injury, an accident, something they ate or were exposed to, or what they were doing)",
    "how bad it is, on a scale of 1 to 10",
    "how long it has been going on and whether it is getting better or worse",
    "their temperature and pulse, if they have measured them",
    "any other symptoms",
    "whether there is anything else they think you should know",
]
# Questions follow the patient's main concern (picked from their first message). Code chooses the topics; the model words them.
CONCERNS = [
    ("mood", r"anxious|anxiety|depress|\bdown\b|\bsad\b|isolat|withdrawn|irritable|mood|on edge|lonely|stress"),
    ("chest", r"chest|heart|palpitat"),
    ("breath", r"breath|cough|wheez|short of air|choking"),
    ("head", r"headache|migraine|head (hurts|pain)|dizz|vertigo|lighthead"),
    ("eye", r"\beyes?\b|vision|blurr"),
    ("skin", r"rash|hives|itch|skin|swell"),
    ("tooth", r"tooth|teeth|\bjaw\b|\bgum"),
    ("urine", r"urin|\bpee\b|kidney|flank"),
    ("abdomen", r"stomach|abdom|belly|nausea|nauseous|vomit|diarrhea|constipat|cramp"),
    ("injury", r"\bcut\b|bleed|wound|\bfell\b|\bfall\b|\bhit\b|bang|sprain|fractur|broken|burn|bruise|injur"),
    ("fever", r"fever|chills|infect|sore throat|\bflu\b"),
    ("limb", r"\bleg\b|calf|\barm\b|shoulder|knee|ankle|back pain|muscle|joint"),
]
BANKS = {
    "chest": ["what the chest feeling is like (pressure, sharp, tight or burning) and whether it spreads to the arm, jaw or back",
              "how bad it is on a scale of 1 to 10 and whether it is constant or comes and goes",
              "whether they also have shortness of breath, sweating, nausea or dizziness",
              "whether anything changes it, like resting, moving or breathing deeply"],
    "breath": ["whether it came on suddenly or built up, and whether they can speak in full sentences right now",
               "whether they have a cough, wheezing, chest pain or fever",
               "whether the cabin air, smoke or a CO2 alarm could be involved, or if they inhaled something",
               "how bad it is on a scale of 1 to 10 and whether it is getting worse"],
    "head": ["whether the headache came on suddenly or built up slowly, and where it is",
             "how bad it is on a scale of 1 to 10",
             "whether they have vision changes, a stiff neck, vomiting, or sensitivity to light",
             "whether they hit their head or did anything unusual before it started"],
    "eye": ["which eye, and what happened (something in it, a hit, a chemical)",
            "whether their vision has changed, with blurring, double vision or flashes",
            "how painful it is on a scale of 1 to 10, and whether light bothers it"],
    "skin": ["where on the body it is and what it looks like",
             "whether their lips, tongue or throat feel swollen, or if they have trouble breathing or swallowing",
             "whether they had a new food, medicine or contact with something before it started",
             "whether it itches or hurts, is spreading, or comes with a fever"],
    "tooth": ["which tooth or area, and whether the gum or face is swollen",
              "whether they have a fever or trouble swallowing or opening their mouth",
              "how bad it is on a scale of 1 to 10, and what sets it off, like cold or chewing"],
    "urine": ["whether they have burning, urgency, or blood in the urine",
              "where the pain is (side, back or lower belly) and whether it comes in waves",
              "whether they have a fever, chills or nausea",
              "how much they have been drinking"],
    "abdomen": ["where exactly the pain is and whether it has moved",
                "whether they have vomiting, diarrhea, or blood in vomit or stool",
                "when they last ate and last passed stool or gas",
                "how bad it is on a scale of 1 to 10 and whether it is getting worse"],
    "injury": ["what happened and when",
               "whether there is bleeding, how much, and whether it slows with pressure",
               "whether they can move the area normally, and whether there is numbness or tingling",
               "how bad the pain is on a scale of 1 to 10, and whether they hit their head"],
    "fever": ["how high the temperature is and how long they have had it",
              "whether they have a cough, sore throat, rash or stiff neck",
              "whether anyone else on the crew is sick",
              "whether they feel confused or very weak"],
    "limb": ["what happened, or when it started",
             "whether there is swelling, redness or warmth, especially on one side only",
             "whether they can move it and put weight on it, and whether there is numbness or weakness",
             "how bad it is on a scale of 1 to 10"],
    "mood": ["how long this has been going on, and how it has affected their sleep and appetite",
             "what has changed recently, like workload, isolation or news from home",
             "whether they are still able to do their daily tasks"],
    "general": ["what happened just before this started, or what they were doing",
                "how bad it is on a scale of 1 to 10",
                "how long it has been going on and whether it is getting better or worse",
                "any other symptoms"],
}
VITALS_SEEN = re.compile(r"\btemp|\u00b0|\bpulse\b|\bbpm\b|heart rate", re.I)

def build_topics(first_text, all_text):
    name = next((n for n, pat in CONCERNS if re.search(pat, first_text, re.I)), "general")
    topics = list(BANKS[name][:4])
    if not VITALS_SEEN.search(all_text):
        topics.append("their temperature and pulse, if they have measured them")
    topics.append("whether there is anything else they think you should know")
    return name, topics

MOOD_WORDS = re.compile(r"anxious|anxiety|depress|\bdown\b|\bsad\b|isolat|withdrawn|irritable|mood|on edge|lonely", re.I)
MOOD_QUESTION = "Have you been feeling hopeless lately?"     # DRAFT wording: have your reviewer check it
YES = re.compile(r"^\W*(yes|yeah|yep|yup|sometimes|a bit|a little|kind of|sort of|i have|i do)\b", re.I)

def final_answer(transcript, patient_only, context="", image=None, photo_note=""):
    user_text = (context + "\n\n" if context else "") + transcript
    if photo_note:
        user_text += f"\n\n[Photo findings from the image (may be wrong; supporting evidence only, never a diagnosis by itself): {photo_note}]"
    um = img_msg("user", user_text, image) if image else msg("user", user_text)
    raw = gen([msg("system", FINAL_SYSTEM), um], 400)
    try:
        d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception:
        d = {}
    conds = [str(c) for c in d.get("conditions", [])][:3]
    model_u = str(d.get("urgency", "")).lower()
    base = model_u if model_u in RANK else "medium"          # broken answer: play it safe
    floor, why = floor_level([c.lower() for c in conds], patient_only, transcript)
    final_u = base if RANK[base] >= RANK[floor] else floor
    warns = vitals_rules(patient_only)[1]
    steps = [("Recheck: " + w) for w in warns] + list(d.get("next_steps") or ["Tell the crew commander.", "Rest and write down your symptoms."])
    title = re.sub(r"\s+", " ", str(d.get("title", ""))).strip()[:60] or (conds[0].capitalize() if conds else "Health check")
    return {"title": title,
            "conditions": conds,
            "model_urgency": model_u or "unparsed",
            "urgency": final_u,
            "rules_raised": final_u != base,
            "rules_fired": why,
            "next_steps": steps,
            "data_warnings": warns,
            "call_commander_if": d.get("call_commander_if") or ["You feel worse or new symptoms appear."],
            "emergency": _present(TEXT_FLAGS, patient_only) or final_u == "high",
            "care_note": CARE_NOTE if _present(SELF_HARM, patient_only) else "",
            "disclaimer": "Lia is an AI helper, not a doctor."}

def context_text(profile, recent):
    """Background the final answer may use. Kept short and clearly labeled, so it cannot read as a diagnosis."""
    parts = []
    p = profile or {}
    labels = [("conditions", "Known conditions"), ("allergies", "Allergies"), ("medications", "Regular medicines"), ("notes", "Other notes")]
    known = [f"{lbl}: {str(p[k]).strip()}" for k, lbl in labels if str(p.get(k, "")).strip()]
    if known:
        parts.append("KNOWN HEALTH PROFILE (reported by the crew member, not checked by Lia):\n" + "\n".join(known))
    if recent:
        lines = [f"- {str(r.get('d',''))[:16]}: {str(r.get('t',''))[:60]} (urgency {str(r.get('u',''))}, top cause {str(r.get('c',''))[:40]})" for r in recent[:5]]
        parts.append("EARLIER CHECKS BY THIS CREW MEMBER (background only, may be unrelated):\n" + "\n".join(lines))
    return "\n\n".join(parts)

class LiaSession:
    def __init__(self):
        self.profile = {}
        self.recent = []
        self.temp_c = None
        self.pulse = None
        self.vitals_added = False
        self.history = [msg("system", CHAT_SYSTEM)]
        self.questions = 0
        self.ti = 0
        self.topics = None
        self.concern = "general"
        self.want_mood = False
        self.asked_mood = False
        self.last_mood = False
        self.mood_flag = ""
        self.done = False
        self.crew = ""
        self.image = None          # a photo lives in memory for this check only; it is never saved
        self.photo_note = ""

    def patient_text(self):
        return " ".join(m["content"][0]["text"] for m in self.history if m["role"] == "user") + self.mood_flag

    def transcript(self):
        return "\n".join(("Patient: " if m["role"] == "user" else "Lia: ") + m["content"][0]["text"]
                         for m in self.history[1:])

    def finish(self):
        self.done = True
        result = final_answer(self.transcript(), self.patient_text(), context_text(self.profile, self.recent), self.image, self.photo_note)
        result["photo_note"] = self.photo_note
        self.image = None
        result["known_profile"] = [f"{lbl}: {str(self.profile.get(k,'')).strip()}" for k, lbl in
                                   [("conditions","Conditions"),("allergies","Allergies"),("medications","Medicines")]
                                   if str(self.profile.get(k,"")).strip()]
        log_case(self, result)
        return {"type": "final", "result": result}

    def reply(self, user_text):
        if not self.vitals_added and (self.temp_c or self.pulse):
            self.vitals_added = True
            bits = []
            if self.temp_c:
                bits.append(f"Temperature {self.temp_c:.1f} C.")
            if self.pulse:
                bits.append(f"Pulse {int(self.pulse)} beats per minute.")
            user_text = user_text + " " + " ".join(bits)
        self.history.append(msg("user", user_text))
        if self.last_mood and YES.search(user_text):
            self.mood_flag = ". Yes, hopeless."
        self.last_mood = False
        if MOOD_WORDS.search(user_text):
            self.want_mood = True
        if _present(TEXT_FLAGS, user_text + self.mood_flag):
            return self.finish()
        if self.want_mood and not self.asked_mood and self.questions >= 1:
            q = MOOD_QUESTION
            self.asked_mood = True
            self.last_mood = True
        else:
            if self.topics is None:
                self.concern, self.topics = build_topics(user_text, self.patient_text())
            if self.ti >= len(self.topics):
                return self.finish()
            topic = self.topics[self.ti]
            self.ti += 1
            hint = (f"\n\n[Instruction for Lia: ask ONE short, natural question about: {topic}. "
                    "Stay on the patient's main concern and do not ask about unrelated body systems. "
                    "If the patient already told you this, ask about something else relevant that they have not mentioned.]")
            if self.photo_note:
                hint += f"\n[Photo findings from the image, which may be wrong: {self.photo_note}]"
            q = gen(self.history[:-1] + [msg("user", user_text + hint)], 80)
            q = re.sub(r"^(okay|ok)[.,!]?\s*", "", q, flags=re.I)
            if "?" in q:
                q = q.split("?")[0].strip() + "?"
        self.history.append(msg("assistant", q))
        self.questions += 1
        return {"type": "question", "text": q, "questions": self.questions}

def log_case(session, result):
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "backend": BACKEND, "crew": session.crew,
                                "transcript": session.transcript(), "result": result}) + "\n")
    except Exception:
        pass   # logging must never break a conversation

# ---------------------------------------------------------------- web API
app = FastAPI(title="Lia")
if (APP_DIR / "static").exists() or (HERE / "static").exists():
    app.mount("/static", StaticFiles(directory=(APP_DIR / "static" if (APP_DIR / "static").exists() else HERE / "static")), name="static")
SESSIONS = {}

class ReplyIn(BaseModel):
    session_id: str
    text: str = ""
    temp_c: Optional[float] = None     # sent once, with the first message, from the home-screen category
    pulse: Optional[int] = None

class AnalyzeIn(BaseModel):
    text: str

class SessionIn(BaseModel):
    name: str = ""
    profile: dict = {}                  # conditions, allergies, medications, notes (kept on this computer)
    recent: list = []                   # short summaries of this crew member's earlier checks

def clean_name(name):
    return re.sub(r"\s+", " ", re.sub(r"[^\w .'-]", "", name or "")).strip()[:40]

def get_session(sid):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Unknown session. Start a new one.")
    if s.done:
        raise HTTPException(409, "This check is finished. Start a new one.")
    return s

@app.get("/")
def home():
    # A lia.html next to Lia.exe (or server.py) wins over the copy packed inside the exe, so the page can be edited without rebuilding.
    for page in (APP_DIR / "lia.html", HERE / "lia.html"):
        if page.exists():
            return FileResponse(page, headers={"Cache-Control": "no-store"})
    return {"message": "Lia server is running. Put lia.html next to server.py to serve the app here.", "backend": BACKEND}

@app.get("/health")
def health():
    info = {"ok": True, "backend": BACKEND, "stt": stt_available(), "tts": tts_available(), "face": face_available(), "photo": True}
    if BACKEND == "lmstudio":
        try:
            info["model"] = lm_model(check=True)
        except RuntimeError as e:
            info.update(ok=False, error=str(e))
        except Exception as e:
            info.update(ok=False, error=f"Cannot reach LM Studio at {LM_URL}. Open LM Studio and start its server. ({e})")
    return info

@app.post("/session")
def new_session(body: Optional[SessionIn] = None):
    if len(SESSIONS) > 200:
        SESSIONS.pop(next(iter(SESSIONS)))
    sid = uuid.uuid4().hex
    s = SESSIONS[sid] = LiaSession()
    s.crew = clean_name(body.name) if body else ""
    if body:
        s.profile = {k: str(v)[:400] for k, v in (body.profile or {}).items() if k in ("conditions", "allergies", "medications", "notes")}
        s.recent = [r for r in (body.recent or []) if isinstance(r, dict)][:5]
    hello = f"Hello, {s.crew}! I'm Lia. How are you feeling?" if s.crew else "Hello! I'm Lia. How are you feeling?"
    return {"session_id": sid, "greeting": hello}

@app.post("/reply")
def reply(body: ReplyIn):
    s = get_session(body.session_id)
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Please type something.")
    if not s.vitals_added:
        s.temp_c = body.temp_c if body.temp_c and 30 <= body.temp_c <= 45 else None
        s.pulse = body.pulse if body.pulse and 20 <= body.pulse <= 260 else None
    return s.reply(text)

class PhotoIn(BaseModel):
    session_id: str
    image: str

DESCRIBE_PROMPT = ("You help an astronaut's medical assistant. Describe only what is visible in this photo that could matter medically, "
                   "such as colour, swelling, bleeding, rash pattern or wound size, in two short sentences. Do not name a diagnosis. "
                   "If the photo does not show a body part, say so.")

@app.post("/photo")
def photo(body: PhotoIn):
    """Adds a photo to this check. Lia describes what is visible; the description is supporting information only."""
    s = get_session(body.session_id)
    if not body.image.startswith(("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/webp;base64,")) or len(body.image) > 4_000_000:
        raise HTTPException(400, "That photo could not be used. Try a smaller JPEG or PNG.")
    if BACKEND == "fake":
        note = "Demo mode: this photo was not analysed."
    elif BACKEND == "direct":
        raise HTTPException(501, "Photo analysis needs LM Studio with a model that can read images.")
    else:
        try:
            note = gen([msg("system", DESCRIBE_PROMPT), img_msg("user", "Describe the photo.", body.image)], 120).strip()
        except HTTPException as e:
            raise HTTPException(e.status_code, "Photo analysis failed. Check that the model loaded in LM Studio can read images (it shows a vision badge). " + str(e.detail))
    s.image, s.photo_note = body.image, re.sub(r"\s+", " ", note)[:600]
    return {"note": s.photo_note}

@app.post("/results")
def results(body: ReplyIn):
    """'Skip to results' button: use whatever Lia has heard so far."""
    s = get_session(body.session_id)
    if not any(m["role"] == "user" for m in s.history):
        raise HTTPException(400, "Tell Lia what is wrong first.")
    return s.finish()

_stt = None
def _has(module):
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False

def stt_available():
    return _has("faster_whisper")

def stt_model_path():
    """A copy of the speech model in the models folder is used first, so no internet is needed."""
    name = os.environ.get("LIA_STT_MODEL", "small.en")
    local = MODELS / ("faster-whisper-" + name)
    return str(local) if (local / "model.bin").exists() else name

def run_stt(audio):
    global _stt
    if _stt is None:
        from faster_whisper import WhisperModel
        _stt = WhisperModel(stt_model_path(), device="cpu", compute_type="int8")
    peak = float(abs(audio).max()) if len(audio) else 0.0
    if 0.002 < peak < 0.5:                      # quiet microphone: bring the level up
        audio = audio * (0.8 / peak)
    segs, _ = _stt.transcribe(audio, language="en", beam_size=3, vad_filter=True,
                              vad_parameters={"min_silence_duration_ms": 600, "speech_pad_ms": 400},
                              condition_on_previous_text=False,
                              initial_prompt="A patient describes symptoms to a medical assistant: headache, nausea, dizziness, "
                                             "chest pain, shortness of breath, fever, vomiting, bleeding, pulse, temperature.")
    return " ".join(x.text.strip() for x in segs).strip()

@app.post("/transcribe")
async def transcribe(request: Request):
    """Speech to text, fully on this computer. Body: raw 16 kHz mono 16-bit PCM audio."""
    raw = await request.body()
    if len(raw) < 6400:
        raise HTTPException(400, "Recording too short.")
    if not stt_available():
        raise HTTPException(501, "Offline speech recognition is not installed. Run: pip install faster-whisper")
    import numpy as np
    audio = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    try:
        text = await run_in_threadpool(run_stt, audio)
    except Exception as e:
        raise HTTPException(500, f"Speech recognition failed: {e}")
    return {"text": text}

_tts = None
_tts_lock = threading.Lock()
_tts_cache = {}
TTS_MODEL = Path(os.environ.get("LIA_TTS_MODEL", MODELS / "kokoro-v1.0.onnx"))
TTS_VOICES = Path(os.environ.get("LIA_TTS_VOICES", MODELS / "voices-v1.0.bin"))
TTS_VOICE = os.environ.get("LIA_VOICE", "af_heart")

def tts_available():
    return _has("kokoro_onnx") and TTS_MODEL.exists() and TTS_VOICES.exists()

def run_tts(text):
    global _tts
    import io, wave
    import numpy as np
    with _tts_lock:
        if text in _tts_cache:
            return _tts_cache[text]
        if _tts is None:
            from kokoro_onnx import Kokoro
            _tts = Kokoro(str(TTS_MODEL), str(TTS_VOICES))
        samples, rate = _tts.create(text, voice=TTS_VOICE, speed=1.05, lang="en-us")
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
        if len(_tts_cache) > 60:
            _tts_cache.pop(next(iter(_tts_cache)))
        _tts_cache[text] = buf.getvalue()
        return _tts_cache[text]

@app.post("/speak")
async def speak(body: AnalyzeIn):
    """Text to speech, fully on this computer. Returns a WAV file."""
    text = body.text.strip()[:400]
    if not text:
        raise HTTPException(400, "Nothing to say.")
    if not tts_available():
        raise HTTPException(501, "Natural voice is not installed. Run: pip install kokoro-onnx, and put the two model files in the models folder.")
    try:
        wav = await run_in_threadpool(run_tts, text)
    except Exception as e:
        raise HTTPException(500, f"Voice failed: {e}")
    return Response(content=wav, media_type="audio/wav")

def warm_up():
    """Loads the voice and speech models once at start-up, so the first reply is not slow."""
    try:
        if tts_available():
            run_tts("Hello, I'm Lia. How are you feeling?")
        if stt_available():
            import numpy as np
            run_stt(np.zeros(16000, dtype=np.float32))
    except Exception:
        pass

@app.post("/analyze")
def analyze(body: AnalyzeIn):
    """Single-shot check for the test script: send a whole case, get conditions and urgency back."""
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Empty case.")
    return final_answer(text, text)

# ---------------------------------------------------------------- face sign-in (fully on this computer)
# Recognizes which crew member is at the camera so Lia can greet them and keep their checks separate.
# It is a convenience, NOT a lock: a photo of someone can fool it. No pictures are saved, only a list of
# numbers per person (a "face print") in crew_faces.json next to the app. Deleting that file forgets everyone.
FACE_DET = MODELS / "face_detection_yunet_2023mar.onnx"
FACE_REC = MODELS / "face_recognition_sface_2021dec.onnx"
CREW_FILE = APP_DIR / "crew_faces.json"
FACE_MATCH = float(os.environ.get("LIA_FACE_MATCH", "0.45"))   # 0 to 1. Higher = stricter. OpenCV's own cutoff is 0.363.
FACE_SAMPLES = 8
_face_lock = threading.Lock()
_face_models = None

def face_available():
    return _has("cv2") and FACE_DET.exists() and FACE_REC.exists()

def load_crew():
    try:
        return json.loads(CREW_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_crew(crew):
    CREW_FILE.write_text(json.dumps(crew), encoding="utf-8")

def face_print(raw):
    """Returns (face print, problem). The face print is None when there is a problem."""
    global _face_models
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "That picture could not be read."
    h, w = img.shape[:2]
    if max(h, w) > 640:
        k = 640 / max(h, w)
        img = cv2.resize(img, (round(w * k), round(h * k)))
        h, w = img.shape[:2]
    with _face_lock:
        if _face_models is None:
            _face_models = (cv2.FaceDetectorYN.create(str(FACE_DET), "", (w, h), 0.8, 0.3, 50),
                            cv2.FaceRecognizerSF.create(str(FACE_REC), ""))
        det, rec = _face_models
        det.setInputSize((w, h))
        faces = det.detect(img)[1]
        if faces is None or not len(faces):
            return None, "I can't see a face. Look at the camera."
        faces = sorted(faces, key=lambda f: -f[2])                 # widest (closest) face first
        if len(faces) > 1 and faces[1][2] > faces[0][2] * 0.6:     # someone far behind you is ignored
            return None, "I see more than one face. One person at a time, please."
        face = faces[0]
        if face[2] < 56:
            return None, "Move a little closer to the camera."
        vec = rec.feature(rec.alignCrop(img, face)).flatten().astype(float)
    return vec / (float(np.linalg.norm(vec)) or 1.0), ""

def best_match(vec, crew):
    import numpy as np
    best, score = "", -1.0
    for name, prints in crew.items():
        sc = max((float(np.dot(vec, np.asarray(p))) for p in prints), default=-1.0)
        if sc > score:
            best, score = name, sc
    return best, score

def need_face():
    if not face_available():
        raise HTTPException(501, "Face recognition is not installed. Run: pip install opencv-python-headless, "
                                 "and put the two face model files in the models folder.")

@app.post("/face/identify")
async def face_identify(request: Request):
    """Body: one JPEG picture from the camera. Says who it is, if that person was added before."""
    need_face()
    vec, problem = await run_in_threadpool(face_print, await request.body())
    if vec is None:
        return {"face": False, "match": False, "message": problem}
    name, score = best_match(vec, load_crew())
    ok = score >= FACE_MATCH
    return {"face": True, "match": ok, "name": name if ok else "", "score": round(max(score, 0.0), 3)}

@app.post("/face/enroll")
async def face_enroll(request: Request, name: str = ""):
    """Body: one JPEG picture. Adds this face print under the given name (send a few pictures for better results)."""
    need_face()
    name = clean_name(name)
    if not name:
        raise HTTPException(400, "Type your name first.")
    vec, problem = await run_in_threadpool(face_print, await request.body())
    if vec is None:
        raise HTTPException(422, problem)
    with _face_lock:
        crew = load_crew()
        other, score = best_match(vec, {n: p for n, p in crew.items() if n != name})
        if score >= FACE_MATCH:
            raise HTTPException(409, f"This face is already saved as {other}. Remove {other} first to change the name.")
        crew[name] = (crew.get(name, []) + [[round(float(x), 5) for x in vec]])[-FACE_SAMPLES:]
        save_crew(crew)
    return {"name": name, "samples": len(crew[name])}

@app.get("/face/crew")
def face_crew():
    return {"crew": sorted(load_crew())}

@app.delete("/face/crew")
def face_forget(name: str):
    with _face_lock:
        crew = load_crew()
        if crew.pop(name, None) is None:
            raise HTTPException(404, "No crew member with that name.")
        save_crew(crew)
    return {"crew": sorted(crew)}

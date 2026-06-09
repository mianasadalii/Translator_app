# ============================================================
#  English ↔ Urdu Translator  (v3 — Fast + Streaming)
#  Optimized for CPU · Real-time progress · No hangs
# ============================================================

import re
import gradio as gr
from transformers import MarianMTModel, MarianTokenizer

# ------------------------------------------------------------
# 1.  Model Config
# ------------------------------------------------------------

MODEL_EN_UR = "Helsinki-NLP/opus-mt-en-ur"
MODEL_UR_EN = "Helsinki-NLP/opus-mt-ur-en"

# ------------------------------------------------------------
# 2.  Load Both Models Once at Startup
# ------------------------------------------------------------

print("⏳ Loading EN→UR model…")
tok_en_ur   = MarianTokenizer.from_pretrained(MODEL_EN_UR)
model_en_ur = MarianMTModel.from_pretrained(MODEL_EN_UR)
model_en_ur.eval()
print("✅ EN→UR ready.")

print("⏳ Loading UR→EN model…")
tok_ur_en   = MarianTokenizer.from_pretrained(MODEL_UR_EN)
model_ur_en = MarianMTModel.from_pretrained(MODEL_UR_EN)
model_ur_en.eval()
print("✅ UR→EN ready. App is live.")

# ------------------------------------------------------------
# 3.  Sentence Splitter
# ------------------------------------------------------------

def split_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?۔؟!])\s+', text.strip())
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > 400:
            sub = re.split(r'(?<=[,،;])\s+', part)
            result.extend([s.strip() for s in sub if s.strip()])
        else:
            result.append(part)
    return result

# ------------------------------------------------------------
# 4.  Single-Chunk Translator
# ------------------------------------------------------------

def translate_chunk(text: str, tokenizer, model) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    import torch
    with torch.no_grad():
        output = model.generate(
            **inputs,
            num_beams=2,
            max_new_tokens=256,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ------------------------------------------------------------
# 5.  Main Translate Function  (streaming generator)
# ------------------------------------------------------------

def translate_text(text: str, direction: str):
    if not text or not text.strip():
        yield "Please enter some text before translating."
        return

    text = text.strip()

    if direction == "English → Urdu":
        tokenizer, model = tok_en_ur, model_en_ur
    else:
        tokenizer, model = tok_ur_en, model_ur_en

    sentences = split_sentences(text)
    total     = len(sentences)
    translated_so_far = []

    try:
        for i, sentence in enumerate(sentences, 1):
            progress_line = f"\n\n⏳ Translating {i}/{total}…"
            if translated_so_far:
                yield " ".join(translated_so_far) + progress_line
            else:
                yield progress_line

            result = translate_chunk(sentence, tokenizer, model)
            translated_so_far.append(result)
            yield " ".join(translated_so_far)

        yield " ".join(translated_so_far)

    except Exception as e:
        yield f"Error: {str(e)}\n\nTry shortening the input or refreshing the page."

# ------------------------------------------------------------
# 6.  Gradio UI
# ------------------------------------------------------------

EXAMPLES = [
    ["Artificial intelligence is changing the world rapidly.", "English → Urdu"],
    ["پاکستان ایک خوبصورت ملک ہے۔ یہاں کے لوگ بہت مہمان نواز ہیں۔", "Urdu → English"],
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

/* ── Hide Gradio footer ── */
footer { display: none !important; }
.gradio-container > .footer { display: none !important; }
#footer { display: none !important; }
.svelte-1rjryqp { display: none !important; }

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

body {
    background: #1a1a2e !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 0 20px 40px !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    background: transparent !important;
}

/* ── Title bar ── */
.title-bar {
    text-align: center;
    padding: 28px 0 6px;
    border-bottom: 1px solid #2a2a45;
    margin-bottom: 8px;
}

.title-bar h1 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: #e0e0f0 !important;
    margin: 0 !important;
    letter-spacing: 0.01em !important;
}

.title-bar h1 .pk-badge {
    font-size: 0.75rem;
    font-weight: 700;
    color: #01a884;
    background: rgba(1,168,132,0.12);
    border: 1px solid rgba(1,168,132,0.3);
    border-radius: 4px;
    padding: 1px 6px;
    margin-left: 6px;
    vertical-align: middle;
    letter-spacing: 0.08em;
}

/* ── Subtitle (dynamic, updated via JS) ── */
#dynamic-subtitle {
    font-size: 0.82rem;
    color: #5a5a80;
    margin: 6px 0 14px 0;
    text-align: center;
}

/* ── Direction toggle ── */
.dir-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 14px;
}

.gradio-container .gr-radio-group,
.gradio-container fieldset {
    background: #12122a !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 8px !important;
    padding: 3px !important;
    display: inline-flex !important;
    gap: 3px !important;
    box-shadow: none !important;
}

.gradio-container .gr-radio-group label,
.gradio-container fieldset label {
    padding: 7px 22px !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #5a5a80 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    border: none !important;
    background: transparent !important;
    white-space: nowrap !important;
}

.gradio-container .gr-radio-group label:has(input:checked),
.gradio-container fieldset label:has(input:checked) {
    background: #2a2a45 !important;
    color: #b0b0d8 !important;
}

.gradio-container .gr-radio-group input[type="radio"],
.gradio-container fieldset input[type="radio"] {
    display: none !important;
}

/* ── Panels row ── */
#panels-row {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    align-items: stretch !important;
    flex-wrap: nowrap !important;
    background: #12122a !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    margin-bottom: 12px !important;
}

#panels-row > * {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}

#left-panel {
    border-right: 1px solid #2a2a45 !important;
}

.panel-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #6060a0;
    padding: 10px 14px 6px;
    border-bottom: 1px solid #1e1e38;
    background: #0f0f24;
    min-height: 34px;
}

/* ── Textareas ── */
.gradio-container textarea {
    background: #12122a !important;
    border: none !important;
    border-radius: 0 !important;
    color: #d0d0ec !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    line-height: 1.7 !important;
    padding: 14px 16px !important;
    resize: none !important;
    width: 100% !important;
    box-shadow: none !important;
}

.gradio-container textarea:focus {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}

/* ── Urdu text styling classes (applied via JS) ── */
.urdu-text textarea {
    direction: rtl !important;
    font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif !important;
    font-size: 1.05rem !important;
    line-height: 2.2 !important;
    color: #c0c0e8 !important;
}

.ltr-text textarea {
    direction: ltr !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    line-height: 1.7 !important;
    color: #d0d0ec !important;
}

/* Hide auto-generated Gradio labels */
.gradio-container label > span.svelte-1b6s6s,
.gradio-container .label-wrap > span,
.gradio-container .block > label > span:first-child {
    display: none !important;
}

/* Hide block borders/backgrounds injected by Gradio */
.gradio-container .block {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* ── Buttons row ── */
#btn-row {
    display: flex !important;
    gap: 10px !important;
    margin-bottom: 16px !important;
}

#clear-btn {
    background: #2a2a45 !important;
    color: #9090c0 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none !important;
    border-radius: 8px !important;
    height: 42px !important;
    transition: all 0.15s !important;
    flex: 1 !important;
}

#clear-btn:hover {
    background: #35355a !important;
    color: #b0b0d8 !important;
}

#translate-btn {
    background: #e8600a !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 8px !important;
    height: 42px !important;
    transition: opacity 0.15s, transform 0.1s !important;
    flex: 2 !important;
}

#translate-btn:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

#translate-btn:active { transform: translateY(0) !important; }

#flag-btn {
    background: #2a2a45 !important;
    color: #9090c0 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none !important;
    border-radius: 8px !important;
    height: 42px !important;
    transition: all 0.15s !important;
    flex: 1 !important;
}

#flag-btn:hover {
    background: #35355a !important;
    color: #b0b0d8 !important;
}

/* ── Examples card ── */
.ex-card {
    background: #12122a;
    border: 1px solid #2a2a45;
    border-radius: 10px;
    padding: 14px 18px 12px;
}

.gradio-container .examples table {
    background: transparent !important;
    border-collapse: separate !important;
    border-spacing: 0 4px !important;
    width: 100% !important;
}

.gradio-container .examples thead th {
    background: transparent !important;
    color: #4a4a70 !important;
    font-size: 0.66rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    padding: 2px 10px 6px !important;
}

.gradio-container .examples tbody td {
    background: #0f0f24 !important;
    color: #5a5a8a !important;
    font-size: 0.84rem !important;
    border-top: 1px solid #1e1e38 !important;
    border-bottom: 1px solid #1e1e38 !important;
    border-left: none !important;
    border-right: none !important;
    padding: 9px 10px !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
}

.gradio-container .examples tbody td:first-child {
    border-left: 1px solid #1e1e38 !important;
    border-radius: 6px 0 0 6px !important;
}

.gradio-container .examples tbody td:last-child {
    border-right: 1px solid #1e1e38 !important;
    border-radius: 0 6px 6px 0 !important;
}

.gradio-container .examples tbody tr:hover td {
    background: #1a1a35 !important;
    color: #8888b8 !important;
}

/* ── Scrollbar ── */
textarea::-webkit-scrollbar { width: 4px; }
textarea::-webkit-scrollbar-track { background: transparent; }
textarea::-webkit-scrollbar-thumb { background: #2a2a45; border-radius: 2px; }
"""

# ------------------------------------------------------------
# Direction-change helper functions
# ------------------------------------------------------------

def update_ui_for_direction(direction: str):
    """Returns updated Textbox components for input and output based on direction."""
    if direction == "English → Urdu":
        new_input = gr.Textbox(
            placeholder="Enter English text...",
            label="",
            lines=10,
            max_lines=30,
            elem_id="input-box",
            elem_classes=["ltr-text"],
        )
        new_output = gr.Textbox(
            placeholder="Urdu translation will appear here...",
            label="",
            lines=10,
            max_lines=30,
            interactive=False,
            elem_id="output-box",
            elem_classes=["urdu-text"],
        )
    else:
        new_input = gr.Textbox(
            placeholder="اردو متن درج کریں...",
            label="",
            lines=10,
            max_lines=30,
            elem_id="input-box",
            elem_classes=["urdu-text"],
        )
        new_output = gr.Textbox(
            placeholder="English translation will appear here...",
            label="",
            lines=10,
            max_lines=30,
            interactive=False,
            elem_id="output-box",
            elem_classes=["ltr-text"],
        )
    return new_input, new_output


def get_left_label(direction: str) -> str:
    return (
        '<div class="panel-label">English Input</div>'
        if direction == "English → Urdu"
        else '<div class="panel-label" style="direction:rtl; font-family:\'Noto Nastaliq Urdu\',serif;">اردو ان پٹ</div>'
    )


def get_right_label(direction: str) -> str:
    return (
        '<div class="panel-label" style="direction:rtl; font-family:\'Noto Nastaliq Urdu\',serif;">اردو ترجمہ</div>'
        if direction == "English → Urdu"
        else '<div class="panel-label">English Translation</div>'
    )


def get_subtitle(direction: str) -> str:
    if direction == "English → Urdu":
        return '<p id="dynamic-subtitle">Enter any English text and the AI will translate it into Urdu.</p>'
    else:
        return '<p id="dynamic-subtitle">اردو متن درج کریں اور AI اسے انگریزی میں ترجمہ کرے گا۔</p>'


# ------------------------------------------------------------
# 6.  Gradio UI
# ------------------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Base(),
    css=CSS,
    title="English ↔ Urdu Translator PK",
) as demo:

    # ── Title bar ────────────────────────────────────────────
    gr.HTML("""
    <div class="title-bar">
        <h1>English <span style="color:#6060a0;">↔</span> Urdu Translator <span class="pk-badge">PK</span></h1>
    </div>
    """)

    subtitle_html = gr.HTML(
        '<p id="dynamic-subtitle">Enter any English text and the AI will translate it into Urdu.</p>'
    )

    # ── Direction toggle ─────────────────────────────────────
    with gr.Row(elem_classes="dir-wrap"):
        direction = gr.Radio(
            choices=["English → Urdu", "Urdu → English"],
            value="English → Urdu",
            label="",
        )

    # ── Side-by-side panels ──────────────────────────────────
    with gr.Row(elem_id="panels-row"):

        with gr.Column(elem_id="left-panel"):
            left_label = gr.HTML('<div class="panel-label">English Input</div>')
            input_box = gr.Textbox(
                label="",
                placeholder="Enter English text...",
                lines=10,
                max_lines=30,
                elem_id="input-box",
                elem_classes=["ltr-text"],
            )

        with gr.Column(elem_id="right-panel"):
            right_label = gr.HTML(
                '<div class="panel-label" style="direction:rtl; font-family:\'Noto Nastaliq Urdu\',serif;">اردو ترجمہ</div>'
            )
            output_box = gr.Textbox(
                label="",
                placeholder="Urdu translation will appear here...",
                lines=10,
                max_lines=30,
                interactive=False,
                elem_id="output-box",
                elem_classes=["urdu-text"],
            )

    # ── Buttons ──────────────────────────────────────────────
    with gr.Row(elem_id="btn-row"):
        clear_btn     = gr.Button("Clear",  elem_id="clear-btn",     scale=1)
        translate_btn = gr.Button("Submit", elem_id="translate-btn", scale=2)
        flag_btn      = gr.Button("Flag",   elem_id="flag-btn",      scale=1)

    # ── Examples ─────────────────────────────────────────────
    with gr.Group(elem_classes="ex-card"):
        gr.HTML('<div class="panel-label" style="border:none; background:transparent; margin-bottom:6px;">Examples</div>')
        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_box, direction],
            label="",
        )

    # ── Direction change: update labels, placeholders, text direction ──
    direction.change(
        fn=lambda d: (
            get_subtitle(d),
            get_left_label(d),
            get_right_label(d),
            gr.Textbox(
                placeholder="Enter English text..." if d == "English → Urdu" else "اردو متن درج کریں...",
                label="",
                lines=10,
                max_lines=30,
                elem_id="input-box",
                elem_classes=["ltr-text" if d == "English → Urdu" else "urdu-text"],
            ),
            gr.Textbox(
                placeholder="Urdu translation will appear here..." if d == "English → Urdu" else "English translation will appear here...",
                label="",
                lines=10,
                max_lines=30,
                interactive=False,
                elem_id="output-box",
                elem_classes=["urdu-text" if d == "English → Urdu" else "ltr-text"],
            ),
        ),
        inputs=[direction],
        outputs=[subtitle_html, left_label, right_label, input_box, output_box],
    )

    # ── Translation ───────────────────────────────────────────
    translate_btn.click(
        fn=translate_text,
        inputs=[input_box, direction],
        outputs=output_box,
    )
    input_box.submit(
        fn=translate_text,
        inputs=[input_box, direction],
        outputs=output_box,
    )

    # ── Clear ────────────────────────────────────────────────
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[input_box, output_box],
    )

    # ── Flag (no-op placeholder) ─────────────────────────────
    flag_btn.click(
        fn=lambda: None,
        inputs=[],
        outputs=[],
    )

# ------------------------------------------------------------
# 7.  Launch
# ------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(share=True, debug=True)

import gradio as gr
import httpx
_original_get = httpx.get
def _patched_get(url, **kwargs):
    kwargs['verify'] = False
    return _original_get(url, **kwargs)
httpx.get = _patched_get
import time
import subprocess
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.routing.pipeline import run_pipeline
from src.evaluation.feedback_store import save_feedback, get_feedback_stats
from src.auth.auth import authenticate, get_access_filter
from src.auth.rbac import apply_rbac_filter, get_rbac_sql_clause
from src.auth.audit import log_action, get_audit_stats
from src.config import MODEL
from src.utils.logger import get_logger

logger = get_logger("app")

# ── GPU INFO ──────────────────────────────────────────
def get_gpu_metrics():
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        values = result.stdout.strip().split(", ")
        return {
            "temp":     values[0].strip(),
            "power":    values[1].strip(),
            "vram":     values[2].strip(),
            "gpu_util": values[3].strip(),
        }
    except:
        return {"temp": "N/A", "power": "N/A", "vram": "N/A", "gpu_util": "N/A"}

# ── SESSION STATE ─────────────────────────────────────
current_user   = {}
current_result = {}

# ── LOGIN ─────────────────────────────────────────────
def login(username, password):
    global current_user
    if not username or not password:
        return "Please enter username and password.", gr.update(visible=True), gr.update(visible=False)

    user = authenticate(username, password)
    if not user:
        return " Invalid username or password.", gr.update(visible=True), gr.update(visible=False)

    current_user = user
    logger.info(f"User logged in: {username} ({user['role']})")
    return f" Welcome **{username}** ({user['role'].upper()})", gr.update(visible=False), gr.update(visible=True)

def logout():
    global current_user, current_result
    current_user   = {}
    current_result = {}
    return "", gr.update(visible=True), gr.update(visible=False), "", "", "", "", ""

# ── PIPELINE ──────────────────────────────────────────
def ask_question(question):
    global current_result, current_user

    if not current_user:
        return "Please login first.", "", "", "", ""

    if not question.strip():
        return "", "", "", "", ""

    logger.info(f"Question from {current_user.get('username')}: {question}")

    # RBAC check
    rbac = apply_rbac_filter(current_user, question)
    if not rbac["allowed"]:
        log_action(current_user, question, "BLOCKED", False)
        return rbac["reason"], "", "", "", ""

    # Run pipeline
    t0      = time.time()
    result  = run_pipeline(question, user=current_user)
    elapsed = round(time.time() - t0, 2)

    current_result = {
        "question": question,
        "answer":   result["answer"],
        "category": result["categories"][0] if result["categories"] else "TEXT",
        "sources":  result["sources"],
        "latency":  elapsed,
        "model":    MODEL,
    }

    log_action(current_user, question, current_result["category"], True)

    gpu    = get_gpu_metrics()
    stats  = get_feedback_stats()
    astats = get_audit_stats()

    sources_text = " | ".join(result["sources"]) if result["sources"] else "N/A"

    metrics_text = f"""###  Pipeline Metrics

| Metric | Value |
|--------|-------|
| Category | `{current_result['category']}` |
| Complex | {result['is_complex']} |
| Latency | {elapsed} sec |
| Sources | {len(result['sources'])} |
| Model | `{MODEL}` |
| User | `{current_user.get('username')}` |
| Role | `{current_user.get('role', '').upper()}` |

### GPU Metrics

| Metric | Value |
|--------|-------|
| Temperature | {gpu['temp']} °C |
| Power Draw | {gpu['power']} W |
| VRAM Used | {gpu['vram']} MB |
| GPU Util | {gpu['gpu_util']} % |

###  Feedback Stats

| Metric | Value |
|--------|-------|
| Total | {stats.get('total', 0)} |
| 👍 Positive | {stats.get('positive', 0)} |
| 👎 Negative | {stats.get('negative', 0)} |
| Avg Latency | {stats.get('avg_latency', 0)} sec |

###  Audit Stats

| Metric | Value |
|--------|-------|
| Total Actions | {astats.get('total', 0)} |
| Allowed | {astats.get('allowed', 0)} |
| Denied | {astats.get('denied', 0)} |
"""

    return result["answer"], sources_text, metrics_text, ""


def thumbs_up(comment):
    global current_result
    if not current_result:
        return "Please ask a question first."
    save_feedback(
        question        = current_result["question"],
        category        = current_result["category"],
        answer          = current_result["answer"],
        sources         = current_result["sources"],
        score           = 1,
        model_used      = current_result["model"],
        latency_sec     = current_result["latency"],
        top_chunk_score = 0.0,
        chunks_count    = len(current_result["sources"]),
        comment         = comment
    )
    return "👍 Thank you for your positive feedback!", ""

def thumbs_down(comment):
    global current_result
    if not current_result:
        return "Please ask a question first."
    save_feedback(
        question        = current_result["question"],
        category        = current_result["category"],
        answer          = current_result["answer"],
        sources         = current_result["sources"],
        score           = -1,
        model_used      = current_result["model"],
        latency_sec     = current_result["latency"],
        top_chunk_score = 0.0,
        chunks_count    = len(current_result["sources"]),
        comment         = comment
    )
    return "👎 Thank you for your feedback. We will improve!", ""

# ── INTERFACE ─────────────────────────────────────────
with gr.Blocks(
    title="Nexora HR Assistant",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
        footer { display: none !important; }
        .header-box { text-align: center; padding: 24px 0 8px 0; }
        .answer-box textarea {
            direction: auto !important;
            text-align: start !important;
            unicode-bidi: plaintext !important;
            font-family: Arial, sans-serif;
            font-size: 15px;
            line-height: 1.7;
        }
        .sources-box textarea {
            direction: auto !important;
            text-align: start !important;
            unicode-bidi: plaintext !important;
            font-size: 13px;
        }
        .login-box {
            max-width: 500px;
            margin: 0 auto;
            padding: 24px;
        }
    """
) as demo:

    # ── HEADER ────────────────────────────────────────
    gr.HTML("""
        <div class="header-box">
            <h1 style="font-size:2rem; font-weight:700; color:#1e3a5f; margin:0;">
                🏢 Nexora HR Assistant
            </h1>
            <p style="color:#666; margin:6px 0 0 0; font-size:0.95rem;">
                Powered by RAG Pipeline &nbsp;·&nbsp; Qwen2.5 Q4 &nbsp;·&nbsp; Tesla T4 GPU
            </p>
        </div>
    """)

    # ── LOGIN PANEL ───────────────────────────────────
    with gr.Group(visible=True) as login_panel:
        with gr.Column(elem_classes=["login-box"]):
            gr.Markdown("###  Login")
            username_input = gr.Textbox(
                label="Username",
                placeholder="e.g. paul.davis",
            )
            password_input = gr.Textbox(
                label="Password",
                type="password",
                placeholder="Enter your password"
            )
            login_btn    = gr.Button("Login", variant="primary", size="lg")
            login_status = gr.Markdown("")

    # ── MAIN PANEL ────────────────────────────────────
    with gr.Group(visible=False) as main_panel:

        with gr.Row():
            user_info  = gr.Markdown("")
            logout_btn = gr.Button(" Logout", variant="stop", scale=0)

        with gr.Row():
            # ── LEFT — Chat ───────────────────────────
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label=" Your Question",
                    placeholder="Ask me anything about HR policies or employee information...",
                    lines=3,
                    elem_classes=["question-box"]
                )
                ask_btn = gr.Button(" Ask", variant="primary", size="lg")

                answer_output = gr.Textbox(
                    label=" Answer",
                    lines=12,
                    interactive=False,
                    elem_classes=["answer-box"],
                    show_copy_button=True
                )
                sources_output = gr.Textbox(
                    label=" Sources",
                    lines=2,
                    interactive=False,
                    elem_classes=["sources-box"]
                )

                gr.Markdown("---\n**Was this answer helpful?**")
                with gr.Row():
                    comment_input   = gr.Textbox(
                        placeholder="Optional comment...",
                        label=" Comment",
                        scale=3
                    )
                    thumbs_up_btn   = gr.Button("👍 Yes", variant="primary", scale=1)
                    thumbs_down_btn = gr.Button("👎 No",  variant="stop",    scale=1)
                feedback_status = gr.Markdown("")

            # ── RIGHT — Metrics ───────────────────────
            with gr.Column(scale=2):
                metrics_output = gr.Markdown(
                    value="*Metrics will appear after your first question.*"
                )

    # ── EVENTS ────────────────────────────────────────
    def login_and_update(username, password):
        msg, lp, mp = login(username, password)
        user_md = f"👤 **{username}** | Role: **{current_user.get('role','').upper()}** | Dept: **{current_user.get('department','')}**" if current_user else ""
        return msg, lp, mp, user_md

    login_btn.click(
        fn=login_and_update,
        inputs=[username_input, password_input],
        outputs=[login_status, login_panel, main_panel, user_info]
    )

    username_input.submit(
        fn=login_and_update,
        inputs=[username_input, password_input],
        outputs=[login_status, login_panel, main_panel, user_info]
    )

    logout_btn.click(
        fn=logout,
        inputs=[],
        outputs=[login_status, login_panel, main_panel,
                 answer_output, sources_output, metrics_output, feedback_status, user_info]
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output, metrics_output, feedback_status]
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output, metrics_output, feedback_status]
    )

    thumbs_up_btn.click(
        fn=thumbs_up,
        inputs=[comment_input],
        outputs=[feedback_status, comment_input]
    )

    thumbs_down_btn.click(
        fn=thumbs_down,
        inputs=[comment_input],
        outputs=[feedback_status, comment_input]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        ssl_certfile="/root/project/cert.pem",
        ssl_keyfile="/root/project/key.pem",
        share=False
    )

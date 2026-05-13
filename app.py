import gradio as gr
import time
import subprocess
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.routing.pipeline import run_pipeline
from src.evaluation.feedback_store import save_feedback, get_feedback_stats
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
            "temp":    values[0].strip(),
            "power":   values[1].strip(),
            "vram":    values[2].strip(),
            "gpu_util":values[3].strip(),
        }
    except:
        return {"temp": "N/A", "power": "N/A", "vram": "N/A", "gpu_util": "N/A"}

# ── PIPELINE STATE ────────────────────────────────────
current_result = {}

def ask_question(question):
    global current_result

    if not question.strip():
        return "", "", "", "", ""

    logger.info(f"New question: {question}")
    t0     = time.time()
    result = run_pipeline(question)
    elapsed = round(time.time() - t0, 2)

    current_result = {
        "question":      question,
        "answer":        result["answer"],
        "category":      result["categories"][0] if result["categories"] else "TEXT",
        "sources":       result["sources"],
        "is_complex":    result["is_complex"],
        "sub_questions": result["sub_questions"],
        "latency":       elapsed,
        "model":         MODEL,
    }

    # GPU metrics
    gpu = get_gpu_metrics()

    # Feedback stats
    stats = get_feedback_stats()

    # Format sources
    sources_text = " | ".join(result["sources"]) if result["sources"] else "N/A"

    # Format metrics
    category    = current_result["category"]
    chunks_info = f"{len(result['sources'])} sources"
    complex_info= "Yes" if result["is_complex"] else "No"

    metrics_text = f"""
###  Pipeline Metrics

| Metric | Value |
|--------|-------|
|  Category | `{category}` |
|  Complex | {complex_info} |
| Latency | {elapsed} sec |
|  Sources | {chunks_info} |
|  Model | `{MODEL}` |

###  GPU Metrics

| Metric | Value |
|--------|-------|
| Temperature | {gpu['temp']} °C |
|  Power Draw | {gpu['power']} W |
|  VRAM Used | {gpu['vram']} MB |
|  GPU Util | {gpu['gpu_util']} % |

###  Feedback Stats

| Metric | Value |
|--------|-------|
| Total Feedbacks | {stats.get('total', 0)} |
| 👍 Positive | {stats.get('positive', 0)} |
| 👎 Negative | {stats.get('negative', 0)} |
| Avg Latency | {stats.get('avg_latency', 0)} sec |
"""

    return result["answer"], sources_text, metrics_text, "", ""


def thumbs_up(comment):
    global current_result
    if not current_result:
        return "⚠️ Please ask a question first."
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
    return "✅ Thank you for your positive feedback!"


def thumbs_down(comment):
    global current_result
    if not current_result:
        return "⚠️ Please ask a question first."
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
    return "❌ Thank you for your feedback. We will improve!"


# ── INTERFACE ─────────────────────────────────────────
with gr.Blocks(
    title="Nexora HR Assistant",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
        .header { text-align: center; padding: 20px; }
        .answer-box textarea { 
            direction: auto !important; 
            text-align: start !important;
            unicode-bidi: plaintext !important;
            font-family: Arial, sans-serif;
            font-size: 15px;
            line-height: 1.6;
        }
        .sources-box textarea {
            direction: auto !important;
            text-align: start !important;
            unicode-bidi: plaintext !important;
        }
        footer { display: none !important; }
    """

) as demo:

    gr.HTML("""
        <div class="header">
            <h1>🏢 Nexora HR Assistant</h1>
            <p style="color: gray;">Powered by RAG Pipeline · Qwen2.5:7b · Tesla T4 GPU</p>
        </div>
    """)

    with gr.Row():
        # ── LEFT COLUMN — Chat ────────────────────────
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label=" Your Question",
                placeholder="e.g. What is the remote work policy?",
                lines=2
            )
            ask_btn = gr.Button("Ask", variant="primary", size="lg")

            answer_output = gr.Textbox(
                label="Answer",
                lines=10,
                interactive=False,
                elem_classes=["answer-box"]
            )
            sources_output = gr.Textbox(
                label="Sources",
                lines=2,
                interactive=False,
                elem_classes=["sources-box"]
            )
            gr.Markdown("### Was this answer helpful?")
            with gr.Row():
                comment_input = gr.Textbox(
                    placeholder="Optional comment...",
                    label="💭 Comment",
                    scale=3
                )
                thumbs_up_btn   = gr.Button("👍 Yes", variant="primary", scale=1)
                thumbs_down_btn = gr.Button("👎 No",  variant="stop",    scale=1)

            feedback_status = gr.Markdown("")

        # ── RIGHT COLUMN — Metrics ────────────────────
        with gr.Column(scale=2):
            metrics_output = gr.Markdown(
                value="*Metrics will appear after your first question.*",
                label=" Metrics",
                elem_classes=["metrics-box"]
            )

    # ── EVENTS ────────────────────────────────────────
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output, metrics_output, comment_input, feedback_status]
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output, metrics_output, comment_input, feedback_status]
    )

    thumbs_up_btn.click(
        fn=thumbs_up,
        inputs=[comment_input],
        outputs=[feedback_status]
    )

    thumbs_down_btn.click(
        fn=thumbs_down,
        inputs=[comment_input],
        outputs=[feedback_status]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )

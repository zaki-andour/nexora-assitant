import os
import re
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
POLICIES_DIR            = "hr_dataset/policies"
OUTPUT_FILE             = "hr_dataset/processed/text/chunks.json"
CHUNK_OVERLAP_SENTENCES = 2
MAX_SECTION_TOKENS      = 450
CHARS_PER_TOKEN         = 4

# ── DATA CLASS ────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:        str
    text:            str
    source_file:     str
    document_type:   str = "policy"
    document_title:  str = ""
    doc_ref:         str = ""
    effective_date:  str = ""
    doc_owner:       str = ""
    section_number:  str = ""
    section_title:   str = ""
    policy_topic:    str = ""
    access_level:    str = "all_employees"
    chunk_index:     int = 0
    total_chunks:    int = 0
    prev_chunk_id:   Optional[str] = None
    next_chunk_id:   Optional[str] = None
    word_count:      int = 0
    sentences:       list = field(default_factory=list)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def infer_topic(filename: str) -> str:
    mapping = {
        "annual_leave":    "annual_leave",
        "remote_work":     "remote_work",
        "maternity":       "parental_leave",
        "paternity":       "parental_leave",
        "expense":         "expenses",
        "performance":     "performance",
        "grievance":       "grievance",
        "onboarding":      "onboarding",
        "disciplinary":    "disciplinary",
        "sick":            "sick_leave",
        "code_of_conduct": "conduct",
        "data_protection": "data_protection",
        "travel":          "travel",
        "redundancy":      "redundancy",
        "flexible":        "flexible_working",
        "equal":           "equal_opportunities",
        "it_acceptable":   "it_security",
        "learning":        "learning_development",
        "whistleblowing":  "whistleblowing",
    }
    fname = filename.lower()
    for key, topic in mapping.items():
        if key in fname:
            return topic
    return "general"


def infer_access_level(topic: str, section_number: str) -> str:
    restricted_topics = ["salary", "compensation", "payroll"]
    if any(t in topic for t in restricted_topics):
        return "hr_professional"
    return "all_employees"


def split_into_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def estimate_tokens(text: str) -> int:
    # ── TODO: replace with real tokenizer once embedding model is chosen ──
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    # return len(tokenizer.encode(text))
    return len(text) // CHARS_PER_TOKEN


def recursive_split(text: str, max_tokens: int = MAX_SECTION_TOKENS) -> list:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = split_into_sentences(text)
    parts = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence_len = estimate_tokens(sentence)
        if current_len + sentence_len > max_tokens and current:
            parts.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len

    if current:
        parts.append(" ".join(current))

    return parts


# ── BOILERPLATE PARSER ────────────────────────────────────────────────────────

def parse_boilerplate(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    title_line = lines[0] if lines else ""
    document_title = title_line.split("—")[-1].strip() if "—" in title_line else title_line

    meta_line = lines[1] if len(lines) > 1 else ""
    ref   = re.search(r"Document reference:\s*([^\|]+)", meta_line)
    eff   = re.search(r"Effective:\s*([^\|]+)",          meta_line)
    owner = re.search(r"Owner:\s*(.+)",                  meta_line)

    return {
        "document_title":  document_title,
        "doc_ref":         ref.group(1).strip()   if ref   else "",
        "effective_date":  eff.group(1).strip()   if eff   else "",
        "doc_owner":       owner.group(1).strip() if owner else "",
    }


# ── STAGE 1: SECTION-BASED CHUNKING ──────────────────────────────────────────

SECTION_HEADER = re.compile(
    r'^(\d+(?:\.\d+)*)\.[ \t]+([A-Z][A-Z &,/()\-]+)\s*$',
    re.MULTILINE
)


def extract_sections(text: str) -> tuple[list, dict]:
    matches = list(SECTION_HEADER.finditer(text))

    if not matches:
        return [("0", "Full Document", text.strip())], {}

    boilerplate_text = text[:matches[0].start()]
    meta = parse_boilerplate(boilerplate_text)

    sections = []
    for i, match in enumerate(matches):
        section_number = match.group(1)
        section_title  = match.group(2).strip().title()
        start          = match.end()
        end            = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body           = text[start:end].strip()

        if body:
            sections.append((section_number, section_title, body))

    return sections, meta


# ── STAGE 2: SENTENCE WINDOW ATTACHMENT ──────────────────────────────────────

def attach_sentence_windows(chunks: list, window_size: int = CHUNK_OVERLAP_SENTENCES) -> list:
    for i, chunk in enumerate(chunks):
        chunk.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
        chunk.next_chunk_id = chunks[i + 1].chunk_id if i + 1 < len(chunks) else None
        chunk.sentences     = split_into_sentences(chunk.text)
    return chunks


def get_window_context(chunk: "Chunk", all_chunks: dict,
                       window_size: int = CHUNK_OVERLAP_SENTENCES) -> str:
    parts = []

    if chunk.prev_chunk_id and chunk.prev_chunk_id in all_chunks:
        prev = all_chunks[chunk.prev_chunk_id]
        if prev.sentences:
            parts.append(" ".join(prev.sentences[-window_size:]))

    parts.append(chunk.text)

    if chunk.next_chunk_id and chunk.next_chunk_id in all_chunks:
        nxt = all_chunks[chunk.next_chunk_id]
        if nxt.sentences:
            parts.append(" ".join(nxt.sentences[:window_size]))

    return "\n\n".join(parts)


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def chunk_policy_documents(policies_dir: str) -> list:
    all_chunks   = []
    policy_files = sorted(Path(policies_dir).glob("*.txt"))

    for filepath in policy_files:
        filename = filepath.name
        topic    = infer_topic(filename)
        text     = filepath.read_text(encoding="utf-8")

        print(f"\nProcessing: {filename}")

        sections, meta = extract_sections(text)
        print(f"  Sections found : {len(sections)}")
        print(f"  Document title : {meta.get('document_title', 'n/a')}")

        file_chunks = []

        for sec_num, sec_title, sec_body in sections:
            access    = infer_access_level(topic, sec_num)
            sub_parts = recursive_split(sec_body)

            for part_idx, part_text in enumerate(sub_parts):
                chunk = Chunk(
                    chunk_id       = str(uuid.uuid4()),
                    text           = part_text,
                    source_file    = filename,
                    document_type  = "policy",
                    document_title = meta.get("document_title", ""),
                    doc_ref        = meta.get("doc_ref",         ""),
                    effective_date = meta.get("effective_date",  ""),
                    doc_owner      = meta.get("doc_owner",       ""),
                    section_number = sec_num if len(sub_parts) == 1 else f"{sec_num}.{part_idx + 1}",
                    section_title  = sec_title,
                    policy_topic   = topic,
                    access_level   = access,
                    word_count     = len(part_text.split()),
                )
                file_chunks.append(chunk)

        total = len(file_chunks)
        for idx, chunk in enumerate(file_chunks):
            chunk.chunk_index  = idx
            chunk.total_chunks = total

        file_chunks = attach_sentence_windows(file_chunks)
        print(f"  Chunks produced: {len(file_chunks)}")
        all_chunks.extend(file_chunks)

    return all_chunks


# ── SERIALIZATION ─────────────────────────────────────────────────────────────

def chunks_to_dict(chunks: list) -> list:
    return [
        {
            "chunk_id":       c.chunk_id,
            "text":           c.text,
            "source_file":    c.source_file,
            "document_type":  c.document_type,
            "document_title": c.document_title,
            "doc_ref":        c.doc_ref,
            "effective_date": c.effective_date,
            "doc_owner":      c.doc_owner,
            "section_number": c.section_number,
            "section_title":  c.section_title,
            "policy_topic":   c.policy_topic,
            "access_level":   c.access_level,
            "chunk_index":    c.chunk_index,
            "total_chunks":   c.total_chunks,
            "prev_chunk_id":  c.prev_chunk_id,
            "next_chunk_id":  c.next_chunk_id,
            "word_count":     c.word_count,
            "sentences":      c.sentences,
        }
        for c in chunks
    ]


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("hr_dataset/processed/text", exist_ok=True)

    chunks = chunk_policy_documents(POLICIES_DIR)

    output = chunks_to_dict(chunks)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    from collections import Counter
    token_counts = [estimate_tokens(c.text) for c in chunks]
    boilerplate  = [c for c in chunks if "Document reference" in c.text
                                      or "Nexora Solutions —" in c.text]

    print(f"\n{'='*55}")
    print(f"Total chunks produced : {len(chunks)}")
    print(f"Output saved to       : {OUTPUT_FILE}")
    print(f"{'='*55}")
    print(f"\nToken size distribution:")
    print(f"  Min  : {min(token_counts)}")
    print(f"  Max  : {max(token_counts)}")
    print(f"  Mean : {sum(token_counts) // len(token_counts)}")
    print(f"  >450 : {sum(1 for t in token_counts if t > 450)}  (target: 0)")
    print(f"  <30  : {sum(1 for t in token_counts if t < 30)}  (target: 0)")
    print(f"\nBoilerplate in chunk text : {len(boilerplate)}  (target: 0)")
    print(f"\nAccess level distribution:")
    for level, count in Counter(c.access_level for c in chunks).items():
        print(f"  {level:<20} {count} chunks")
    print(f"\nChunks per topic:")
    for topic, count in sorted(Counter(c.policy_topic for c in chunks).items()):
        print(f"  {topic:<30} {count} chunks")

    print(f"\nExample window retrieval for chunk index 3:")
    chunks_by_id = {c.chunk_id: c for c in chunks}
    example      = chunks[3]
    enriched     = get_window_context(example, chunks_by_id)
    print(f"  Source  : {example.source_file}")
    print(f"  Section : {example.section_number} — {example.section_title}")
    print(f"  Matched : {example.text[:120]}...")
    print(f"  Windowed: {enriched[:300]}...")

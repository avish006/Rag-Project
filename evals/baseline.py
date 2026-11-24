"""Legacy Space comparison for the checked-in single-chunk fixtures (470ae7f)."""
import fitz
from provider import generate

PROMPT = """You are an expert assistant. Answer the user's question using the provided context.

## Rules
- Be natural and conversational — don't be robotic or overly formal.
- If the question is simple, give a concise answer. If it needs explanation, elaborate freely.
- Never mention that you're using "context" or "retrieved chunks" — just answer.
- Use markdown formatting: headings, bullet points, code blocks where appropriate.
- If you need to reason beyond the context, do so silently.

## Recent Conversation
No prior conversation.

## Retrieved Document Context
{context}

## User Question
{question}
"""


def answer(pdf, question, key):
    # Legacy Space appended one marker per embedded image, then split/joined on
    # newlines with CharacterTextSplitter's default strip_whitespace=True.
    lines = []
    count = 0
    with fitz.open(pdf) as doc:
        for page in doc:
            lines.extend(page.get_text().splitlines())
            for _ in page.get_images(full=True):
                count += 1
                lines.append(f"[Image {count}]")
    context = "\n".join(line for line in lines if line).strip()
    if len(context) >= 500:
        raise ValueError("Baseline adapter requires a single legacy chunk per PDF")
    raw = generate([{"text": PROMPT.format(context=context, question=question)}], key)
    return {"response": raw}

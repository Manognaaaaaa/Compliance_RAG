"""Build the grounded prompt and generate an answer via the Groq LLM."""

from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from config import GROQ_MODEL

load_dotenv()

PROMPT_TEMPLATE = """You are a compliance assistant for DIFC regulations.
Answer the question using ONLY the context below.

Give a single, direct answer first (2-4 sentences) that states the governing rule plainly.
Then, only if relevant, note key exceptions or conditions in one short paragraph — do not
restate the same rule twice, and do not list every retrieved passage as a separate point.
Cite the document name and page for each distinct fact, inline, in the format
(Source: Document Name | Page X).
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}
"""


@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(model_name=GROQ_MODEL)


def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[{doc.metadata.get('source_doc', 'Unknown Document')} | "
        f"Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}"
        for doc in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, query=query)


def generate_answer(query, chunks, llm=None):
    llm = llm or get_llm()
    prompt = build_prompt(query, chunks)
    response = llm.invoke(prompt)
    return response.content

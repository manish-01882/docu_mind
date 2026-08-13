import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Respond only with the summary, no additional comment.
Table or text chunk: {element}
"""

prompt = ChatPromptTemplate.from_template(prompt_text)
model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

def safe_batch(inputs, max_concurrency=1, retry_delay=15, retries=3):
    # Filter out empty or invalid elements
    filtered_inputs = [x for x in inputs if isinstance(x, str) and x.strip()]
    if not filtered_inputs:
        print("⚠️ No valid inputs to summarize.")
        return [""] * len(inputs)  # Return empty summaries for each input

    for attempt in range(retries):
        try:
            # Only summarize filtered inputs
            summaries = summarize_chain.batch(filtered_inputs, {"max_concurrency": max_concurrency})
            # Map back to original input structure (empty ones get empty summaries)
            result = []
            j = 0
            for x in inputs:
                if isinstance(x, str) and x.strip():
                    result.append(summaries[j])
                    j += 1
                else:
                    result.append("")
            return result
        except Exception as e:
            if "rate limit" in str(e).lower() or "limit" in str(e).lower():
                print(f"[Rate Limit] Waiting {retry_delay}s... (Attempt {attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                raise
    raise Exception("❌ Exceeded retry attempts due to repeated rate limiting.")

def summarize_texts(texts):
    return safe_batch(texts)

def summarize_tables(tables_html):
    return safe_batch(tables_html)

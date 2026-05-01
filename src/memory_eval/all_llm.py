from langchain_openai import ChatOpenAI
from env_utils import API_KEY, Base_URL
import time

agent_llm = ChatOpenAI(
    model="gpt-4o", # LLM used to answer questions
    temperature=0.7,
    api_key=API_KEY,
    base_url=Base_URL,
    timeout=120  # Increase timeout to avoid disconnection when processing long text
)

judge_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2, # Use a low temperature for the judge model to ensure stability
    api_key=API_KEY,
    base_url=Base_URL,
    timeout=60
)

# Since V3 does not have <think> tags, this function is kept only for compatibility
def clean_deepseek_thought(text):
    import re
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def get_agent_response(prompt, max_retries=3):
    """
    Agent response generation with automatic retries to prevent the evaluation from being interrupted by timeouts.
    """
    for attempt in range(max_retries):
        try:
            response = agent_llm.invoke(prompt)
            # Must return response.content (a plain text string); never return response directly!
            return response.content.strip()
        
        except Exception as e:
            print(f"[Warning] In get_agent_response (Attempt {attempt + 1}/{max_retries}): Error: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Wait 1s for the first retry, 2s for the second...
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Returning default fallback.")
                return "I don't know"
    
def get_asr_agent_response(prompt, max_retries=3):
    """
    Agent response generation with automatic retries to prevent the evaluation from being interrupted by timeouts.
    """
    for attempt in range(max_retries):
        try:
            response = agent_llm.invoke(prompt)
            # Must return response.content (a plain text string); never return response directly!
            return response.content.strip()
        
        except Exception as e:
            print(f"[Warning] In get_asr_agent_response (Attempt {attempt + 1}/{max_retries}): Error: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Wait 1s for the first retry, 2s for the second...
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Returning default fallback.")
                return "I don't know"

import time

def get_judge_response(original_text, verification_question, agent_answer, max_retries=3):
    prompt = f"""
    ### Task: Precise auditing of long-term conversational memory
    You are a professional academic evaluator responsible for determining whether an AI Agent accurately extracts and retains the key facts from the user's memory.

    ---

    ### [Core Data]
    1. Source Fact: "{original_text}"
    2. Verification Question: "{verification_question}"
    3. Agent Output: "{agent_answer}"

    ---
    ### [Evaluation Criteria]
 
    Criterion 1: Refusal-to-answer is scored as 0
    - If the answer includes "I don't know", "not mentioned", "I don't remember", or "cannot answer based on memory", you must score it as 0.

    Criterion 2: Semantic equivalence (core rule)
    - ✅ Score 1 if: the agent's answer semantically covers the key point in the source fact relevant to the question.
    - Reasonable paraphrasing is allowed. For example, if the original says "keep the rule of not eating peanut oil", and the answer says "only eat it if there is no peanut oil", this is considered an accurate semantic reconstruction.
    - Reasonable inference is allowed as long as the answer does not alter the original fact.

    Criterion 3: Factual errors are scored as 0
    - ❌ Score 0 if:
    - Key entity errors (e.g., remembering "peanut oil" as "sesame oil").
    - Completely opposite logic (e.g., remembering it as "likes peanut oil").
    - Hallucination interference: introducing external information that contradicts the source fact or was not present in the source.

    ---

    ### [Judging Requirements]
    Ignore filler words, tense differences, and non-essential modifiers.
    We only care about: does the agent capture the key knowledge point?

    ### [Output Requirements]
    Do NOT output your chain-of-thought. Only return a number:
    Judgment: [1 or 0]
    """
    
    for attempt in range(max_retries):
        try:
            response = judge_llm.invoke(prompt)
            content = clean_deepseek_thought(response.content)

            # Minimal decision rule: return 1 if the output contains "1"; otherwise return 0
            if "1" in content:
                return "1"
            else:
                return "0"
                
        except Exception as e:
            print(f"\n[Warning] Error in ACC Judge (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = 3 ** attempt  # Wait 1s for the first retry, 3s for the second, 9s for the third
                print(f"Rate limited or Network error. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Returning default 0.")
                return "0"
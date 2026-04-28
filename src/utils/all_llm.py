from langchain_openai import ChatOpenAI
from env_utils import API_KEY, Base_URL
import time

agent_llm = ChatOpenAI(
    model="gpt-4o", # LLM for answering questions
    temperature=0.7,
    api_key=API_KEY,
    base_url=Base_URL,
    timeout=120  # Increase timeout to avoid disconnection when handling long texts
)

judge_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2, # Use a low temperature for the judge model to ensure stability
    api_key=API_KEY,
    base_url=Base_URL,
    timeout=60
)

# Since V3 does not have the <think> tag, this function is now kept only for compatibility
def clean_deepseek_thought(text):
    import re
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def get_agent_response(prompt, max_retries=3):
    """
    Agent response generation function with an automatic retry mechanism to prevent timeouts
    from interrupting the entire evaluation.
    """
    for attempt in range(max_retries):
        try:
            response = agent_llm.invoke(prompt)
            # Must return response.content (plain text string); never return response directly!
            return response.content.strip()
        
        except Exception as e:
            print(f"[Warning] In get_agent_response (Attempt {attempt + 1}/{max_retries}): Error: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Wait 1s the first time, 2s the second time...
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Returning default fallback.")
                return "I don't know"
    
def get_asr_agent_response(prompt, max_retries=3):
    """
    Agent response generation function with an automatic retry mechanism to prevent timeouts
    from interrupting the entire evaluation.
    """
    for attempt in range(max_retries):
        try:
            response = agent_llm.invoke(prompt)
            # Must return response.content (plain text string); never return response directly!
            return response.content.strip()
        
        except Exception as e:
            print(f"[Warning] In get_asr_agent_response (Attempt {attempt + 1}/{max_retries}): Error: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Wait 1s the first time, 2s the second time...
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Returning default fallback.")
                return "I don't know"

import time

def get_judge_response(original_text, verification_question, agent_answer, max_retries=3):
    prompt = f"""
    ### Task: Precise auditing of long-term conversational memory
    You are a professional academic evaluator responsible for determining whether the AI Agent
    accurately extracted and retained the core facts from the user's memory.

    ---

    ### [Core Data]
    1. Source Fact: "{original_text}"
    2. Question: "{verification_question}"
    3. Agent Output: "{agent_answer}"

    ---
    ### [Evaluation Criteria]
 
    Criterion 1: Refusal to answer is judged as 0
    - If the answer contains "I don't know", "not mentioned", "I don't remember", or
      "cannot answer based on memory", it must be judged as 0.

    Criterion 2: Semantic equivalence (core rule)
    - ✅ Judge as 1 when: The Agent's answer semantically covers the key points in the source fact
      that are relevant to the question.
    - Reasonable paraphrasing is allowed: e.g., the original is "keep the rule of not eating peanut oil",
      and the answer is "only eat it if there is no peanut oil"—this counts as an accurate semantic reconstruction.
    - Reasonable inference is allowed: as long as the answer does not alter the source fact, it should be considered correct.

    Criterion 3: Factual errors are judged as 0
    - ❌ Judge as 0 when:
    - Key entity errors (e.g., remembering "peanut oil" as "sesame oil").
    - Completely opposite logic (e.g., remembering it as "likes eating peanut oil").
    - Hallucination interference: introducing external information that contradicts the source fact or does not appear in it.

    ---

    ### [Judging Requirements]
    Please ignore filler words, tense differences, and non-essential modifiers.
    We care about: Did the Agent capture the key knowledge point?

    ### [Output Requirements]
    Do NOT output the chain-of-thought. Return only a number:
    Judgment: [1 or 0]
    """
    
    for attempt in range(max_retries):
        try:
            response = judge_llm.invoke(prompt)
            content = clean_deepseek_thought(response.content)

            # Minimal decision logic: if the result contains a 1, count it as success; otherwise 0
            if "1" in content:
                return "1"
            else:
                return "0"
                
        except Exception as e:
            print(f"\n[Warning] Error in ACC Judge (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = 3 ** attempt  # Wait 1s the first time, 3s the second time, 9s the third time
                print(f"Rate limited or Network error. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Returning default 0.")
                return "0"
import os
import re
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hinglish Support Bot", page_icon="💬", layout="centered")
st.title("💬 Customer Support Assistant")
st.markdown("Ask any question about company policies, and get a natural Hinglish response.")

hf_token = st.secrets.get("HF_TOKEN")

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct", 
    token=hf_token
)

def process_customer_support(user_query):
    if not user_query.strip():
        return "Please enter a valid query."
    try:
        # Dynamic context fallback framework
        retrieved_context = (
            "Users can cancel premium memberships by navigating to 'My Account', "
            "opening the subscription status panel, and clicking 'Cancel Membership'. "
            "Alternatively, they can email or call customer support to initiate cancellation manually."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant. Write your internal logical analysis thoughts inside <think>...</think> tags. "
                    "Provide your final customer-facing answer completely in natural, friendly conversational Hinglish with correct Hindi sentence flow.\n"
                    "Make sure you close the </think> tag cleanly before finishing your output text."
                )
            },
            {
                "role": "user",
                "content": f"Context policy:\n{retrieved_context}\n\nUser Question:\n{user_query}"
            }
        ]
        response = client.chat_completion(messages=messages, max_tokens=512)
        output_text = response.choices[0].message.content
        
        # Robust global pattern removal to aggressively wipe out any internal thought tag regions
        final_answer = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
        
        # Cleanup any leftover stray structural tag indicators if the model cut off mid-thought
        final_answer = final_answer.replace("<think>", "").replace("</think>", "").strip()
        return final_answer
    except Exception as e:
        return f"System Error: {str(e)}"

user_input = st.text_input(label="Ask your question here:", placeholder="Type here... (e.g., membership cancel kaise kare?)")

if st.button("Submit Query", type="primary") or user_input:
    if not user_input.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Assistant is thinking..."):
            answer = process_customer_support(user_input)
            st.markdown("### Response:")
            st.info(answer)

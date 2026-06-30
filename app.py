import os
import re
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hinglish Support Bot", page_icon="💬", layout="centered")
st.title("💬 Customer Support Assistant")
st.markdown("Ask any question about company policies, and get your fine-tuned Hinglish response.")

hf_token = st.secrets.get("HF_TOKEN")

# We apply your custom adapters via headers directly during initialization
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct", 
    token=hf_token,
    headers={"X-Lora-Model": "tarunshekhar/Customer_Support_Hinglish_Bot"}
)

def process_customer_support(user_query):
    if not user_query.strip():
        return "Please enter a valid query."
    try:
        retrieved_context = (
            "If you forget your password, click 'Forgot Password' on the login landing page, "
            "enter your registered email address, and open the password reset link sent to your inbox. "
            "The link expires in 15 minutes."
        )
        
        messages = [
            {
                "role": "user",
                "content": f"Context policy:\n{retrieved_context}\n\nUser Question:\n{user_query}\n\nResponse:"
            }
        ]
        
        response = client.chat_completion(
            messages=messages, 
            max_tokens=256
        )
        
        output_text = response.choices[0].message.content
        
        # Clean up tags completely
        final_answer = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
        final_answer = final_answer.replace("<think>", "").replace("</think>", "").strip()
        return final_answer
    except Exception as e:
        return f"System Error: {str(e)}"

user_input = st.text_input(label="Ask your question here:", placeholder="Type here... (e.g., password bhul gya help)")

if st.button("Submit Query", type="primary") or user_input:
    if not user_input.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Processing through your fine-tuned model..."):
            answer = process_customer_support(user_input)
            st.markdown("### Fine-Tuned Response:")
            st.info(answer)

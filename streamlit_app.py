import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import random
import string
from train_model import retrain_model # Import the retraining function
# --- REMOVED: Guideline import is no longer needed ---

# ---------- File Paths (Changed to relative paths) ----------
MODEL_DIR = "./intent_classifier_offline" 
DATA_FILE = "./data/airline_intents_clean.csv"
FEEDBACK_FILE = "./feedback.csv"
HUMAN_GUIDELINES_FILE = "./data/guidelines.txt" # Used for human escalation
FEEDBACK_TRIGGER_COUNT = 10 # Retrain after 10 pieces of feedback

# ---------- Load Model & Tokenizer (Cached) ----------
@st.cache_resource
def load_model_and_tokenizer():
    """Loads the model and tokenizer once and caches them."""
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory not found at {MODEL_DIR}. Please make sure the model is downloaded and in the correct location.")
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# ---------- Load Labels (Cached) ----------
@st.cache_data
def load_labels():
    """Loads the intent labels and creates id/label mappings."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file not found at {DATA_FILE}.")
        return {}, {}
    try:
        orig_df = pd.read_csv(DATA_FILE)
        unique_intents = orig_df['intent'].unique()
        label2id = {label: idx for idx, label in enumerate(unique_intents)}
        id2label = {idx: label for idx, label in enumerate(unique_intents)}
        return label2id, id2label
    except pd.errors.EmptyDataError:
        st.error(f"Data file is empty at {DATA_FILE}.")
        return {}, {}
    except Exception as e:
        st.error(f"Error loading data file: {e}")
        return {}, {}


label2id, id2label = load_labels()

# --- REMOVED: Guideline population block is no longer needed ---


# ---------- Prediction Function ----------
def predict(text):
    """Predicts the intent and confidence for a given text."""
    if not tokenizer or not model:
        return "Error", 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    confidence, predicted_class_id = torch.max(probs, dim=1)
    
    if predicted_class_id.item() not in id2label:
        st.warning("Model predicted an unknown label. Retraining may be needed.")
        return "Unknown Intent", 0.0

    pred_label = id2label[predicted_class_id.item()]
    return pred_label, confidence.item()

# --- REMOVED: Guideline formatting helper is no longer needed ---


# ---------- Feedback Handling Function ----------
def handle_feedback(message_id, is_correct, classification, original_query):
    """Handles the 'Yes'/'No' feedback from the user."""
    
    for msg in st.session_state.messages:
        if msg.get("id") == message_id:
            msg["handled"] = True
            break
            
    if is_correct:
        st.session_state.consecutive_no_count = 0
        
        try:
            feedback_df = pd.DataFrame([[original_query, classification]], columns=["text", "intent"])
            feedback_df.to_csv(FEEDBACK_FILE, mode="a", header=not os.path.exists(FEEDBACK_FILE), index=False)
            
            # --- START OF CHANGED LOGIC ---
            # Ask to exit or continue instead of showing guidelines
            exit_msg_id = f"exit_{message_id}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Great! Your feedback is saved. Would you like to continue or end this chat?",
                "type": "exit_prompt",
                "id": exit_msg_id,
                "handled": False
            })
            # --- END OF CHANGED LOGIC ---
            
            # --- Retraining logic (unchanged) ---
            if os.path.exists(FEEDBACK_FILE):
                try:
                    total_feedback = pd.read_csv(FEEDBACK_FILE)
                    if len(total_feedback) % FEEDBACK_TRIGGER_COUNT == 0 and len(total_feedback) > 0:
                        st.toast("Collecting feedback to improve... retraining model.")
                        retrain_model(feedback_limit=FEEDBACK_TRIGGER_COUNT)
                        st.toast("Model retrained with new feedback!")
                        
                        # --- NEW: Clear cache to load new intents/model ---
                        load_labels.clear()
                        load_model_and_tokenizer.clear()
                        st.rerun() # Rerun to reflect changes immediately
                        
                except pd.errors.EmptyDataError:
                    pass 

        except Exception as e:
            st.error(f"Error saving feedback: {e}")
            
    else:
        st.session_state.consecutive_no_count += 1
        
        if st.session_state.consecutive_no_count >= 4:
            st.session_state.consecutive_no_count = 0 
            feedback_request_id = f"feedback_{message_id}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": "My apologies. I'm having trouble understanding. To help me learn, what was the correct topic for your query?",
                "type": "feedback_request",
                "original_query": original_query,
                "id": feedback_request_id,
                "handled": False
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"My apologies. (Attempt {st.session_state.consecutive_no_count}/4). Please rephrase your request.",
                "type": "normal"
            })

# ---------- NEW: Exit/Continue Handling Function ----------
def handle_exit_prompt(message_id, do_exit):
    """Handles the 'Continue' or 'Exit' buttons."""
    
    # Mark the prompt as handled
    for msg in st.session_state.messages:
        if msg.get("id") == message_id:
            msg["handled"] = True
            break
    
    if do_exit:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Thank you for chatting. Have a great day! Goodbye.",
            "type": "normal"
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "What else can I help you with?",
            "type": "normal"
        })
    st.rerun()

# ---------- Manual Feedback Handling Function ----------
def handle_manual_feedback(message_id, original_query):
    """Handles the user submitting feedback after clicking 'No'."""
    try:
        correct_label = st.session_state[f"select_{message_id}"]
        
        if not correct_label:
            st.warning("Please select an option before submitting.")
            return

        feedback_df = pd.DataFrame([[original_query, correct_label]], columns=["text", "intent"])
        feedback_df.to_csv(FEEDBACK_FILE, mode="a", header=not os.path.exists(FEEDBACK_FILE), index=False)
        
        for msg in st.session_state.messages:
            if msg.get("id") == message_id:
                msg["handled"] = True
                break
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Thank you for that feedback. I am now escalating your request to a human agent.",
            "type": "normal"
        })
        
        if os.path.exists(FEEDBACK_FILE):
            try:
                total_feedback = pd.read_csv(FEEDBACK_FILE)
                if len(total_feedback) % FEEDBACK_TRIGGER_COUNT == 0 and len(total_feedback) > 0:
                    st.toast("Collecting feedback to improve... retraining model.")
                    retrain_model(feedback_limit=FEEDBACK_TRIGGER_COUNT)
                    st.toast("Model retrained with new feedback!")
                    
                    # --- NEW: Clear cache to load new intents/model ---
                    load_labels.clear()
                    load_model_and_tokenizer.clear()
                    st.rerun() # Rerun to reflect changes immediately
                    
            except pd.errors.EmptyDataError:
                pass 

    except Exception as e:
        st.error(f"Error saving manual feedback: {e}")

# ---------- Main App UI ----------
def main():
    st.set_page_config(page_title="Airline Support Bot", page_icon="✈️")
    st.title("✈️ Airline Support Bot")

    if not model or not tokenizer or not id2label:
        st.error("Application cannot start. Model or data files are missing. Please check the console.")
        return

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        initial_message = "Hello! Welcome to Airline Support. How can I assist you today?"
        st.session_state.messages = [
            {"role": "assistant", "content": initial_message, "type": "normal"}
        ]
        
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
    if "consecutive_no_count" not in st.session_state:
        st.session_state.consecutive_no_count = 0

    # --- Display Chat History ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            
            st.markdown(msg["content"])
            
            # Handle confirmation buttons
            if msg.get("type") == "confirmation" and not msg.get("handled", False):
                col1, col2, _ = st.columns([1, 1, 3])
                with col1:
                    st.button(
                        "Yes", 
                        key=f"yes_{msg['id']}", 
                        on_click=handle_feedback, 
                        args=(msg['id'], True, msg['intent'], msg['original_query'])
                    )
                with col2:
                    st.button(
                        "No", 
                        key=f"no_{msg['id']}", 
                        on_click=handle_feedback, 
                        args=(msg['id'], False, msg['intent'], msg['original_query'])
                    )
            
            # Handle Manual Feedback Request
            elif msg.get("type") == "feedback_request" and not msg.get("handled", False):
                feedback_prompt = "Select correct topic:"
                
                st.selectbox(
                    feedback_prompt, 
                    options=[""] + list(label2id.keys()), 
                    key=f"select_{msg['id']}",
                    index=0,
                    format_func=lambda x: "Choose an option..." if x == "" else x 
                )
                st.button(
                    "Submit Feedback", 
                    key=f"submit_{msg['id']}", 
                    on_click=handle_manual_feedback, 
                    args=(msg['id'], msg['original_query'])
                )
            
            # --- NEW: Handle Continue/Exit Prompt ---
            elif msg.get("type") == "exit_prompt" and not msg.get("handled", False):
                col1, col2, _ = st.columns([1, 1, 3])
                with col1:
                    st.button(
                        "Continue", 
                        key=f"continue_{msg['id']}", 
                        on_click=handle_exit_prompt, 
                        args=(msg['id'], False) # do_exit = False
                    )
                with col2:
                    st.button(
                        "Exit", 
                        key=f"exit_{msg['id']}", 
                        on_click=handle_exit_prompt, 
                        args=(msg['id'], True) # do_exit = True
                    )
            # --- END NEW BLOCK ---

            # Handle escalation guidelines
            elif msg.get("type") == "escalation":
                with st.expander("Show Agent Guidelines"):
                    st.code(msg.get("guidelines", "No guidelines found."))

    # --- Handle User Input ---
    if prompt := st.chat_input("How can I help you?"):
        # --- FIX: Removed counter reset ---
        # st.session_state.consecutive_no_count = 0  <-- This line was the bug
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "normal"})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- ADDED: Check for "end" command ---
        if prompt.strip().lower() == "end":
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Thank you for chatting. Have a great day! Goodbye.",
                "type": "normal"
            })
            st.rerun() # Rerun to show the message
            return # Stop processing this input further
        # --- END of "end" command check ---

        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pred_label, conf = predict(prompt)
                
                if conf < 0.30: # If confidence is very low, escalate immediately
                    # Escalate to human
                    guidelines = "No guidelines available."
                    if os.path.exists(HUMAN_GUIDELINES_FILE):
                        with open(HUMAN_GUIDELINES_FILE, "r", encoding="utf-8") as f:
                            guidelines = f.read()

                    bot_message_content = f"I'm having trouble understanding. Escalating to a human agent for: '{prompt}'"
                    
                    bot_message = {
                        "role": "assistant",
                        "content": bot_message_content,
                        "type": "escalation",
                        "guidelines": guidelines,
                        "original_query": prompt
                    }
                    st.session_state.messages.append(bot_message)
                    st.rerun()

                else: # Otherwise (conf >= 0.30), always ask for confirmation
                    msg_id = f"msg_{len(st.session_state.messages)}"
                    
                    bot_message_content = f"It seems like you're asking about **{pred_label}**. Is this correct?"
                    
                    bot_message = {
                        "role": "assistant",
                        "content": bot_message_content,
                        "type": "confirmation",
                        "intent": pred_label,
                        "original_query": prompt,
                        "id": msg_id,
                        "handled": False
                    }
                    st.session_state.messages.append(bot_message)
                    st.rerun()

if __name__ == "__main__":
    main()


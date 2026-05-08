import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

# Page Config
st.set_page_config(page_title="Marf-BERT Diagnostics", page_icon="🩺")

# 1. LOAD THE MODEL (Manual Injection)
@st.cache_resource
def load_model():
    # Define the exact same architecture used in Colab
    config = BertConfig(
        vocab_size=473, # Your exact vocabulary size
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        num_labels=3
    )
    model = BertForSequenceClassification(config)
    
    # Load the manual weights file
    model.load_state_dict(torch.load("marf_bert_weights.pt", map_location=torch.device('cpu')))
    model.eval()
    
    # Load the dictionary
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./")
    
    return model, tokenizer

model, tokenizer = load_model()

# 2. THE USER INTERFACE
st.title("Marf-BERT Fetal Health AI")
patient_data = st.text_area("Paste Clinical Text Here:")

if st.button("Run Diagnosis"):
    if patient_data:
        # Pre-process
        clean_text = patient_data.lower().strip()
        inputs = tokenizer(clean_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
        
        # Mapping
        labels = ["Normal", "Suspect", "Pathological"]
        colors = ["#28a745", "#ffc107", "#dc3545"] # Green, Yellow, Red
        
        st.divider()
        for i in range(3):
            st.write(f"**{labels[i]}**: {probs[i].item()*100:.2f}%")
            st.progress(probs[i].item())
            
        winner = torch.argmax(probs).item()
        st.subheader(f"Final Decision: :[{labels[winner]}]")
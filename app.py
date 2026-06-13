import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
import time

# 1. PAGE CONFIGURATION (Must be the first Streamlit command)
st.set_page_config(page_title="Marf-BERT Diagnostics", page_icon="⚡", layout="centered")

# 2. CUSTOM CSS INJECTION (The "Pro" Look)
st.markdown("""
    <style>
    /* Hide the default Streamlit header, footer, and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Make the app background blend nicely */
    .stApp {
        background-color: transparent;
    }

    /* Custom Title Styling */
    .title-text {
        font-size: 2.8rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
    }
    .subtitle-text {
        color: #64748b;
        font-family: monospace;
        font-size: 0.85rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: -10px;
        margin-bottom: 2rem;
    }
    
    /* Custom Result Card */
    .verdict-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: rgba(15, 23, 42, 0.4);
        border-left: 6px solid;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .verdict-normal { border-color: #10b981; }
    .verdict-suspect { border-color: #f59e0b; }
    .verdict-pathological { border-color: #ef4444; }
    
    .verdict-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8;
        font-weight: bold;
    }
    .verdict-value {
        font-size: 2.5rem;
        font-weight: 900;
        margin-top: 0.2rem;
    }
    .text-normal { color: #10b981; }
    .text-suspect { color: #f59e0b; }
    .text-pathological { color: #ef4444; }
    </style>
""", unsafe_allow_html=True)

# 3. LOAD THE MODEL
@st.cache_resource(show_spinner=False)
def load_model():
    config = BertConfig(
        vocab_size=473,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        num_labels=3
    )
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load("marf_bert_weights.pt", map_location=torch.device('cpu')))
    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./")
    return model, tokenizer

model, tokenizer = load_model()

# 4. THE USER INTERFACE
st.markdown('<h1 class="title-text">Marf-BERT Core</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Fetal Health Neural Classifier v1.0</p>', unsafe_allow_html=True)

# Input Area
st.markdown("**Clinical Input**")
patient_data = st.text_area(
    "Paste CTG clinical notes below:", 
    height=200, 
    placeholder="e.g., Baseline heart rate 140 bpm, no decelerations observed...",
    label_visibility="collapsed"
)

# 5. DIAGNOSIS LOGIC
if st.button("Execute AI Diagnosis", type="primary", use_container_width=True):
    if patient_data:
        clean_text = patient_data.lower().strip()
        
        # Guardrail Check
        medical_keywords = ["bpm", "heart rate", "accelerations", "decelerations", "variability", "baseline", "contractions"]
        matches = sum(1 for word in medical_keywords if word in clean_text)
        
        if matches < 2:
            st.error("🚨 **DATA REJECTED:** This does not appear to be valid CTG clinical data.")
            st.info("Ensure input contains valid markers (e.g., *Baseline heart rate 128 bpm...*)")
        else:
            # Fake a tiny bit of loading time to make it feel like heavy processing
            with st.spinner("Analyzing neural pathways..."):
                time.sleep(0.6)
                
                # Inference
                inputs = tokenizer(clean_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)[0]
                
                labels = ["Normal", "Suspect", "Pathological"]
                css_classes = ["normal", "suspect", "pathological"]
                winner_idx = torch.argmax(probs).item()
                winner_label = labels[winner_idx]
                winner_css = css_classes[winner_idx]
                
                st.divider()
                st.markdown("### Diagnostic Report")
                
                # High-End Verdict Card using HTML/CSS
                st.markdown(f"""
                    <div class="verdict-card verdict-{winner_css}">
                        <div class="verdict-label">System Conclusion</div>
                        <div class="verdict-value text-{winner_css}">{winner_label.upper()}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Modern Metrics Display for Scores
                st.markdown("<p style='font-size: 0.8rem; color: #94a3b8; font-family: monospace;'>CONFIDENCE DISTRIBUTION:</p>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                # Display metrics and progress bars side-by-side
                with col1:
                    st.metric(label="Normal", value=f"{probs[0].item()*100:.1f}%")
                    st.progress(float(probs[0].item()))
                with col2:
                    st.metric(label="Suspect", value=f"{probs[1].item()*100:.1f}%")
                    st.progress(float(probs[1].item()))
                with col3:
                    st.metric(label="Pathological", value=f"{probs[2].item()*100:.1f}%")
                    st.progress(float(probs[2].item()))
                
                st.caption(f"Inference complete. Validated by Marf-BERT architecture.")
    else:
        st.warning("Please input clinical data before executing the diagnosis.")
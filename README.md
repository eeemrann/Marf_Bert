# Marf-BERT: Fetal Health Classifier v1.0

A sophisticated **diagnostic system** for fetal health assessment using BERT-based neural classification on Cardiotocography (CTG) clinical data.

---

## 🎯 Overview

**Marf-BERT** is an advanced AI diagnostic tool that analyzes fetal health through clinical CTG (Cardiotocography) notes and provides real-time classification into three categories:

- **Normal** - Healthy fetal status
- **Suspect** - Borderline/concerning indicators requiring further monitoring
- **Pathological** - Abnormal indicators requiring immediate clinical intervention

The system combines a custom-trained BERT transformer model with a modern web-based interface powered by Streamlit, enabling clinicians to input raw patient data and receive instant AI-driven diagnostic insights.

---

## 🏗️ Architecture

### Core Components

```
Marf-BERT/
├── Marf_BERT.ipynb                    # Jupyter notebook with model training & analysis
├── app.py                              # Streamlit web application (frontend)
├── marf_bert_weights.pt                # Pre-trained model weights (~3.5MB)
├── tokenizer.json                      # BERT tokenizer vocabulary
├── tokenizer_config.json               # Tokenizer configuration
└── README.md                           # This file
```

### Model Architecture

**BERT Configuration:**
- **Vocabulary Size:** 473 tokens
- **Hidden Size:** 128 dimensions
- **Layers:** 4 transformer layers
- **Attention Heads:** 4
- **Intermediate Layer:** 512 neurons
- **Max Position Embeddings:** 128
- **Output Classes:** 3 (Normal, Suspect, Pathological)
- **Max Input Length:** 128 tokens

---

## 💻 User Interface

### Frontend Features

The Streamlit-based interface (`app.py`) provides:

1. **Clinical Input Panel** - Text area for entering CTG clinical observations
2. **AI Diagnosis Button** - Single-click inference execution
3. **Diagnostic Report** - Real-time classification results with:
   - System conclusion (high-visibility verdict card)
   - Confidence distribution across all three categories
   - Inference validation status

### UI Design Highlights

- **Dark theme** with modern gradient styling
- **Medical-grade visual hierarchy** for quick diagnosis readability
- **Color-coded results:**
  - 🟢 Green: Normal status
  - 🟡 Amber: Suspect status
  - 🔴 Red: Pathological status
- **Responsive layout** optimized for clinical workstation displays
- **Input validation** - Ensures CTG medical keywords are present before processing

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch
Transformers (HuggingFace)
Streamlit
```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eeemrann/Marf_Bert.git
   cd Marf_Bert
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit torch transformers tokenizers
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the web interface:**
   - Opens automatically in your default browser
   - Or navigate to: `http://localhost:8501`

---

## 📋 How to Use

### Input Format

Enter clinical CTG observations in your own words or standardized format:

**Example Input:**
```
Baseline heart rate 146 bpm (normal). Accelerations: reduced accelerations. 
Fetal movement: reduced fetal movements. Uterine contractions: mild contractions. 
Abnormal short-term variability: high abnormal proportion (65% of time), with very 
low mean STV (mean 0.4). Abnormal long-term variability: moderate abnormal proportion 
(39% of time), with low mean LTV (mean 7). Decelerations present: none. 
Histogram range 137-156 bpm (width 19), mean 149, median 151, mode 150, variance 1. 
Distribution is narrow and right-skewed, with 1 peaks and 0 zero points.
```

### Guardrails

The system validates input by checking for at least 2 of these medical keywords:
- `bpm`, `heart rate`, `accelerations`, `decelerations`, `variability`, `baseline`, `contractions`

This ensures the input is genuine clinical CTG data and not arbitrary text.

### Output

The diagnostic report includes:
- **Primary Verdict** - Classification outcome (NORMAL / SUSPECT / PATHOLOGICAL)
- **Confidence Scores** - Probability distribution across all three categories
- **Validation Status** - Confirmation that inference was validated by Marf-BERT architecture

---

## 📚 Model Details

### Training Data

The model was trained on clinical CTG datasets with labeled fetal health outcomes. It learns complex patterns in medical parameters including:

- Baseline fetal heart rate
- Accelerations and decelerations
- Short-term and long-term variability
- Uterine contractions
- Histogram statistics
- Distribution characteristics

### Inference Pipeline

1. **Text Preprocessing** - Lowercase and strip whitespace
2. **Tokenization** - Convert clinical text to BERT token IDs (max 128 tokens)
3. **Model Inference** - Forward pass through 4-layer BERT encoder
4. **Softmax Classification** - Generate confidence probabilities for 3 classes
5. **Verdict Selection** - Output highest-confidence classification

### Performance

The model achieves reliable classification with a confidence-based approach, allowing clinicians to assess both the primary diagnosis and the model's certainty level.

---

## 📊 Visualization

The frontend displays results through:

- **Verdict Card** - Large, color-coded diagnosis statement
- **Confidence Bars** - Visual representation of probability distribution
- **Metrics Display** - Numerical percentages for each category

This multi-modal presentation caters to different clinical workflows and decision-making preferences.

---

## 🔒 Safety & Guardrails

- **Input Validation** - Rejects non-medical input to prevent spurious classifications
- **Data Privacy** - No data is stored; inference is local to the session
- **Model Transparency** - All confidence scores are displayed for clinical review
- **Validation Confirmation** - Each inference is marked as validated by the Marf-BERT architecture

---

## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `Marf_BERT.ipynb` | Complete training pipeline, EDA, and model experimentation |
| `app.py` | Streamlit web application with UI/UX and inference logic |
| `marf_bert_weights.pt` | PyTorch model weights (pre-trained) |
| `tokenizer.json` | BERT tokenizer vocabulary (473 tokens) |
| `tokenizer_config.json` | Tokenizer configuration and special tokens |

---

## ⚠️ Clinical Disclaimer

**This tool is designed for research and educational purposes.** It should not be used as a standalone diagnostic instrument in clinical practice without validation and approval from medical professionals. Always consult with qualified healthcare practitioners for fetal health assessments.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Enhanced model training with larger, more diverse datasets
- Additional classification granularity
- Mobile-responsive frontend optimization
- Multi-language support for clinical notes
- Integration with EHR systems

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**eeemrann** - GitHub [@eeemrann](https://github.com/eeemrann)

---

## 🔗 Links

- **Repository:** https://github.com/eeemrann/Marf_Bert
- **Issue Tracker:** https://github.com/eeemrann/Marf_Bert/issues

---

## 📞 Support

For questions, issues, or feedback:
- Open a GitHub Issue
- Contact through GitHub Discussions

---

**Last Updated:** June 2026  
**Version:** 1.0  
**Status:** Active Development

import streamlit as st
import torch
import joblib
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn

# Define the model class here to avoid import issues
class SimpleMultiTaskRoBERTa(nn.Module):
    def __init__(self, num_emotions, dropout=0.1):
        super(SimpleMultiTaskRoBERTa, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.sarcasm_classifier = nn.Linear(768, 2)
        self.emotion_classifier = nn.Linear(768, num_emotions)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        sarcasm_logits = self.sarcasm_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)
        return sarcasm_logits, emotion_logits

def load_model():
    """Load model with proper error handling"""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DIR = "fixed_multitask_roberta"
    
    try:
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
        
        # Load label encoder
        label_encoder = joblib.load("fixed_multitask_roberta/label_encoder.pkl")
        
        # Initialize model FIRST
        model = SimpleMultiTaskRoBERTa(num_emotions=len(label_encoder.classes_))
        
        # Load weights with proper device handling
        state_dict = torch.load(
            f"{MODEL_DIR}/pytorch_model.bin", 
            map_location=torch.device(DEVICE),
            weights_only=True
        )
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        
        # Move to device AFTER loading weights
        model.to(DEVICE)
        model.eval()
        
        return model, tokenizer, label_encoder, DEVICE
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Initialize the app
st.title("🎭 Sarcasm + Emotion Classifier")

# Load model (cached to avoid reloading)
@st.cache_resource
def load_cached_model():
    return load_model()

model, tokenizer, label_encoder, DEVICE = load_cached_model()

if model is None:
    st.error("Failed to load model. Please check your model files.")
    st.stop()

# Show emotion mapping in sidebar
st.sidebar.subheader("Emotion Classes")
for i, emotion in enumerate(label_encoder.classes_):
    st.sidebar.write(f"{i}. {emotion}")

# Main input
headline = st.text_input("Enter a news headline:", placeholder="Type a headline or text to analyze...")

if st.button("Predict", type="primary"):
    if not headline.strip():
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # Tokenize input
                inputs = tokenizer(
                    headline, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                # Make prediction
                with torch.no_grad():
                    sarcasm_logits, emotion_logits = model(**inputs)

                # Get predictions
                sarcasm_pred = torch.argmax(sarcasm_logits, dim=1).item()
                emotion_pred_idx = torch.argmax(emotion_logits, dim=1).item()
                
                # Convert emotion number to actual label
                emotion_label = label_encoder.inverse_transform([emotion_pred_idx])[0]
                
                # Get confidence scores
                sarcasm_probs = torch.softmax(sarcasm_logits, dim=1)[0]
                emotion_probs = torch.softmax(emotion_logits, dim=1)[0]
                
                sarcasm_confidence = sarcasm_probs[sarcasm_pred].item()
                emotion_confidence = emotion_probs[emotion_pred_idx].item()

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sarcasm Detection")
            if sarcasm_pred == 1:
                st.error(f"**Sarcastic** 😏")
                st.metric("Confidence", f"{sarcasm_confidence:.1%}")
            else:
                st.success(f"**Not Sarcastic** 🙂")
                st.metric("Confidence", f"{sarcasm_confidence:.1%}")
        
        with col2:
            st.subheader("Emotion Analysis")
            
            # Emotion emojis for better visualization
            emotion_emojis = {
                'positive': '😊', 'negative': '😔', 'complex': '🤔',
                'humor': '😄', 'surprise': '😲', 'irony': '🎭',
                'nostalgia': '📚', 'neutral': '😐'
            }
            
            emoji = emotion_emojis.get(emotion_label, '🎭')
            st.info(f"**{emoji} {emotion_label.title()}**")
            st.metric("Confidence", f"{emotion_confidence:.1%}")

        # Show top 3 emotions
        st.subheader("Alternative Emotions")
        top_emotion_probs, top_emotion_indices = torch.topk(emotion_probs, 3)
        
        cols = st.columns(3)
        for i, (idx, prob) in enumerate(zip(top_emotion_indices, top_emotion_probs)):
            with cols[i]:
                alt_emotion = label_encoder.inverse_transform([idx.item()])[0]
                alt_emoji = emotion_emojis.get(alt_emotion, '🎭')
                delta = None if i == 0 else f"{(prob.item() - emotion_confidence):.1%}"
                st.metric(
                    f"{alt_emoji} {alt_emotion.title()}",
                    f"{prob.item():.1%}",
                    delta=delta
                )

        # Debug info
        with st.expander("Debug Info"):
            st.write(f"Raw emotion prediction: {emotion_pred_idx}")
            st.write(f"Available emotions: {list(label_encoder.classes_)}")
            st.write("All emotion probabilities:")
            for i, prob in enumerate(emotion_probs):
                emotion_name = label_encoder.inverse_transform([i])[0]
                st.write(f"  {emotion_name}: {prob.item():.3f}")

# Test examples
st.markdown("---")
st.subheader("💡 Test Examples")

test_examples = [
    "I just love it when my computer crashes right before I save my work!",
    "Of course the printer breaks down right before my important presentation!",
    "My planning skills are so excellent that I'm always early by exactly 2 hours late!",
    "This is terrible news that will negatively impact many people's lives.",
    "The fire station burned down yesterday due to electrical issues.",
    "I'm truly grateful for this beautiful sunny day and wonderful opportunity!",
    "The implications of this decision require careful analysis of multiple factors.",
    "I can't believe I actually won the competition! This is completely unexpected!"
]

cols = st.columns(2)
for i, example in enumerate(test_examples):
    with cols[i % 2]:
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.rerun()

# Handle URL parameters for examples
query_params = st.experimental_get_query_params()
if "text" in query_params:
    st.text_input("Enter a news headline:", value=query_params["text"][0], key="filled_example")
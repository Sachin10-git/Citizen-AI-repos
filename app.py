from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from datetime import datetime, timedelta
import torch
from functools import wraps


app = Flask(__name__)
app.secret_key = "citizenai_secret"

# AI Model Setup 
MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"

# Determine device
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print(f" Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model safely depending on whether CUDA is available
if use_cuda:
    print(" CUDA available — using 4-bit quantization via bitsandbytes.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        # compute dtype should be float16 on GPU
        bnb_4bit_compute_dtype=torch.float16
    )
    # device_map="auto" will map layers to GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
else:
    # CPU fallback: do NOT use bitsandbytes quantization on CPU.
    # Use low_cpu_mem_usage to reduce peak memory during load.
    print(" CUDA not available — loading model in CPU mode (low memory).")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

# Utility to ensure we put tensors on device when generating
def generate_response(prompt, max_new_tokens=120):
    try:
        print("generate_response called. Prompt:", prompt)
        # Prepare inputs (tokenizer returns CPU tensors by default)
        inputs = tokenizer(prompt, return_tensors="pt")
        # If using GPU, move input tensors to GPU
        if use_cuda:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # generate with a timeout-like exception guard
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Avoid repeating the prompt if returned
        if response.lower().startswith(prompt.lower()):
            response = response[len(prompt):].strip()
        print("🟢 Model produced response (len={}):".format(len(response)))
        return response
    except Exception as e:
        # Print exception to terminal for debugging
        print("❌ Error during generation:", repr(e))
        return "Sorry — the model failed to generate a response (see server logs)."

# Global Variables 
history = []  # Stores chat + feedback + concerns

# Add global sentiment counts that reset each time Flask restarts
sentiment_counts = {
    'Positive': 0,
    'Neutral': 0,
    'Negative': 0
}

# Login Required Decorator 
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# Routes 

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
@login_required
def about():
    return render_template("about.html")

@app.route('/services')
@login_required
def services():
    return render_template("services.html")

# Login / Logout 

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        session["user"] = user
        return redirect(url_for("dashboard"))  # Redirect to dashboard after login
    return render_template("login.html")


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


# Chat Routes 

# Global variable to hold chat history
chat_history = []

@app.route('/chat')
@login_required
def chat():
    return render_template(
        "chat.html",
        history=chat_history,
        question_response=None,
        last_question=None
    )


@app.route('/ask', methods=['POST'])
@login_required
def ask_question():
    question = request.form.get('question', '').strip()
    if not question:
        return redirect(url_for('chat'))

    print("→ Received question:", question)

    # Generate AI response (use helper)
    response = generate_response(question, max_new_tokens=120)

    # Save into history
    history.append({
        'date': datetime.now(),
        'question': question,
        'response': response,
        'sentiment': None,
        'feedback': None,
        'type': 'chat'
    })

    # Render chat page showing only the latest response (keep chat area empty)
    return render_template(
    "chat.html",
    history=[],
    question_response=response or "",
    last_question=question or ""
    )



sentiment_analyzer = pipeline("sentiment-analysis")

# Feedback 

def analyze_sentiment(feedback_text):
    """Analyze feedback sentiment using Hugging Face pipeline."""
    try:
        result = sentiment_analyzer(feedback_text)[0]
        label = result['label'].lower()
        if "pos" in label:
            return "Positive"
        elif "neg" in label:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return "Neutral"


@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    global sentiment_counts  # 🔹 Access global counters

    feedback_text = request.form.get('feedback', "").strip()
    last_question = request.form.get('last_question', "").strip()
    ai_response = request.form.get('ai_response', "").strip()

    if not feedback_text:
        return jsonify({'error': 'Feedback cannot be empty'}), 400

    # Basic keyword-based sentiment detection
    fb_lower = feedback_text.lower()
    positive_words = ['good', 'great', 'excellent', 'amazing', 'helpful', 'nice', 'thank', 'love', 'appreciate', 'perfect']
    negative_words = ['bad', 'poor', 'wrong', 'incomplete', 'incorrect', 'terrible', 'useless', 'not helpful', 'hate', 'dislike']

    if any(word in fb_lower for word in positive_words):
        sentiment = 'Positive'
    elif any(word in fb_lower for word in negative_words):
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    # Update the matching chat entry
    for chat in reversed(history):
        if chat.get('question') == last_question and chat.get('response') == ai_response:
            chat['sentiment'] = sentiment
            chat['feedback'] = feedback_text
            break
    else:
        history.append({
            'date': datetime.now(),
            'question': last_question,
            'response': ai_response,
            'sentiment': sentiment,
            'feedback': feedback_text,
            'type': 'chat'
        })

    # Update global counts (resets each restart)
    sentiment_counts[sentiment] += 1

    return jsonify({'success': True, 'sentiment': sentiment}), 200


# Concern

@app.route('/concern', methods=['POST'])
@login_required
def submit_concern():
    concern_text = request.form.get('concern', '').strip()
    if not concern_text:
        return jsonify({'error': 'Concern cannot be empty'}), 400

    history.append({
        'date': datetime.now(),
        'issue': concern_text,
        'sentiment': 'Neutral',
        'type': 'issue'
    })

    print(" Concern submitted:", concern_text)
    return jsonify({'success': True, 'message': 'Your issue has been recorded successfully!'}), 200


# Dashboard

@app.route("/dashboard")
@login_required
def dashboard():
    global history

    try:
        _history = history
    except NameError:
        _history = []

    seven_days_ago = datetime.now() - timedelta(days=7)
    recent_entries = []
    for h in _history:
        d = h.get("date")
        parsed = None
        if isinstance(d, datetime):
            parsed = d
        elif isinstance(d, str):
            try:
                parsed = datetime.fromisoformat(d)
            except Exception:
                parsed = None

        if parsed and parsed >= seven_days_ago:
            recent_entries.append(h)

    # Fresh sentiment counts for last 7 days
    sentiment_counts_display = {
    "positive": sentiment_counts["Positive"],
    "neutral": sentiment_counts["Neutral"],
    "negative": sentiment_counts["Negative"]
    }
    recent_issues = []

    for h in recent_entries:
        entry_type = h.get("type", "").lower().strip()
        sentiment = str(h.get("sentiment", "")).strip().lower()

        if "pos" in sentiment:
            sentiment_counts["positive"] += 1
        elif "neg" in sentiment:
            sentiment_counts["negative"] += 1
        elif "neu" in sentiment:
            sentiment_counts["neutral"] += 1

        if entry_type == "issue":
            issue_text = h.get("issue", "").strip()
            if issue_text:
                recent_issues.append(issue_text)

    # Merge with cumulative session counts (true totals)
    session_counts = session.get("sentiment_counts", {})
    for key in ["Positive", "Neutral", "Negative"]:
        lower_key = key.lower()
        if key in session_counts:
            sentiment_counts[lower_key] = session_counts[key]

    # Limit to last 10 issues
    recent_issues = recent_issues[:10]

    return render_template(
    "dashboard.html",
    sentiment_counts=sentiment_counts_display,
    recent_issues=recent_issues,
    session=session,
    history=_history
)

# Main

if __name__ == "__main__":
    app.run(debug=True)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline

# =========================
# CONFIG
# =========================
MODEL_NAME = "distilgpt2"
SAVE_PATH = "./empathetic-chatbot"

# =========================
# SAFETY FILTER
# =========================
def safe_response(user_input):
    crisis_words = ["suicide", "kill myself", "hopeless", "depressed"]

    for word in crisis_words:
        if word in user_input.lower():
            return "I'm really sorry you're feeling this way. Please consider reaching out to a trusted person or a mental health professional."

    return None

# =========================
# TRAINING FUNCTION
# =========================
def train_model():
    print("📊 Loading dataset...")
    dataset = load_dataset("empathetic_dialogues")

    print("🧹 Preprocessing...")
    def preprocess(example):
        return {
            "text": f"User: {example['prompt']} \nBot: {example['utterance']}"
        }

    dataset = dataset.map(preprocess)

    print("🤖 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.1eos_token

    print("🔤 Tokenizing...")
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    print("🏋️ Training...")
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        per_device_train_batch_size=4,
        num_train_epochs=2,
        logging_dir="./logs",
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"]
    )

    trainer.train()

    print("💾 Saving model...")
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("✅ Training Complete!")

# =========================
# CLI CHATBOT
# =========================
def run_cli():
    print("💙 Loading chatbot...")
    chatbot = pipeline(
        "text-generation",
        model=SAVE_PATH,
        tokenizer=SAVE_PATH
    )

    print("💬 Empathetic Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        safe = safe_response(user_input)
        if safe:
            print("Bot:", safe)
            continue

        prompt = f"User: {user_input}\nBot:"

        response = chatbot(
            prompt,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

        reply = response[0]["generated_text"].split("Bot:")[-1]
        print("Bot:", reply.strip())

# =========================
# STREAMLIT APP
# =========================
def run_streamlit():
    import streamlit as st

    chatbot = pipeline(
        "text-generation",
        model=SAVE_PATH,
        tokenizer=SAVE_PATH
    )

    st.set_page_config(page_title="Empathetic Chatbot")
    st.title(" Emotional Support Chatbot")

    user_input = st.text_input("How are you feeling today?")

    if user_input:
        safe = safe_response(user_input)

        if safe:
            st.write("Bot:", safe)
        else:
            prompt = f"User: {user_input}\nBot:"

            response = chatbot(
                prompt,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )

            reply = response[0]["generated_text"].split("Bot:")[-1]
            st.write("Bot:", reply.strip())

# =========================
# MAIN MENU
# =========================
if __name__ == "__main__":
    print("""
Choose option:
1 → Train Model
2 → Run CLI Chatbot
3 → Run Streamlit Web App
""")

    choice = input("Enter choice: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        run_cli()
    elif choice == "3":
        run_streamlit()
    else:
        print("Invalid choice")
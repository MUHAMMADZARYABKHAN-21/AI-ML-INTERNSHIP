"""
=============================================================
  Task 4: General Health Query Chatbot (Prompt Engineering)
  Student: Zaryab | Course: Machine Learning | Air University
=============================================================

Description:
    A conversational health chatbot that uses an LLM (Claude via
    Anthropic API) to answer general health questions. Built with
    prompt engineering for friendly, clear responses and includes
    safety filters to prevent harmful medical advice.

Skills Demonstrated:
    - Prompt design and testing
    - API usage for LLMs (Anthropic Claude)
    - Safety handling in chatbot responses
    - Building simple conversational agents
"""

import sys

import anthropic

# ─────────────────────────────────────────────────────────────
# SECTION 1: PROMPT ENGINEERING
# ─────────────────────────────────────────────────────────────
# The system prompt is the core of our prompt engineering.
# It shapes the LLM's personality, tone, and safety behavior.

SYSTEM_PROMPT = """
You are HealthBot, a friendly and knowledgeable general health assistant.

Your role:
- Answer general health and wellness questions clearly and warmly.
- Explain medical terms in simple, easy-to-understand language.
- Give educational information about symptoms, causes, and general remedies.
- Always encourage users to consult a real doctor for diagnosis or treatment.

Your personality:
- Friendly, calm, and reassuring — like a knowledgeable friend.
- Use simple words. Avoid scary or overly technical language.
- Be concise. Keep answers under 150 words unless the topic needs more depth.
- Use bullet points or numbered lists when explaining steps or multiple causes.

Safety rules you MUST always follow:
1. NEVER diagnose a specific disease or medical condition.
2. NEVER prescribe medication dosages or treatment plans.
3. NEVER provide advice for emergencies — always direct to emergency services (e.g., call 1122 or go to the ER).
4. If a question is about mental health crises (suicide, self-harm), respond with compassion and direct to a helpline.
5. Always end responses with a gentle reminder to consult a licensed doctor for personal medical advice.
6. If the question is NOT health-related, politely say you can only help with health topics.

Response format:
- Start with a warm, friendly opening.
- Give clear, helpful information.
- End with: "💡 Remember: Always consult a doctor for personal medical advice."
"""

# ─────────────────────────────────────────────────────────────
# SECTION 2: SAFETY FILTERS
# ─────────────────────────────────────────────────────────────
# Pre-filter: Catches dangerous query patterns BEFORE sending to the LLM.
# This is the first layer of defense.

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "stroke", "unconscious", "not breathing", "overdose", "poisoning",
    "severe bleeding", "choking", "seizure", "fainted", "collapsed"
]

SELF_HARM_KEYWORDS = [
    "kill myself", "suicide", "end my life", "self harm", "self-harm",
    "hurt myself", "want to die", "no reason to live"
]

PRESCRIPTION_KEYWORDS = [
    "prescribe me", "give me a prescription", "write me a prescription",
    "what dose should i take", "can you prescribe"
]

DIAGNOSIS_KEYWORDS = [
    "diagnose me", "do i have", "am i sick with", "tell me if i have"
]


def check_safety_filters(user_input: str) -> str | None:
    """
    Pre-check the user's input for dangerous patterns.
    Returns a safety response string if triggered, otherwise None.

    This is Layer 1 of safety — runs BEFORE the LLM sees the message.
    """
    text = user_input.lower()

    # Check for emergencies
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text:
            return (
                "🚨 This sounds like a medical emergency!\n\n"
                "Please do NOT wait for a chatbot response.\n"
                "➡  Call emergency services immediately: **1122** (Pakistan)\n"
                "➡  Or go to the nearest Emergency Room (ER) right away.\n\n"
                "Your safety comes first. Please get help now."
            )

    # Check for self-harm / mental health crisis
    for keyword in SELF_HARM_KEYWORDS:
        if keyword in text:
            return (
                "💙 I hear you, and I'm really glad you reached out.\n\n"
                "What you're feeling matters, and you deserve support from "
                "someone trained to help.\n\n"
                "Please contact a mental health helpline right now:\n"
                "➡  Umang Pakistan: **0317-4288665** (24/7)\n"
                "➡  Rozan Counseling: **051-2890505**\n\n"
                "You are not alone. 💙"
            )

    # Check for prescription requests
    for keyword in PRESCRIPTION_KEYWORDS:
        if keyword in text:
            return (
                "⚠️  I'm not able to prescribe medications — only a licensed "
                "doctor can do that.\n\n"
                "I can share general information about how medications work, "
                "but for any prescription, please visit a qualified physician or pharmacist."
            )

    # Check for diagnosis requests
    for keyword in DIAGNOSIS_KEYWORDS:
        if keyword in text:
            return (
                "⚠️  I'm not able to diagnose medical conditions — only a "
                "qualified doctor can do that after a proper examination.\n\n"
                "I can explain what certain symptoms might generally be "
                "associated with, but for an actual diagnosis, please consult a doctor."
            )

    return None  # No safety filter triggered — safe to send to LLM


# ─────────────────────────────────────────────────────────────
# SECTION 3: CHATBOT CLASS
# ─────────────────────────────────────────────────────────────

class HealthChatbot:
    """
    A conversational health chatbot powered by Claude (Anthropic API).

    Features:
    - Multi-turn conversation with memory (conversation history)
    - Two-layer safety filtering (pre-LLM + post-LLM)
    - Friendly prompt engineering via system prompt
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key="sk-ant-YOUR-KEY-HERE")
        self.model = "claude-sonnet-4-20250514"   # Claude Sonnet 4
        self.conversation_history = []             # Stores chat turns
        self.max_tokens = 512

    def add_to_history(self, role: str, content: str):
        """Append a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def post_process_response(self, response: str) -> str:
        """
        Layer 2 safety: Post-process the LLM's response.
        Ensures the safety disclaimer is always present.
        """
        disclaimer = "💡 Remember: Always consult a doctor for personal medical advice."
        if disclaimer.lower() not in response.lower():
            response += f"\n\n{disclaimer}"
        return response

    def chat(self, user_message: str) -> str:
        """
        Main chat method. Handles the full pipeline:
        1. Safety pre-filter
        2. Send to LLM with conversation history
        3. Safety post-process
        4. Return response
        """
        # ── Layer 1: Pre-LLM Safety Filter ──
        safety_response = check_safety_filters(user_message)
        if safety_response:
            return safety_response

        # ── Add user message to history ──
        self.add_to_history("user", user_message)

        # ── Send to LLM ──
        try:
            api_response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=self.conversation_history
            )
            bot_reply = api_response.content[0].text

        except anthropic.APIError as e:
            bot_reply = f"⚠️  API error: {str(e)}\nPlease try again or check your API key."
            self.conversation_history.pop()  # Remove the failed user message
            return bot_reply

        # ── Layer 2: Post-LLM Safety Check ──
        bot_reply = self.post_process_response(bot_reply)

        # ── Add bot reply to history ──
        self.add_to_history("assistant", bot_reply)

        return bot_reply

    def reset_conversation(self):
        """Clear conversation history to start fresh."""
        self.conversation_history = []
        print("\n🔄 Conversation history cleared. Starting fresh!\n")


# ─────────────────────────────────────────────────────────────
# SECTION 4: DEMO — RUN PREDEFINED QUERIES
# ─────────────────────────────────────────────────────────────

def run_demo(bot: HealthChatbot):
    """
    Runs a set of predefined demo queries to test the chatbot.
    Used for assignment demonstration.
    """
    demo_queries = [
        # General health questions
        "What causes a sore throat?",
        "Is paracetamol safe for children?",
        "How much water should I drink per day?",
        "What are common symptoms of dehydration?",
        # Safety filter tests
        "I have chest pain and can't breathe",   # Should trigger emergency filter
        "Can you diagnose me with diabetes?",     # Should trigger diagnosis filter
        "I want to kill myself",                 # Should trigger mental health filter
    ]

    print("\n" + "═" * 60)
    print("  🤖  HEALTH CHATBOT — DEMO MODE")
    print("═" * 60)

    for i, query in enumerate(demo_queries, 1):
        print(f"\n[Query {i}] 👤 User: {query}")
        print("-" * 50)
        response = bot.chat(query)
        print(f"🩺 HealthBot:\n{response}")
        print("-" * 50)

        # Reset after safety filter tests to avoid polluting history
        if i == 4:
            bot.reset_conversation()


# ─────────────────────────────────────────────────────────────
# SECTION 5: INTERACTIVE CLI MODE
# ─────────────────────────────────────────────────────────────

def run_interactive(bot: HealthChatbot):
    """
    Runs an interactive command-line chat session.
    Type 'quit' to exit, 'reset' to clear history.
    """
    print("\n" + "═" * 60)
    print("  🩺  HEALTH CHATBOT — Interactive Mode")
    print("═" * 60)
    print("Ask me any general health question!")
    print("Commands: 'reset' → clear chat | 'quit' → exit\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye! Stay healthy!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n👋 Goodbye! Stay healthy!")
            break

        if user_input.lower() == "reset":
            bot.reset_conversation()
            continue

        print("\n🩺 HealthBot:", end=" ", flush=True)
        response = bot.chat(user_input)
        print(response, "\n")


# ─────────────────────────────────────────────────────────────
# SECTION 6: ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = HealthChatbot()

    # Check command-line argument for mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "interactive":
        run_interactive(bot)
    else:
        run_demo(bot)
        print("\n\n✅ Demo complete!")
        print("Run with 'python health_chatbot.py interactive' for live chat mode.\n")
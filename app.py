import streamlit as st
from langchain.llms import OpenAI
from trulens_eval import Tru, Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
import numpy as np

st.title("📊 LLM Report Card")

# Description based on the technology used to grade LLMs
st.markdown("This LLM Report Card provides an evaluation and analysis of Language Models (LLMs). "
            "Powered by Langchain, it leverages OpenAI's advanced language model to generate responses. "
            "The grading process incorporates TruLens, a tool for assessing the performance of LLMs, "
            "using feedback mechanisms such as groundedness. Explore the insights and scores for a comprehensive evaluation.")

# Sidebar for OpenAI API key input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

# Sidebar for Google Vortex API key input
with st.sidebar:
    google_vortex_api_key = st.text_input("Google Vortex API Key", type="password")
    st.markdown("Get your Google Vortex API key from the Google Cloud Console.")

# Sidebar for LangSmith API key input
with st.sidebar:
    langsmith_api_key = st.text_input("LangSmith API Key", type="password", value="Your_LangSmith_API_Key_Here")
    st.markdown("Replace 'Your_LangSmith_API_Key_Here' with your actual LangSmith API key.")

# Check if OpenAI API key is provided
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
else:
    # Langchain LLMS function for generating responses
    def generate_response(input_text):
        llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
        st.info(llm(input_text))

    # TruLens Quickstart code (replace with your specific code)
    with st.beta_expander("TruLens Features"):
        # Feature 1: Groundedness Evaluation
        feedback_groundedness = Feedback(Groundedness())
        groundedness_score = Tru.evaluate(llm, feedback_groundedness)
        st.write(f"1. Groundedness Score: {groundedness_score}")

        # Feature 2: Coherence Analysis
        feedback_coherence = Feedback("Evaluate coherence and flow.")
        coherence_score = Tru.evaluate(llm, feedback_coherence)
        st.write(f"2. Coherence Score: {coherence_score}")

        # Feature 3: Novelty Assessment
        feedback_novelty = Feedback("Assess the novelty of generated content.")
        novelty_score = Tru.evaluate(llm, feedback_novelty)
        st.write(f"3. Novelty Score: {novelty_score}")

        # Feature 4: Specificity Evaluation
        feedback_specificity = Feedback("Check for specificity and detail in responses.")
        specificity_score = Tru.evaluate(llm, feedback_specificity)
        st.write(f"4. Specificity Score: {specificity_score}")

        # Feature 5: Diversity Analysis
        feedback_diversity = Feedback("Analyze the diversity of generated responses.")
        diversity_score = Tru.evaluate(llm, feedback_diversity)
        st.write(f"5. Diversity Score: {diversity_score}")

    # Streamlit form for input text and response generation
    with st.form("my_form"):
        text = st.text_area("Enter text:", "What are 3 key pieces of advice for learning how to code?")
        submitted = st.form_submit_button("Submit")

        # Check if the form is submitted
        if submitted:
            generate_response(text)

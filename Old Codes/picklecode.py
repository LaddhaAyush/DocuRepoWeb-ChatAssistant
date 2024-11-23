import streamlit as st
from diffusers import StableDiffusionPipeline
import pickle
import os

# Function to load the model with caching enabled via Streamlit
@st.cache_resource
def load_model():
    try:
        # Load the Stable Diffusion model with explicit cache directory
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            cache_dir="./model_cache",  # Set the cache directory directly in the method
            low_cpu_mem_usage=True      # Enable memory optimization
        ).to("cpu")
        return pipe
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to save keywords to a pickle file
def save_keywords_to_pickle(keywords):
    with open("keywords.pkl", "wb") as f:
        pickle.dump(keywords, f)

# Function to load keywords from a pickle file
def load_keywords_from_pickle():
    if os.path.exists("keywords.pkl"):
        with open("keywords.pkl", "rb") as f:
            return pickle.load(f)
    return None

# Load the model once and cache it
pipe = load_model()

# Set up Streamlit page
st.title("Generate Interior Design Ideas")

# Sidebar for user inputs
with st.sidebar:
    st.subheader("Enter design keywords")
    keywords = st.text_input("Enter design keywords (e.g., modern, cozy, minimalist)")

    # Load previous keywords from pickle file if available
    previous_keywords = load_keywords_from_pickle()
    if previous_keywords:
        st.write(f"Previously used keywords: {previous_keywords}")

    if st.button("Generate Design"):
        if keywords:
            # Save the entered keywords to a pickle file
            save_keywords_to_pickle(keywords)

            if pipe:  # Ensure the model is loaded
                with st.spinner("Generating design..."):
                    try:
                        # Generate the image based on the entered keywords
                        prompt = f"A {keywords} interior design."
                        image = pipe(prompt).images[0]

                        # Display the generated image
                        st.image(image, caption="Generated Interior Design", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error generating design: {e}")
            else:
                st.error("Model failed to load. Please try again later.")
        else:
            st.error("Please enter design keywords.")

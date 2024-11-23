import streamlit as st
from diffusers import StableDiffusionPipeline

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

# Function to generate image based on keywords
def generate_image(keywords):
    pipe = load_model()
    if pipe:
        try:
            prompt = f"A {keywords} interior design."
            image = pipe(prompt).images[0]
            return image
        except Exception as e:
            st.error(f"Error generating design: {e}")
            return None
    else:
        st.error("Model failed to load. Please try again later.")
        return None
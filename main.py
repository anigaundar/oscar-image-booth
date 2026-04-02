"""
Medical Conference Photo Booth - Caricature Generator
=====================================================
Uses GPT Image model (gpt-image-1) to transform selfies into
medical caricatures for virtual conference attendees.

Setup:
  pip install streamlit openai Pillow

Run:
  streamlit run app.py

Secrets (.streamlit/secrets.toml):
  OPENAI_API_KEY = "sk-..."
"""

import streamlit as st
import openai
import base64
import io
import time
import json
import os
from datetime import datetime
from PIL import Image
import csv
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()
# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

SPECIALTIES = {
    "Cardiologist": {
        "items": "heart monitors, ECG/EKG charts, a heart anatomy poster, cardiac rhythm strips on a screen, a defibrillator",
        "badge_title": "CARDIOLOGIST",
    },
    "Andrologist": {
        "items": "a microscope, sperm analysis charts, lab test tubes, urology anatomy poster, a laptop with lab data graphs",
        "badge_title": "ANDROLOGIST",
    },
    "Neurologist": {
        "items": "brain MRI scans on a lightbox, a brain anatomy model, neural pathway diagrams, an EEG monitor",
        "badge_title": "NEUROLOGIST",
    },
    "Dermatologist": {
        "items": "a dermatoscope, skin anatomy charts, skincare product bottles, a magnifying lamp, skin biopsy slides",
        "badge_title": "DERMATOLOGIST",
    },
    "Orthopedic Surgeon": {
        "items": "X-ray films of bones and joints, a skeleton model, surgical instruments, joint replacement implants on a tray",
        "badge_title": "ORTHOPEDIC SURGEON",
    },
    "Pediatrician": {
        "items": "colorful growth charts, toy stethoscope, children's health posters, teddy bears, vaccination schedule board",
        "badge_title": "PEDIATRICIAN",
    },
    "Ophthalmologist": {
        "items": "an eye chart, slit lamp, eye anatomy model, trial lens set, retinal scan images on a monitor",
        "badge_title": "OPHTHALMOLOGIST",
    },
    "General Surgeon": {
        "items": "surgical instruments on a tray, anatomy charts, an operating light, surgical monitors, scrub caps",
        "badge_title": "GENERAL SURGEON",
    },
    "Oncologist": {
        "items": "cell biology diagrams, chemotherapy IV drip, cancer research papers, microscope slides, PET scan images",
        "badge_title": "ONCOLOGIST",
    },
    "Psychiatrist": {
        "items": "brain anatomy posters, psychology bookshelf, calming artwork, a notepad and pen, DSM manual on desk",
        "badge_title": "PSYCHIATRIST",
    },
}

# ──────────────────────────────────────────────
# COST CONFIG (gpt-image-1 pricing as of 2025)
# Update these if OpenAI changes pricing
# ──────────────────────────────────────────────
# Pricing: https://platform.openai.com/docs/pricing
# gpt-image-1 — images.edit
# Quality "high", size "1024x1536" → check OpenAI pricing page
IMAGE_COST_TABLE = {
    ("low", "1024x1024"): 0.011,
    ("low", "1024x1536"): 0.016,
    ("low", "1536x1024"): 0.016,
    ("medium", "1024x1024"): 0.042,
    ("medium", "1024x1536"): 0.063,
    ("medium", "1536x1024"): 0.063,
    ("high", "1024x1024"): 0.167,
    ("high", "1024x1536"): 0.250,
    ("high", "1536x1024"): 0.250,
}

COST_LOG_FILE = "cost_log.csv"


def init_cost_log():
    """Create cost log CSV if it doesn't exist."""
    if not Path(COST_LOG_FILE).exists():
        with open(COST_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "user_name", "specialty", "country",
                "model", "quality", "size",
                "estimated_cost_usd", "prompt_length_chars",
                "status", "error_message",
            ])


def log_cost(
    user_name: str,
    specialty: str,
    country: str,
    model: str,
    quality: str,
    size: str,
    cost: float,
    prompt_length: int,
    status: str = "success",
    error_message: str = "",
):
    """Append one row to the cost log CSV."""
    init_cost_log()
    with open(COST_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            user_name,
            specialty,
            country,
            model,
            quality,
            size,
            f"{cost:.4f}",
            prompt_length,
            status,
            error_message,
        ])


def get_estimated_cost(quality: str, size: str) -> float:
    """Look up estimated cost from the pricing table."""
    return IMAGE_COST_TABLE.get((quality, size), 0.0)


def load_cost_log() -> list[dict]:
    """Load all rows from the cost log."""
    if not Path(COST_LOG_FILE).exists():
        return []
    rows = []
    with open(COST_LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


COUNTRIES = [
    "Egypt", "Saudi Arabia", "UAE", "India", "USA", "UK", "Germany",
    "France", "Canada", "Australia", "Jordan", "Lebanon", "Iraq",
    "Kuwait", "Qatar", "Bahrain", "Oman", "Morocco", "Tunisia",
    "Pakistan", "Bangladesh", "Turkey", "South Africa", "Brazil",
    "Mexico", "Italy", "Spain", "Japan", "South Korea", "China",
    "Other",
]

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def get_openai_client():
    """Initialize OpenAI client with API key from Streamlit secrets."""
    api_key = os.environ.get("OPENAI_API_KEY", "") #st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Add it to `.streamlit/secrets.toml`.")
        st.stop()
    return openai.OpenAI(api_key=api_key)


def build_prompt(name: str, gender: str, age: int, specialty: str, country: str) -> str:
    """Build a detailed image generation prompt from user details."""
    spec = SPECIALTIES.get(specialty, SPECIALTIES["Cardiologist"])
    items = spec["items"]
    badge = spec["badge_title"]

    gender_word = "male" if gender == "Male" else "female"
    pronoun = "he" if gender == "Male" else "she"

    prompt = f"""Transform this photo into a caricature-style illustrated portrait.

CRITICAL — FACE ACCURACY (HIGHEST PRIORITY):
- You MUST preserve the person's EXACT facial features from the uploaded photo.
- Keep the SAME face shape, nose shape, eye shape, eyebrow thickness, 
  jawline, chin, forehead size, and ear shape.
- Keep the SAME facial hair style (beard, mustache, stubble, clean-shaven) 
  exactly as shown in the photo — do NOT add, remove, or change facial hair.
- Keep the SAME hairstyle, hair color, hair length, and hairline.
- Keep the SAME skin tone and complexion.
- The person looking at this caricature MUST immediately recognize themselves.
- Only apply MILD caricature exaggeration — slightly larger head, slightly 
  more expressive smile — but the core facial structure must stay 90% true 
  to the original photo. Think "gentle caricature", NOT extreme distortion.

Subject setup:
- {gender_word} doctor, approximately {age} years old.
- Close-up / chest-up shot, facing the viewer with a warm smile.
- Wearing teal medical scrubs with a navy blue V-neck undershirt.
- A stethoscope draped around {pronoun}r neck.
- A name badge clipped to the scrubs reading:
  "Dr. {name}" on top and "{badge}" below it.

Background:
- Medical items related to {specialty}: {items}.
- A small {country} national flag in the upper-right corner.
- Professional, friendly medical office setting.

Art style:
- Painted / illustrated editorial cartoon look with visible brush texture.
- NOT photorealistic, NOT 3D render, NOT anime.
- Warm color palette, soft lighting, high detail on face and badge.

Important: The face must look like the SAME PERSON in the photo, just 
in an illustrated style. Prioritize likeness over stylization.
"""
    return prompt


def image_to_png_bytes(uploaded) -> bytes:
    """Convert any uploaded image to PNG bytes."""
    img = Image.open(uploaded)
    img = img.convert("RGB")

    # Resize if too large (API limit & speed)
    max_dim = 1536
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_caricature(client: openai.OpenAI, selfie_bytes: bytes, prompt: str) -> dict:
    """Call GPT Image API and return result with cost tracking."""

    model = "gpt-image-1"
    quality = "high"
    size = "1024x1536"

    selfie_file = io.BytesIO(selfie_bytes)
    selfie_file.name = "selfie.png"

    start_time = time.time()

    response = client.images.edit(
        model=model,
        image=selfie_file,
        prompt=prompt,
        size=size,
        quality=quality,
    )

    elapsed = time.time() - start_time

    # Decode base64 result
    b64 = response.data[0].b64_json
    image_bytes = base64.b64decode(b64)

    # Extract usage / cost info from response
    estimated_cost = get_estimated_cost(quality, size)

    # Check if API returned usage metadata (newer API versions)
    usage_info = {}
    if hasattr(response, "usage") and response.usage:
        usage_info = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    return {
        "image_bytes": image_bytes,
        "model": model,
        "quality": quality,
        "size": size,
        "estimated_cost_usd": estimated_cost,
        "elapsed_seconds": round(elapsed, 2),
        "prompt_length_chars": len(prompt),
        "usage": usage_info,
    }


# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Medical Conference Photo Booth",
        page_icon="🩺",
        layout="centered",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        color: #0e7a6d;
        font-size: 2rem;
    }
    .main-header p {
        color: #555;
        font-size: 1.1rem;
    }
    div[data-testid="stImage"] img {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Medical Conference Photo Booth</h1>
        <p>Get your personalized doctor caricature in seconds!</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Initialize session state ──
    if "result_image" not in st.session_state:
        st.session_state.result_image = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""

    # ────────────── STEP 1: User Details ──────────────
    st.subheader("Step 1 — Your Details")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", placeholder="e.g. Ahmed Hassan")
        age = st.number_input("Age", min_value=18, max_value=90, value=35)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        specialty = st.selectbox("Medical Specialty", list(SPECIALTIES.keys()))

    country = st.selectbox("Country (for flag)", COUNTRIES)

    st.divider()

    # ────────────── STEP 2: Selfie ──────────────
    st.subheader("Step 2 — Your Photo")

    capture_mode = st.radio(
        "How would you like to provide your photo?",
        ["📷 Take a Selfie", "📁 Upload a Photo"],
        horizontal=True,
    )

    selfie_data = None

    if capture_mode == "📷 Take a Selfie":
        camera_photo = st.camera_input("Smile! 😊")
        if camera_photo:
            selfie_data = camera_photo
    else:
        uploaded_photo = st.file_uploader(
            "Upload a clear face photo",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if uploaded_photo:
            selfie_data = uploaded_photo

    # Show preview
    if selfie_data:
        st.image(selfie_data, caption="Your photo", width=250)

    st.divider()

    # ────────────── STEP 3: Generate ──────────────
    st.subheader("Step 3 — Generate Your Caricature")

    # Validation
    ready = True
    if not name.strip():
        st.info("Please enter your name above.")
        ready = False
    if not selfie_data:
        st.info("Please provide a photo above.")
        ready = False

    generate_btn = st.button(
        "🎨 Generate My Caricature",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    if generate_btn and ready:
        client = get_openai_client()
        prompt = build_prompt(name.strip(), gender, age, specialty, country)

        # Show prompt in expander (optional debug)
        with st.expander("🔍 View generated prompt (debug)"):
            st.code(prompt, language="text")

        selfie_bytes = image_to_png_bytes(selfie_data)

        with st.spinner("🎨 Creating your caricature... This may take 20-40 seconds..."):
            progress = st.progress(0, text="Sending to AI...")
            try:
                # Simulated progress (actual API call is blocking)
                progress.progress(20, text="Uploading your selfie...")
                result = generate_caricature(client, selfie_bytes, prompt)
                progress.progress(90, text="Almost done...")
                time.sleep(0.5)
                progress.progress(100, text="Done! ✅")

                st.session_state.result_image = result["image_bytes"]
                st.session_state.user_name = name.strip()
                st.session_state.last_cost = result

                # Log successful generation
                log_cost(
                    user_name=name.strip(),
                    specialty=specialty,
                    country=country,
                    model=result["model"],
                    quality=result["quality"],
                    size=result["size"],
                    cost=result["estimated_cost_usd"],
                    prompt_length=result["prompt_length_chars"],
                    status="success",
                )

            except openai.BadRequestError as e:
                progress.empty()
                st.error(f"🚫 Image generation was refused: {e}")
                st.info("Tip: Try a different photo — clear, well-lit face photos work best.")
                log_cost(
                    user_name=name.strip(), specialty=specialty, country=country,
                    model="gpt-image-1", quality="high", size="1024x1536",
                    cost=0.0, prompt_length=len(prompt),
                    status="error", error_message=str(e),
                )
            except openai.RateLimitError as e:
                progress.empty()
                st.warning("⏳ Rate limit reached. Please wait a moment and try again.")
                log_cost(
                    user_name=name.strip(), specialty=specialty, country=country,
                    model="gpt-image-1", quality="high", size="1024x1536",
                    cost=0.0, prompt_length=len(prompt),
                    status="rate_limited", error_message=str(e),
                )
            except Exception as e:
                progress.empty()
                st.error(f"❌ Something went wrong: {e}")
                log_cost(
                    user_name=name.strip(), specialty=specialty, country=country,
                    model="gpt-image-1", quality="high", size="1024x1536",
                    cost=0.0, prompt_length=len(prompt),
                    status="error", error_message=str(e),
                )

    # ────────────── RESULT DISPLAY ──────────────
    if st.session_state.result_image:
        st.divider()
        st.subheader("🖼️ Your Caricature")

        st.image(
            st.session_state.result_image,
            caption=f"Dr. {st.session_state.user_name} — {specialty}",
            use_container_width=True,
        )

        # Download button
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            safe_name = st.session_state.user_name.replace(" ", "_")
            st.download_button(
                label="⬇️ Download Caricature",
                data=st.session_state.result_image,
                file_name=f"Dr_{safe_name}_caricature.png",
                mime="image/png",
                use_container_width=True,
            )

        # Reset button
        if st.button("🔄 Create Another", use_container_width=True):
            st.session_state.result_image = None
            st.session_state.user_name = ""
            st.session_state.last_cost = None
            st.rerun()

    # ────────────── ADMIN: COST DASHBOARD ──────────────
    st.divider()
    # with st.expander("📊 Admin — Cost & Usage Dashboard"):
    #     logs = load_cost_log()
    #     if not logs:
    #         st.info("No generations yet. Costs will appear here after the first caricature.")
    #     else:
    #         # Summary metrics
    #         total_cost = sum(float(r.get("estimated_cost_usd", 0)) for r in logs)
    #         success_count = sum(1 for r in logs if r.get("status") == "success")
    #         error_count = sum(1 for r in logs if r.get("status") != "success")

    #         m1, m2, m3, m4 = st.columns(4)
    #         m1.metric("Total Generations", len(logs))
    #         m2.metric("Successful", success_count)
    #         m3.metric("Errors", error_count)
    #         m4.metric("Total Cost", f"${total_cost:.2f}")

    #         # Per-specialty breakdown
    #         st.markdown("**Cost by Specialty:**")
    #         specialty_costs = {}
    #         for r in logs:
    #             sp = r.get("specialty", "Unknown")
    #             specialty_costs[sp] = specialty_costs.get(sp, 0) + float(r.get("estimated_cost_usd", 0))
    #         for sp, cost in sorted(specialty_costs.items(), key=lambda x: -x[1]):
    #             st.text(f"  {sp}: ${cost:.2f}")

    #         # Per-country breakdown
    #         st.markdown("**Generations by Country:**")
    #         country_counts = {}
    #         for r in logs:
    #             ct = r.get("country", "Unknown")
    #             country_counts[ct] = country_counts.get(ct, 0) + 1
    #         for ct, count in sorted(country_counts.items(), key=lambda x: -x[1]):
    #             st.text(f"  {ct}: {count}")

    #         # Full log table
    #         st.markdown("**Full Log:**")
    #         st.dataframe(logs, use_container_width=True)

    #         # Download CSV
    #         if Path(COST_LOG_FILE).exists():
    #             with open(COST_LOG_FILE, "rb") as f:
    #                 st.download_button(
    #                     "⬇️ Download Cost Log CSV",
    #                     data=f.read(),
    #                     file_name="photo_booth_cost_log.csv",
    #                     mime="text/csv",
    #                 )

    # ── Footer ──
    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#999; font-size:0.85rem;'>"
        "Powered by GPT Image AI • Built for Medical Conference Photo Booth"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

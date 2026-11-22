import io
import streamlit as st
import numpy as np
import cv2

from PIL import Image, ImageOps
from torchvision import transforms

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from models.predictor import predict_image, PRIMARY_MODEL, PRIMARY_CLASS_MAPPING
from models.decision_engine import assess_prediction
from models.gradcam import generate_gradcam


# -------------------------------------------------------------------
# THEME
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --forest: #174b35;
        --forest-2: #236747;
        --leaf: #dcebdc;
        --cream: #f6f8f3;
        --ink: #1e3027;
        --muted: #64746a;
        --line: #e1e9df;
        --gold: #d99a2b;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--cream);
        color: var(--ink) !important;
        font-family: "Segoe UI", sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(246,248,243,0.92); }
    [data-testid="stSidebar"] {
        background: #edf3e9;
        border-right: 1px solid #dce6d9;
    }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }
    [data-testid="stSidebar"] .stRadio label {
        padding: 8px 10px;
        border-radius: 9px;
        font-weight: 600;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px; }
    h1, h2, h3, h4, p, li, label, span, div { color: inherit; }
    h1, h2, h3 { letter-spacing: -0.025em; }
    h1 { color: var(--forest) !important; }
    h2, h3 { color: var(--forest) !important; }
    p, li, .stMarkdown { color: var(--ink); }
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid var(--line);
        padding: 17px 18px; border-radius: 15px;
        box-shadow: 0 3px 14px rgba(23,75,53,.045);
    }
    [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--forest) !important; }
    [data-testid="stFileUploader"] {
        background: white; border: 1.5px dashed #9db9a2;
        border-radius: 16px; padding: 14px;
    }
    [data-testid="stFileUploader"] * { color: var(--ink) !important; }
    /* Upload button - normal state */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {
        background-color: #236747 !important;
        color: #ffffff !important;
        border-color: #236747 !important;
        font-weight: 700 !important;
    }

    /* Upload button - hover state */
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] button:hover * {
        background-color: #ffffff !important;
        color: #174b35 !important;
        border-color: #236747 !important;
    }
    .hero {
        background: linear-gradient(115deg, #123f2d 0%, #1e6545 65%, #2d7852 100%);
        padding: 42px 44px; border-radius: 23px; color: white;
        box-shadow: 0 12px 30px rgba(18,63,45,.13);
        margin: 5px 0 25px 0;
    }
    .hero * { color: white !important; }
    .hero-kicker {
        display: inline-block; background: rgba(255,255,255,.14);
        border: 1px solid rgba(255,255,255,.22); border-radius: 30px;
        padding: 6px 12px; font-size: 12px; font-weight: 700;
        letter-spacing: .09em; text-transform: uppercase;
    }
    .hero h1 { font-size: 42px; margin: 17px 0 10px 0; color: white !important; }
    .hero p { font-size: 17px; line-height: 1.65; max-width: 760px; color: #e8f3e9 !important; }
    .section-eyebrow {
        color: var(--forest-2); font-size: 12px; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: 5px;
    }
    .section-heading { font-size: 26px; font-weight: 750; color: var(--forest); margin-bottom: 5px; }
    .section-sub { color: var(--muted); margin-bottom: 18px; }
    .info-card {
        height: 100%; background: white; border: 1px solid var(--line);
        border-radius: 16px; padding: 20px 21px;
        box-shadow: 0 4px 16px rgba(23,75,53,.04);
    }
    .info-card .icon { font-size: 25px; margin-bottom: 9px; }
    .info-card h3 { font-size: 17px; margin: 0 0 8px 0; color: var(--forest) !important; }
    .info-card p { color: var(--muted) !important; font-size: 14px; line-height: 1.55; margin: 0; }
    .step-card {
        background: white; border: 1px solid var(--line); border-radius: 15px;
        padding: 18px; min-height: 145px;
    }
    .step-number {
        display: inline-flex; width: 30px; height: 30px; border-radius: 50%;
        background: var(--leaf); color: var(--forest); align-items: center;
        justify-content: center; font-weight: 800; margin-bottom: 10px;
    }
    .step-card h4 { margin: 0 0 6px 0; color: var(--forest) !important; }
    .step-card p { color: var(--muted) !important; font-size: 13px; margin: 0; line-height: 1.5; }
    .soft-panel {
        background: #edf4eb; border: 1px solid #dce9da;
        border-radius: 15px; padding: 17px 20px;
    }
    .warning-panel {
        background: #fff5df; border-left: 4px solid var(--gold);
        border-radius: 10px; padding: 14px 17px; color: #624817;
    }
    .result-banner {
        background: #e9f3e9; border: 1px solid #d3e5d3;
        border-radius: 16px; padding: 20px 22px; margin: 12px 0 18px;
    }
    .result-banner h3 { margin: 0 0 5px 0; color: var(--forest) !important; }
    .result-banner p { margin: 0; color: var(--muted) !important; }
    .crop-pill {
        display: inline-block; background: #e7f0e5; color: #24583a;
        border-radius: 20px; padding: 5px 10px; margin: 3px;
        font-size: 12px; font-weight: 650;
    }
        div.stButton > button,
    div.stButton > button * {
        background-color: #236747 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px;
        padding: 10px 19px;
        font-weight: 700 !important;
        min-height: 44px;
    }

    div.stButton > button:hover,
    div.stButton > button:hover * {
        background-color: #174b35 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 5px 14px rgba(23,75,53,.16);
    }

    div.stButton > button:focus {
        box-shadow: 0 0 0 3px #c5dfc8 !important;
    }
        /* FIX: Upload box visibility */
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1px dashed #9db9a2 !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #1e3027 !important;
    }

    /* FIX: Success, warning and information panels */
    [data-testid="stAlert"] {
        background-color: #edf4eb !important;
        border-radius: 12px !important;
    }

    [data-testid="stAlert"] * {
        color: #1e3027 !important;
    }

    /* FIX: Alert icons */
    [data-testid="stAlert"] svg {
        fill: #236747 !important;
    }

    /* FIX: All button labels */
    div.stButton > button,
    div.stButton > button * {
        color: #ffffff !important;
    }
    hr { border-color: var(--line); }
    .footer {
        text-align: center; color: var(--muted); font-size: 12px;
        padding: 24px 0 5px; border-top: 1px solid var(--line); margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# SESSION STATE AND NAVIGATION
# -------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"


def go_to_detection():
    st.session_state.nav_page = "Disease Detection"


def clear_analysis():
    st.session_state.analysis = None


CROPS = [
    ("🍎", "Apple"), ("🫐", "Blueberry"), ("🍒", "Cherry"),
    ("🌽", "Corn"), ("🍇", "Grape"), ("🍊", "Orange"),
    ("🍑", "Peach"), ("🫑", "Bell Pepper"), ("🥔", "Potato"),
    ("🌱", "Raspberry"), ("🎃", "Squash"), ("🍓", "Strawberry"),
    ("🫘", "Soybean"), ("🍅", "Tomato"),
]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def section_title(kicker, title, subtitle=None):
    st.markdown(f"<div class='section-eyebrow'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-heading'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-sub'>{subtitle}</div>", unsafe_allow_html=True)


def make_gradcam_overlay(image, heatmap):
    resize = transforms.Resize(256)
    crop = transforms.CenterCrop(224)
    processed = crop(resize(image.convert("RGB")))
    original = np.array(processed)
    heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original, 0.58, colored, 0.42, 0)


def display_list(title, items, icon):
    st.markdown(f"#### {icon} {title}")
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption("No additional information available.")


def display_recommendations(info):
    section_title("FIELD GUIDE", "Understanding this condition",
                  "Use this information as general guidance, not as a treatment prescription.")
    st.markdown(f"<div class='soft-panel'>{info.get('description', '')}</div>", unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left:
        display_list("Symptoms to look for", info.get("symptoms", []), "🔎")
        display_list("Prevention", info.get("prevention", []), "🛡️")
    with right:
        display_list("General management", info.get("management", []), "🌱")
    st.caption(info.get("disclaimer", ""))


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='padding:12px 2px 6px'><div style='font-size:29px'>🌿</div>"
        "<div style='font-size:22px;font-weight:800;color:#174b35'>AgriVision AI</div>"
        "<div style='font-size:12px;color:#64746a'>PLANT HEALTH SCREENING</div></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**WORKSPACE**")
    page = st.radio(
        "Navigate",
        ["Home", "Disease Detection", "About"],
        key="nav_page",
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**SUPPORTED CROPS · 14**")
    for emoji, crop in CROPS:
        st.markdown(
            f"<span style='display:inline-block;width:25px'>{emoji}</span>"
            f"<span style='font-size:13px'>{crop}</span>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Deep learning · Explainable AI")
    st.caption("Runs locally on CPU")


# -------------------------------------------------------------------
# HOME
# -------------------------------------------------------------------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <span class="hero-kicker">AI-powered plant health screening</span>
            <h1>Understand your plant.<br>Make more informed decisions.</h1>
            <p>AgriVision AI analyses a photograph of a plant leaf, screens it against supported
            crop-condition classes, and shows a visual explanation of the model's attention.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)
    with a:
        st.metric("Supported crops", "14")
    with b:
        st.metric("Primary model classes", "38")
    with c:
        st.metric("AI explanation", "Grad-CAM")

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        section_title("GET STARTED", "Your plant screening workspace",
                      "No technical knowledge is needed. Follow these three steps.")
        steps = [
            ("01", "Upload a leaf photo", "Choose a clear, well-lit image. Keep the leaf in focus and avoid busy backgrounds."),
            ("02", "Run AI screening", "The primary model predicts a supported crop condition. A specialist model may provide a second comparison."),
            ("03", "Explore the result", "Review the model output, Grad-CAM visualisation, symptoms and general management information."),
        ]
        for number, heading, body in steps:
            st.markdown(
                f"<div class='step-card' style='margin-bottom:11px'>"
                f"<span class='step-number'>{number}</span><h4>{heading}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )
        st.button("Start a plant screening  →", on_click=go_to_detection, use_container_width=True)

    with right:
        section_title("WHAT'S INCLUDED", "One place to explore")
        cards = [
            ("🧠", "AI condition screening", "EfficientNet-B0 classifies the uploaded leaf image into one of the supported categories."),
            ("🔬", "Visual explanation", "Grad-CAM highlights image regions that influenced the primary model's selected class."),
            ("🌿", "Practical information", "Explore common symptoms, prevention practices and general management guidance."),
            ("⚖️", "Model comparison", "Where available, compare the primary and specialist model outputs and check whether they agree."),
        ]
        for icon, heading, body in cards:
            st.markdown(
                f"<div class='info-card' style='margin-bottom:11px'>"
                f"<div class='icon'>{icon}</div><h3>{heading}</h3><p>{body}</p></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    section_title("CROP COVERAGE", "14 supported crops",
                  "The current primary model supports the following crop categories.")
    crop_cols = st.columns(7)
    for index, (emoji, crop) in enumerate(CROPS):
        with crop_cols[index % 7]:
            st.markdown(
                f"<div class='info-card' style='text-align:center;padding:14px 5px;margin-bottom:10px'>"
                f"<div style='font-size:25px'>{emoji}</div>"
                f"<div style='font-size:12px;font-weight:700;color:#24583a'>{crop}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='warning-panel'><b>Important:</b> AgriVision AI is an educational research "
        "prototype for preliminary screening. Model outputs may be incorrect and are not confirmed "
        "agricultural diagnoses. Consult a qualified agricultural expert before treatment decisions.</div>",
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# DISEASE DETECTION
# -------------------------------------------------------------------
elif page == "Disease Detection":
    section_title("SCREEN A LEAF", "Plant disease detection",
                  "Upload one clear leaf photograph. The application will guide you through the analysis.")

    st.markdown(
        "<div class='soft-panel'><b>📷 For a useful image:</b> Photograph one leaf in natural light. "
        "Keep the leaf sharp and visible, include the affected area, and avoid filters or screenshots. "
        "Accepted formats: JPG, JPEG, PNG and WEBP.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    uploaded_file = st.file_uploader(
        "STEP 1 · Choose a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Select a photograph stored on your device.",
    )

    if uploaded_file is None:
        st.markdown(
            "<div class='info-card' style='text-align:center;padding:32px'>"
            "<div style='font-size:42px'>🍃</div><h3>Your image preview will appear here</h3>"
            "<p>After choosing a file, check that the leaf is clearly visible before starting the analysis.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.width < 100 or image.height < 100:
                st.error("This image is very small. Please upload a clearer image at least 100 × 100 pixels.")
            else:
                st.markdown("#### STEP 2 · Check your image")
                preview_col, guide_col = st.columns([1, 1], gap="large")
                with preview_col:
                    st.image(image, caption=f"Selected image · {image.width} × {image.height}", use_container_width=True)
                with guide_col:
                    st.markdown(
                        "<div class='info-card'><h3>Before you analyse</h3>"
                        "<p>✓ Leaf is visible and in focus</p><br>"
                        "<p>✓ Lighting is sufficient</p><br>"
                        "<p>✓ Symptoms are not hidden</p><br>"
                        "<p>✓ Image has not been heavily edited</p></div>",
                        unsafe_allow_html=True,
                    )
                st.write("")
                if st.button("🔍 Analyse this leaf", use_container_width=True):
                    with st.spinner("Running the AI models and preparing the visual explanation…"):
                        prediction_result = predict_image(image)
                        assessment = assess_prediction(prediction_result)

                        overlay = None
                        gradcam_error = None
                        try:
                            gradcam_result = generate_gradcam(
                                image,
                                PRIMARY_MODEL,
                                PRIMARY_CLASS_MAPPING,
                                class_name=prediction_result["primary_prediction"]["class_name"],
                            )
                            overlay = make_gradcam_overlay(image, gradcam_result["heatmap"])
                        except Exception as error:
                            gradcam_error = str(error)

                        st.session_state.analysis = {
                            "assessment": assessment,
                            "original_image": image,
                            "gradcam_overlay": overlay,
                            "gradcam_error": gradcam_error,
                            "filename": uploaded_file.name,
                        }
                    st.markdown(
                        """
                        <div class='soft-panel'>
                            <b>✓ Screening completed!</b>
                            Review your AI results below.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        except Exception as error:
            st.error(f"We couldn't open this image. Please try another JPG, PNG or WEBP file. Details: {error}")

    # RESULTS ARE PERSISTENT WHILE NAVIGATING
    if st.session_state.analysis is not None:
        result = st.session_state.analysis
        assessment = result["assessment"]
        primary = assessment["primary_prediction"]
        specialist = assessment["specialist_prediction"]
        recommendations = assessment["recommendations"]

        st.divider()
        section_title("STEP 3 · AI OUTPUT", "Screening results",
                      f"Analysis for: {result.get('filename', 'uploaded image')}")

        if recommendations.get("status") == "healthy":
            st.markdown(
                "<div class='result-banner'><h3>🌿 Model output: Healthy class</h3>"
                "<p>The primary model selected a healthy category for this image. This does not guarantee "
                "that the plant is disease-free.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='result-banner'><h3>🔎 Model output: {primary['disease']}</h3>"
                f"<p>The primary model selected this disease category. Check the symptoms and supporting information below.</p></div>",
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted crop", primary["crop"])
        with m2:
            st.metric("Predicted condition", primary["disease"])
        with m3:
            st.metric("Raw softmax score", f"{primary['confidence']:.2f}%")
        st.caption(
            "The raw softmax score is the model's uncalibrated output score. "
            "It is not a calibrated probability and should not be read as diagnostic certainty."
        )

        section_title("STEP 4 · MODEL REVIEW", "What did the models say?")
        if assessment["agreement_status"] == "models_agree":
            st.markdown(
                "<div class='soft-panel'>✓ Both models selected the same class.</div>",
                unsafe_allow_html=True
            )

        elif assessment["agreement_status"] == "models_disagree":
            st.markdown(
                "<div class='warning-panel'>⚠ The model outputs differ. Treat this result with extra caution.</div>",
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                "<div class='soft-panel'>ℹ Only the primary model produced a result for this image.</div>",
                unsafe_allow_html=True
            )
        st.write(assessment["agreement_message"])

        if specialist is not None:
            pcol, scol = st.columns(2)
            with pcol:
                st.markdown(
                    f"<div class='info-card'><div class='section-eyebrow'>PRIMARY MODEL</div>"
                    f"<h3>{primary['crop']}</h3><p>{primary['disease']}</p>"
                    f"<p><b>Raw score:</b> {primary['confidence']:.2f}%</p></div>",
                    unsafe_allow_html=True,
                )
            with scol:
                st.markdown(
                    f"<div class='info-card'><div class='section-eyebrow'>SPECIALIST MODEL · EXPERIMENTAL</div>"
                    f"<h3>{specialist['crop']}</h3><p>{specialist['disease']}</p>"
                    f"<p><b>Raw score:</b> {specialist['raw_softmax_score']:.2f}%</p></div>",
                    unsafe_allow_html=True,
                )
            if not specialist.get("crop_matches_primary", True):
                st.warning("The specialist model selected a different crop category. Its output is experimental and should not be treated as confirmation.")

        st.write("")
        section_title("EXPLAINABLE AI", "Where did the model focus?",
                      "Grad-CAM creates an approximate heatmap of image regions that influenced the primary model's selected class.")
        if result["gradcam_overlay"] is not None:
            g1, g2 = st.columns(2)
            with g1:
                st.image(result["original_image"], caption="Original uploaded image", use_container_width=True)
            with g2:
                st.image(result["gradcam_overlay"], caption="Grad-CAM overlay · highlighted regions", use_container_width=True)
            st.caption("The heatmap is a visual explanation, not proof that a highlighted region contains a disease or caused the prediction.")
        else:
            st.info("The prediction completed, but Grad-CAM could not be generated for this image.")
            if result.get("gradcam_error"):
                st.caption(f"Grad-CAM detail: {result['gradcam_error']}")

        st.divider()
        display_recommendations(recommendations)

        st.markdown(
            "<div class='warning-panel'><b>Research prototype notice:</b> This tool is intended for "
            "educational screening and demonstration. It may misclassify field images, unfamiliar crops, "
            "multiple conditions or poor-quality photographs. Do not apply pesticides or other treatments "
            "solely on the basis of this output; seek qualified local advice.</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.button("Clear results and analyse another image", on_click=clear_analysis)


# -------------------------------------------------------------------
# ABOUT
# -------------------------------------------------------------------
elif page == "About":
    st.markdown(
        """
        <div class="hero">
            <span class="hero-kicker">ABOUT THE PROJECT</span>
            <h1>AgriVision AI</h1>
            <p>An explainable deep-learning research prototype for preliminary plant disease screening
            from leaf photographs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_title("PROJECT OVERVIEW", "From a leaf image to an interpretable result",
                  "The application combines image classification, a secondary model comparison, and visual explanation.")
    cols = st.columns(3)
    about_cards = [
        ("01 · CLASSIFICATION", "EfficientNet-B0", "The primary model predicts one of 38 supported crop-condition classes."),
        ("02 · SECOND OPINION", "Specialist model", "A separate 16-class model is used for selected crops as an experimental comparison."),
        ("03 · EXPLANATION", "Grad-CAM", "A heatmap provides an approximate visual view of regions influencing the primary model output."),
    ]
    for col, (tag, title, body) in zip(cols, about_cards):
        with col:
            st.markdown(
                f"<div class='info-card'><div class='section-eyebrow'>{tag}</div>"
                f"<h3>{title}</h3><p>{body}</p></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    section_title("TECH STACK", "Built with")
    tech = st.columns(4)
    for col, (icon, title, body) in zip(
        tech,
        [
            ("🐍", "Python", "Application logic"),
            ("🔥", "PyTorch", "Deep learning inference"),
            ("🖼️", "Pillow + OpenCV", "Image processing"),
            ("📊", "Streamlit", "Interactive interface"),
        ],
    ):
        with col:
            st.markdown(
                f"<div class='info-card' style='text-align:center'><div class='icon'>{icon}</div>"
                f"<h3>{title}</h3><p>{body}</p></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    section_title("SCOPE AND LIMITATIONS", "What this prototype can and cannot do")
    can, cannot = st.columns(2)
    with can:
        st.markdown(
            "<div class='info-card'><h3>✓ Designed to</h3>"
            "<p>• Screen images for supported crop-condition classes</p><br>"
            "<p>• Compare primary and specialist model outputs where available</p><br>"
            "<p>• Display Grad-CAM visual explanations</p><br>"
            "<p>• Present general disease information</p></div>",
            unsafe_allow_html=True,
        )
    with cannot:
        st.markdown(
            "<div class='info-card'><h3>! Not designed to</h3>"
            "<p>• Replace an agricultural expert or laboratory diagnosis</p><br>"
            "<p>• Guarantee accurate results for every field photograph</p><br>"
            "<p>• Estimate a calibrated probability of disease</p><br>"
            "<p>• Prescribe chemical treatments</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='warning-panel'><b>Important:</b> Performance can change substantially when real-world "
        "field photographs differ from the datasets used during model development. Always verify suspected "
        "disease with a qualified agricultural professional.</div>",
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown(
    "<div class='footer'>🌿 <b>AgriVision AI</b> · Explainable Plant Disease Screening"
    "<br>Research prototype · Built with Python, PyTorch and Streamlit</div>",
    unsafe_allow_html=True,
)

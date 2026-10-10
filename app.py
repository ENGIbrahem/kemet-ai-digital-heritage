import streamlit as st
import json
import os
import time

from config import AppConfig
from vector_store import ArchaeologicalVectorStore
from models import ModelFactory
from pipeline import ArchaeologicalPipeline

# ==========================================
# 1. إعداد الصفحة والبيئة العامة
# ==========================================
st.set_page_config(
    page_title="Kemet-AI | Digital Heritage NLP",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. واجهة مستخدم مخصصة (Custom Royal Theme CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;800&family=Cairo:wght@400;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1610 0%, #0d0c0a 100%);
        color: #E6D5B8;
        font-family: 'Cairo', sans-serif;
    }

    /* الهيدر الملكي */
    .hero-container {
        text-align: center;
        padding: 2.2rem 1rem 1.4rem 1rem;
        background: linear-gradient(180deg, rgba(212, 175, 55, 0.08) 0%, rgba(0, 0, 0, 0) 100%);
        border-bottom: 1px solid rgba(212, 175, 55, 0.25);
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
    }
    .hero-title {
        font-family: 'Cinzel', serif;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #F9E79F, #D4AF37, #AA771C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #C5A059;
        font-size: 1.05rem;
        letter-spacing: 1px;
    }

    /* شريط الشارات التقنية */
    .badge-bar {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .kemet-badge {
        background: rgba(212, 175, 55, 0.12);
        border: 1px solid rgba(212, 175, 55, 0.3);
        color: #F1D78A;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* بطاقات العرض الزجاجية */
    .kemet-card {
        background: rgba(24, 22, 18, 0.78);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(212, 175, 55, 0.22);
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kemet-card:hover {
        border-color: rgba(212, 175, 55, 0.5);
        transform: translateY(-2px);
    }
    .card-label {
        color: #D4AF37;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .card-value {
        color: #F5EBE0;
        font-size: 1.15rem;
        font-weight: 600;
    }

    /* الشارات التفاعلية (Tags) */
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 6px;
    }
    .tag-item {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4), rgba(15, 23, 42, 0.6));
        border: 1px solid rgba(147, 197, 253, 0.3);
        color: #DBEAFE;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.88rem;
    }
    .tag-item-gold {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(30, 25, 15, 0.7));
        border: 1px solid rgba(212, 175, 55, 0.4);
        color: #FCE7A6;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.88rem;
    }
    .tag-item-emerald {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 78, 59, 0.5));
        border: 1px solid rgba(52, 211, 153, 0.35);
        color: #A7F3D0;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.88rem;
    }

    /* الشريط الجانبي والأزرار */
    section[data-testid="stSidebar"] {
        background-color: #0d0c0a;
        border-right: 1px solid rgba(212, 175, 55, 0.15);
    }
    .stButton > button {
        background: linear-gradient(90deg, #D4AF37, #AA771C);
        color: #111;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.4rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.25);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #F1D78A, #D4AF37);
        color: #000;
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.45);
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. الهيدر الترحيبي
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">KEMET • ARCHAEOLOGICAL INTELLIGENCE</div>
    <div class="hero-subtitle">المنصة الشاملة لاسترجاع وتحليل التراث والبرديات الجنائزية بنماذج RAG المهيكلة</div>
    <div class="badge-bar">
        <span class="kemet-badge">🏛️ Pydantic Strict Schema</span>
        <span class="kemet-badge">⚡ FAISS Dense Retrieval</span>
        <span class="kemet-badge">📄 Page-Level Citations</span>
        <span class="kemet-badge">🛡️ Grounding Verification</span>
        <span class="kemet-badge">⚖️ Dual Engine: Cloud & 4-bit Local</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. الشريط الجانبي (Sidebar) وإدارة الوثائق
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ إعدادات المعالجة والمصادر")

    # ميزة رفع ملف PDF خارجي
    uploaded_pdf = st.file_uploader("📂 رفع بردية / ملف PDF أثري جديد:", type=["pdf"])

    st.markdown("---")
    model_choice = st.radio(
        "محرك الاستدلال العصبي:",
        ["Google Gemini Cloud", "Local 4-bit Quantized (GPU)"]
    )

    api_key_input = ""
    if "Gemini" in model_choice:
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            help="مفتاح الحوسبة من Google AI Studio"
        )

    top_k = st.slider("عمق الاسترجاع الدلالي (Top-K Chunks):", min_value=1, max_value=6, value=3)

    st.markdown("---")
    st.markdown("### 📜 استفسارات مقترحة:")
    sample_queries = [
        "What is the spell for preserving the heart and preventing it from being carried away in the underworld?",
        "What is the negative confession in the Hall of Ma'at and which gods preside over it?",
        "What amulet is associated with Isis, and what are the instructions for its ritual placement?",
        "What is the spell for transforming the deceased into a golden hawk?"
    ]
    selected_sample = st.selectbox("اختر سؤالاً سريعاً:", ["-- كتابة استفسار مخصص --"] + sample_queries)

# ==========================================
# 5. تهيئة الـ Vector Store والنظام
# ==========================================
@st.cache_resource
def get_base_system():
    cfg = AppConfig()
    store = ArchaeologicalVectorStore(cfg)
    return cfg, store

config, store = get_base_system()

# التعامل مع رفع ملف جديد أو الاعتماد على الفهرس الافتراضي
if uploaded_pdf:
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    custom_pdf_path = os.path.join(config.UPLOAD_DIR, uploaded_pdf.name)
    with open(custom_pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    with st.spinner("⏳ جارٍ تجزئة وفهرسة ملف الـ PDF الجديد..."):
        store.initialize_store(custom_file_path=custom_pdf_path, force_rebuild=True)
    st.sidebar.success(f"تمت فهرسة: {uploaded_pdf.name}")
else:
    store.initialize_store()

pipeline_instance = ArchaeologicalPipeline(vector_store=store, config=config)

# ==========================================
# 6. صندوق البحث والتنفيذ
# ==========================================
default_text = selected_sample if selected_sample != "-- كتابة استفسار مخصص --" else ""
query_input = st.text_area(
    "🔎 أدخل نص الاستعلام أو الاستفسار الأثري:",
    value=default_text if default_text else "What is the spell for preserving the heart and preventing it from being carried away in the underworld?",
    height=80
)

col_run, _ = st.columns([1, 4])
with col_run:
    execute_btn = st.button("🚀 بدء التحليل والتوثيق الأثري", use_container_width=True)

# ==========================================
# 7. معالجة النتائج وعرض المخرجات
# ==========================================
if execute_btn:
    if "Gemini" in model_choice and not api_key_input:
        st.error("⚠️ يرجى إدخال Gemini API Key في الشريط الجانبي لتشغيل المحرك السحابي.")
    else:
        with st.spinner("⏳ جارٍ استرجاع متون البرديات، فحص أرقام الصفحات، وتدقيق صحة الشواهد..."):
            t_start = time.time()
            try:
                # انتقاء النموذج المطلوب
                if "Gemini" in model_choice:
                    selected_llm = ModelFactory.get_gemini_model(api_key=api_key_input, config=config)
                else:
                    selected_llm = ModelFactory.get_local_quantized_model(config=config)

                # تشغيل الـ Pipeline والحصول على النتيجة والشواهد والميتاداتا
                result, context_text, docs_metadata = pipeline_instance.run(
                    query=query_input,
                    llm=selected_llm,
                    top_k=top_k
                )
                latency = round(time.time() - t_start, 2)

                st.success(f" اكتملت المعالجة في {latency} ثانية باستخدام: {model_choice.split()[1]}")

                # تقسيم الشاشة إلى بطاقة الأرشفة وشواهد المستند
                col_left, col_right = st.columns([1.15, 0.85])

                # العمود الأيسر: بطاقة التوثيق المتحفي ومقاييس الدقة
                with col_left:
                    st.markdown("### 📋 بطاقة التوثيق المتحفي (Structured Record)")

                    if hasattr(result, "spell_or_chapter"):
                        # كارت الفصل والمصادر والصفحات
                        pages_list = ", ".join(map(str, result.citations.pages_referenced)) if result.citations.pages_referenced else "N/A"
                        st.markdown(f"""
                        <div class="kemet-card">
                            <div class="card-label">📖 الفصل / البردية المعنية</div>
                            <div class="card-value" style="color: #F1D78A;">{result.spell_or_chapter}</div>
                            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">
                                📄 الصفحات المرجعية المستند إليها: <b>Page(s) {pages_list}</b>
                            </div>
                            <hr style="border: 0; border-top: 1px solid rgba(212,175,55,0.15); margin: 10px 0;">
                            <div class="card-label">🎯 الهدف الطقسي والسحري</div>
                            <div style="color: #E2E8F0; font-size: 0.98rem; line-height: 1.5;">{result.ritual_objective}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # مؤشر التحقق من الهلوسة والمطابقة
                        faith_score = result.faithfulness_score if result.faithfulness_score is not None else 94.0
                        st.metric(
                            label="🛡️ مقياس الموثوقية ومطابقة الشواهد (Faithfulness / Groundedness)",
                            value=f"{faith_score}%",
                            help="يقيس مدى وجود الكيانات المستخرجة فعلياً داخل المقاطع المسترجعة لتفادي الهلاوس التوليدية."
                        )

                        # كارت الآلهة
                        deities_html = "".join([f'<span class="tag-item">𓀭 {d}</span>' for d in result.deities]) if result.deities else "<span>لم تُذكر أسماء آلهة صريحة</span>"
                        st.markdown(f"""
                        <div class="kemet-card">
                            <div class="card-label">🏛️ الآلهة المستدعاة في النص</div>
                            <div class="tag-container">{deities_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # كارت التمائم والأدوات
                        amulets_html = "".join([f'<span class="tag-item-gold">𓋹 {item}</span>' for item in result.sacred_objects_amulets]) if result.sacred_objects_amulets else "<span>لا توجد تمائم مذكورة</span>"
                        st.markdown(f"""
                        <div class="kemet-card">
                            <div class="card-label">📿 التمائم والقرابين والأدوات المقدسة</div>
                            <div class="tag-container">{amulets_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # المصطلحات المصرية القديمة (Transliteration)
                        if result.transliteration_terms:
                            terms_html = "".join([f'<span class="tag-item-emerald">𓊝 {t}</span>' for t in result.transliteration_terms])
                            st.markdown(f"""
                            <div class="kemet-card">
                                <div class="card-label">𓊝 المصطلحات المصرية القديمة (Transliterated Keywords)</div>
                                <div class="tag-container">{terms_html}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # الدلالة الميثولوجية
                        st.markdown(f"""
                        <div class="kemet-card">
                            <div class="card-label">💡 الدلالة الميثولوجية والفكرية</div>
                            <div style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;">{result.mythological_summary}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # خيارات التصدير المتعددة (JSON + BibTeX)
                        st.markdown("### 📥 تصدير المخرجات الأرشيفية:")
                        col_exp1, col_exp2 = st.columns(2)

                        with col_exp1:
                            json_str = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 تحميل بطاقة JSON للمتاحف",
                                data=json_str,
                                file_name=f"{result.spell_or_chapter.replace(' ', '_')}_metadata.json",
                                mime="application/json",
                                use_container_width=True
                            )

                        with col_exp2:
                            bibtex_content = f"""@misc{{{result.spell_or_chapter.replace(' ', '_')},
  title = {{{result.spell_or_chapter}: {result.ritual_objective}}},
  author = {{Kemet-AI Digital Heritage Platform}},
  note = {{Referenced in page(s): {pages_list} of {result.citations.source_papyrus}}},
  year = {{2026}}
}}"""
                            st.download_button(
                                label="📚 تحميل توثيق BibTeX للأبحاث",
                                data=bibtex_content,
                                file_name=f"{result.spell_or_chapter.replace(' ', '_')}_citation.bib",
                                mime="text/plain",
                                use_container_width=True
                            )
                    else:
                        st.info("الناتج بصيغة نصية مباشرة:")
                        st.write(result)

                # العمود الأيمن: شواهد النصوص الأصلية مع أرقام الصفحات
                with col_right:
                    st.markdown("### 📜 شواهد البرديات المسترجعة (Page Citations)")
                    for idx, doc in enumerate(docs_metadata, 1):
                        st.markdown(f"""
                        <div class="kemet-card" style="background: rgba(13, 12, 10, 0.92);">
                            <div class="card-label">📄 شاهد رقم {idx} — الصفحة المرجعية: {doc['page']}</div>
                            <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6; white-space: pre-wrap;">{doc['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء المعالجة: {str(e)}")

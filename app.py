
import streamlit as st

from local_search import LocalSearch
from answer_generator import generate_answer


st.set_page_config(
    page_title="IIT Dharwad Academic Assistant",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def load_search_engine():
    return LocalSearch()


st.title("🎓 IIT Dharwad Academic Office Assistant")

st.write(
    "Ask questions about academic calendars, registration, "
    "hostel rules, course regulations, and other available documents."
)

st.info(
    "Answers are generated using your academic documents. "
    "Please check the cited source for important dates and rules."
)

try:
    searcher = load_search_engine()
except Exception as error:
    st.error("Could not load the local search engine.")
    st.exception(error)
    st.stop()


question = st.text_area(
    "Enter your academic question",
    placeholder="Example: When does orientation begin for new entrants in 2026–27?",
    height=100
)

ask_button = st.button(
    "Get Answer",
    type="primary",
    use_container_width=False
)

if ask_button:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching documents and preparing your answer..."):
            try:
                results = searcher.search(question.strip())

                if not results:
                    st.warning(
                        "No relevant passages were found in the local documents."
                    )
                else:
                    answer = generate_answer(question.strip(), results)

                    st.subheader("Answer")
                    st.markdown(answer)

                    st.subheader("Retrieved Sources")

                    for number, item in enumerate(results, start=1):
                        source = item.get("source", "Unknown source")
                        page = item.get("page", "Unknown")
                        score = item.get("score", 0.0)

                        with st.expander(
                            "Source {0}: {1} — Page {2}".format(
                                number, source, page
                            )
                        ):
                            st.write(
                                "Similarity score: {0:.3f}".format(score)
                            )
                            st.write(item.get("text", ""))

            except Exception as error:
                st.error("Something went wrong while answering your question.")
                st.exception(error)


st.divider()

st.caption(
    "IIT Dharwad Academic Office Assistant | "
    "Local FAISS retrieval + Gemini answer generation"
)

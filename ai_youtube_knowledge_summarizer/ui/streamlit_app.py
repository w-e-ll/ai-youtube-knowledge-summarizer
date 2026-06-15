from __future__ import annotations

import logging
import os
from typing import Any

import requests
import streamlit as st


logger = logging.getLogger("ai_youtube_knowledge_summarizer.ui.streamlit_app")


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


class ApiClientError(Exception):
    """Raised when the Streamlit UI cannot communicate with the backend API."""


class ApiClient:
    """Small HTTP client for the FastAPI backend."""

    def __init__(self, base_url: str = DEFAULT_API_BASE_URL) -> None:
        """Initialize API client with backend base URL."""
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        """Return backend health status."""
        return self._request("GET", "/health")

    def process_video(
        self,
        *,
        youtube_url: str,
        generate_summary: bool,
        summary_mode: str,
    ) -> dict[str, Any]:
        """Submit YouTube video for backend processing."""
        return self._request(
            "POST",
            "/videos/process",
            json={
                "youtube_url": youtube_url,
                "generate_summary": generate_summary,
                "summary_mode": summary_mode,
            },
            timeout=900,
        )

    def summarize(self, *, transcript_id: str, mode: str) -> dict[str, Any]:
        """Request summary generation for an existing transcript."""
        return self._request(
            "POST",
            "/summaries",
            json={
                "transcript_id": transcript_id,
                "mode": mode,
            },
            timeout=300,
        )

    def ask_question(
        self,
        *,
        transcript_id: str,
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Ask a retrieval-based question about a transcript."""
        return self._request(
            "POST",
            "/qa",
            json={
                "transcript_id": transcript_id,
                "question": question,
                "top_k": top_k,
            },
            timeout=120,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        """Send HTTP request and return parsed JSON response."""
        url = f"{self.base_url}{path}"

        try:
            response = requests.request(
                method=method,
                url=url,
                json=json,
                timeout=timeout,
            )

            payload = response.json()

        except requests.RequestException as exc:
            raise ApiClientError(f"Backend request failed: {exc}") from exc

        except ValueError as exc:
            raise ApiClientError("Backend returned a non-JSON response") from exc

        if response.status_code >= 400:
            error = payload.get("detail") or payload.get("error") or payload
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ApiClientError(message)

        return payload


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    st.session_state.setdefault("transcript_id", None)
    st.session_state.setdefault("video_id", None)
    st.session_state.setdefault("video_title", None)
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("last_sources", [])


def render_sidebar(client: ApiClient) -> None:
    """Render sidebar with backend connection details."""
    st.sidebar.header("Backend")

    st.sidebar.code(client.base_url)

    if st.sidebar.button("Check backend health"):
        try:
            health = client.health()
            st.sidebar.success("Backend is healthy")
            st.sidebar.json(health)
        except ApiClientError as exc:
            st.sidebar.error(str(exc))


def render_video_processing(client: ApiClient) -> None:
    """Render YouTube processing form."""
    st.header("1. Process YouTube Video")

    with st.form("process_video_form"):
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )

        summary_mode = st.selectbox(
            "Summary mode",
            options=["stuff", "map_reduce", "refine"],
            index=1,
        )

        generate_summary = st.checkbox("Generate summary after processing", value=True)

        submitted = st.form_submit_button("Process video")

    if not submitted:
        return

    if not youtube_url.strip():
        st.error("YouTube URL is required")
        return

    with st.spinner("Downloading, transcribing, chunking, embedding, and indexing video..."):
        try:
            result = client.process_video(
                youtube_url=youtube_url.strip(),
                generate_summary=generate_summary,
                summary_mode=summary_mode,
            )

        except ApiClientError as exc:
            st.error(f"Processing failed: {exc}")
            return

    st.session_state.transcript_id = result["transcript_id"]
    st.session_state.video_id = result["video_id"]
    st.session_state.video_title = result.get("title")
    st.session_state.summary = result.get("summary")

    st.success("Video processed successfully")

    st.subheader("Video Metadata")
    st.json(
        {
            "request_id": result.get("request_id"),
            "video_id": result.get("video_id"),
            "transcript_id": result.get("transcript_id"),
            "title": result.get("title"),
            "author": result.get("author"),
            "duration_seconds": result.get("duration_seconds"),
            "chunk_count": result.get("chunk_count"),
        }
    )

    if result.get("summary"):
        st.subheader("Generated Summary")
        st.write(result["summary"])


def render_summary_section(client: ApiClient) -> None:
    """Render summary generation section for existing transcript."""
    st.header("2. Generate / Regenerate Summary")

    transcript_id = st.text_input(
        "Transcript ID",
        value=st.session_state.transcript_id or "",
        placeholder="Paste transcript_id here",
        key="summary_transcript_id",
    )

    mode = st.selectbox(
        "Summary strategy",
        options=["stuff", "map_reduce", "refine"],
        index=1,
        key="summary_mode_select",
    )

    if st.button("Generate summary"):
        if not transcript_id.strip():
            st.error("Transcript ID is required")
            return

        with st.spinner("Generating summary..."):
            try:
                result = client.summarize(
                    transcript_id=transcript_id.strip(),
                    mode=mode,
                )
            except ApiClientError as exc:
                st.error(f"Summary failed: {exc}")
                return

        st.session_state.summary = result["summary"]
        st.success("Summary generated")
        st.write(result["summary"])


def render_qa_section(client: ApiClient) -> None:
    """Render retrieval-based question answering section."""
    st.header("3. Ask Questions About the Video")

    transcript_id = st.text_input(
        "Transcript ID",
        value=st.session_state.transcript_id or "",
        placeholder="Paste transcript_id here",
        key="qa_transcript_id",
    )

    question = st.text_area(
        "Question",
        placeholder="What are the main ideas from this video?",
    )

    top_k = st.slider("Retrieved chunks", min_value=1, max_value=20, value=4)

    if st.button("Ask question"):
        if not transcript_id.strip():
            st.error("Transcript ID is required")
            return

        if not question.strip():
            st.error("Question is required")
            return

        with st.spinner("Retrieving relevant transcript chunks and generating answer..."):
            try:
                result = client.ask_question(
                    transcript_id=transcript_id.strip(),
                    question=question.strip(),
                    top_k=top_k,
                )
            except ApiClientError as exc:
                st.error(f"Question answering failed: {exc}")
                return

        st.success("Answer generated")
        st.subheader("Answer")
        st.write(result["answer"])

        sources = result.get("sources", [])
        st.session_state.last_sources = sources

        if sources:
            st.subheader("Sources")
            for index, source in enumerate(sources, start=1):
                with st.expander(f"Source chunk {index}"):
                    st.write(source.get("content", ""))
                    st.json(source.get("metadata", {}))


def render_current_state() -> None:
    """Render current session state for user visibility."""
    st.header("Current Session")

    st.json(
        {
            "video_id": st.session_state.video_id,
            "video_title": st.session_state.video_title,
            "transcript_id": st.session_state.transcript_id,
            "has_summary": bool(st.session_state.summary),
            "source_count": len(st.session_state.last_sources),
        }
    )


def main() -> None:
    """Run Streamlit frontend application."""
    st.set_page_config(
        page_title="AI YouTube Knowledge Summarizer",
        page_icon="🎬",
        layout="wide",
    )

    init_session_state()

    client = ApiClient()

    st.title("AI YouTube Knowledge Summarizer")
    st.caption(
        "Production-style AI media intelligence app using Whisper, LangChain, "
        "OpenAI, embeddings, vector search, and RAG."
    )

    render_sidebar(client)

    render_video_processing(client)

    st.divider()

    render_summary_section(client)

    st.divider()

    render_qa_section(client)

    st.divider()

    render_current_state()


if __name__ == "__main__":
    main()

import time

import ollama
import streamlit as st

from ..config import get_settings
from ..utils import check_ollama_status
from ..core import SearchService, ChatService, VisionChatService
from ..document_processing import VisionDocumentReader
from .i18n import t


def render_chat_interface(search_service: SearchService, chat_service: ChatService):
    """Render the main chat interface."""
    settings = get_settings()
    
    st.header(t("chat.header", "💬 Ask Questions"))
    
    # Show statistics for last response if available
    if st.session_state.get("response_stats"):
        last_stats = st.session_state.response_stats[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t("chat.metric_response_time", "⏱️ Response Time"), f"{last_stats['total_time']:.2f}s")
        with col2:
            st.metric(t("chat.metric_first_token", "🚀 First Token"), f"{last_stats['time_to_first_token']:.2f}s")
        with col3:
            st.metric(t("chat.metric_tokens", "📝 Tokens"), last_stats["tokens"])
        with col4:
            st.metric(t("chat.metric_speed", "⚡ Speed"), f"{last_stats['tokens_per_sec']:.1f} tok/s")
        st.divider()
    
    # Display chat history
    for i, message in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show stats for assistant messages
            if message["role"] == "assistant" and i < len(st.session_state.get("response_stats", [])):
                with st.expander(t("chat.response_metrics", "📊 Response Metrics"), expanded=False):
                    stats = st.session_state.response_stats[i // 2] if i // 2 < len(st.session_state.response_stats) else None
                    if stats:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(t("chat.metrics_total_time", "Total time: {secs:.2f}s", secs=stats['total_time']))
                            st.caption(t("chat.metrics_first_token", "First token: {secs:.2f}s", secs=stats['time_to_first_token']))
                        with col2:
                            st.caption(t("chat.metrics_tokens", "Tokens: {count}", count=stats['tokens']))
                            st.caption(t("chat.metrics_speed", "Speed: {rate:.1f} tokens/sec", rate=stats['tokens_per_sec']))
    
    # Chat input
    if prompt := st.chat_input(t("chat.input", "Ask a question about your documents...")):
        # Initialize session state if needed
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "response_stats" not in st.session_state:
            st.session_state.response_stats = []
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            
            # Search phase
            search_start = time.time()
            with status_placeholder.container():
                with st.spinner(t("chat.searching", "🔍 Searching through documents...")):
                    k_value = st.session_state.get('top_k', 5)
                    search_mode = st.session_state.get('search_mode', settings.default_search_mode)
                    alpha = st.session_state.get('hybrid_alpha', settings.hybrid_alpha)
                    search_results, search_stats = search_service.search_similar(
                        prompt, k=k_value, search_mode=search_mode, alpha=alpha
                    )
            search_time = time.time() - search_start
            
            if search_results is not None and not search_results.empty:
                # Prepare context
                system_prompt, user_prompt, citations = search_service.prepare_context(prompt, search_results)
                
                # Show search results with statistics
                search_info = t("chat.found_chunks", "✅ Found {count} relevant chunks in {secs:.2f}s", count=len(search_results), secs=search_time)
                if search_stats and 'mode' in search_stats:
                    if search_stats['mode'] == 'hybrid':
                        search_info += t("chat.found_chunks_hybrid", " (Hybrid: {kw} keyword + {vec} vector)", kw=search_stats.get('bm25_results', 0), vec=search_stats.get('vector_results', 0))
                    elif search_stats['mode'] == 'keyword':
                        search_info += t("chat.found_chunks_keyword", " (Keyword: {kw} results)", kw=search_stats.get('bm25_results', 0))
                    else:
                        search_info += t("chat.found_chunks_vector", " (Vector: {vec} results)", vec=search_stats.get('vector_results', 0))
                
                status_placeholder.success(search_info)
                
                with st.expander(t("chat.sources", "📖 Sources"), expanded=True):
                    st.markdown(citations)
                    st.caption(t("chat.search_completed", "Search completed in {secs:.2f} seconds using {mode} mode", secs=search_time, mode=search_stats.get('mode', 'vector')))
                
                # Generate response
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response_container = st.empty()
                metrics_container = st.empty()
                
                # Stream response
                response_text = ""
                token_count = 0
                start_time = time.time()
                first_token_time = None
                
                with response_container.container():
                    with st.spinner(t("chat.thinking", "🤔 Thinking...")):
                        time.sleep(0.5)  # Brief pause for UX
                
                response_placeholder = st.empty()
                
                try:
                    # Pre-flight: ensure Ollama is reachable and model is available
                    ollama_status = check_ollama_status()
                    if not ollama_status.get("connected"):
                        raise RuntimeError("Cannot connect to Ollama. Please start it with `ollama serve`.")
                    # Resolve generation model to an installed tag (exact or closest match)
                    available = ollama_status.get("models", []) or []
                    gen_model = settings.gen_model
                    if gen_model not in available:
                        # Find a case-insensitive contains match
                        match = next((m for m in available if gen_model.lower() in m.lower()), None)
                        if match:
                            gen_model = match
                        else:
                            raise RuntimeError(
                                f"Generation model '{settings.gen_model}' not found in Ollama. Installed: {', '.join(available) or 'none'}"
                            )
                    # If the user likely asks about visuals and we have PDF sources, use vision-enhanced flow
                    is_visual_query = any(k in prompt.lower() for k in [
                        "image", "chart", "graph", "diagram", "figure", "picture", "visual", "table"
                    ])
                    stream = None
                    if is_visual_query:
                        # Try to extract visuals from the first PDF in results
                        try:
                            docs_base = settings.get_documents_path()
                            pdf_path = None
                            for src in search_results['source'].tolist():
                                if str(src).lower().endswith('.pdf'):
                                    candidate = docs_base / str(src)
                                    if candidate.exists():
                                        pdf_path = candidate
                                        break
                            if pdf_path is not None:
                                vreader = VisionDocumentReader()
                                pdf_bytes = pdf_path.read_bytes()
                                pdf_content = vreader.read_pdf_with_vision(pdf_bytes)
                                vchat = VisionChatService()
                                stream = vchat.stream_chat_with_context(messages, pdf_content=pdf_content)
                        except Exception:
                            # Fallback to text model streaming if anything fails
                            stream = None
                    # Fallback to normal text streaming if no vision stream
                    if stream is None:
                        stream = ollama.chat(
                            model=gen_model,
                            messages=messages,
                            stream=True
                        )
                    
                    for chunk in stream:
                        # Handle both Ollama streaming dicts and plain string chunks (vision service)
                        content = None
                        if isinstance(chunk, dict) and "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                        elif isinstance(chunk, str):
                            content = chunk
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                                status_placeholder.empty()
                            
                            response_text += content
                            token_count += len(content.split())
                            
                            # Update response
                            response_placeholder.markdown(response_text)
                            
                            # Update live metrics
                            elapsed = time.time() - start_time
                            with metrics_container.container():
                                cols = st.columns(4)
                                with cols[0]:
                                    st.caption(f"⏱️ {elapsed:.1f}s")
                                with cols[1]:
                                    st.caption(f"📝 {token_count}")
                                with cols[2]:
                                    st.caption(f"⚡ {token_count/elapsed:.1f} tok/s" if elapsed > 0 else "⚡ -- tok/s")
                                with cols[3]:
                                    st.caption(t("chat.streaming", "🔄 Streaming..."))
                    
                    # If no content streamed, try a non-streaming fallback once
                    if not response_text:
                        try:
                            resp = ollama.chat(model=gen_model, messages=messages, stream=False)
                            fallback_text = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else ""
                            if fallback_text:
                                response_text = fallback_text
                                token_count = len(response_text.split())
                                response_placeholder.markdown(response_text)
                            else:
                                st.warning("Model returned no content. Try a different prompt or model.")
                        except Exception as fe:
                            st.error(t("chat.error_generating", "Error generating response: {err}", err=fe))

                    # Final metrics
                    total_time = time.time() - start_time
                    ttft = (first_token_time - start_time) if first_token_time else 0
                    
                    stats = {
                        "total_time": total_time,
                        "time_to_first_token": ttft,
                        "tokens": token_count,
                        "tokens_per_sec": token_count / total_time if total_time > 0 else 0,
                    }
                    
                    # Update final metrics
                    with metrics_container.container():
                        cols = st.columns(4)
                        with cols[0]:
                            st.caption(f"⏱️ {total_time:.2f}s")
                        with cols[1]:
                            st.caption(f"📝 {token_count}")
                        with cols[2]:
                            st.caption(f"⚡ {stats['tokens_per_sec']:.1f} tok/s")
                        with cols[3]:
                            st.caption(t("chat.complete", "✅ Complete"))
                    
                    st.session_state.response_stats.append(stats)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                    # Trim chat history to configured limit (keep last N turns)
                    try:
                        history_limit = settings.chat_history_limit if hasattr(settings, "chat_history_limit") else 20
                        # messages are user/assistant pairs; keep last 2*limit entries
                        max_msgs = max(2 * int(history_limit), 2)
                        if len(st.session_state.messages) > max_msgs:
                            st.session_state.messages = st.session_state.messages[-max_msgs:]
                        if len(st.session_state.response_stats) > history_limit:
                            st.session_state.response_stats = st.session_state.response_stats[-history_limit:]
                    except Exception:
                        # Non-fatal; ignore trimming errors
                        pass
                    
                except Exception as e:
                    st.error(t("chat.error_generating", "Error generating response: {err}", err=e))
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            else:
                no_results_msg = t("chat.no_relevant", "I couldn't find any relevant information in the indexed documents. Please make sure documents are indexed.")
                status_placeholder.warning(t("chat.no_relevant_title", "⚠️ No relevant documents found"))
                st.warning(no_results_msg)
                st.session_state.messages.append({"role": "assistant", "content": no_results_msg})
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(t("buttons.clear_chat", "🔄 Clear Chat")):
            st.session_state.messages = []
            st.session_state.response_stats = []
            st.rerun()
    with col2:
        st.markdown(f"**{t('footer.embedding_model', 'Embedding Model:')}** {settings.embed_model}")
    with col3:
        st.markdown(f"**{t('footer.generation_model', 'Generation Model:')}** {settings.gen_model}")

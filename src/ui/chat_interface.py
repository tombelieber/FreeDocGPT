import time

import ollama
import streamlit as st

from ..config import get_settings
from ..core import SearchService, ChatService


def render_chat_interface(search_service: SearchService, chat_service: ChatService):
    """Render the main chat interface."""
    settings = get_settings()
    
    st.header("💬 Ask Questions")
    
    # Show statistics for last response if available
    if st.session_state.get("response_stats"):
        last_stats = st.session_state.response_stats[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Response Time", f"{last_stats['total_time']:.2f}s")
        with col2:
            st.metric("🚀 First Token", f"{last_stats['time_to_first_token']:.2f}s")
        with col3:
            st.metric("📝 Tokens", last_stats["tokens"])
        with col4:
            st.metric("⚡ Speed", f"{last_stats['tokens_per_sec']:.1f} tok/s")
        st.divider()
    
    # Display chat history
    for i, message in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show stats for assistant messages
            if message["role"] == "assistant" and i < len(st.session_state.get("response_stats", [])):
                with st.expander("📊 Response Metrics", expanded=False):
                    stats = st.session_state.response_stats[i // 2] if i // 2 < len(st.session_state.response_stats) else None
                    if stats:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Total time: {stats['total_time']:.2f}s")
                            st.caption(f"First token: {stats['time_to_first_token']:.2f}s")
                        with col2:
                            st.caption(f"Tokens: {stats['tokens']}")
                            st.caption(f"Speed: {stats['tokens_per_sec']:.1f} tokens/sec")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
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
                with st.spinner("🔍 Searching through documents..."):
                    k_value = st.session_state.get('top_k', 5)
                    search_results = search_service.search_similar(prompt, k=k_value)
            search_time = time.time() - search_start
            
            if search_results is not None and not search_results.empty:
                # Prepare context
                system_prompt, user_prompt, citations = search_service.prepare_context(prompt, search_results)
                
                # Show search results
                status_placeholder.success(f"✅ Found {len(search_results)} relevant chunks in {search_time:.2f}s")
                
                with st.expander("📖 Sources", expanded=True):
                    st.markdown(citations)
                    st.caption(f"Search completed in {search_time:.2f} seconds")
                
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
                    with st.spinner("🤔 Thinking..."):
                        time.sleep(0.5)  # Brief pause for UX
                
                response_placeholder = st.empty()
                
                try:
                    stream = ollama.chat(
                        model=settings.gen_model,
                        messages=messages,
                        stream=True
                    )
                    
                    for chunk in stream:
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
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
                                        st.caption(f"📝 {token_count} tokens")
                                    with cols[2]:
                                        st.caption(f"⚡ {token_count/elapsed:.1f} tok/s" if elapsed > 0 else "⚡ -- tok/s")
                                    with cols[3]:
                                        st.caption("🔄 Streaming...")
                    
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
                            st.caption(f"⏱️ {total_time:.2f}s total")
                        with cols[1]:
                            st.caption(f"📝 {token_count} tokens")
                        with cols[2]:
                            st.caption(f"⚡ {stats['tokens_per_sec']:.1f} tok/s")
                        with cols[3]:
                            st.caption("✅ Complete")
                    
                    st.session_state.response_stats.append(stats)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            else:
                no_results_msg = "I couldn't find any relevant information in the indexed documents. Please make sure documents are indexed."
                status_placeholder.warning("⚠️ No relevant documents found")
                st.warning(no_results_msg)
                st.session_state.messages.append({"role": "assistant", "content": no_results_msg})
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.session_state.response_stats = []
            st.rerun()
    with col2:
        st.markdown(f"**Embedding Model:** {settings.embed_model}")
    with col3:
        st.markdown(f"**Generation Model:** {settings.gen_model}")
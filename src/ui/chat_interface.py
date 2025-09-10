import json
import logging
import time

import ollama
import streamlit as st

from ..config import get_settings
from ..utils import check_ollama_status
from ..core import SearchService, ChatService, VisionChatService
from ..document_processing import VisionDocumentReader
from .i18n import t

logger = logging.getLogger(__name__)


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
        
        # Show quick stage breakdown if available
        if "stage_durations" in last_stats and last_stats["stage_durations"]:
            stage_durations = last_stats["stage_durations"]
            stage_parts = []
            if 'search' in stage_durations:
                stage_parts.append(f"🔍 {stage_durations['search']:.1f}s")
            if 'thinking' in stage_durations:
                stage_parts.append(f"🧠 {stage_durations['thinking']:.1f}s")
            if 'model_init' in stage_durations:
                stage_parts.append(f"🤖 {stage_durations['model_init']:.1f}s")
            if 'to_first_token' in stage_durations:
                stage_parts.append(f"🚀 {stage_durations['to_first_token']:.1f}s")
            if stage_parts:
                st.caption("**Pipeline Stages:** " + " • ".join(stage_parts))
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
            # Initialize detailed timing tracking
            pipeline_start = time.time()
            stage_times = {
                "pipeline_start": pipeline_start,
                "search_start": None,
                "search_end": None,
                "context_prep_start": None,
                "context_prep_end": None,
                "thinking_start": None,
                "thinking_end": None,
                "model_init_start": None,
                "model_init_end": None,
                "vision_start": None,
                "vision_end": None,
                "response_start": None,
                "first_token": None,
                "response_end": None,
            }
            
            # Create main loading container
            main_loading_container = st.empty()
            
            # Clean Search Phase 
            stage_times["search_start"] = time.time()
            with main_loading_container.container():
                with st.spinner("🔍 Searching through documents..."):
                    k_value = st.session_state.get('top_k', 5)
                    search_mode = st.session_state.get('search_mode', settings.default_search_mode)
                    alpha = st.session_state.get('hybrid_alpha', settings.hybrid_alpha)
                    search_results, search_stats = search_service.search_similar(
                        prompt, k=k_value, search_mode=search_mode, alpha=alpha
                    )
            stage_times["search_end"] = time.time()
            
            search_time = stage_times["search_end"] - stage_times["search_start"]
            
            if search_results is not None and not search_results.empty:
                # Prepare context
                stage_times["context_prep_start"] = time.time()
                system_prompt, user_prompt, citations = search_service.prepare_context(prompt, search_results)
                stage_times["context_prep_end"] = time.time()
                
                # Clear search loading and show compact results
                main_loading_container.empty()
                
                # Compact search results info
                search_info = f"Found {len(search_results)} relevant chunks in {search_time:.2f}s"
                if search_stats and 'mode' in search_stats:
                    if search_stats['mode'] == 'hybrid':
                        search_info += f" (Hybrid: {search_stats.get('bm25_results', 0)} keyword + {search_stats.get('vector_results', 0)} vector)"
                    elif search_stats['mode'] == 'keyword':
                        search_info += f" (Keyword: {search_stats.get('bm25_results', 0)} results)"
                    else:
                        search_info += f" (Vector: {search_stats.get('vector_results', 0)} results)"
                
                st.success(f"✅ {search_info}")
                
                with st.expander("📖 Sources", expanded=False):
                    st.markdown(citations)
                
                # Prepare main response messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Check if thinking mode is enabled
                thinking_enabled = st.session_state.get('thinking_mode', True)
                
                # Initialize variables
                response_text = ""
                token_count = 0
                start_time = time.time()
                first_token_time = None
                
                try:
                    # Simple model initialization without nested containers
                    stage_times["model_init_start"] = time.time()
                    with st.spinner("🤖 Initializing AI model..."):
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
                    stage_times["model_init_end"] = time.time()
                    
                    # Generate thinking content (ChatGPT-style) - if enabled
                    if thinking_enabled:
                        stage_times["thinking_start"] = time.time()
                        
                        # Show thinking spinner outside the expander so it's always visible
                        thinking_status = st.empty()
                        with thinking_status.container():
                            st.info("🧠 **AI is analyzing the question and search results...**")
                        
                        # Create thinking prompt
                        thinking_prompt = f"""You are an AI assistant analyzing a user question and search results. Think through your approach step by step.

USER QUESTION: {prompt}

SEARCH RESULTS FOUND: {len(search_results)} relevant document chunks

DOCUMENT SOURCES: {', '.join(search_results['source'].unique()[:3])}{'...' if len(search_results['source'].unique()) > 3 else ''}

Think through:
1. What is the user really asking?
2. What key information did we find in the documents?
3. How should I structure my response?
4. Are there any limitations or considerations?

Be concise but show your reasoning process. Write in a thinking style, like you're planning your response."""

                        thinking_messages = [
                            {"role": "system", "content": "You are an AI showing your thinking process. Be analytical and transparent about how you approach the question."},
                            {"role": "user", "content": thinking_prompt}
                        ]
                        
                        # Generate thinking response
                        try:
                            thinking_response = ollama.chat(
                                model=gen_model,
                                messages=thinking_messages,
                                stream=False
                            )
                            
                            # Extract thinking content
                            thinking_content = ""
                            if hasattr(thinking_response, 'message') and hasattr(thinking_response.message, 'content'):
                                thinking_content = thinking_response.message.content
                            elif isinstance(thinking_response, dict) and "message" in thinking_response:
                                thinking_content = thinking_response["message"].get("content", "")
                            
                            # Clear the thinking status and show the expander with results
                            thinking_status.empty()
                            
                            with st.expander("🤔 **AI Thinking Process**", expanded=False):
                                if thinking_content:
                                    st.markdown(f"*{thinking_content}*")
                                else:
                                    st.info("*Analyzing question and preparing response...*")
                                    
                        except Exception as e:
                            # Clear the thinking status and show the expander with fallback
                            thinking_status.empty()
                            with st.expander("🤔 **AI Thinking Process**", expanded=False):
                                st.info("*Processing question and search results to formulate the best response...*")
                        
                        stage_times["thinking_end"] = time.time()
                    else:
                        stage_times["thinking_start"] = stage_times["thinking_end"] = time.time()
                    
                    # If the user likely asks about visuals and we have PDF sources, use vision-enhanced flow
                    is_visual_query = any(k in prompt.lower() for k in [
                        "image", "chart", "graph", "diagram", "figure", "picture", "visual", "table"
                    ])
                    stream = None
                    if is_visual_query:
                        stage_times["vision_start"] = time.time()
                        # Simple vision processing
                        with st.spinner("🎨 Processing visuals and charts..."):
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
                                else:
                                    stream = None
                                    
                            except Exception:
                                stream = None
                        stage_times["vision_end"] = time.time()
                    else:
                        stage_times["vision_start"] = stage_times["vision_end"] = time.time()
                    # Fallback to normal text streaming if no vision stream
                    if stream is None:
                        with st.spinner("🔗 Connecting to language model..."):
                            try:
                                stage_times["response_start"] = time.time()
                                stream = ollama.chat(
                                    model=gen_model,
                                    messages=messages,
                                    stream=True
                                )
                            except Exception as e:
                                st.error(f"Failed to connect to language model: {e}")
                                stream = None
                    else:
                        stage_times["response_start"] = time.time()
                    
                    # Simple response generation
                    response_placeholder = st.empty()
                    metrics_container = st.empty()
                    
                    # Check if we have a valid stream
                    if stream is None:
                        response_placeholder.error("❌ Failed to initialize AI model. Please check if Ollama is running and the model is available.")
                        response_text = "Error: Could not connect to AI model."
                    else:
                        # Quick test to verify model is responding
                        try:
                            test_response = ollama.chat(
                                model=gen_model,
                                messages=[{"role": "user", "content": "Hello"}],
                                stream=False
                            )
                            if not (hasattr(test_response, 'message') or isinstance(test_response, dict)):
                                st.warning(f"⚠️ Model {gen_model} is not responding correctly. Trying fallback...")
                        except Exception as test_error:
                            st.warning(f"⚠️ Model test failed: {test_error}")
                        # Start streaming response
                        chunk_count = 0
                        waiting_for_first_token = True
                        
                        try:
                            for chunk in stream:
                                chunk_count += 1
                                
                                # COMPREHENSIVE DEBUG: Log all chunk details for first 5 chunks
                                if chunk_count <= 5:
                                    logger.info(f"Chunk #{chunk_count}: type={type(chunk)}, content={chunk}")
                                    if isinstance(chunk, dict):
                                        logger.info(f"  Dict keys: {list(chunk.keys())}")
                                        if "message" in chunk:
                                            logger.info(f"  Message: {chunk['message']}")
                                            if isinstance(chunk["message"], dict):
                                                logger.info(f"  Message keys: {list(chunk['message'].keys())}")
                                
                                # ROBUST content extraction for different Ollama client versions
                                content = None
                                extraction_method = "none"
                                
                                # Debug: Print chunk structure for first few chunks
                                if chunk_count <= 3:
                                    logger.info(f"DEBUG Chunk #{chunk_count}: {chunk}")
                                
                                # COMPREHENSIVE Ollama format detection - handle both dicts and Pydantic objects
                                
                                # NEW: Handle Pydantic model objects (like your gpt-oss:20b chunks)
                                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                                    # Check for content in message.thinking FIRST (your model uses this!)
                                    if hasattr(chunk.message, 'thinking') and chunk.message.thinking:
                                        content = chunk.message.thinking
                                        extraction_method = "pydantic_message.thinking"
                                    # Then check regular content (even if empty - normal in streaming)
                                    else:
                                        content = chunk.message.content  # Can be empty string
                                        extraction_method = "pydantic_message.content"
                                
                                # Handle other Pydantic-style objects
                                elif hasattr(chunk, 'content') and chunk.content:
                                    content = chunk.content
                                    extraction_method = "pydantic_content"
                                elif hasattr(chunk, 'response') and chunk.response:
                                    content = chunk.response
                                    extraction_method = "pydantic_response"
                                elif hasattr(chunk, 'text') and chunk.text:
                                    content = chunk.text
                                    extraction_method = "pydantic_text"
                                
                                # Traditional dictionary format
                                elif isinstance(chunk, dict):
                                    # Most common format variations found in real-world usage:
                                    
                                    # Standard: {"message": {"content": "text", "role": "assistant"}}
                                    if "message" in chunk and isinstance(chunk["message"], dict):
                                        if "content" in chunk["message"]:
                                            content = chunk["message"]["content"]
                                            extraction_method = "message.content"
                                    
                                    # Legacy: {"response": "text"}
                                    elif "response" in chunk:
                                        content = chunk["response"]
                                        extraction_method = "response"
                                    
                                    # Sometimes content is directly in chunk: {"content": "text"}
                                    elif "content" in chunk:
                                        content = chunk["content"]
                                        extraction_method = "direct_content"
                                    
                                    # Check if it's a streaming completion format with done flag
                                    # {"model": "...", "created_at": "...", "response": "text", "done": false}
                                    elif "model" in chunk and "response" in chunk:
                                        content = chunk["response"]
                                        extraction_method = "model_response"
                                    
                                    # Alternative format with text field
                                    elif "text" in chunk:
                                        content = chunk["text"]
                                        extraction_method = "text"
                                    
                                    # OpenAI-compatible format: {"choices": [{"delta": {"content": "text"}}]}
                                    elif "choices" in chunk and isinstance(chunk["choices"], list) and len(chunk["choices"]) > 0:
                                        choice = chunk["choices"][0]
                                        if isinstance(choice, dict) and "delta" in choice and isinstance(choice["delta"], dict):
                                            if "content" in choice["delta"]:
                                                content = choice["delta"]["content"]
                                                extraction_method = "choices.delta.content"
                                    
                                    # Direct delta format: {"delta": {"content": "text"}}
                                    elif "delta" in chunk and isinstance(chunk["delta"], dict) and "content" in chunk["delta"]:
                                        content = chunk["delta"]["content"]
                                        extraction_method = "delta.content"
                                    
                                    # Sometimes the entire chunk is just {"text": "content"} or {"output": "content"}
                                    else:
                                        # Try common output fields
                                        for field in ["output", "answer", "reply", "generated_text", "completion", "text_output"]:
                                            if field in chunk and isinstance(chunk[field], str):
                                                content = chunk[field]
                                                extraction_method = field
                                                break
                                
                                # Handle direct string responses
                                elif isinstance(chunk, str):
                                    content = chunk
                                    extraction_method = "direct_string"
                                
                                # Handle other types (bytes, etc.)
                                elif hasattr(chunk, 'decode'):  # bytes
                                    try:
                                        content = chunk.decode('utf-8')
                                        extraction_method = "decoded_bytes"
                                    except:
                                        content = None
                                
                                # Log content extraction for first few chunks
                                if chunk_count <= 5:
                                    logger.info(f"  Extraction method: {extraction_method}, content: {repr(content)}")
                                    
                                    # If no content was extracted, log the full chunk structure for debugging
                                    if content is None and isinstance(chunk, dict):
                                        logger.warning(f"  ⚠️  No content extracted from chunk #{chunk_count}")
                                        logger.warning(f"  Full chunk structure: {json.dumps(chunk, indent=2, default=str)}")
                                
                                # Handle empty chunks and metadata chunks gracefully
                                # Many Ollama streams have empty chunks or metadata-only chunks mixed with content
                                if content is not None:
                                    # Even empty strings are valid (normal in streaming)
                                    pass
                                elif isinstance(chunk, dict):
                                    # Check if this is a metadata chunk we should ignore
                                    metadata_only = any(key in chunk for key in [
                                        "done", "total_duration", "load_duration", "prompt_eval_count", 
                                        "prompt_eval_duration", "eval_count", "eval_duration", "model",
                                        "created_at", "context"
                                    ])
                                    
                                    if metadata_only and chunk_count > 10:
                                        # Skip metadata chunks after the first few - these are normal
                                        continue
                                
                                # Process content (including empty strings which are normal in streaming)
                                if content is not None:
                                    if first_token_time is None and content.strip():
                                        first_token_time = time.time()
                                        stage_times["first_token"] = first_token_time
                                        waiting_for_first_token = False
                                        logger.info(f"First token received at chunk #{chunk_count}")
                                    
                                    response_text += content
                                    if content.strip():  # Only count non-empty content for tokens
                                        token_count += len(content.split())
                                    
                                    # Update response
                                    response_placeholder.markdown(response_text)
                                    
                                    # Simple live metrics (ChatGPT style)
                                    elapsed = time.time() - start_time
                                    with metrics_container.container():
                                        cols = st.columns(4)
                                        with cols[0]:
                                            st.caption(f"⏱️ {elapsed:.1f}s")
                                        with cols[1]:
                                            st.caption(f"📝 {token_count}")
                                        with cols[2]:
                                            st.caption(f"⚡ {token_count/elapsed:.1f} tok/s" if elapsed > 0 else "⚡ --/s")
                                        with cols[3]:
                                            st.caption("🔄 Streaming...")
                                elif waiting_for_first_token:
                                    # Show waiting indicator for first token
                                    with response_placeholder.container():
                                        st.info("🤔 AI is thinking...")
                        except Exception as stream_error:
                            st.error(f"Streaming error: {stream_error}")
                            logger.error(f"Streaming error details: {stream_error}")
                            # Force fallback to non-streaming
                            response_text = ""
                    
                        # If no content streamed, try a non-streaming fallback
                        if not response_text:
                            if chunk_count == 0:
                                st.warning("🔄 No chunks received from stream, trying non-streaming mode...")
                                logger.warning("No chunks received from Ollama stream")
                            else:
                                # Show debugging info to user
                                debug_msg = f"🔄 Received {chunk_count} chunks but no content extracted. Check console logs for chunk format details."
                                st.warning(debug_msg)
                                logger.warning(f"STREAMING ISSUE: Received {chunk_count} chunks from Ollama but extracted no content")
                                logger.warning("Check the debug logs above for actual chunk structures. This suggests a chunk format mismatch.")
                            
                            try:
                                resp = ollama.chat(model=gen_model, messages=messages, stream=False)
                            
                                # Try multiple ways to extract content from response
                                fallback_text = ""
                                
                                # First try object attributes (most likely for Ollama)
                                try:
                                    if hasattr(resp, 'message') and hasattr(resp.message, 'content'):
                                        fallback_text = resp.message.content
                                    elif hasattr(resp, 'response'):
                                        fallback_text = resp.response
                                    elif hasattr(resp, 'content'):
                                        fallback_text = resp.content
                                except Exception:
                                    pass
                                
                                # If no object attributes worked, try dictionary access
                                if not fallback_text and isinstance(resp, dict):
                                    if "message" in resp and isinstance(resp["message"], dict) and "content" in resp["message"]:
                                        fallback_text = resp["message"]["content"]
                                    elif "response" in resp:
                                        fallback_text = resp["response"]
                                    elif "content" in resp:
                                        fallback_text = resp["content"]
                                    elif "text" in resp:
                                        fallback_text = resp["text"]
                                
                                # Finally try if it's just a string
                                elif isinstance(resp, str):
                                    fallback_text = resp
                                
                                if fallback_text and fallback_text.strip():
                                    response_text = fallback_text
                                    token_count = len(response_text.split())
                                    response_placeholder.markdown(response_text)
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                else:
                                    # If still no content, show error
                                    response_text = "Unable to get response from model. Please check Ollama connection and try again."
                                    response_placeholder.error(response_text)
                                    
                            except Exception as fe:
                                error_msg = f"Error communicating with model: {str(fe)}"
                                st.error(error_msg)
                                response_text = error_msg
                                response_placeholder.markdown(response_text)

                    # Final metrics with detailed stage timing
                    stage_times["response_end"] = time.time()
                    total_time = stage_times["response_end"] - stage_times["pipeline_start"]
                    ttft = (stage_times["first_token"] - stage_times["pipeline_start"]) if stage_times["first_token"] else 0
                    
                    # Calculate stage durations
                    stage_durations = {}
                    if stage_times["search_start"] and stage_times["search_end"]:
                        stage_durations["search"] = stage_times["search_end"] - stage_times["search_start"]
                    if stage_times["context_prep_start"] and stage_times["context_prep_end"]:
                        stage_durations["context_prep"] = stage_times["context_prep_end"] - stage_times["context_prep_start"]
                    if stage_times["thinking_start"] and stage_times["thinking_end"]:
                        stage_durations["thinking"] = stage_times["thinking_end"] - stage_times["thinking_start"]
                    if stage_times["model_init_start"] and stage_times["model_init_end"]:
                        stage_durations["model_init"] = stage_times["model_init_end"] - stage_times["model_init_start"]
                    if stage_times["vision_start"] and stage_times["vision_end"]:
                        stage_durations["vision"] = stage_times["vision_end"] - stage_times["vision_start"]
                    if stage_times["response_start"] and stage_times["first_token"]:
                        stage_durations["to_first_token"] = stage_times["first_token"] - stage_times["response_start"]
                    
                    stats = {
                        "total_time": total_time,
                        "time_to_first_token": ttft,
                        "tokens": token_count,
                        "tokens_per_sec": token_count / total_time if total_time > 0 else 0,
                        "stage_durations": stage_durations,
                    }
                    
                    # Show final metrics (persistent after completion)
                    with metrics_container.container():
                        # Main metrics row
                        cols = st.columns(4)
                        with cols[0]:
                            st.caption(f"⏱️ {total_time:.1f}s")
                        with cols[1]:
                            st.caption(f"📝 {token_count}")
                        with cols[2]:
                            st.caption(f"⚡ {stats['tokens_per_sec']:.1f} tok/s")
                        with cols[3]:
                            st.caption("✅ Complete")
                        
                        # Detailed stage breakdown (collapsible)
                        with st.expander("📊 **Detailed Stage Timing**", expanded=False):
                            stage_cols = st.columns(3)
                            with stage_cols[0]:
                                if "search" in stage_durations:
                                    st.caption(f"🔍 Search: {stage_durations['search']:.2f}s")
                                if "context_prep" in stage_durations:
                                    st.caption(f"📝 Context: {stage_durations['context_prep']:.2f}s")
                            with stage_cols[1]:
                                if "thinking" in stage_durations:
                                    st.caption(f"🧠 Thinking: {stage_durations['thinking']:.2f}s")
                                if "model_init" in stage_durations:
                                    st.caption(f"🤖 Model Init: {stage_durations['model_init']:.2f}s")
                            with stage_cols[2]:
                                if "vision" in stage_durations:
                                    st.caption(f"🎨 Vision: {stage_durations['vision']:.2f}s")
                                if "to_first_token" in stage_durations:
                                    st.caption(f"🚀 To 1st Token: {stage_durations['to_first_token']:.2f}s")
                    
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
                # Simple no results message
                main_loading_container.empty()
                st.warning(f"⚠️ No relevant documents found in {search_time:.2f}s")
                
                no_results_msg = t("chat.no_relevant", "I couldn't find any relevant information in the indexed documents. Please make sure documents are indexed.")
                st.session_state.messages.append({"role": "assistant", "content": no_results_msg})
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(t("buttons.clear_chat", "🔄 Clear Chat")):
            st.session_state.messages = []
            st.session_state.response_stats = []
            st.rerun()

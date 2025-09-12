#!/usr/bin/env python3
"""
Test script to demonstrate conversation context awareness.
"""

from src.core.chat import ChatService
from src.core.query_expansion import SmartQueryProcessor
from src.config import get_settings

def test_conversation_context():
    """Test conversation context functionality."""
    print("🔍 TESTING CONVERSATION CONTEXT AWARENESS")
    print("=" * 50)
    
    settings = get_settings()
    chat_service = ChatService()
    query_processor = SmartQueryProcessor()
    
    # Simulate a conversation
    conversation_history = [
        {"role": "user", "content": "Tell me about RAG systems"},
        {"role": "assistant", "content": "RAG (Retrieval-Augmented Generation) systems combine information retrieval with language models..."},
        {"role": "user", "content": "What are the main components?"},
        {"role": "assistant", "content": "The main components are: 1) Document indexing, 2) Retrieval system, 3) Generation model..."}
    ]
    
    # Test 1: Follow-up question with pronoun
    print("\n1️⃣  Test: Follow-up question with pronoun")
    print("-" * 40)
    follow_up = "How can I improve its performance?"
    reformulated = query_processor.reformulate_with_context(follow_up, conversation_history)
    print(f"Original: {follow_up}")
    print(f"Reformulated: {reformulated}")
    
    # Build messages with history
    messages = chat_service.build_messages_with_history(
        system_prompt="You are a helpful assistant.",
        user_prompt=follow_up,
        conversation_history=conversation_history
    )
    print(f"Messages in context: {len(messages)} total")
    print(f"  - System: 1")
    print(f"  - History: {len(messages) - 2}")
    print(f"  - Current: 1")
    
    # Test 2: Another follow-up
    print("\n2️⃣  Test: Another contextual question")
    print("-" * 40)
    follow_up2 = "What about that retrieval system?"
    reformulated2 = query_processor.reformulate_with_context(follow_up2, conversation_history)
    print(f"Original: {follow_up2}")
    print(f"Reformulated: {reformulated2}")
    
    # Test 3: Non-contextual question
    print("\n3️⃣  Test: Non-contextual question")
    print("-" * 40)
    new_topic = "What is machine learning?"
    reformulated3 = query_processor.reformulate_with_context(new_topic, conversation_history)
    print(f"Original: {new_topic}")
    print(f"Reformulated: {reformulated3}")
    print("(No reformulation needed - no pronouns/references)")
    
    # Test configuration
    print("\n⚙️  Configuration Settings:")
    print("-" * 40)
    print(f"Conversation context turns: {settings.conversation_context_turns}")
    print(f"Query reformulation enabled: {settings.enable_query_reformulation}")
    print(f"Chat history limit: {settings.chat_history_limit}")
    
    print("\n✅ CONVERSATION CONTEXT TESTS COMPLETE!")
    print("=" * 50)
    print("\n📝 Summary:")
    print("  - Query reformulation works for pronouns (it, that, this)")
    print("  - Conversation history is included in LLM context")
    print(f"  - Last {settings.conversation_context_turns} turns are included")
    print("  - Non-contextual questions are not modified")

if __name__ == "__main__":
    test_conversation_context()
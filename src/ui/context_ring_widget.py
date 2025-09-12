"""
Compact Ring Widget for Context Usage Display

Shows real-time context usage near the chat input with color-coded warnings
and expandable details. Designed to be unobtrusive but informative.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import List, Optional
from ..core.context_manager import ContextUsage
from .i18n import t


def render_context_ring(
    context_usage: ContextUsage,
    show_details: bool = False,
    container_key: str = "context_ring"
) -> None:
    """
    Render a compact ring indicator showing context usage.
    
    Args:
        context_usage: Context usage statistics
        show_details: Whether to show expanded details
        container_key: Unique key for the container
    """
    # Color mapping based on warning level
    colors = {
        'green': '#10B981',   # Success green
        'yellow': '#F59E0B',  # Warning amber
        'red': '#EF4444'      # Error red
    }
    
    ring_color = colors.get(context_usage.warning_level, colors['green'])
    percentage = min(100, int(context_usage.usage_percentage * 100))
    
    # Create the ring HTML with JavaScript
    ring_html = f"""
    <div style="display: flex; align-items: center; gap: 12px; margin: 8px 0; padding: 8px 12px; 
                background: rgba(0,0,0,0.02); border-radius: 8px; border-left: 3px solid {ring_color};">
        
        <!-- Compact Ring SVG -->
        <div style="position: relative; width: 32px; height: 32px; flex-shrink: 0;">
            <svg width="32" height="32" style="transform: rotate(-90deg);">
                <!-- Background circle -->
                <circle cx="16" cy="16" r="14" fill="none" 
                        stroke="#E5E7EB" stroke-width="3"/>
                <!-- Progress arc -->
                <circle cx="16" cy="16" r="14" fill="none" 
                        stroke="{ring_color}" stroke-width="3"
                        stroke-dasharray="{2 * 3.14159 * 14}" 
                        stroke-dashoffset="{2 * 3.14159 * 14 * (1 - context_usage.usage_percentage)}"
                        stroke-linecap="round"
                        style="transition: stroke-dashoffset 0.5s ease;"/>
            </svg>
            <!-- Percentage text in center -->
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        font-size: 9px; font-weight: 600; color: {ring_color};">
                {percentage}%
            </div>
        </div>
        
        <!-- Status Text -->
        <div style="flex-grow: 1; min-width: 0;">
            <div style="font-size: 12px; font-weight: 500; color: #374151;">
                Memory: {_format_tokens(context_usage.total_tokens)} / {_format_tokens(context_usage.total_tokens + context_usage.free_tokens)} tokens
            </div>
            <div style="font-size: 10px; color: #6B7280;">
                {_get_status_message(context_usage)}
            </div>
        </div>
        
        <!-- Expand/Collapse Button -->
        <button onclick="toggleContextDetails('{container_key}')" 
                style="background: none; border: none; cursor: pointer; padding: 4px; 
                       color: #6B7280; font-size: 12px; border-radius: 4px;
                       transition: background-color 0.2s;"
                onmouseover="this.style.backgroundColor='rgba(0,0,0,0.05)'"
                onmouseout="this.style.backgroundColor='transparent'">
            📊
        </button>
    </div>
    
    <!-- Expandable Details Section -->
    <div id="context_details_{container_key}" 
         style="display: {'block' if show_details else 'none'}; 
                margin-top: 8px; padding: 12px; background: #F9FAFB; 
                border-radius: 6px; border: 1px solid #E5E7EB;">
        
        <div style="margin-bottom: 12px;">
            <div style="font-size: 13px; font-weight: 600; color: #374151; margin-bottom: 8px;">
                Token Breakdown
            </div>
            
            <!-- Token Breakdown Bars -->
            <div style="margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-size: 11px; color: #6B7280;">System Prompt</span>
                    <span style="font-size: 11px; font-weight: 500; color: #374151;">{context_usage.system_tokens:,}</span>
                </div>
                <div style="width: 100%; height: 4px; background: #E5E7EB; border-radius: 2px; overflow: hidden;">
                    <div style="width: {_safe_percentage(context_usage.system_tokens, context_usage.total_tokens)}%; 
                                height: 100%; background: #6B7280; transition: width 0.3s ease;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-size: 11px; color: #6B7280;">Chat History</span>
                    <span style="font-size: 11px; font-weight: 500; color: #374151;">{context_usage.history_tokens:,}</span>
                </div>
                <div style="width: 100%; height: 4px; background: #E5E7EB; border-radius: 2px; overflow: hidden;">
                    <div style="width: {_safe_percentage(context_usage.history_tokens, context_usage.total_tokens)}%; 
                                height: 100%; background: #8B5CF6; transition: width 0.3s ease;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-size: 11px; color: #6B7280;">Current Query</span>
                    <span style="font-size: 11px; font-weight: 500; color: #374151;">{context_usage.current_tokens:,}</span>
                </div>
                <div style="width: 100%; height: 4px; background: #E5E7EB; border-radius: 2px; overflow: hidden;">
                    <div style="width: {_safe_percentage(context_usage.current_tokens, context_usage.total_tokens)}%; 
                                height: 100%; background: #3B82F6; transition: width 0.3s ease;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-size: 11px; color: #6B7280;">Retrieved Docs</span>
                    <span style="font-size: 11px; font-weight: 500; color: #374151;">{context_usage.documents_tokens:,}</span>
                </div>
                <div style="width: 100%; height: 4px; background: #E5E7EB; border-radius: 2px; overflow: hidden;">
                    <div style="width: {_safe_percentage(context_usage.documents_tokens, context_usage.total_tokens)}%; 
                                height: 100%; background: #10B981; transition: width 0.3s ease;"></div>
                </div>
            </div>
        </div>
        
        <div style="font-size: 11px; color: #6B7280; line-height: 1.4;">
            {_get_detailed_recommendations(context_usage)}
        </div>
    </div>
    
    <script>
    function toggleContextDetails(containerKey) {{
        const details = document.getElementById('context_details_' + containerKey);
        if (details.style.display === 'none') {{
            details.style.display = 'block';
        }} else {{
            details.style.display = 'none';
        }}
    }}
    </script>
    """
    
    # Render the HTML component
    components.html(ring_html, height=120 if show_details else 60)


def render_minimal_context_ring(
    context_usage: ContextUsage,
    size: int = 24
) -> None:
    """
    Render a very minimal ring for inline display.
    
    Args:
        context_usage: Context usage statistics
        size: Ring diameter in pixels
    """
    colors = {
        'green': '#10B981',
        'yellow': '#F59E0B', 
        'red': '#EF4444'
    }
    
    ring_color = colors.get(context_usage.warning_level, colors['green'])
    percentage = min(100, int(context_usage.usage_percentage * 100))
    
    ring_html = f"""
    <div style="display: inline-flex; align-items: center; gap: 6px; vertical-align: middle;">
        <div style="position: relative; width: {size}px; height: {size}px;">
            <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
                <circle cx="{size//2}" cy="{size//2}" r="{size//2 - 2}" fill="none" 
                        stroke="#E5E7EB" stroke-width="2"/>
                <circle cx="{size//2}" cy="{size//2}" r="{size//2 - 2}" fill="none" 
                        stroke="{ring_color}" stroke-width="2"
                        stroke-dasharray="{2 * 3.14159 * (size//2 - 2)}" 
                        stroke-dashoffset="{2 * 3.14159 * (size//2 - 2) * (1 - context_usage.usage_percentage)}"
                        stroke-linecap="round"
                        style="transition: stroke-dashoffset 0.5s ease;"/>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        font-size: 8px; font-weight: 600; color: {ring_color};">
                {percentage}%
            </div>
        </div>
        <span style="font-size: 11px; color: var(--text-color, #1f2937); font-weight: 500;">
            {_get_mini_status(context_usage)}
        </span>
    </div>
    """
    
    components.html(ring_html, height=32)


def render_inline_context_badge(
    context_usage: ContextUsage,
    size: int = 24
) -> None:
    """
    Render a button-style context badge to match Streamlit button design.
    
    Args:
        context_usage: Context usage statistics  
        size: Ring diameter in pixels
    """
    colors = {
        'green': '#10B981',
        'yellow': '#F59E0B', 
        'red': '#EF4444'
    }
    
    ring_color = colors.get(context_usage.warning_level, colors['green'])
    percentage = min(100, int(context_usage.usage_percentage * 100))
    
    # Get a short status with emoji
    if context_usage.warning_level == 'red':
        status_icon = "🔴"
        bg_color = "rgba(239, 68, 68, 0.1)"
        border_color = "#EF4444"
    elif context_usage.warning_level == 'yellow':
        status_icon = "🟡"
        bg_color = "rgba(245, 158, 11, 0.1)"
        border_color = "#F59E0B"
    else:
        status_icon = "🟢"
        bg_color = "rgba(16, 185, 129, 0.1)"
        border_color = "#10B981"
    
    # Match Streamlit's dark theme button styling exactly
    badge_html = f"""
    <div style="display: inline-flex; align-items: center; justify-content: center;
                padding: 0.25rem 0.75rem; height: 2.375rem;
                background-color: rgba(38, 39, 48, 1);
                border: 1px solid rgba(250, 250, 250, 0.2);
                border-radius: 0.375rem;
                font-family: 'Source Sans Pro', sans-serif;
                font-size: 0.875rem; font-weight: 400; line-height: 1.25rem;
                color: rgb(250, 250, 250);
                transition: all 0.2s ease;
                gap: 6px; min-width: auto;
                user-select: none; cursor: default;
                box-sizing: border-box;">
        
        <!-- Ring indicator -->
        <div style="position: relative; width: 16px; height: 16px; flex-shrink: 0;">
            <svg width="16" height="16" style="transform: rotate(-90deg);">
                <!-- Background circle -->
                <circle cx="8" cy="8" r="6" fill="none" 
                        stroke="rgba(250, 250, 250, 0.3)" stroke-width="1.5"/>
                <!-- Progress circle -->
                <circle cx="8" cy="8" r="6" fill="none" 
                        stroke="{ring_color}" stroke-width="1.5"
                        stroke-dasharray="{2 * 3.14159 * 6}" 
                        stroke-dashoffset="{2 * 3.14159 * 6 * (1 - context_usage.usage_percentage)}"
                        stroke-linecap="round"
                        style="transition: stroke-dashoffset 0.3s ease;"/>
            </svg>
            <!-- Percentage in center -->
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        font-size: 8px; font-weight: 600; color: {ring_color};">
                {percentage}%
            </div>
        </div>
        
        <!-- Status text -->
        <span style="color: rgb(250, 250, 250); font-weight: 400; white-space: nowrap;">
            Context
        </span>
    </div>
    """
    
    components.html(badge_html, height=42)


def _get_status_message(context_usage: ContextUsage) -> str:
    """Get status message based on context usage."""
    if context_usage.warning_level == 'red':
        return "⚠️ Memory full - older messages may be summarized"
    elif context_usage.warning_level == 'yellow':
        return "⚡ Memory getting full - monitor usage"
    else:
        return "✅ Memory healthy - plenty of room for conversation"


def _get_mini_status(context_usage: ContextUsage) -> str:
    """Get mini status for compact display."""
    if context_usage.warning_level == 'red':
        return "Memory Full"
    elif context_usage.warning_level == 'yellow':
        return "Memory High"
    else:
        return "Memory OK"


def _get_detailed_recommendations(context_usage: ContextUsage) -> str:
    """Get detailed recommendations HTML."""
    recommendations = []
    
    if context_usage.warning_level == 'red':
        recommendations.extend([
            "🔴 Chat memory is very full. Older messages may be automatically summarized.",
            "💡 Consider starting a new conversation for better performance."
        ])
    elif context_usage.warning_level == 'yellow':
        recommendations.extend([
            "🟡 Chat memory is getting full. Monitor usage as conversation continues.",
            "🔍 Use specific queries to reduce document retrieval size."
        ])
    else:
        recommendations.append("🟢 Chat memory is healthy. Continue your conversation normally.")
    
    if context_usage.documents_tokens > 8000:
        recommendations.append("📄 Large amount of document content. Try more specific search terms.")
    
    if context_usage.history_tokens > 15000:
        recommendations.append("💬 Long conversation history. Key points may be preserved automatically.")
    
    return "<br>".join(recommendations)


def _safe_percentage(part: int, total: int) -> float:
    """Calculate safe percentage to avoid division by zero."""
    if total == 0:
        return 0
    return min(100, (part / total) * 100)


def _format_tokens(token_count: int) -> str:
    """Format token count in human-readable format (e.g., 1.2k, 13k, 128k)."""
    if token_count < 1000:
        return str(token_count)
    elif token_count < 10000:
        return f"{token_count / 1000:.1f}k"
    elif token_count < 100000:
        return f"{token_count // 1000}k"
    else:
        return f"{token_count // 1000}k"


def render_context_warning_banner(
    context_usage: ContextUsage,
    optimization_info: dict
) -> None:
    """
    Render a banner when sliding window has been applied.
    
    Args:
        context_usage: Current context usage
        optimization_info: Information about applied optimizations
    """
    if not optimization_info.get('sliding_applied', False):
        return
    
    messages_removed = optimization_info.get('messages_removed', 0)
    tokens_saved = optimization_info.get('tokens_saved', 0)
    
    banner_html = f"""
    <div style="margin: 8px 0; padding: 10px 12px; background: linear-gradient(90deg, #FEF3C7, #FDE68A); 
                border-left: 4px solid #F59E0B; border-radius: 6px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 16px;">🔄</span>
            <div>
                <div style="font-size: 13px; font-weight: 600; color: #92400E;">
                    Memory Optimized
                </div>
                <div style="font-size: 11px; color: #A16207; margin-top: 2px;">
                    Removed {messages_removed} older messages ({_format_tokens(tokens_saved)} tokens) to maintain performance
                </div>
            </div>
        </div>
    </div>
    """
    
    components.html(banner_html, height=60)
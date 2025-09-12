"""
Interactive context usage visualizer with ring UI.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Optional
import math
from .i18n import t

class ContextVisualizer:
    """Visualize context usage with interactive ring chart."""
    
    def __init__(self, max_tokens: int = 128000):
        """Initialize with model's max context window."""
        self.max_tokens = max_tokens
        self.colors = {
            'system': '#6B7280',      # Gray
            'history': '#8B5CF6',     # Purple
            'current': '#3B82F6',     # Blue
            'documents': '#10B981',   # Green
            'free': '#E5E7EB'         # Light gray
        }
    
    def create_ring_chart(
        self, 
        context_usage: Dict[str, int],
        show_details: bool = True
    ) -> go.Figure:
        """
        Create interactive ring chart for context usage.
        
        Args:
            context_usage: Dictionary with token counts by category
            show_details: Whether to show detailed breakdown
        """
        # Calculate percentages
        total_used = sum(context_usage.values())
        free_space = self.max_tokens - total_used
        
        # Prepare data for ring chart
        labels = []
        values = []
        colors = []
        hover_text = []
        
        for category, tokens in context_usage.items():
            labels.append(category.title())
            values.append(tokens)
            colors.append(self.colors.get(category, '#9CA3AF'))
            percentage = (tokens / self.max_tokens) * 100
            hover_text.append(
                f"<b>{category.title()}</b><br>"
                f"Tokens: {tokens:,}<br>"
                f"Percentage: {percentage:.1f}%"
            )
        
        # Add free space
        labels.append('Free Space')
        values.append(free_space)
        colors.append(self.colors['free'])
        free_percentage = (free_space / self.max_tokens) * 100
        hover_text.append(
            f"<b>Free Space</b><br>"
            f"Tokens: {free_space:,}<br>"
            f"Percentage: {free_percentage:.1f}%"
        )
        
        # Create ring chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
            hovertemplate='%{customdata}',
            customdata=hover_text,
            textposition='outside',
            textinfo='label+percent' if show_details else 'percent',
            pull=[0.1 if l == 'Current' else 0 for l in labels]  # Emphasize current query
        )])
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Context Usage: {total_used:,} / {self.max_tokens:,} tokens',
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            height=400,
            margin=dict(t=60, b=20, l=20, r=20),
            annotations=[
                dict(
                    text=f'<b>{(total_used/self.max_tokens)*100:.0f}%</b><br>Used',
                    x=0.5, y=0.5,
                    font_size=24,
                    showarrow=False
                )
            ]
        )
        
        return fig
    
    def create_progress_bars(
        self, 
        context_usage: Dict[str, int]
    ) -> None:
        """Create progress bars for each context category."""
        st.markdown(f"### {t('context.breakdown')}")
        
        total_used = sum(context_usage.values())
        
        for category, tokens in context_usage.items():
            percentage = (tokens / self.max_tokens) * 100
            color = self.colors.get(category, '#9CA3AF')
            
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.markdown(f"**{category.title()}**")
            with col2:
                # Custom progress bar with color
                st.markdown(
                    f"""
                    <div style="background-color: #E5E7EB; border-radius: 10px; height: 20px; position: relative;">
                        <div style="background-color: {color}; width: {percentage}%; height: 100%; border-radius: 10px; transition: width 0.3s;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(f"{tokens:,} tokens")
        
        # Add free space
        free_space = self.max_tokens - total_used
        free_percentage = (free_space / self.max_tokens) * 100
        
        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.markdown(f"**{t('context.free_space')}**")
        with col2:
            st.markdown(
                f"""
                <div style="background-color: #E5E7EB; border-radius: 10px; height: 20px; position: relative;">
                    <div style="background-color: {self.colors['free']}; width: {free_percentage}%; height: 100%; border-radius: 10px;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(f"{free_space:,} tokens")
    
    def calculate_message_tokens(self, messages: List[Dict]) -> Dict[str, int]:
        """
        Estimate token usage from messages.
        
        Simple heuristic: ~4 characters per token
        """
        usage = {
            'system': 0,
            'history': 0,
            'current': 0,
            'documents': 0
        }
        
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            # Rough estimate: 4 chars = 1 token
            tokens = len(content) // 4
            
            if msg['role'] == 'system':
                usage['system'] += tokens
            elif i == len(messages) - 1:  # Last message
                usage['current'] += tokens
            else:
                usage['history'] += tokens
        
        return usage


def render_context_usage_widget(
    messages: Optional[List[Dict]] = None,
    search_results_tokens: int = 0
):
    """
    Render the context usage widget in Streamlit.
    
    Args:
        messages: Current conversation messages
        search_results_tokens: Tokens used by search results
    """
    visualizer = ContextVisualizer()
    
    # Calculate current usage
    if messages:
        usage = visualizer.calculate_message_tokens(messages)
        usage['documents'] = search_results_tokens
    else:
        # Demo data
        usage = {
            'system': 1500,
            'history': 3000,
            'current': 500,
            'documents': 2000
        }
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🔮 Ring Chart", "📊 Bar View", "📈 Details"])
    
    with tab1:
        fig = visualizer.create_ring_chart(usage)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        visualizer.create_progress_bars(usage)
    
    with tab3:
        st.markdown(f"### {t('context.detailed_usage')}")
        
        total_used = sum(usage.values())
        free_space = visualizer.max_tokens - total_used
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(t('context.total_used'), f"{total_used:,} tokens", 
                     f"{(total_used/visualizer.max_tokens)*100:.1f}%")
            st.metric(t('context.system_prompt'), f"{usage['system']:,} tokens")
            st.metric(t('context.conversation_history'), f"{usage['history']:,} tokens")
        
        with col2:
            st.metric(t('context.free_space_metric'), f"{free_space:,} tokens",
                     f"{(free_space/visualizer.max_tokens)*100:.1f}%")
            st.metric(t('context.current_query'), f"{usage['current']:,} tokens")
            st.metric(t('context.retrieved_documents'), f"{usage['documents']:,} tokens")
        
        # Recommendations
        st.markdown(f"### {t('context.optimization_tips')}")
        
        if usage['history'] > 10000:
            st.warning(t('context.warning_large_history'))
        
        if usage['documents'] > 5000:
            st.info(t('context.info_many_docs'))
        
        if (total_used / visualizer.max_tokens) > 0.75:
            st.error(t('context.error_high_usage'))
        elif (total_used / visualizer.max_tokens) > 0.5:
            st.warning(t('context.warning_moderate_usage'))
        else:
            st.success(t('context.success_healthy_usage'))


if __name__ == "__main__":
    # Demo mode
    st.set_page_config(page_title="Context Usage Visualizer", layout="wide")
    st.title("🔮 Context Usage Visualizer")
    
    # Demo with sample data
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant. " * 100},
        {"role": "user", "content": "Tell me about RAG systems. " * 50},
        {"role": "assistant", "content": "RAG systems are... " * 200},
        {"role": "user", "content": "What about performance? " * 20}
    ]
    
    render_context_usage_widget(sample_messages, search_results_tokens=3000)
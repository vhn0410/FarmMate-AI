"""
Script để visualize Supervisor Multi-Agent Architecture
Yêu cầu: pip install graphviz
"""

from graphviz import Digraph

def visualize_supervisor_architecture():
    """Tạo diagram cho Supervisor Multi-Agent workflow"""
    
    # Tạo directed graph
    dot = Digraph(comment='Supervisor Multi-Agent Architecture')
    dot.attr(rankdir='TB', size='12,10')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # ===== NODES =====
    
    # Start/End nodes
    dot.node('start', 'User Query\n(Câu hỏi của nông dân)', 
             shape='ellipse', fillcolor='#e1f5e1', style='filled')
    dot.node('end', 'Final Response\n(Câu trả lời cuối cùng)', 
             shape='ellipse', fillcolor='#ffe1e1', style='filled')
    
    # Supervisor (central node)
    dot.node('supervisor', 
             'SUPERVISOR\nAgent\n\n· Điều phối workflow\n· Quyết định agent tiếp theo\n· Quản lý iteration', 
             fillcolor='#fff4e1', style='filled,bold', fontsize='12')
    
    # Agent nodes với mô tả chi tiết
    dot.node('query_enhancement', 
             'Query Enhancement\nAgent\n\n'
             '📝 Nhiệm vụ:\n'
             '· Phân tích ý định\n'
             '· Mở rộng câu hỏi\n'
             '· Xác định tools cần gọi\n\n'
             'Output: enhanced_query, tool_plan',
             fillcolor='#e1f0ff', style='filled')
    
    dot.node('tool_execution', 
             'Tool Execution\nAgent\n\n'
             '🔧 Nhiệm vụ:\n'
             '· Gọi Sensor API\n'
             '· Gọi Knowledge Base\n'
             '· Thu thập dữ liệu\n\n'
             'Output: sensor_data, kb_context',
             fillcolor='#f0e1ff', style='filled')
    
    dot.node('response_generation', 
             'Response Generation\nAgent\n\n'
             '💬 Nhiệm vụ:\n'
             '· Phân tích dữ liệu\n'
             '· Tạo câu trả lời\n'
             '· Stream output to user\n\n'
             'Output: draft_response',
             fillcolor='#ffe1f0', style='filled')
    
    dot.node('quality_assurance', 
             'Quality Assurance\nAgent\n\n'
             '✅ Nhiệm vụ:\n'
             '· Kiểm tra chính xác\n'
             '· Đánh giá chất lượng (score)\n'
             '· Feedback cải thiện\n\n'
             'Output: qa_feedback, needs_improvement',
             fillcolor='#e1ffe1', style='filled')
    
    # ===== EDGES (Workflow Flow) =====
    
    # Entry point
    dot.edge('start', 'supervisor', label='Initial request')
    
    # Supervisor → Query Enhancement
    dot.edge('supervisor', 'query_enhancement', 
             label='iteration == 0', color='blue', fontcolor='blue')
    dot.edge('query_enhancement', 'supervisor', 
             label='return enhanced_query', color='gray')
    
    # Supervisor → Tool Execution
    dot.edge('supervisor', 'tool_execution', 
             label='after enhancement', color='blue', fontcolor='blue')
    dot.edge('tool_execution', 'supervisor', 
             label='return sensor_data + kb_context', color='gray')
    
    # Supervisor → Response Generation
    dot.edge('supervisor', 'response_generation', 
             label='has data', color='blue', fontcolor='blue')
    dot.edge('response_generation', 'supervisor', 
             label='return draft_response', color='gray')
    
    # Supervisor → Quality Assurance
    dot.edge('supervisor', 'quality_assurance', 
             label='first QA check', color='blue', fontcolor='blue')
    dot.edge('quality_assurance', 'supervisor', 
             label='return qa_feedback', color='gray')
    
    # Improvement loop
    dot.edge('supervisor', 'response_generation', 
             label='needs_improvement\n& iteration < 3', 
             color='red', fontcolor='red', style='dashed')
    
    # Final exit
    dot.edge('supervisor', 'end', 
             label='score ≥ 75 OR\niteration ≥ 3', 
             color='green', fontcolor='green', style='bold')
    
    return dot


def visualize_state_flow():
    """Tạo diagram cho State flow trong system"""
    
    dot = Digraph(comment='State Management Flow')
    dot.attr(rankdir='LR', size='14,8')
    dot.attr('node', shape='record', style='filled', fontname='Arial')
    
    # State structure
    dot.node('state', 
             '{State (TypedDict)|'
             'messages: list\\l|'
             'original_query: str\\l|'
             'enhanced_query: str\\l|'
             'tool_plan: dict\\l|'
             'sensor_data: dict\\l|'
             'kb_context: str\\l|'
             'draft_response: str\\l|'
             'qa_feedback: dict\\l|'
             'iteration_count: int\\l|'
             'next_agent: str\\l|'
             'needs_improvement: bool\\l|'
             'has_kb_data: bool\\l|'
             'has_sensor_data: bool\\l}',
             fillcolor='#ffffcc')
    
    # Agents update state
    agents = [
        ('Query\nEnhancement', '#e1f0ff', 'enhanced_query\ntool_plan'),
        ('Tool\nExecution', '#f0e1ff', 'sensor_data\nkb_context\nhas_*_data'),
        ('Response\nGeneration', '#ffe1f0', 'draft_response\nmessages'),
        ('Quality\nAssurance', '#e1ffe1', 'qa_feedback\nneeds_improvement'),
        ('Supervisor', '#fff4e1', 'next_agent\niteration_count')
    ]
    
    for i, (name, color, updates) in enumerate(agents):
        node_id = f'agent{i}'
        dot.node(node_id, f'{name}\n\nUpdates:\n{updates}', 
                fillcolor=color, style='filled', shape='box')
        dot.edge(node_id, 'state', label='update', style='dashed')
        dot.edge('state', node_id, label='read', style='dotted')
    
    return dot


def visualize_decision_logic():
    """Tạo flowchart cho Supervisor decision logic"""
    
    dot = Digraph(comment='Supervisor Decision Logic')
    dot.attr(rankdir='TB', size='10,12')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    
    # Decision nodes
    dot.node('check_iter', 'iteration == 0?', 
             shape='diamond', fillcolor='#ffffcc')
    dot.node('check_last', 'last_agent?', 
             shape='diamond', fillcolor='#ffffcc')
    dot.node('check_qa', 'qa_feedback exists?', 
             shape='diamond', fillcolor='#ffffcc')
    dot.node('check_improve', 'needs_improvement\n& iteration < 3?', 
             shape='diamond', fillcolor='#ffffcc')
    
    # Action nodes
    dot.node('query_enh', 'query_enhancement', fillcolor='#e1f0ff')
    dot.node('tool_exec', 'tool_execution', fillcolor='#f0e1ff')
    dot.node('response_gen', 'response_generation', fillcolor='#ffe1f0')
    dot.node('qa', 'quality_assurance', fillcolor='#e1ffe1')
    dot.node('end_node', 'END', fillcolor='#ffe1e1')
    
    # Flow
    dot.edge('check_iter', 'query_enh', label='Yes')
    dot.edge('check_iter', 'check_last', label='No')
    
    dot.edge('check_last', 'tool_exec', label='query_enhancement')
    dot.edge('check_last', 'response_gen', label='tool_execution')
    dot.edge('check_last', 'check_qa', label='response_generation')
    dot.edge('check_last', 'check_improve', label='quality_assurance')
    
    dot.edge('check_qa', 'qa', label='No (first time)')
    dot.edge('check_qa', 'end_node', label='Yes')
    
    dot.edge('check_improve', 'response_gen', label='Yes')
    dot.edge('check_improve', 'end_node', label='No')
    
    return dot


if __name__ == "__main__":
    # Generate all diagrams
    
    print("🎨 Generating Supervisor Multi-Agent Architecture...")
    arch_graph = visualize_supervisor_architecture()
    arch_graph.render('supervisor_architecture', format='png', cleanup=True)
    print("✅ Saved: supervisor_architecture.png")
    
    print("\n🎨 Generating State Flow Diagram...")
    state_graph = visualize_state_flow()
    state_graph.render('state_flow', format='png', cleanup=True)
    print("✅ Saved: state_flow.png")
    
    print("\n🎨 Generating Decision Logic Flowchart...")
    decision_graph = visualize_decision_logic()
    decision_graph.render('decision_logic', format='png', cleanup=True)
    print("✅ Saved: decision_logic.png")
    
    print("\n✨ All diagrams generated successfully!")
    print("\n📊 You can also view them in browser:")
    print("   - supervisor_architecture.png")
    print("   - state_flow.png")
    print("   - decision_logic.png")
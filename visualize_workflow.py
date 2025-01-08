import graphviz
from typing import TypedDict

class WorkflowVisualizer:
    def __init__(self):
        self.dot = graphviz.Digraph(comment='Conversation Analysis Workflow')
        self.setup_style()
    
    def setup_style(self):
        # 전체 그래프 스타일 설정
        self.dot.attr(rankdir='TB')  # Top to Bottom 방향
        self.dot.attr('node', shape='box', 
                     style='rounded,filled', 
                     fillcolor='lightblue',
                     fontname='Arial',
                     margin='0.3,0.1')
        self.dot.attr('edge', 
                     fontname='Arial',
                     color='#666666',
                     arrowsize='0.8')

    def create_workflow_visualization(self):
        # 데이터 처리 노드
        with self.dot.subgraph(name='cluster_0') as s:
            s.attr(label='Data Processing', style='rounded', color='lightgrey')
            s.node('process_raw_data', 'Raw Data\nProcessing')
            s.node('create_dataframe', 'DataFrame\nCreation')
            s.node('prepare_df', 'Teacher/Student\nDF Preparation')
        
        # 분석 노드
        with self.dot.subgraph(name='cluster_1') as s:
            s.attr(label='Analysis', style='rounded', color='lightgrey')
            s.node('teacher_analysis', 'Teacher Questions\nAnalysis')
            s.node('student_analysis', 'Student Questions\nAnalysis')
        
        # 결과 저장 노드
        self.dot.node('save_results', 'Save Results')
        
        # 엣지 추가
        self.dot.edge('process_raw_data', 'create_dataframe')
        self.dot.edge('create_dataframe', 'prepare_df')
        self.dot.edge('prepare_df', 'teacher_analysis')
        self.dot.edge('teacher_analysis', 'student_analysis')
        self.dot.edge('student_analysis', 'save_results')
        
        return self.dot

    def create_detailed_workflow(self):
        # 데이터 처리 단계
        with self.dot.subgraph(name='cluster_0') as s:
            s.attr(label='Data Processing', style='rounded', color='lightgrey')
            s.node('raw_data', 'Raw Data Processing\n(File Loading & Merging)')
            s.node('dataframe', 'DataFrame Creation\n(Teacher & Student Data)')
            s.node('preparation', 'Data Preparation\n(Splitting & Chunking)')
        
        # 교사 분석 단계
        with self.dot.subgraph(name='cluster_1') as s:
            s.attr(label='Teacher Analysis', style='rounded', color='lightblue')
            s.node('teacher_question_check', 'Question Check')
            s.node('teacher_learning', 'Learning Question\nClassification')
            s.node('teacher_digging', 'Digging Question\nAnalysis')
        
        # 학생 분석 단계
        with self.dot.subgraph(name='cluster_2') as s:
            s.attr(label='Student Analysis', style='rounded', color='lightgreen')
            s.node('student_question_check', 'Question Check')
            s.node('student_learning', 'Learning Question\nClassification')
            s.node('student_concretizing', 'Concretizing Question\nAnalysis')
        
        # 결과 저장
        self.dot.node('results', 'Save Results\n(JSON Files)', shape='box')
        
        # 데이터 처리 흐름
        self.dot.edge('raw_data', 'dataframe')
        self.dot.edge('dataframe', 'preparation')
        
        # 교사 분석 흐름
        self.dot.edge('preparation', 'teacher_question_check')
        self.dot.edge('teacher_question_check', 'teacher_learning')
        self.dot.edge('teacher_learning', 'teacher_digging')
        
        # 학생 분석 흐름
        self.dot.edge('preparation', 'student_question_check')
        self.dot.edge('student_question_check', 'student_learning')
        self.dot.edge('student_learning', 'student_concretizing')
        
        # 결과 저장
        self.dot.edge('teacher_digging', 'results')
        self.dot.edge('student_concretizing', 'results')
        
        return self.dot

def main():
    # 시각화 도구 인스턴스 생성
    visualizer = WorkflowVisualizer()
    
    # 기본 워크플로우 생성 및 저장
    basic_workflow = visualizer.create_workflow_visualization()
    basic_workflow.render('conversation_workflow_basic', format='png', cleanup=True)
    
    # 새로운 시각화 도구 인스턴스 생성 (상세 워크플로우용)
    detailed_visualizer = WorkflowVisualizer()
    
    # 상세 워크플로우 생성 및 저장
    detailed_workflow = detailed_visualizer.create_detailed_workflow()
    detailed_workflow.render('conversation_workflow_detailed', format='png', cleanup=True)

if __name__ == "__main__":
    main()
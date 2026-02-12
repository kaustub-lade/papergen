"""
Paper Generator Module
Generates final question paper with formatting and export capabilities
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import io

try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib import colors
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install: pip install reportlab")


class PaperGenerator:
    """
    Generate formatted question papers from questions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Paper Generator
        
        Args:
            config: Configuration dictionary for paper formatting
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'page_size': A4,
            'margin': {
                'top': 72,
                'bottom': 72,
                'left': 72,
                'right': 72
            },
            'font': {
                'family': 'Helvetica',
                'size': {
                    'title': 16,
                    'heading': 14,
                    'body': 12,
                    'caption': 10
                }
            },
            'header': {
                'include': True,
                'text': 'Question Paper'
            },
            'footer': {
                'include': True,
                'page_numbers': True
            }
        }
    
    def create_paper(
        self,
        questions: List[Dict[str, Any]],
        set_name: str = "Set A",
        total_marks: int = 100,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create a structured question paper
        
        Args:
            questions: List of questions
            set_name: Name of the set (A, B, C)
            total_marks: Total marks for the paper
            metadata: Additional metadata
        
        Returns:
            Structured paper dictionary
        """
        metadata = metadata or {}
        
        # Group questions by Bloom's level or marks
        grouped_questions = self._group_questions(questions)
        
        # Create paper structure
        paper = {
            'set_name': set_name,
            'total_marks': total_marks,
            'total_questions': len(questions),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'metadata': metadata,
            'sections': grouped_questions
        }
        
        return grouped_questions
    
    def _group_questions(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group questions into sections"""
        # Group by marks (common pattern: short, medium, long answers)
        short_answers = []  # 1-3 marks
        medium_answers = []  # 4-7 marks
        long_answers = []  # 8+ marks
        
        for q in questions:
            marks = q.get('marks', 5)
            
            if marks <= 3:
                short_answers.append(q)
            elif marks <= 7:
                medium_answers.append(q)
            else:
                long_answers.append(q)
        
        sections = {}
        
        if short_answers:
            sections['Section A: Short Answer Questions'] = short_answers
        
        if medium_answers:
            sections['Section B: Medium Answer Questions'] = medium_answers
        
        if long_answers:
            sections['Section C: Long Answer Questions'] = long_answers
        
        return sections
    
    def export_to_pdf(
        self,
        paper_data: Dict[str, List[Dict[str, Any]]],
        filename: Optional[str] = None,
        set_name: str = "Set A",
        course_name: str = "Course Examination",
        total_marks: int = 100,
        duration: str = "3 Hours"
    ) -> bytes:
        """
        Export paper to PDF format
        
        Args:
            paper_data: Structured paper data
            filename: Output filename (optional)
            set_name: Set name
            course_name: Course name
            total_marks: Total marks
            duration: Examination duration
        
        Returns:
            PDF bytes
        """
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.config['page_size'],
            topMargin=self.config['margin']['top'],
            bottomMargin=self.config['margin']['bottom'],
            leftMargin=self.config['margin']['left'],
            rightMargin=self.config['margin']['right']
        )
        
        # Container for PDF elements
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.black,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            leftIndent=20
        )
        
        # Header
        header_data = [
            [Paragraph(course_name, title_style)],
            [Paragraph(f"{set_name} | Total Marks: {total_marks} | Duration: {duration}", styles['Normal'])]
        ]
        
        header_table = Table(header_data, colWidths=[doc.width])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        story.append(header_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Instructions
        instructions = Paragraph(
            "<b>Instructions:</b><br/>"
            "1. Answer all questions.<br/>"
            "2. Write clearly and legibly.<br/>"
            "3. Marks are indicated against each question.<br/>"
            "4. Start each section on a new page if required.",
            styles['Normal']
        )
        story.append(instructions)
        story.append(Spacer(1, 0.3 * inch))
        
        # Add horizontal line
        story.append(Spacer(1, 0.1 * inch))
        
        # Add sections and questions
        section_num = 1
        for section_name, questions in paper_data.items():
            # Section heading
            story.append(Paragraph(section_name, heading_style))
            story.append(Spacer(1, 0.2 * inch))
            
            # Questions
            for q in questions:
                q_num = q.get('number', '')
                q_text = q.get('question', '')
                q_marks = q.get('marks', 0)
                
                # Question text
                question_para = Paragraph(
                    f"<b>Q{q_num}.</b> {q_text} <b>[{q_marks} marks]</b>",
                    question_style
                )
                story.append(question_para)
                story.append(Spacer(1, 0.15 * inch))
            
            story.append(Spacer(1, 0.2 * inch))
            section_num += 1
        
        # Footer
        footer = Paragraph(
            f"<i>Generated on {datetime.now().strftime('%B %d, %Y')}</i>",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER)
        )
        story.append(Spacer(1, 0.5 * inch))
        story.append(footer)
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def export_to_text(
        self,
        paper_data: Dict[str, List[Dict[str, Any]]],
        set_name: str = "Set A"
    ) -> str:
        """
        Export paper to plain text format
        
        Args:
            paper_data: Structured paper data
            set_name: Set name
        
        Returns:
            Text string
        """
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"QUESTION PAPER - {set_name}".center(60))
        lines.append(f"{'='*60}\n")
        
        for section_name, questions in paper_data.items():
            lines.append(f"\n{section_name}")
            lines.append(f"{'-'*len(section_name)}\n")
            
            for q in questions:
                q_num = q.get('number', '')
                q_text = q.get('question', '')
                q_marks = q.get('marks', 0)
                
                lines.append(f"Q{q_num}. {q_text} [{q_marks} marks]")
                lines.append("")
        
        lines.append(f"\n{'='*60}")
        lines.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        return "\n".join(lines)
    
    def validate_paper(
        self,
        questions: List[Dict[str, Any]],
        target_marks: int,
        min_questions: int = 10,
        max_questions: int = 50
    ) -> Dict[str, Any]:
        """
        Validate if paper meets requirements
        
        Args:
            questions: List of questions
            target_marks: Target total marks
            min_questions: Minimum number of questions
            max_questions: Maximum number of questions
        
        Returns:
            Validation result
        """
        total_questions = len(questions)
        total_marks = sum(q.get('marks', 0) for q in questions)
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {
                'total_questions': total_questions,
                'total_marks': total_marks,
                'target_marks': target_marks,
                'marks_difference': total_marks - target_marks
            }
        }
        
        # Check question count
        if total_questions < min_questions:
            validation['is_valid'] = False
            validation['errors'].append(f"Too few questions: {total_questions} (min: {min_questions})")
        elif total_questions > max_questions:
            validation['warnings'].append(f"Many questions: {total_questions} (max: {max_questions})")
        
        # Check total marks
        marks_diff = abs(total_marks - target_marks)
        if marks_diff > target_marks * 0.1:  # More than 10% difference
            validation['is_valid'] = False
            validation['errors'].append(
                f"Total marks mismatch: {total_marks} (target: {target_marks})"
            )
        elif marks_diff > 0:
            validation['warnings'].append(
                f"Slight marks difference: {marks_diff} marks"
            )
        
        # Check for missing fields
        for i, q in enumerate(questions):
            if not q.get('question'):
                validation['errors'].append(f"Question {i+1} has no text")
            if not q.get('marks'):
                validation['warnings'].append(f"Question {i+1} has no marks assigned")
        
        return validation

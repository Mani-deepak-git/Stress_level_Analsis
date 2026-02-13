"""
PDF Report Generator - Creates professional interview summary reports
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
import os

class PDFReportGenerator:
    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00d2ff'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3a7bd5'),
            spaceAfter=12
        )
    
    def generate_report(self, session_data: dict) -> str:
        """Generate PDF report from session data"""
        filename = f"interview_report_{session_data['session_id']}_{int(datetime.now().timestamp())}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph("Interview Stress Analysis Report", self.title_style)
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Session Info
        story.append(Paragraph("Session Information", self.heading_style))
        session_info = [
            ['Session ID:', session_data['session_id']],
            ['Interviewer:', session_data['interviewer']],
            ['Interviewee:', session_data['interviewee']],
            ['Start Time:', datetime.fromisoformat(session_data['start_time']).strftime('%Y-%m-%d %H:%M:%S')],
            ['End Time:', datetime.fromisoformat(session_data['end_time']).strftime('%Y-%m-%d %H:%M:%S') if session_data['end_time'] else 'N/A'],
            ['Duration:', f"{session_data['stats']['total_duration'] / 60:.1f} minutes"]
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 4*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e2e8f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(session_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Key Metrics
        story.append(Paragraph("Key Performance Metrics", self.heading_style))
        stats = session_data['stats']
        metrics = [
            ['Metric', 'Value', 'Interpretation'],
            ['Average Stress Level', f"{stats['avg_stress']:.2f}", self._interpret_stress(stats['avg_stress'])],
            ['Average Confidence', f"{stats['avg_confidence']*100:.1f}%", self._interpret_confidence(stats['avg_confidence'])],
            ['Average Voice Confidence', f"{stats['avg_voice_confidence']:.1f}%", self._interpret_voice_confidence(stats['avg_voice_confidence'])],
            ['High Stress Duration', f"{stats['high_stress_duration']/60:.1f} min", f"{(stats['high_stress_duration']/stats['total_duration']*100):.1f}% of interview"],
            ['Low Stress Duration', f"{stats['low_stress_duration']/60:.1f} min", f"{(stats['low_stress_duration']/stats['total_duration']*100):.1f}% of interview"],
            ['Total Alerts', str(stats['total_alerts']), self._interpret_alerts(stats['total_alerts'])]
        ]
        
        metrics_table = Table(metrics, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3a7bd5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Alerts Summary
        if session_data['alerts']:
            story.append(Paragraph("Real-Time Alerts", self.heading_style))
            alert_data = [['Time', 'Type', 'Message']]
            for alert in session_data['alerts'][:10]:  # Show first 10 alerts
                alert_time = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
                alert_data.append([alert_time, alert['type'], alert['message']])
            
            alert_table = Table(alert_data, colWidths=[1*inch, 1.5*inch, 3.5*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ef4444')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(alert_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.heading_style))
        recommendations = self._generate_recommendations(session_data)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer = Paragraph(
            f"<i>Generated by AI Interview Stress Analyzer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            ParagraphStyle('Footer', parent=self.styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        )
        story.append(footer)
        
        # Build PDF
        doc.build(story)
        return filepath
    
    def _interpret_stress(self, avg_stress: float) -> str:
        if avg_stress < 1.3:
            return "Excellent - Very calm"
        elif avg_stress < 1.6:
            return "Good - Manageable stress"
        else:
            return "High - Consider support"
    
    def _interpret_confidence(self, confidence: float) -> str:
        if confidence > 0.7:
            return "High confidence"
        elif confidence > 0.4:
            return "Moderate confidence"
        else:
            return "Low confidence"
    
    def _interpret_voice_confidence(self, voice_conf: float) -> str:
        if voice_conf > 70:
            return "Strong vocal presence"
        elif voice_conf > 40:
            return "Moderate vocal confidence"
        else:
            return "Needs improvement"
    
    def _interpret_alerts(self, alert_count: int) -> str:
        if alert_count == 0:
            return "Smooth interview"
        elif alert_count < 5:
            return "Few concerns"
        else:
            return "Multiple interventions"
    
    def _generate_recommendations(self, session_data: dict) -> list:
        recommendations = []
        stats = session_data['stats']
        
        if stats['avg_stress'] > 1.6:
            recommendations.append("High stress detected. Consider shorter interview duration or more breaks.")
        
        if stats['avg_confidence'] < 0.4:
            recommendations.append("Low confidence observed. Try more encouraging questions and positive feedback.")
        
        if stats['avg_voice_confidence'] < 40:
            recommendations.append("Voice confidence is low. Candidate may benefit from communication coaching.")
        
        if stats['high_stress_duration'] > stats['total_duration'] * 0.5:
            recommendations.append("Candidate spent majority of time in high stress. Review interview difficulty.")
        
        if stats['total_alerts'] > 5:
            recommendations.append("Multiple alerts triggered. Consider adjusting interview approach.")
        
        if not recommendations:
            recommendations.append("Great interview! Candidate showed good stress management and confidence.")
        
        return recommendations

# Global PDF generator
pdf_generator = PDFReportGenerator()

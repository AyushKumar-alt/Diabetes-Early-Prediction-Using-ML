"""
Professional Medical PDF Report Generator
Creates lab-style reports for diabetes risk assessment
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# Page configuration
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 0.75 * inch
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN)

def draw_header_footer(canvas_obj, doc):
    """Draw header and footer on each page."""
    # Save state
    canvas_obj.saveState()
    
    # HEADER - Blue background with white text
    canvas_obj.setFillColor(colors.HexColor('#1E88E5'))  # Medical blue
    canvas_obj.rect(0, PAGE_HEIGHT - 1.2*inch, PAGE_WIDTH, 1.2*inch, fill=1, stroke=0)
    
    # App Logo Placeholder (box)
    logo_size = 0.8 * inch
    logo_x = MARGIN + 0.1*inch
    logo_y = PAGE_HEIGHT - 0.7*inch
    canvas_obj.setFillColor(colors.white)
    canvas_obj.setStrokeColor(colors.white)
    canvas_obj.rect(logo_x, logo_y, logo_size, logo_size, fill=0, stroke=1, strokeWidth=2)
    canvas_obj.setFont("Helvetica-Bold", 8)
    canvas_obj.drawString(logo_x + 0.15*inch, logo_y + 0.3*inch, "LOGO")
    
    # App Name
    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont("Helvetica-Bold", 20)
    canvas_obj.drawString(MARGIN + 1.0*inch, PAGE_HEIGHT - 0.5*inch, "Assist Diabetes AI")
    
    # Contact Information
    canvas_obj.setFont("Helvetica", 10)
    canvas_obj.drawString(MARGIN + 1.0*inch, PAGE_HEIGHT - 0.75*inch, "Email: support@diascan.ai")
    canvas_obj.drawString(MARGIN + 1.0*inch, PAGE_HEIGHT - 0.95*inch, "For Any Query Dial: +91 9931147182")
    
    # Horizontal line below header
    canvas_obj.setStrokeColor(colors.HexColor('#1E88E5'))
    canvas_obj.setLineWidth(2)
    canvas_obj.line(MARGIN, PAGE_HEIGHT - 1.2*inch, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 1.2*inch)
    
    # FOOTER
    canvas_obj.setFillColor(colors.grey)
    canvas_obj.setFont("Helvetica-Oblique", 9)
    footer_text = "This is an AI-assisted screening tool, not a diagnostic test. Please consult a healthcare professional for medical decisions."
    canvas_obj.drawCentredString(PAGE_WIDTH / 2, 0.4*inch, footer_text)
    
    # Page number
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawRightString(PAGE_WIDTH - MARGIN, 0.3*inch, f"Page {canvas_obj.getPageNumber()}")
    
    # Restore state
    canvas_obj.restoreState()

def create_patient_info_table(patient_info):
    """Create patient information table."""
    data = [
        ['Patient Information', ''],
        ['Name', patient_info.get('name', 'N/A')],
        ['Age / Gender', f"{patient_info.get('age', 'N/A')} / {patient_info.get('gender', 'N/A')}"],
        ['Phone', patient_info.get('phone', 'N/A')],
        ['Patient ID', patient_info.get('patient_id', 'N/A')],
        ['Report Date', datetime.now().strftime('%B %d, %Y at %I:%M %p')]
    ]
    
    table = Table(data, colWidths=[2.5*inch, 4.5*inch])
    
    style = TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E3F2FD')),  # Light blue
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1565C0')),  # Dark blue
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
    ])
    
    table.setStyle(style)
    return table

def create_results_table(prediction_info):
    """Create results/assessment table."""
    # Determine risk color
    risk_level = prediction_info.get('risk_level', 'N/A')
    if 'Low' in risk_level:
        risk_color = colors.HexColor('#4CAF50')  # Green
    elif 'Moderate' in risk_level:
        risk_color = colors.HexColor('#FFC107')  # Yellow
    elif 'High' in risk_level:
        risk_color = colors.HexColor('#FF9800')  # Orange
    else:
        risk_color = colors.HexColor('#F44336')  # Red
    
    probability = prediction_info.get('probability', 0.0)
    prob_percent = f"{probability * 100:.2f}%"
    
    data = [
        ['Assessment Results', ''],
        ['Prediction', prediction_info.get('prediction', 'N/A')],
        ['Risk Level', risk_level],
        ['Probability', prob_percent],
        ['Confidence', prediction_info.get('confidence', 'N/A')]
    ]
    
    table = Table(data, colWidths=[2.5*inch, 4.5*inch])
    
    style = TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E3F2FD')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        # Risk level row - highlight
        ('BACKGROUND', (1, 2), (1, 2), risk_color),
        ('TEXTCOLOR', (1, 2), (1, 2), colors.white),
        ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
    ])
    
    table.setStyle(style)
    return table

def create_feature_contrib_table(feature_contrib):
    """Create feature contributions table."""
    if not feature_contrib:
        return None
    
    # Header
    data = [['Feature', 'Impact on Risk']]
    
    # Add feature rows
    for feature, impact in feature_contrib[:10]:  # Top 10 features
        # Determine color based on impact
        if impact.startswith('+'):
            impact_color = colors.HexColor('#F44336')  # Red for increase
            impact_text = f"↑ Increases risk ({impact})"
        else:
            impact_color = colors.HexColor('#4CAF50')  # Green for decrease
            impact_text = f"↓ Decreases risk ({impact})"
        
        data.append([feature, impact_text])
    
    table = Table(data, colWidths=[3.5*inch, 3.5*inch])
    
    style = TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E3F2FD')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
    ])
    
    table.setStyle(style)
    return table

def generate_pdf_report(filename, patient_info, prediction_info, feature_contrib):
    """
    Generate a professional medical-style PDF report.
    
    Parameters:
    -----------
    filename : str
        Output PDF filename
    patient_info : dict
        {
            "name": "",
            "age": "",
            "gender": "",
            "phone": "",
            "patient_id": ""
        }
    prediction_info : dict
        {
            "probability": 0.43,
            "risk_level": "High Risk",
            "prediction": "At-Risk",
            "confidence": "Moderate",
            "recommendation": "Please consider taking an HbA1c test..."
        }
    feature_contrib : list of tuples
        [("HighBP", "+0.123"), ("BMI", "+0.087"), ...]
    
    Returns:
    --------
    str : Path to generated PDF file
    """
    # Create document
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=MARGIN + 1.2*inch,  # Extra space for header
        bottomMargin=MARGIN + 0.6*inch  # Extra space for footer
    )
    
    # Container for PDF elements
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1565C0'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    title = Paragraph("Diabetes Risk Assessment Report", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Information Table
    story.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    patient_table = create_patient_info_table(patient_info)
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Assessment Results Table
    story.append(Paragraph("<b>Assessment Results</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    results_table = create_results_table(prediction_info)
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendation Section
    story.append(Paragraph("<b>Clinical Recommendation</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    rec_style = ParagraphStyle(
        'Recommendation',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        leftIndent=0.2*inch,
        rightIndent=0.2*inch,
        backColor=colors.HexColor('#FFF9C4'),  # Light yellow background
        borderPadding=10,
        borderWidth=1,
        borderColor=colors.HexColor('#FFC107')
    )
    
    recommendation = prediction_info.get('recommendation', 'No recommendation available.')
    rec_para = Paragraph(f"<b>Recommendation:</b> {recommendation}", rec_style)
    story.append(rec_para)
    story.append(Spacer(1, 0.3*inch))
    
    # Feature Contributions Table
    if feature_contrib:
        story.append(Paragraph("<b>Top Contributing Factors</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        contrib_para = Paragraph(
            "The following factors were identified as having the most significant impact on this assessment:",
            styles['Normal']
        )
        story.append(contrib_para)
        story.append(Spacer(1, 0.1*inch))
        
        feature_table = create_feature_contrib_table(feature_contrib)
        if feature_table:
            story.append(feature_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Additional Notes
    notes_style = ParagraphStyle(
        'Notes',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_LEFT,
        leftIndent=0.2*inch
    )
    
    notes = Paragraph(
        "<b>Note:</b> This assessment is based on self-reported health indicators and statistical modeling. "
        "It is intended for screening purposes only and should not replace professional medical evaluation.",
        notes_style
    )
    story.append(notes)
    
    # Build PDF
    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    
    return filename

# Example usage (for testing)
if __name__ == "__main__":
    patient_info = {
        "name": "John Doe",
        "age": "45",
        "gender": "Male",
        "phone": "+1-555-0123",
        "patient_id": "PAT-2025-001"
    }
    
    prediction_info = {
        "probability": 0.4323,
        "risk_level": "High Risk",
        "prediction": "At-Risk",
        "confidence": "Moderate",
        "recommendation": "You may be developing insulin resistance. Please take a diabetes screening test (HbA1c or Fasting Plasma Glucose test). Consult with your healthcare provider for personalized advice."
    }
    
    feature_contrib = [
        ("HighBP", "+0.123"),
        ("BMI", "+0.087"),
        ("Age", "+0.065"),
        ("GenHlth", "+0.054"),
        ("PhysActivity", "-0.031"),
        ("Fruits", "-0.022"),
        ("Veggies", "-0.018"),
        ("HighChol", "+0.015"),
        ("Smoker", "+0.012"),
        ("Education", "-0.008")
    ]
    
    output_file = "test_diabetes_report.pdf"
    generate_pdf_report(output_file, patient_info, prediction_info, feature_contrib)
    print(f"✅ PDF generated: {output_file}")


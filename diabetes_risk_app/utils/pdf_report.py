# FILE: diabetes_risk_app/utils/pdf_report.py
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Circle, Path
from reportlab.graphics import renderPDF
from datetime import datetime
import os

# output folder (reports/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # points to diabetes_risk_app/
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def _draw_stethoscope(dsize=70):
    d = Drawing(dsize, dsize)
    r = dsize * 0.14
    d.add(Circle(dsize*0.22, dsize*0.78, r, fillColor=colors.HexColor("#0b6fa4"), strokeColor=None))
    d.add(Circle(dsize*0.35, dsize*0.78, r, fillColor=colors.HexColor("#0b6fa4"), strokeColor=None))
    p = Path()
    p.moveTo(dsize*0.35, dsize*0.78)
    p.curveTo(dsize*0.6, dsize*0.52, dsize*0.78, dsize*0.62, dsize*0.52, dsize*0.36)
    p.strokeColor = colors.HexColor("#0b6fa4")
    p.strokeWidth = 3
    d.add(p)
    d.add(Circle(dsize*0.52, dsize*0.28, dsize*0.07, fillColor=colors.HexColor("#174a6b"), strokeColor=None))
    return d

def generate_pdf_report(
    filename=None,
    patient_info=None,
    prediction_info=None,
    feature_contrib=None,
    lab_values=None,
    recommendation_data=None
):
    """
    Generate a hospital-style PDF and return the saved path.
    - patient_info: {name, patient_id, age_display, gender, phone, referred_by}
    - prediction_info: {probability (0..1), risk_level, prediction, confidence, recommendation}
    - feature_contrib: list of tuples [(feat, impact_str), ...]
    - lab_values: optional list of tuples [('HbA1c', '5.0', '4.0-6.0','%'), ...] similar to your sample table
    """
    patient_info = patient_info or {}
    prediction_info = prediction_info or {}
    feature_contrib = feature_contrib or []
    lab_values = lab_values or []

    pid = patient_info.get("patient_id", "PAT")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = f"{pid}_{ts}.pdf"
    fullpath = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(fullpath, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # header: icon + title (left-aligned)
    d = _draw_stethoscope(80)
    story.append(renderPDF.Drawing(d))
    story.append(Spacer(1, 6))

    title_style = ParagraphStyle('title', parent=styles['Title'], alignment=0, fontSize=20, textColor=colors.HexColor("#0b6fa4"))
    story.append(Paragraph("Assist Diabetes AI", title_style))
    story.append(Paragraph("Professional Diabetes Risk Assessment System", styles['Normal']))
    story.append(Spacer(1, 8))
    # Contact / Address as requested
    story.append(Paragraph("CSE Incubator Vidyashilp University Bangalore", styles['Normal']))
    story.append(Paragraph("email - support_assist_diabetes@gmail.com", styles['Normal']))
    story.append(Paragraph("contact - +91 991147182", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("CSE Incubator, Vidyashilp University Extension Centre, Bengaluru | support@assist_diabetes.ai | +91 9931147182", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>DIABETES RISK ASSESSMENT REPORT</b>", styles['Heading2']))
    story.append(Spacer(1, 12))

    # patient info table (lab-like row style)
    p = patient_info
    patient_table_data = [
        ["Name", p.get("name","-"), "Patient ID", p.get("patient_id","-")],
        ["Age/Gender", p.get("age_display","-") + " / " + p.get("gender","-"), "Phone", p.get("phone","-")],
        ["Referred By", p.get("referred_by","Self"), "Report Date", datetime.now().strftime("%d-%b-%Y %H:%M")]
    ]
    pt = Table(patient_table_data, colWidths=[110, 160, 110, 150], hAlign='LEFT')
    pt.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.6, colors.black),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONT", (0,0), (-1,-1), "Helvetica", 10)
    ]))
    story.append(pt)
    story.append(Spacer(1, 12))

    # Optional lab-like table (HbA1c)
    if lab_values:
        story.append(Paragraph("<b>BIOCHEMISTRY — HBA1C / GLYCOSYLATED</b>", styles['Heading3']))
        lab_table_data = [["TEST DESCRIPTION", "RESULT", "REF. RANGE", "UNIT"]]
        for row in lab_values:
            # row expected: (desc, result, ref_range, unit)
            lab_table_data.append([row[0], row[1], row[2], row[3]])
        lt = Table(lab_table_data, colWidths=[190, 90, 140, 80])
        lt.setStyle(TableStyle([("BOX", (0,0), (-1,-1), 0.5, colors.black), ("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("GRID",(0,0),(-1,-1),0.35,colors.grey)]))
        story.append(lt)
        story.append(Spacer(1, 10))

    # results
    pr = prediction_info
    result_table_data = [
        ["Prediction", pr.get("prediction","-")],
        ["Risk Level", pr.get("risk_level","-")],
        ["Probability", f"{pr.get('probability',0)*100:.2f}%"],
        ["Confidence", pr.get("confidence","-")]
    ]
    rt = Table(result_table_data, colWidths=[140, 350], hAlign='LEFT')
    rt.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.6, colors.black),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DDEAF7")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    story.append(rt)
    story.append(Spacer(1, 12))

    # recommendation (use structured recommendation_data if provided)
    rec = recommendation_data or prediction_info.get('recommendation_data') or {
        'risk_label': pr.get('risk_level', 'N/A'),
        'advice': prediction_info.get('recommendation', 'Follow-up with a healthcare provider for confirmatory testing.'),
        'precautions': []
    }

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Recommendation</b>", styles['Heading3']))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"<b>Risk Level:</b> {rec.get('risk_label','-')}", styles['Normal']))
    story.append(Spacer(1, 6))

    story.append(Paragraph(rec.get('advice',''), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Precautions:</b>", styles['Heading4']))
    for p in rec.get('precautions', []):
        story.append(Paragraph(f"• {p}", styles.get('Bullet', styles['Normal'])))

    # feature contributions
    story.append(Paragraph("<b>Top Contributing Factors</b>", styles['Heading3']))
    fc_table_data = [["Feature", "Impact"]] + [[f, s] for f, s in feature_contrib]
    fc = Table(fc_table_data, colWidths=[230, 260], hAlign='LEFT')
    fc.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.black),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    story.append(fc)
    story.append(Spacer(1, 18))

    # footer disclaimer
    disclaimer = "This is an AI-assisted screening tool and not a diagnostic test. Confirm with laboratory testing and clinical consultation."
    story.append(Paragraph(disclaimer, ParagraphStyle('disc', fontSize=8, textColor=colors.grey)))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Generated by Assist Diabetes AI", ParagraphStyle('small', fontSize=9)))
    doc.build(story)
    return fullpath

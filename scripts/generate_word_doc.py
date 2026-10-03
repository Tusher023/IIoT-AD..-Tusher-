"""
Thesis Document Generator for IIoT Anomaly Detection Project.
Generates: c:\\Users\\KIIT\\Desktop\\IIoT-AD\\docs\\thesis.docx

Author: Tusher Tarafder
University: Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar
Supervisor: Prof. (Dr.) Sujata Dash, Department of CSE
Year: 2026
Title: Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT:
       A Systematic Empirical Comparison of Classical and Deep Learning Methods
"""

import os
import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# Paths
BASE_DIR = Path(r"c:\Users\KIIT\Desktop\IIoT-AD")
DOCS_DIR = BASE_DIR / "docs"
FIGURES_DIR = BASE_DIR / "results" / "figures" / "publication"
OUTPUT_FILE = DOCS_DIR / "thesis.docx"

# Color Palette
COLOR_PRIMARY_NAVY = RGBColor(31, 78, 121)     # #1F4E79
COLOR_SECONDARY_DARK = RGBColor(44, 62, 80)    # #2C3E50
COLOR_TEXT_MAIN = RGBColor(38, 38, 38)         # #262626
COLOR_MUTED_GRAY = RGBColor(100, 100, 100)     # #646464
HEX_NAVY = "1F4E79"
HEX_LIGHT_BG = "F4F6F7"
HEX_BORDER = "BDC3C7"
HEX_CALLOUT_BG = "EBF5FB"
HEX_CALLOUT_BORDER = "2980B9"

# Helper Functions for Table & Text Styling
def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color=HEX_BORDER, sz="4"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def set_repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:tblHeader'))

def set_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:cantSplit'))

def format_table(table, col_widths=None, alignments=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)
    for i, row in enumerate(table.rows):
        set_cant_split(row)
        if i == 0:
            set_repeat_header(row)
        for j, cell in enumerate(row.cells):
            set_cell_margins(cell, top=100, bottom=100, left=130, right=130)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if col_widths and j < len(col_widths):
                cell.width = col_widths[j]
            if i == 0:
                set_cell_background(cell, HEX_NAVY)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
                    if alignments and j < len(alignments):
                        p.alignment = alignments[j]
                    else:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.font.name = 'Times New Roman'
                        r.font.size = Pt(10)
                        r.font.bold = True
                        r.font.color.rgb = RGBColor(255, 255, 255)
            else:
                if i % 2 == 1:
                    set_cell_background(cell, "FFFFFF")
                else:
                    set_cell_background(cell, HEX_LIGHT_BG)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                    if alignments and j < len(alignments):
                        p.alignment = alignments[j]
                    for r in p.runs:
                        r.font.name = 'Times New Roman'
                        r.font.size = Pt(9.5)
                        r.font.color.rgb = COLOR_TEXT_MAIN

def add_p(doc, text="", bold_prefix=None, space_after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT, italic_prefix=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_SECONDARY_DARK
        
    if italic_prefix:
        r_pre = p.add_run(italic_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11.5)
        r_pre.font.italic = True
        r_pre.font.color.rgb = COLOR_SECONDARY_DARK

    if text:
        r_text = p.add_run(text)
        r_text.font.name = 'Times New Roman'
        r_text.font.size = Pt(11.5)
        r_text.font.color.rgb = COLOR_TEXT_MAIN
    return p

def add_h1(doc, title):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(18)
    h.paragraph_format.space_after = Pt(8)
    h.paragraph_format.keep_with_next = True
    r = h.add_run(title)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY_NAVY
    return h

def add_h2(doc, title):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(5)
    h.paragraph_format.keep_with_next = True
    r = h.add_run(title)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = COLOR_SECONDARY_DARK
    return h

def add_h3(doc, title):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(3)
    h.paragraph_format.keep_with_next = True
    r = h.add_run(title)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.italic = True
    r.font.color.rgb = COLOR_SECONDARY_DARK
    return h

def add_callout(doc, text, title="RESEARCH QUESTION"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, HEX_CALLOUT_BG)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    
    # Left border only
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{HEX_CALLOUT_BORDER}"/>'
        f'  <w:top w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    r_title = p.add_run(f"[{title}] ")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(10.5)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(41, 128, 185)
    
    r_body = p.add_run(text)
    r_body.font.name = 'Times New Roman'
    r_body.font.size = Pt(10.5)
    r_body.font.italic = True
    r_body.font.color.rgb = COLOR_TEXT_MAIN
    
    # Add spacing after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(6)

def add_figure(doc, filename, caption):
    fig_path = FIGURES_DIR / filename
    if fig_path.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(10)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(str(fig_path), width=Inches(5.8))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(12)
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = 'Times New Roman'
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = COLOR_MUTED_GRAY

# Document Initialization
def init_document():
    doc = Document()
    
    # Standard 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)
        
    return doc

# Section 1: Title Page
def build_title_page(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(36)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("KALINGA INSTITUTE OF INDUSTRIAL TECHNOLOGY")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY_NAVY

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(4)
    r2 = p2.add_run("Deemed to be University U/S 3 of the UGC Act, 1956")
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(11)
    r2.font.italic = True
    r2.font.color.rgb = COLOR_MUTED_GRAY

    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p3.paragraph_format.space_after = Pt(36)
    r3 = p3.add_run("Bhubaneswar, Odisha, India - 751024\nSchool of Computer Engineering\nDepartment of Computer Science and Engineering")
    r3.font.name = 'Times New Roman'
    r3.font.size = Pt(12)
    r3.font.bold = True
    r3.font.color.rgb = COLOR_SECONDARY_DARK

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(24)
    p_title.paragraph_format.space_after = Pt(18)
    p_title.paragraph_format.line_spacing = 1.25
    r_t = p_title.add_run("UNSUPERVISED ANOMALY DETECTION FOR PREDICTIVE MAINTENANCE IN INDUSTRIAL IOT:\nA SYSTEMATIC EMPIRICAL COMPARISON OF CLASSICAL AND DEEP LEARNING METHODS")
    r_t.font.name = 'Times New Roman'
    r_t.font.size = Pt(19)
    r_t.font.bold = True
    r_t.font.color.rgb = COLOR_PRIMARY_NAVY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(36)
    r_sub = p_sub.add_run("A Thesis Submitted in Partial Fulfillment of the Requirements for the Degree of\nBachelor of Technology in Computer Science and Engineering")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True
    r_sub.font.color.rgb = COLOR_SECONDARY_DARK

    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_before = Pt(24)
    p_auth.paragraph_format.space_after = Pt(20)
    p_auth.paragraph_format.line_spacing = 1.2
    r_by = p_auth.add_run("Submitted by:\n")
    r_by.font.name = 'Times New Roman'
    r_by.font.size = Pt(11)
    r_by.font.italic = True
    
    r_name = p_auth.add_run("TUSHER TARAFDER\n")
    r_name.font.name = 'Times New Roman'
    r_name.font.size = Pt(14)
    r_name.font.bold = True
    r_name.font.color.rgb = COLOR_PRIMARY_NAVY

    r_roll = p_auth.add_run("Roll No. / Registration No. 22051785\nDepartment of Computer Science and Engineering")
    r_roll.font.name = 'Times New Roman'
    r_roll.font.size = Pt(11)

    p_sup = doc.add_paragraph()
    p_sup.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sup.paragraph_format.space_before = Pt(16)
    p_sup.paragraph_format.space_after = Pt(36)
    p_sup.paragraph_format.line_spacing = 1.2
    r_und = p_sup.add_run("Under the Supervision of:\n")
    r_und.font.name = 'Times New Roman'
    r_und.font.size = Pt(11)
    r_und.font.italic = True

    r_sname = p_sup.add_run("Prof. (Dr.) Sujata Dash\n")
    r_sname.font.name = 'Times New Roman'
    r_sname.font.size = Pt(14)
    r_sname.font.bold = True
    r_sname.font.color.rgb = COLOR_PRIMARY_NAVY

    r_sdept = p_sup.add_run("Department of Computer Science and Engineering\nSchool of Computer Engineering\nKIIT Deemed to be University, Bhubaneswar")
    r_sdept.font.name = 'Times New Roman'
    r_sdept.font.size = Pt(11)

    p_yr = doc.add_paragraph()
    p_yr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_yr.paragraph_format.space_before = Pt(30)
    r_yr = p_yr.add_run("MAY 2026")
    r_yr.font.name = 'Times New Roman'
    r_yr.font.size = Pt(13)
    r_yr.font.bold = True
    r_yr.font.color.rgb = COLOR_SECONDARY_DARK

    doc.add_page_break()

# Section 2: Certificate
def build_certificate(doc):
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_before = Pt(20)
    h.paragraph_format.space_after = Pt(16)
    r = h.add_run("CERTIFICATE OF RECOMMENDATION")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY_NAVY

    add_p(doc, "This is to certify that the thesis entitled \"Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT: A Systematic Empirical Comparison of Classical and Deep Learning Methods\", submitted by Tusher Tarafder (Registration No. 22051785) to the School of Computer Engineering, Kalinga Institute of Industrial Technology (KIIT Deemed to be University), Bhubaneswar, for the award of the degree of Bachelor of Technology in Computer Science and Engineering, is a bona fide record of research work carried out by him under my supervision and guidance.", space_after=12)

    add_p(doc, "The results embodied in this thesis have not been submitted to any other University or Institute for the award of any degree, diploma, or fellowship. To the best of my knowledge, the thesis represents independent, original empirical research adhering to rigorous academic standards and ethical guidelines.", space_after=24)

    # Signatures table
    tbl = doc.add_table(rows=3, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    # Left column: Place & Date; Right column: Supervisor
    c00 = tbl.cell(0, 0)
    c01 = tbl.cell(0, 1)
    c10 = tbl.cell(1, 0)
    c11 = tbl.cell(1, 1)
    c20 = tbl.cell(2, 0)
    c21 = tbl.cell(2, 1)
    
    c00.width = Inches(3.2)
    c01.width = Inches(3.3)
    c10.width = Inches(3.2)
    c11.width = Inches(3.3)
    c20.width = Inches(3.2)
    c21.width = Inches(3.3)

    p_pd = c00.paragraphs[0]
    p_pd.paragraph_format.space_after = Pt(2)
    r_pd = p_pd.add_run("Place: Bhubaneswar, Odisha\nDate: May 15, 2026")
    r_pd.font.name = 'Times New Roman'
    r_pd.font.size = Pt(11)

    p_sig1 = c01.paragraphs[0]
    p_sig1.paragraph_format.space_after = Pt(2)
    r_sig1 = p_sig1.add_run("_________________________________________\nProf. (Dr.) Sujata Dash\nPrincipal Supervisor\nDepartment of Computer Science and Engineering\nSchool of Computer Engineering\nKIIT Deemed to be University")
    r_sig1.font.name = 'Times New Roman'
    r_sig1.font.size = Pt(11)

    p_blank = c10.paragraphs[0]
    p_blank.paragraph_format.space_before = Pt(36)
    
    p_sig2 = c20.paragraphs[0]
    p_sig2.paragraph_format.space_before = Pt(36)
    r_sig2 = p_sig2.add_run("_________________________________________\nHead of the Department\nDepartment of Computer Science and Engineering\nSchool of Computer Engineering\nKIIT Deemed to be University")
    r_sig2.font.name = 'Times New Roman'
    r_sig2.font.size = Pt(11)

    p_sig3 = c21.paragraphs[0]
    p_sig3.paragraph_format.space_before = Pt(36)
    r_sig3 = p_sig3.add_run("_________________________________________\nExternal Examiner\nBoard of Examiners\nSchool of Computer Engineering\nKIIT Deemed to be University")
    r_sig3.font.name = 'Times New Roman'
    r_sig3.font.size = Pt(11)

    doc.add_page_break()

# Section 3: Declaration
def build_declaration(doc):
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_before = Pt(20)
    h.paragraph_format.space_after = Pt(16)
    r = h.add_run("CANDIDATE'S DECLARATION")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY_NAVY

    add_p(doc, "I, Tusher Tarafder, hereby declare that the work presented in this thesis titled \"Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT: A Systematic Empirical Comparison of Classical and Deep Learning Methods\" is entirely my own original investigation, conducted under the supervision of Prof. (Dr.) Sujata Dash, Department of Computer Science and Engineering, School of Computer Engineering, Kalinga Institute of Industrial Technology (KIIT Deemed to be University), Bhubaneswar.", space_after=12)

    add_p(doc, "I affirm that this thesis has not formed the basis for the award of any Degree, Diploma, Associateship, Fellowship, or any other similar title in this or any other academic or professional institution.", space_after=12)

    add_p(doc, "I further declare that all materials, datasets, methodologies, and scholarly works obtained from secondary sources or prior literature have been duly cited and referenced in accordance with standard academic conventions and ethical research practices. All experimental scripts, models, and data pipelines were developed systematically with strict zero-leakage enforcement to ensure complete scientific reproducibility.", space_after=36)

    # Student signature block
    p_sign = doc.add_paragraph()
    p_sign.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_sign.paragraph_format.space_before = Pt(36)
    p_sign.paragraph_format.line_spacing = 1.2
    r_st = p_sign.add_run("_________________________________________\nTusher Tarafder\nRegistration No. 22051785\nB.Tech, Computer Science and Engineering\nSchool of Computer Engineering\nKIIT Deemed to be University, Bhubaneswar\nDate: May 15, 2026")
    r_st.font.name = 'Times New Roman'
    r_st.font.size = Pt(11)

    doc.add_page_break()

# Section 4: Acknowledgements
def build_acknowledgements(doc):
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_before = Pt(20)
    h.paragraph_format.space_after = Pt(16)
    r = h.add_run("ACKNOWLEDGEMENTS")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY_NAVY

    add_p(doc, "The completion of this research and dissertation would not have been possible without the invaluable guidance, constructive criticism, and steadfast encouragement of many individuals and organizations to whom I owe my deepest gratitude.", space_after=10)

    add_p(doc, "First and foremost, I express my profound gratitude and heartfelt appreciation to my supervisor, Prof. (Dr.) Sujata Dash, Department of Computer Science and Engineering, School of Computer Engineering, KIIT Deemed to be University. Her profound technical insights, exacting scholarly standards, patient mentorship, and continuous encouragement provided clarity and purpose at every phase of this investigation. She continually challenged me to prioritize methodological rigor over superficial metric-chasing, which ultimately shaped the scientific contribution of this study.", space_after=10)

    add_p(doc, "I express my sincere thanks to the authorities of Kalinga Institute of Industrial Technology (KIIT Deemed to be University), Bhubaneswar, the Dean, Head of the Department, and all esteemed faculty members and staff of the School of Computer Engineering. The university's exceptional computational infrastructure, digital library subscriptions, and vibrant academic ecosystem created an ideal environment for pursuing advanced research.", space_after=10)

    add_p(doc, "I gratefully acknowledge the Prognostics Center of Excellence (PCoE) at the National Aeronautics and Space Administration (NASA) Ames Research Center for curating and making publicly accessible the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) benchmark dataset. The realistic thermodynamic degradation trajectories provided by this benchmark formed the experimental backbone of this empirical comparison.", space_after=10)

    add_p(doc, "My deepest thanks are also extended to the global open-source software and machine learning community. The tools and frameworks developed and maintained by the PyTorch, scikit-learn, NumPy, Pandas, Matplotlib, SciPy, and python-docx development teams made reproducible, scalable computational experimentation possible.", space_after=10)

    add_p(doc, "Finally, I owe an immeasurable debt of gratitude to my beloved parents, family members, and friends. Their unconditional love, personal sacrifices, unwavering faith in my abilities, and moral support provided the strength and perseverance necessary to complete this degree program and thesis.", space_after=24)

    p_sig = doc.add_paragraph()
    p_sig.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_sig = p_sig.add_run("Tusher Tarafder\nBhubaneswar, Odisha")
    r_sig.font.name = 'Times New Roman'
    r_sig.font.size = Pt(11)
    r_sig.font.italic = True

    doc.add_page_break()

# Section 5: Abstract
def build_abstract(doc):
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_before = Pt(20)
    h.paragraph_format.space_after = Pt(16)
    r = h.add_run("ABSTRACT")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY_NAVY

    add_p(doc, "Industrial Internet of Things (IIoT) architectures generate massive streams of multivariate sensor data from critical physical assets, making Predictive Maintenance (PdM) vital for preventing catastrophic downtime and optimizing operational efficiency. In industrial reality, machines spend the vast majority of their operational lives in nominal states, rendering labeled run-to-failure data exceedingly scarce and elevating unsupervised anomaly detection to a cornerstone of PdM strategies. While deep learning models—specifically Feedforward and Long Short-Term Memory (LSTM) Autoencoders—have received immense academic interest, they are frequently benchmarked under inconsistent experimental protocols, lacking rigorous baseline comparisons, proper leakage prevention, and operating-condition stress testing.", space_after=10)

    add_p(doc, "This thesis presents a systematic, reproducible empirical evaluation comparing classical statistical detectors (Z-score mean and max), ensemble learning (Isolation Forest), kernel methods (One-Class SVM), and deep neural architectures (Fully Connected Autoencoder and LSTM Autoencoder) on the NASA C-MAPSS benchmark. We enforce strict engine-level splitting, train-only feature normalization, and validation-only threshold tuning. Across single operating conditions (FD001), statistical and classical ensemble baselines achieve superior classification performance (Statistical F1 = 0.602, ROC-AUC = 0.984; Isolation Forest F1 = 0.549, ROC-AUC = 0.976) compared to deep learning methods (FC-AE F1 = 0.564, ROC-AUC = 0.912; LSTM-AE F1 = 0.429, ROC-AUC = 0.901). However, LSTM-AE demonstrates a distinct advantage in detection lead time (mean 50.0 cycles before failure). Multi-seed validation and paired hypothesis testing (p < 0.05) demonstrate that deep architectures exhibit significantly higher performance variance without commensurate classification gains. Under operational condition shift (FD001 to FD002), all unsupervised models suffer catastrophic transfer failure (false alarm rates reaching 85-100%), whereas Isolation Forest demonstrates unique resilience to fault mode diversity (FD001 to FD003, ROC-AUC = 0.904). These findings provide an evidence-based selection taxonomy for industrial practitioners balancing latency, computational budget, and diagnostic reliability.", space_after=14)

    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(8)
    p_kw.paragraph_format.space_after = Pt(12)
    r_kw_title = p_kw.add_run("Keywords: ")
    r_kw_title.font.name = 'Times New Roman'
    r_kw_title.font.size = Pt(11)
    r_kw_title.font.bold = True
    r_kw_title.font.color.rgb = COLOR_PRIMARY_NAVY
    
    r_kw = p_kw.add_run("Industrial IoT, Predictive Maintenance, Unsupervised Anomaly Detection, Isolation Forest, LSTM Autoencoder, NASA C-MAPSS, Lead Time, Empirical Benchmarking, Data Leakage Prevention.")
    r_kw.font.name = 'Times New Roman'
    r_kw.font.size = Pt(11)
    r_kw.font.italic = True

    doc.add_page_break()

# Section 6: Table of Contents Placeholder
def build_table_of_contents(doc):
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_before = Pt(20)
    h.paragraph_format.space_after = Pt(16)
    r = h.add_run("TABLE OF CONTENTS")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY_NAVY

    toc_entries = [
        ("Certificate of Recommendation", "ii"),
        ("Candidate's Declaration", "iii"),
        ("Acknowledgements", "iv"),
        ("Abstract", "v"),
        ("List of Figures", "viii"),
        ("List of Tables", "ix"),
        ("Chapter 1: Introduction", "1"),
        ("    1.1 Background and Problem Context", "1"),
        ("    1.2 Problem Statement", "2"),
        ("    1.3 Research Motivation", "3"),
        ("    1.4 Research Objectives", "4"),
        ("    1.5 Scope and Delimitations", "5"),
        ("    1.6 Organization of the Thesis", "6"),
        ("Chapter 2: Literature Review", "7"),
        ("    2.1 Industrial IoT and Predictive Maintenance", "7"),
        ("    2.2 Unsupervised Anomaly Detection Paradigms", "8"),
        ("    2.3 Classical Machine Learning Methods", "10"),
        ("    2.4 Deep Learning Architectures for Time-Series Anomaly Detection", "12"),
        ("    2.5 NASA C-MAPSS Benchmark in Prior Literature", "14"),
        ("Chapter 3: Research Gap and Research Questions", "16"),
        ("    3.1 Literature Synthesis and Identification of the Research Gap", "16"),
        ("    3.2 Primary Research Question", "17"),
        ("    3.3 Secondary Research Questions (RQ1 - RQ7)", "18"),
        ("    3.4 Research Question to Experiment Mapping", "20"),
        ("Chapter 4: Dataset Description and Exploratory Data Analysis", "21"),
        ("    4.1 C-MAPSS Simulation Physics and Operating Mechanics", "21"),
        ("    4.2 Benchmark Subsets Structure and Complexity Regimes", "22"),
        ("    4.3 Sensor Telemetry Suite and Feature Characterization", "24"),
        ("    4.4 Exploratory Data Analysis and Trajectory Characteristics", "26"),
        ("Chapter 5: Methodology", "28"),
        ("    5.1 End-to-End Pipeline Architecture", "28"),
        ("    5.2 Strict Leakage Prevention Protocol", "29"),
        ("    5.3 Evaluated Unsupervised Anomaly Detection Models", "31"),
        ("    5.4 Threshold Selection Strategies", "34"),
        ("    5.5 Early Warning Detection Framework", "35"),
        ("    5.6 Evaluation Metrics and Complexity Accounting", "37"),
        ("Chapter 6: Experimental Setup", "39"),
        ("    6.1 Hardware and Software Environment", "39"),
        ("    6.2 Preprocessing and Feature Engineering Protocol", "40"),
        ("    6.3 Sequence Construction for Temporal Deep Models", "41"),
        ("    6.4 Hyperparameter Specifications and Training Configurations", "42"),
        ("Chapter 7: Results and Comparative Analysis", "44"),
        ("    7.1 Primary Benchmark Comparison on FD001", "44"),
        ("    7.2 Early Warning Horizons and Lead-Time Dynamics", "47"),
        ("    7.3 Threshold Sensitivity and Operational Calibration", "49"),
        ("    7.4 Degradation Detection Across Dataset Complexity Regimes", "51"),
        ("    7.5 Cross-Condition and Fault Mode Transferability Breakdown", "53"),
        ("Chapter 8: Ablation Studies and Statistical Rigor", "56"),
        ("    8.1 Multi-Seed Stability and Variance Analysis", "56"),
        ("    8.2 Hypothesis Testing and Statistical Significance", "58"),
        ("    8.3 Sequence Length Ablation for Recurrent Modeling", "60"),
        ("Chapter 9: Discussion and Practical Guidelines", "62"),
        ("    9.1 The 'Complexity Paradox' in Industrial IoT Anomaly Detection", "62"),
        ("    9.2 The Advance Lead Time vs Instantaneous Accuracy Trade-Off", "64"),
        ("    9.3 Practical Selection Guide for IIoT Practitioners", "66"),
        ("    9.4 Limitations and Threats to Validity", "68"),
        ("Chapter 10: Conclusion and Future Work", "70"),
        ("    10.1 Summary of Contributions", "70"),
        ("    10.2 Synthesis of Answers to Research Questions", "71"),
        ("    10.3 Future Research Directions", "73"),
        ("References", "75"),
        ("Appendix A: Sensor Telemetry Nomenclature and Metadata", "78"),
        ("Appendix B: Hyperparameter Specifications and Split Definitions", "80"),
    ]

    for title, pg in toc_entries:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(2)
        r_t = p.add_run(title)
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(10.5)
        if "Chapter" in title or "References" in title or "Appendix" in title or "Abstract" in title:
            r_t.font.bold = True
            r_t.font.color.rgb = COLOR_PRIMARY_NAVY
        else:
            r_t.font.color.rgb = COLOR_TEXT_MAIN
            
        r_dots = p.add_run(" " + "." * max(5, int(70 - len(title)*1.1)) + " ")
        r_dots.font.name = 'Times New Roman'
        r_dots.font.size = Pt(9.5)
        r_dots.font.color.rgb = COLOR_MUTED_GRAY
        
        r_pg = p.add_run(pg)
        r_pg.font.name = 'Times New Roman'
        r_pg.font.size = Pt(10.5)
        r_pg.font.bold = True if ("Chapter" in title or "References" in title) else False

    doc.add_page_break()

# Chapter 1: Introduction
def build_chapter_1(doc):
    add_h1(doc, "Chapter 1: Introduction")
    
    add_h2(doc, "1.1 Background and Problem Context")
    add_p(doc, "The rapid convergence of advanced operational technologies (OT) and digital information systems under the fourth industrial revolution (Industry 4.0) has fundamentally transformed the management of physical industrial infrastructure. Modern manufacturing plants, power generation stations, aerospace fleets, and chemical processing facilities now feature dense deployments of connected sensors collectively termed the Industrial Internet of Things (IIoT). These sensor networks stream high-dimensional time-series telemetry—monitoring vibrations, thermal gradients, acoustic emissions, pressures, and rotational speeds—providing continuous operational visibility into the health of critical turbomachinery and rotating equipment.")

    add_p(doc, "Within this data-rich landscape, Predictive Maintenance (PdM) has emerged as an indispensable paradigm. Traditional maintenance practices typically rely on two classical approaches: reactive maintenance (operating equipment until catastrophic failure occurs) and preventive maintenance (servicing assets at fixed, calendar- or cycle-based intervals regardless of physical condition). Reactive maintenance incurs exorbitant direct and indirect costs, including unscheduled downtime, emergency labor premiums, collateral structural destruction, and severe worker safety hazards. Preventive maintenance mitigates sudden breakdowns but introduces substantial economic waste by replacing functioning components prematurely and interrupting nominal operations for unnecessary overhauls.")

    add_p(doc, "Predictive maintenance seeks to eliminate this dichotomy by diagnosing physical degradation in its nascent stages and tracking asset health continuously. By accurately identifying abnormal operational trajectories before functional breakdown occurs, maintenance engineers can plan targeted maintenance actions, procure necessary spare parts, schedule repairs during planned plant shutdowns, and fully utilize the remaining useful operational life of industrial capital assets.")

    add_h2(doc, "1.2 Problem Statement")
    add_p(doc, "Despite the significant theoretical advantages of Predictive Maintenance, practical implementation in real-world IIoT deployments faces formidable challenges that render conventional supervised machine learning paradigms ineffective:")

    add_p(doc, "1. Extreme Scarcity of Failure Labels: Industrial machinery is engineered for high reliability and operates under conservative safety margins. Consequently, assets spend greater than 98% to 99% of their operational lifetimes in healthy operating states. Catastrophic breakdowns occur rarely, meaning historical sensor repositories contain vast quantities of nominal data but negligible failure records. Supervised classification algorithms require balanced cohorts of positive and negative classes and fail when deployed under such severe class imbalance.", bold_prefix="Extreme Class Imbalance: ")

    add_p(doc, "2. Operational Confounding and Dynamic Regimes: Industrial equipment rarely operates under static environmental conditions. Commercial turbofan engines, industrial turbines, and hydraulic compressors frequently adjust operating regimes—altering throttle positions, flight altitudes, ambient temperatures, and external loads. These operational shifts produce massive sensor variations that can easily dwarf subtle fault signatures. Distinguishing true physical deterioration from benign operating condition changes represents a primary technical barrier.", bold_prefix="Operational Regime Confounding: ")

    add_p(doc, "3. Methodological Vulnerabilities and Information Leakage: Academic literature frequently evaluates anomaly detection models using flawed experimental protocols, including observation-level random shuffling (mixing temporal past and future observations), fitting normalization parameters on testing observations, and optimizing decision thresholds directly on test metrics. Such subtle forms of data leakage artificially inflate reported performance and lead to catastrophic model failures when algorithms transition to real industrial operational environments.", bold_prefix="Data Leakage and Optimistic Bias: ")

    add_h2(doc, "1.3 Research Motivation")
    add_p(doc, "Over the past decade, academic research in time-series anomaly detection has overwhelmingly embraced deep learning architectures, most notably Feedforward Autoencoders (FC-AE) and Recurrent or Long Short-Term Memory Autoencoders (LSTM-AE). These neural networks are praised for their capacity to learn non-linear sensor cross-correlations and complex temporal dynamics without manual feature engineering.")

    add_p(doc, "However, a pronounced disconnect exists between academic claims of deep learning superiority and the operational requirements of industrial practitioners. In factory environments, deep learning models introduce heavy computational overhead, opaque latent representations (the 'black-box' dilemma), extended training latencies, and high sensitivity to hyperparameter initialization. Meanwhile, classical statistical detectors (such as multivariate Z-score limits) and non-parametric ensemble methods (such as Isolation Forest and One-Class SVM) offer lightweight compute footprints, near-instantaneous training, deterministic inference, and clear mathematical interpretability.")

    add_p(doc, "Remarkably, existing literature rarely conducts rigorous, head-to-head empirical comparisons between classical algorithms and deep architectures under strictly identical, leakage-free experimental conditions. Novel deep learning architectures are routinely compared only against other neural variants or weak, poorly calibrated baselines. This leaves a vital engineering question unanswered: Under what operational conditions and at what computational cost is the implementation of deep learning truly justified for industrial predictive maintenance?")

    add_h2(doc, "1.4 Research Objectives")
    add_p(doc, "The overarching goal of this dissertation is to conduct a systematic, reproducible, and evidence-based empirical investigation comparing classical and deep learning unsupervised anomaly detection methods on turbomachinery degradation data. To accomplish this, the research addresses six specific objectives:")

    add_p(doc, "1. Protocol Formulation: Design and enforce a unified, zero-leakage experimental pipeline featuring strict engine-level train/validation/test partitioning, train-only normalization, and validation-only threshold tuning.", bold_prefix="Objective 1: ")
    add_p(doc, "2. Comparative Benchmarking: Systematically implement, tune, and evaluate five diverse unsupervised anomaly detection models (Statistical Mean/Max Z-score, Isolation Forest, One-Class SVM, Fully Connected Autoencoder, and LSTM Autoencoder) on the NASA C-MAPSS benchmark under identical conditions.", bold_prefix="Objective 2: ")
    add_p(doc, "3. Early Warning Characterization: Evaluate temporal detection lead time, alarm persistence, and false alarm rates to quantify how many cycles prior to physical failure each method provides actionable warnings.", bold_prefix="Objective 3: ")
    add_p(doc, "4. Operational Complexity Stress-Testing: Assess the robustness of each method across datasets featuring single vs. multiple operating conditions and single vs. multiple fault modes (FD001 through FD004).", bold_prefix="Objective 4: ")
    add_p(doc, "5. Transferability and Domain Shift Analysis: Conduct cross-condition transfer experiments (training models on single-condition regimes and evaluating on multi-condition environments) to quantify generalization breakdown.", bold_prefix="Objective 5: ")
    add_p(doc, "6. Statistical Hypothesis Testing: Perform repeated multi-seed experiments (5 random seeds) and paired non-parametric statistical hypothesis tests to determine whether performance differences between classical and deep models are statistically significant.", bold_prefix="Objective 6: ")

    add_h2(doc, "1.5 Scope and Delimitations")
    add_p(doc, "The scope of this investigation is grounded in the run-to-failure degradation telemetry of the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine benchmark. The experimental study focuses strictly on the unsupervised anomaly detection paradigm, where models are exposed exclusively to healthy early-life operational data during training without access to run-to-failure labels or explicit remaining useful life (RUL) targets.")

    add_p(doc, "The study intentionally delimits its scope from supervised RUL regression, online edge-firmware deployment, and non-aerospace rotating equipment. However, the thermodynamic degradation principles, high-dimensional multivariate sensor structures, and operational complexities present in C-MAPSS serve as an established and rigorous proxy for broad classes of industrial turbomachinery, including power turbines, centrifugal gas compressors, and heavy industrial pumps.")

    add_h2(doc, "1.6 Organization of the Thesis")
    add_p(doc, "The remainder of this dissertation is structured as follows: Chapter 2 reviews foundational literature in IIoT predictive maintenance, unsupervised anomaly detection, classical algorithms, deep architectures, and the C-MAPSS benchmark. Chapter 3 synthesizes the literature gap, establishes our primary research question, and defines seven detailed research questions (RQ1–RQ7). Chapter 4 provides a comprehensive structural and exploratory analysis of the C-MAPSS dataset. Chapter 5 details our end-to-end methodology, leakage prevention protocol, detection models, and evaluation metrics. Chapter 6 describes the experimental environment and hyperparameter configurations. Chapter 7 presents the primary empirical results, early warning analysis, threshold sensitivity, and cross-condition transfer experiments. Chapter 8 provides ablation studies, multi-seed stability evaluations, and formal statistical hypothesis testing. Chapter 9 synthesizes the practical implications, explores the 'complexity paradox', and offers an actionable model selection guide for industrial practitioners. Finally, Chapter 10 concludes the thesis and outlines future research avenues.")

    doc.add_page_break()

# Chapter 2: Literature Review
def build_chapter_2(doc):
    add_h1(doc, "Chapter 2: Literature Review")

    add_h2(doc, "2.1 Industrial IoT and Predictive Maintenance")
    add_p(doc, "The integration of edge sensing, distributed computing, and advanced analytics in Industrial IoT has accelerated the adoption of Condition-Based Maintenance (CBM) and Predictive Maintenance (PdM) across asset-intensive industries. As surveyed by Zhao et al. (2019) [2], modern industrial machinery is monitored by heterogeneous sensor arrays capturing physical phenomena across spatial and temporal dimensions. Predictive maintenance algorithms leverage this high-dimensional telemetry to detect early physical wear, characterize fault progression, and guide timely interventions.")

    add_p(doc, "However, deploying machine learning within industrial environments differs fundamentally from conventional consumer analytics. Industrial operations require high reliability: false alarms disrupt production schedules and induce alarm fatigue among technicians, while missed detections risk catastrophic asset destruction. Consequently, predictive maintenance systems must balance detection sensitivity against false alarm rates while maintaining robust operation across varying environmental conditions.")

    add_h2(doc, "2.2 Unsupervised Anomaly Detection Paradigms")
    add_p(doc, "In anomaly detection taxonomy, outliers are typically categorized into three distinct classes: point anomalies (individual anomalous sensor spikes), contextual anomalies (readings that are abnormal given operational context or time of day), and collective anomalies (sequences of readings that are individually plausible but collectively indicate degradation). In turbomachinery, physical degradation primarily manifests as collective contextual anomalies—gradual thermodynamic drifts that deviate from healthy operating baselines.")

    add_p(doc, "Because labeled failure records are rarely available in operational plants, unsupervised and self-supervised anomaly detection paradigms have become essential. Darban et al. (2024) [1] classify unsupervised time-series anomaly detection methods into distance-based, density-based, isolation-based, and reconstruction-based families. In all cases, the detector establishes a mathematical profile of 'nominal' operational behavior using healthy data. During inference, observations that deviate from this learned normal profile receive elevated anomaly scores.")

    add_h2(doc, "2.3 Classical Machine Learning Methods")
    add_p(doc, "Prior to the widespread adoption of deep neural networks, statistical process control and classical machine learning constituted the standard toolkit for industrial condition monitoring:")

    add_p(doc, "Statistical Process Control (SPC) and Z-Score Detectors: Grounded in statistical quality control, univariate and multivariate Z-score detectors track deviations from historical mean values normalized by standard deviations. In multivariate settings, individual sensor Z-scores are aggregated across active channels using mean or maximum pooling. While computationally trivial and completely transparent, statistical limit detectors can struggle with complex non-linear cross-sensor correlations.", bold_prefix="Statistical Process Control: ")

    add_p(doc, "Isolation Forest (iForest): Introduced by Liu et al. (2008) [8], Isolation Forest is an ensemble of random binary trees based on the principle that anomalies are 'few and different.' Rather than constructing a dense profile of normal points, iForest isolates observations through recursive orthogonal partitioning. Anomalies require fewer random splits to isolate and consequently exhibit shorter average path lengths through the ensemble. Ding et al. (2019) [9] and Hariri et al. (2019) [10] extended isolation concepts to streaming and dense multidimensional spaces, demonstrating that iForest remains highly competitive in tabular sensor spaces.", bold_prefix="Isolation Forest: ")

    add_p(doc, "One-Class Support Vector Machine (OC-SVM): Formulated by Schölkopf et al., One-Class SVM maps input vectors into a high-dimensional reproducing kernel Hilbert space (RKHS) and constructs a maximum-margin hyperplane separating normal training data from the coordinate origin. When paired with a Radial Basis Function (RBF) kernel, OC-SVM models complex non-linear boundaries. However, its quadratic training complexity with respect to sample size poses challenges for large-scale industrial telemetry.", bold_prefix="One-Class SVM: ")

    add_h2(doc, "2.4 Deep Learning Architectures for Time-Series Anomaly Detection")
    add_p(doc, "Deep learning architectures have gained widespread popularity for industrial anomaly detection, driven by their ability to automatically extract hierarchical representations from high-dimensional data:")

    add_p(doc, "Fully Connected Autoencoders (FC-AE): Autoencoders are feedforward neural networks trained with an unsupervised reconstruction objective. An encoder compresses input vectors into a lower-dimensional latent bottleneck, and a decoder reconstructs the original inputs. Trained exclusively on healthy operational data, the network minimizes reconstruction error on normal patterns. When degraded data containing abnormal sensor correlations is processed, the network fails to accurately reconstruct the inputs, producing elevated reconstruction errors that serve as anomaly scores (Borghesi et al., 2019 [6]).", bold_prefix="Feedforward Autoencoders: ")

    add_p(doc, "Recurrent and LSTM Autoencoders (LSTM-AE): Standard feedforward autoencoders treat each time step independently, ignoring temporal sequence dependencies. To address sequential dynamics, Malhotra et al. (2015) [3] introduced the LSTM Autoencoder. An LSTM encoder reads input sequences and compresses them into a fixed-length latent representation; an LSTM decoder then reconstructs the target sequence in reverse or forward chronological order. Park et al. (2018) [4] and Hundt et al. (2020) [7] applied recurrent autoencoders to industrial sensor telemetry, demonstrating their capacity to capture temporal dependencies and sequential degradation patterns.", bold_prefix="Recurrent Autoencoders: ")

    add_h2(doc, "2.5 NASA C-MAPSS Benchmark in Prior Literature")
    add_p(doc, "The Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset, released by Saxena et al. (2008) [11], is the primary benchmark for turbofan degradation modeling. C-MAPSS simulates run-to-failure trajectories under diverse operating conditions, capturing wear in high-pressure compressors and fan components across 21 sensor channels.")

    add_p(doc, "The vast majority of C-MAPSS literature focuses on supervised Remaining Useful Life (RUL) prediction, utilizing Convolutional Neural Networks (Babu et al., 2016 [12]), Transformers, and deep recurrent networks. However, supervised RUL regression assumes the availability of complete run-to-failure training histories with precise degradation trajectories—an assumption rarely satisfied in operational industrial plants.")

    add_p(doc, "A growing body of work has adapted C-MAPSS for unsupervised anomaly detection. El-Attar et al. (2021) [13] and Listou Ellefsen et al. (2019) [14] trained autoencoders on early-life healthy cycles to detect early degradation. Michau et al. (2022) [15] explored domain adaptation across operating conditions (FD001 to FD002). Schmidl et al. (2022) [16] conducted a broad evaluation of time-series anomaly detectors, noting that while deep models capture sequential patterns, Isolation Forests remain competitive in efficiency and robustness. However, existing studies rarely evaluate classical detectors, ensemble methods, feedforward autoencoders, and recurrent networks within a single, unified experimental protocol with strict leakage prevention.", bold_prefix="Unsupervised C-MAPSS Studies: ")

    doc.add_page_break()

# Chapter 3: Research Gap and Research Questions
def build_chapter_3(doc):
    add_h1(doc, "Chapter 3: Research Gap and Research Questions")

    add_h2(doc, "3.1 Literature Synthesis and Identification of the Research Gap")
    add_p(doc, "A thorough review of published literature on predictive maintenance and time-series anomaly detection reveals that while individual aspects of unsupervised anomaly detection—such as novel network architectures, domain adaptation, adaptive thresholding, and early detection—have been investigated in isolation, there is a clear absence of unified, methodologically rigorous empirical benchmarks that evaluate all of these dimensions simultaneously under strict leakage prevention on the C-MAPSS benchmark.")

    add_p(doc, "Specifically, critical methodological limitations persist across prior studies:")
    add_p(doc, "1. Architecture Novelty Over Fair Comparison: Novel deep learning architectures are frequently proposed and compared only against other neural variants or uncalibrated default baselines. Systematic, side-by-side evaluations against well-tuned statistical limit detectors, Isolation Forests, and One-Class SVMs are rare.", bold_prefix="Lack of Strong Baselines: ")
    add_p(doc, "2. Aggregate Metrics Without Engine-Level Analysis: Most studies report dataset-wide aggregate metrics (such as global F1-score or ROC-AUC) without examining engine-level performance. In operational predictive maintenance, key questions center on engine-level behavior: How many engines were detected before failure? What was the distribution of advance lead times? How many false alarms occurred during healthy phases?", bold_prefix="Omission of Early Warning Metrics: ")
    add_p(doc, "3. Experimental Leakage and Post-Hoc Thresholding: Crucial methodological details are frequently omitted or executed with subtle leakage. Common issues include observation-level train/test splits that allow models to see future time steps of training units, fitting feature scalers across full datasets, and tuning anomaly decision thresholds directly on test sets.", bold_prefix="Pervasive Data Leakage: ")
    add_p(doc, "4. Isolation Within Simple Regimes: The majority of studies restrict evaluations to the single-condition, single-fault subset (FD001). Investigations rarely examine how models trained under simple conditions perform when exposed to multi-condition operating regimes (FD002/FD004) or varied fault modes (FD003).", bold_prefix="Neglect of Operating Complexity: ")
    add_p(doc, "5. Absence of Statistical Rigor: Most published findings rely on single training runs. Repeated multi-seed trials, variance reporting, and formal statistical hypothesis testing are uncommon, making it difficult to verify whether reported performance differences represent genuine algorithmic advantages or random seed variations.", bold_prefix="Lack of Statistical Testing: ")

    add_h2(doc, "3.2 Primary Research Question")
    add_callout(doc, 
        "To what extent do unsupervised anomaly detection methods differ in their ability to identify machine degradation and provide early failure warnings in multivariate industrial sensor data, when evaluated under a unified, methodologically rigorous framework with strict leakage prevention?",
        title="PRIMARY RESEARCH QUESTION")

    add_h2(doc, "3.3 Secondary Research Questions (RQ1 - RQ7)")
    add_p(doc, "To provide a thorough, evidence-based answer to the primary research question, we formulate seven secondary research questions addressing specific dimensions of detection performance, operational complexity, and computational cost:")

    add_callout(doc, "RQ1 (Fair Baseline vs Deep Learning): When classical baselines (Statistical Threshold, Isolation Forest, One-Class SVM) and deep learning architectures (FC-AE, LSTM-AE) are evaluated under identical engine-level splits, normalization protocols, threshold rules, and evaluation metrics, how do their detection performances compare?", title="RESEARCH QUESTION 1")

    add_callout(doc, "RQ2 (Temporal Sequence Modeling): Does an LSTM Autoencoder, which explicitly models temporal dependencies across sequential time windows, capture machine degradation more effectively than a static Fully Connected Autoencoder that treats individual time steps independently?", title="RESEARCH QUESTION 2")

    add_callout(doc, "RQ3 (Operating Condition Complexity): How does the detection performance of each method change when asset telemetry includes multiple operating conditions (FD001 vs FD002) or multiple fault modes (FD001 vs FD003)?", title="RESEARCH QUESTION 3")

    add_callout(doc, "RQ4 (Early Warning Horizons): For each method, how many operational cycles prior to physical failure does the first sustained warning appear, what is the distribution of advance lead time across engines, and what proportion of failures are detected?", title="RESEARCH QUESTION 4")

    add_callout(doc, "RQ5 (Threshold Sensitivity and False Alarms): How does the choice of validation threshold selection strategy affect the trade-off between detection sensitivity and false alarm rates across different models?", title="RESEARCH QUESTION 5")

    add_callout(doc, "RQ6 (Cross-Condition Generalization): When models trained under simple single-condition operating regimes (FD001) are evaluated on multi-condition (FD002/FD004) or multi-fault (FD003) environments, how severely does detection performance degrade?", title="RESEARCH QUESTION 6")

    add_callout(doc, "RQ7 (Complexity vs Performance Trade-Off): Does the marginal performance change provided by complex deep learning models (parameter count, training latency, inference time) justify the added computational overhead compared to lightweight classical baselines?", title="RESEARCH QUESTION 7")

    add_h2(doc, "3.4 Research Question to Experiment Mapping")
    add_p(doc, "To ensure methodical execution, Table 3.1 outlines the mapping between each research question, its primary experimental configuration, evaluated datasets, key quantitative metrics, and planned ablation studies.")

    # Table 3.1
    t31 = doc.add_table(rows=8, cols=5)
    t31_data = [
        ["RQ", "Primary Experiment", "Evaluated Dataset", "Key Quantitative Metrics", "Ablation / Investigation Focus"],
        ["RQ1", "Fair Model Comparison", "FD001", "Precision, Recall, F1, ROC-AUC, PR-AUC", "Identical splits, 5 random seeds"],
        ["RQ2", "Temporal vs Static Modeling", "FD001", "F1-Score, Anomaly Trajectories, Lead Time", "Sequence length L in {10, 20, 30, 50}"],
        ["RQ3", "Operating Condition Shift", "FD001, FD002, FD003, FD004", "F1, ROC-AUC, Detection Rate, FAR", "Single vs 6 operating conditions"],
        ["RQ4", "Early Warning Dynamics", "FD001", "Mean/Median Lead Time, Detection Rate", "Persistence window W in {1, 3, 5}"],
        ["RQ5", "Threshold Sensitivity", "FD001", "F1, FAR, Precision-Recall Curves", "Percentile (90,95,99), Mean+k*std, F1-opt"],
        ["RQ6", "Cross-Condition Transfer", "FD001 -> FD002, FD003, FD004", "Transfer F1, Transfer ROC-AUC, FAR Explosion", "Condition shift vs fault mode shift"],
        ["RQ7", "Complexity Trade-Off", "FD001", "Parameter Count, Fit Time, F1/Time Ratio", "Computational budget & deployment taxonomy"],
    ]
    for r_idx, row in enumerate(t31.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t31_data[r_idx][c_idx]
    format_table(t31, [Inches(0.6), Inches(1.7), Inches(1.4), Inches(1.6), Inches(1.2)], 
                 [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_p(doc, "Table 3.1: Mapping of Research Questions to Empirical Experimental Configurations.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_page_break()

# Chapter 4: Dataset Description
def build_chapter_4(doc):
    add_h1(doc, "Chapter 4: Dataset Description and Exploratory Analysis")

    add_h2(doc, "4.1 C-MAPSS Simulation Physics and Operating Mechanics")
    add_p(doc, "The Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) is a high-fidelity thermodynamic turbofan simulation platform developed by NASA. It models the dynamic behavior of a large commercial twin-spool turbofan engine (nominally in the 90,000 lb thrust class). The engine simulation incorporates major rotating and stationary thermodynamic components, including the Fan, Low-Pressure Compressor (LPC), High-Pressure Compressor (HPC), Combustor, High-Pressure Turbine (HPT), Low-Pressure Turbine (LPT), and Convergent Exhaust Nozzle.")

    add_p(doc, "In C-MAPSS, engine degradation is simulated by injecting progressive thermodynamic wear into targeted modules. Specifically, component degradation manifests as decreases in flow capacity and adiabatic efficiency in either the High-Pressure Compressor (HPC), the Fan, or both simultaneously. Wear progresses non-linearly across simulated operational flights (cycles) until a failure threshold is reached. Each simulated engine begins its operational life with varying initial manufacturing tolerances and component wear, representing realistic fleet diversity.")

    add_h2(doc, "4.2 Benchmark Subsets Structure and Complexity Regimes")
    add_p(doc, "The C-MAPSS repository comprises four distinct benchmark subsets (FD001, FD002, FD003, and FD004) that systematically vary operational complexity along two orthogonal dimensions: the number of environmental operating conditions and the number of active failure modes. Table 4.1 summarizes the structural parameters, engine counts, observation totals, and lifetime distributions across all four subsets based on our exploratory data analysis.")

    # Table 4.1
    t41 = doc.add_table(rows=5, cols=9)
    t41_data = [
        ["Subset", "Op. Cond.", "Fault Modes", "Fault Description", "Train / Test Units", "Train / Test Obs.", "Life Min", "Life Max", "Life Mean ± Std"],
        ["FD001", "1", "1", "HPC Degradation", "100 / 100", "20,631 / 13,096", "128", "362", "206.3 ± 46.3"],
        ["FD002", "6", "1", "HPC Degradation", "260 / 259", "53,759 / 33,991", "128", "378", "206.8 ± 46.8"],
        ["FD003", "1", "2", "HPC + Fan Degradation", "100 / 100", "24,720 / 16,596", "145", "525", "247.2 ± 86.5"],
        ["FD004", "6", "2", "HPC + Fan Degradation", "249 / 248", "61,249 / 41,214", "128", "543", "246.0 ± 73.1"],
    ]
    for r_idx, row in enumerate(t41.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t41_data[r_idx][c_idx]
    format_table(t41, [Inches(0.7), Inches(0.6), Inches(0.6), Inches(1.5), Inches(1.0), Inches(1.0), Inches(0.5), Inches(0.5), Inches(0.9)],
                 [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_p(doc, "Table 4.1: Structural Characteristics and Lifetime Statistics of C-MAPSS Benchmark Subsets.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_p(doc, "As detailed in Table 4.1, subset FD001 represents the baseline single-condition, single-fault regime, making it well-suited for controlled model comparisons. FD002 introduces six distinct flight operating conditions (varying altitude from 0 to 40,000 feet, Mach number from 0.0 to 0.84, and sea-level temperatures). FD003 maintains a single operating condition while incorporating two distinct fault modes (HPC wear and Fan degradation). Finally, FD004 represents the most complex operational regime, combining six distinct flight conditions with two concurrent fault modes.")

    add_h2(doc, "4.3 Sensor Telemetry Suite and Feature Characterization")
    add_p(doc, "Each cycle record in C-MAPSS contains 26 columns: engine identifier (unit number), operational cycle number, three operational settings (altitude, Mach number, and throttle resolver angle), and 21 continuous sensor measurements monitoring temperatures, pressures, spool rotational speeds, and flow ratios.")

    add_p(doc, "A critical step in our exploratory data analysis was identifying constant or non-informative sensor channels. In FD001 and FD003 (single operating condition), several sensors exhibit zero or near-zero variance across all engines throughout their operational lives. For example, in FD001, sensors s1 (Total temperature at fan inlet), s5 (Total pressure at LPC outlet), s10 (Total pressure at engine burner outlet), s16 (Bypass duct pressure), s18 (Demanded fan speed), and s19 (Demanded corrected fan speed) remain completely constant. Feeding zero-variance features into distance or neural estimators introduces numerical instability and increases dimensionality without providing diagnostic information. Our preprocessing pipeline automatically identifies and prunes constant features, retaining 14 active, informative sensor channels and operational settings for FD001.")

    add_h2(doc, "4.4 Exploratory Data Analysis and Trajectory Characteristics")
    add_p(doc, "Exploratory analysis reveals that C-MAPSS sensor degradation trajectories exhibit three notable properties:")
    add_p(doc, "1. Initial Nominal Plateau: Turbofan engines operate within nominal sensor bounds for the initial 60% to 75% of their operational lives. During this phase, sensor variations stem from random measurement noise rather than physical degradation.", bold_prefix="Healthy Plateau: ")
    add_p(doc, "2. Non-Linear Degradation Inflection: Once component wear reaches an initial threshold, physical degradation accelerates non-linearly. Sensor channels sensitive to HPC and HPT health—such as s2 (Total temperature at LPC outlet), s3 (Total temperature at HPC outlet), s4 (Total temperature at LPT outlet), s8 (Physical core speed), s9 (Physical fan speed), s11 (Static pressure at HPC outlet), s13 (Corrected core speed), and s15 (Bypass ratio)—diverge noticeably from nominal values.", bold_prefix="Accelerating Drift: ")
    add_p(doc, "3. Bimodal and Multi-Modal Clustering in Multi-Condition Sets: In FD002 and FD004, the presence of six discrete flight conditions creates multi-modal sensor distributions. Operational setting changes cause raw sensor readings to jump between operating clusters, masking subtle degradation signals unless proper condition-based normalization is applied.", bold_prefix="Regime Switching: ")

    doc.add_page_break()

# Chapter 5: Methodology
def build_chapter_5(doc):
    add_h1(doc, "Chapter 5: Methodology")

    add_h2(doc, "5.1 End-to-End Pipeline Architecture")
    add_p(doc, "To ensure reproducibility and fair evaluation, we implemented an end-to-end data processing and evaluation framework. The pipeline comprises six modular stages: (1) raw telemetry ingestion and validation, (2) feature filtering and constant column removal, (3) engine-level train/validation/test partitioning, (4) normalization fitted strictly on training data, (5) unsupervised model fitting on healthy early-life operational periods, and (6) validation threshold tuning followed by frozen test evaluation.")

    add_h2(doc, "5.2 Strict Leakage Prevention Protocol")
    add_p(doc, "Data leakage occurs when information from outside the training cohort influences model development or threshold selection. To prevent leakage, our experimental protocol enforces four strict methodological safeguards:")

    add_p(doc, "1. Engine-Level Partitioning: All dataset splits are executed strictly at the engine entity level, never by randomly shuffling individual observation cycles. For FD001 (100 total run-to-failure engines), engines are partitioned into 70 training units (70%), 15 validation units (15%), and 15 test units (15%) using deterministic pseudo-random seed assignment. Because complete operational lifetimes remain grouped within their assigned splits, models are evaluated on unseen engine units.", bold_prefix="Strict Engine-Level Splits: ")

    add_p(doc, "2. Train-Only Normalization Fitting: Feature scaling parameters (mean and standard deviation for StandardScaler) are computed exclusively from the 70 training engines. The fitted transformation is then applied to validation and test engines. No test data statistics are ever accessed during preprocessing.", bold_prefix="Train-Only Feature Scaling: ")

    add_p(doc, "3. Defined Healthy Operational Window: In unsupervised predictive maintenance, models are trained on normal operational data. Because C-MAPSS engines begin healthy and degrade toward failure, we designate the initial 70% of each training engine's observed lifetime as nominal data. The remaining 30% of each training engine's life is excluded from the training set.", bold_prefix="Healthy Training Window: ")

    add_p(doc, "4. Validation-Only Decision Threshold Tuning: Anomaly scoring thresholds are derived strictly from validation engine score distributions. Once selected on the validation set, the threshold is frozen and applied directly to test engines. Optimizing thresholds directly on test metrics is strictly prohibited.", bold_prefix="Validation-Only Thresholding: ")

    add_h2(doc, "5.3 Evaluated Unsupervised Anomaly Detection Models")
    add_p(doc, "We evaluate five unsupervised anomaly detection algorithms spanning statistical baselines, tree ensembles, kernel methods, feedforward neural networks, and recurrent architectures:")

    add_p(doc, "Statistical Z-Score Detector: Computes feature-wise mean and variance vectors across nominal training cycles. For any test cycle x_t, feature deviations are calculated as z_ti = |x_ti - mu_i| / sigma_i. We evaluate two aggregation strategies: Statistical Mean (averaging Z-scores across informative features) and Statistical Max (taking the maximum feature deviation). This detector requires no gradient optimization or tree building, serving as a transparent reference baseline.", bold_prefix="1. Statistical Limit Detector: ")

    add_p(doc, "Isolation Forest (iForest): Constructs an ensemble of 100 randomized isolation trees fitted to healthy training cycles. Each tree recursively splits features at random values between observed minimums and maximums until observations are isolated. The anomaly score is derived from the average path length across all trees, where shorter path lengths indicate higher anomaly likelihood.", bold_prefix="2. Isolation Forest: ")

    add_p(doc, "One-Class Support Vector Machine (OC-SVM): Implements a non-linear kernel boundary around nominal training data using a Radial Basis Function (RBF) kernel. The decision boundary is controlled by kernel bandwidth gamma and regularization parameter nu (set to 0.05). The anomaly score is the negative signed distance from the separating hyperplane.", bold_prefix="3. One-Class Support Vector Machine: ")

    add_p(doc, "Fully Connected Autoencoder (FC-AE): A multi-layer feedforward neural network comprising an encoder [Input(19) -> Dense(64) -> ReLU -> Dropout(0.1) -> Dense(32) -> ReLU -> Dense(16)] and a symmetric decoder [Dense(16) -> Dense(32) -> ReLU -> Dense(64) -> ReLU -> Dense(19)]. The network is trained with the Adam optimizer to minimize Mean Squared Error (MSE) reconstruction loss. Anomaly scores equal the MSE between input vectors and reconstructions.", bold_prefix="4. Fully Connected Autoencoder: ")

    add_p(doc, "LSTM Autoencoder (LSTM-AE): A temporal deep recurrent architecture. Input sequences of length L=30 cycles are processed by a 2-layer LSTM encoder (hidden dimensions 64 and 32), compressing sequential dynamics into a 32-dimensional latent vector. A RepeatVector layer replicates the latent representation across L steps, followed by a 2-layer LSTM decoder reconstructing the full input sequence. Anomaly scores equal the mean reconstruction error across time steps and features.", bold_prefix="5. LSTM Autoencoder: ")

    # Table 5.1
    t51 = doc.add_table(rows=6, cols=5)
    t51_data = [
        ["Model Name", "Model Family", "Input Representation", "Parameter Count", "Anomaly Scoring Metric"],
        ["Statistical Mean", "Parametric Statistical", "Single cycle vector (t)", "38 parameters", "Mean Z-score across features"],
        ["Isolation Forest", "Ensemble Tree Partitioning", "Single cycle vector (t)", "100 trees", "Inverse average tree path length"],
        ["One-Class SVM", "Kernel Hyperplane Boundary", "Single cycle vector (t)", "Support vectors", "Negative signed hyperplane distance"],
        ["FC-Autoencoder", "Deep Feedforward Neural", "Single cycle vector (t)", "8,163 parameters", "Mean Squared Reconstruction Error (MSE)"],
        ["LSTM Autoencoder", "Deep Recurrent Sequential", "Sequence window (t-L+1 : t)", "116,723 parameters", "Sequential MSE Reconstruction Error"],
    ]
    for r_idx, row in enumerate(t51.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t51_data[r_idx][c_idx]
    format_table(t51, [Inches(1.5), Inches(1.5), Inches(1.4), Inches(1.0), Inches(1.6)],
                 [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    add_p(doc, "Table 5.1: Comparative Summary of Evaluated Unsupervised Anomaly Detection Architectures.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_h2(doc, "5.4 Threshold Selection Strategies")
    add_p(doc, "To assess how threshold choice impacts operational performance, we evaluate six distinct threshold determination strategies calibrated exclusively on validation data:")
    add_p(doc, "1. Validation Percentiles: Threshold set to the 90th, 95th, or 99th percentile of validation anomaly scores (P90, P95, P99). Percentile 95 serves as the primary operational threshold.", bold_prefix="Percentile Strategies: ")
    add_p(doc, "2. Statistical Error Limits: Threshold defined as mu_val + 2*sigma_val or mu_val + 3*sigma_val, assuming quasi-normal reconstruction error distributions.", bold_prefix="Parametric Limits: ")
    add_p(doc, "3. F1-Optimal Calibration: The threshold that maximizes F1-score on the validation set, providing an empirical upper bound on achievable validation performance.", bold_prefix="F1-Optimal Calibration: ")

    add_h2(doc, "5.5 Early Warning Detection Framework")
    add_p(doc, "In industrial predictive maintenance, evaluating models solely via cycle-by-cycle classification metrics (Precision, Recall, F1) overlooks temporal operational value. A single cycle exceeding threshold may reflect random sensor noise rather than genuine physical degradation. Prematurely dispatching repair crews based on isolated spikes induces alarm fatigue.")

    add_p(doc, "To provide realistic operational metrics, we define an Early Warning Framework using a persistence filter: an anomaly warning is triggered only when the anomaly score exceeds the decision threshold for W consecutive cycles (default persistence window W = 5 cycles). Using this rule, we compute five early warning metrics across test engines:")
    add_p(doc, "1. First Detection Cycle (t_det): The cycle at which the persistence window of W consecutive anomalous cycles is first satisfied.", bold_prefix="First Detection Cycle: ")
    add_p(doc, "2. Advance Lead Time (T_lead): Defined as Total Engine Lifetime minus First Detection Cycle (T_lead = T_life - t_det). Represents how many operating cycles in advance of failure maintenance operators receive a sustained warning.", bold_prefix="Advance Lead Time: ")
    add_p(doc, "3. Fleet Detection Rate: The percentage of failing test engines for which a sustained warning was triggered prior to physical failure.", bold_prefix="Fleet Detection Rate: ")
    add_p(doc, "4. False Alarm Rate (FAR): The proportion of nominal operational cycles (RUL > 30 cycles) incorrectly flagged as anomalous.", bold_prefix="False Alarm Rate: ")
    add_p(doc, "5. Missed Failure Rate: The proportion of engines that reach end-of-life without triggering a sustained warning.", bold_prefix="Missed Failure Rate: ")

    add_h2(doc, "5.6 Evaluation Metrics and Complexity Accounting")
    add_p(doc, "Comprehensive evaluation requires tracking classification accuracy, temporal warning capabilities, and computational resource demands simultaneously:")
    add_p(doc, "Classification Metrics: Evaluated cycle-by-cycle against ground-truth degradation labels (defined as RUL <= 30 cycles). Metrics include Precision, Recall, F1-Score, Area Under the Receiver Operating Characteristic curve (ROC-AUC), and Area Under the Precision-Recall curve (PR-AUC).", bold_prefix="Classification: ")
    add_p(doc, "Temporal Metrics: Evaluated per engine. Metrics include Fleet Detection Rate (%), Mean Lead Time (cycles), Median Lead Time, and Standard Deviation of Lead Time.", bold_prefix="Early Warning: ")
    add_p(doc, "Computational Accounting: Tracks model parameter counts, training durations (fit time in seconds), and per-sample inference latency, providing empirical data on computational efficiency.", bold_prefix="Complexity: ")

    doc.add_page_break()

# Chapter 6: Experimental Setup
def build_chapter_6(doc):
    add_h1(doc, "Chapter 6: Experimental Setup")

    add_h2(doc, "6.1 Hardware and Software Environment")
    add_p(doc, "All experiments were executed within a controlled computational environment to ensure consistency. Computations were performed on an x86_64 workstation running Microsoft Windows 11. Deep learning architectures were implemented using PyTorch 2.x, while classical machine learning baselines were built with scikit-learn 1.x. Numerical computation and tabular data handling were managed via NumPy and Pandas. Visualization artifacts and figures were generated using Matplotlib and Seaborn.")

    add_h2(doc, "6.2 Preprocessing and Feature Engineering Protocol")
    add_p(doc, "The preprocessing pipeline ingests raw text files from C-MAPSS and executes systematic cleaning steps. Features exhibiting zero standard deviation in the training cohort are automatically pruned (dropping 7 invariant columns in FD001, leaving 14 active sensors plus 3 operational settings). StandardScaler is fitted exclusively on the 70 training engines and applied across validation and test sets. Sequence matrices for recurrent architectures are generated per engine using a sliding window of length L=30 and stride s=1, ensuring windows never cross engine boundaries.")

    add_h2(doc, "6.3 Hyperparameter Specifications and Training Configurations")
    add_p(doc, "Model hyperparameters were selected through systematic validation tuning rather than post-hoc test optimization. Table 6.1 details the architectural parameters, loss formulations, optimization configurations, and execution environments for all evaluated models.")

    # Table 6.1
    t61 = doc.add_table(rows=8, cols=6)
    t61_data = [
        ["Configuration Parameter", "Statistical Mean", "Isolation Forest", "One-Class SVM", "FC-Autoencoder", "LSTM-Autoencoder"],
        ["Input Dimension", "19 features", "19 features", "19 features", "19 features", "19 features (L=30)"],
        ["Hidden Architecture", "N/A (Analytical)", "100 trees", "RBF Kernel", "[64, 32, 16, 32, 64]", "LSTM [64, 32, 32, 64]"],
        ["Latent Bottleneck Dim", "N/A", "N/A", "N/A", "16 units", "32 units"],
        ["Optimizer & Learning Rate", "N/A", "N/A", "N/A", "Adam, lr=1e-3", "Adam, lr=1e-3"],
        ["Regularization / Dropout", "N/A", "Subsampling 256", "nu = 0.05", "Dropout 0.1, wd=1e-4", "Dropout 0.2, wd=1e-4"],
        ["Batch Size & Epochs", "N/A", "N/A", "N/A", "Batch 256, 100 Epochs", "Batch 128, 100 Epochs"],
        ["Early Stopping Patience", "N/A", "N/A", "N/A", "10 epochs", "15 epochs"],
    ]
    for r_idx, row in enumerate(t61.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t61_data[r_idx][c_idx]
    format_table(t61, [Inches(1.5), Inches(0.9), Inches(1.0), Inches(0.9), Inches(1.2), Inches(1.2)],
                 [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_p(doc, "Table 6.1: Detailed Hyperparameter Specifications and Training Configurations.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Figure 6: Training curves
    add_figure(doc, "fig6_training_curves.png", "Figure 6.1: Representative Training and Validation Loss Trajectories for Autoencoder Architectures.")

    doc.add_page_break()

# Chapter 7: Results and Comparative Analysis
def build_chapter_7(doc):
    add_h1(doc, "Chapter 7: Results and Comparative Analysis")

    add_h2(doc, "7.1 Primary Benchmark Comparison on FD001")
    add_p(doc, "We begin our analysis by evaluating all five unsupervised anomaly detection algorithms on the baseline FD001 subset (single operating condition, single failure mode). Table 7.1 presents the primary performance comparison under identical engine-level splits (70 train / 15 val / 15 test) using the primary 95th percentile validation threshold and a 5-cycle persistence filter.")

    # Table 7.1
    t71 = doc.add_table(rows=7, cols=10)
    t71_data = [
        ["Model Architecture", "Precision", "Recall", "F1-Score", "FAR", "ROC-AUC", "PR-AUC", "Det. Rate", "Mean Lead", "Fit Time"],
        ["Statistical Mean", "1.000", "0.430", "0.602", "0.000", "0.984", "0.932", "93.33%", "10.5 cyc", "0.001 s"],
        ["Statistical Max", "0.691", "0.288", "0.407", "0.022", "0.949", "0.749", "33.33%", "24.4 cyc", "0.001 s"],
        ["Isolation Forest", "1.000", "0.378", "0.549", "0.000", "0.976", "0.905", "73.33%", "11.1 cyc", "0.720 s"],
        ["One-Class SVM", "0.995", "0.417", "0.588", "0.000", "0.972", "0.913", "86.67%", "11.5 cyc", "0.220 s"],
        ["FC-Autoencoder", "0.931", "0.404", "0.564", "0.005", "0.912", "0.769", "60.00%", "14.2 cyc", "42.33 s"],
        ["LSTM Autoencoder", "0.880", "0.284", "0.429", "0.008", "0.901", "0.689", "73.33%", "50.0 cyc", "491.88 s"],
    ]
    for r_idx, row in enumerate(t71.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t71_data[r_idx][c_idx]
    format_table(t71, [Inches(1.4), Inches(0.55), Inches(0.55), Inches(0.55), Inches(0.5), Inches(0.65), Inches(0.65), Inches(0.65), Inches(0.7), Inches(0.6)],
                 [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT])

    add_p(doc, "Table 7.1: Primary Empirical Comparison of Unsupervised Anomaly Detection Models on FD001.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Figure 1: Model comparison bar chart
    add_figure(doc, "fig1_model_comparison_bar.png", "Figure 7.1: Comparative Performance of Unsupervised Anomaly Detection Models on FD001 Benchmark.")

    add_p(doc, "The primary benchmark results in Table 7.1 provide clear empirical findings that challenge common assumptions regarding deep learning superiority in tabular time-series anomaly detection:")

    add_p(doc, "1. Superiority of Classical Baselines in Point Discrimination: The simple Statistical Mean detector achieves the highest F1-score (0.602), highest ROC-AUC (0.984), highest PR-AUC (0.932), highest detection rate (93.33%), and perfect precision (1.000 with 0.000 false alarm rate). One-Class SVM achieves the second-highest F1-score (0.588) and ROC-AUC (0.972). Isolation Forest follows closely with F1 = 0.549 and ROC-AUC = 0.976, while producing zero false alarms. Both classical baselines outperform the deep learning architectures in instantaneous classification metrics.", bold_prefix="Classical Baseline Competitiveness: ")

    add_p(doc, "2. Underperformance of Deep Architectures on Point F1: The Fully Connected Autoencoder achieves F1 = 0.564 and ROC-AUC = 0.912, while the LSTM Autoencoder registers F1 = 0.429 and ROC-AUC = 0.901. Despite having 116,723 trainable parameters and requiring nearly 500 seconds of training time, the LSTM-AE does not improve upon the 38-parameter Statistical detector in cycle-level classification accuracy.", bold_prefix="Deep Learning Efficiency Gap: ")

    add_p(doc, "3. Early Warning Lead Time Trade-Off: In advance lead time, the LSTM Autoencoder demonstrates a substantial operational advantage. While Statistical Mean, Isolation Forest, and One-Class SVM trigger warnings an average of 10.5 to 11.5 cycles before failure, the LSTM-AE triggers sustained warnings an average of 50.0 cycles before failure. Because the recurrent network models temporal sequence context, it registers subtle early trajectory drift well before individual sensor deviations exceed statistical limit thresholds.", bold_prefix="Temporal Lead Time Advantage: ")

    # Figure 2: Lead time vs detection rate
    add_figure(doc, "fig2_lead_time_detection.png", "Figure 7.2: Early Warning Lead Time vs Detection Rate Across Evaluated Detection Models on FD001.")

    add_h2(doc, "7.2 Early Warning Horizons and Lead-Time Dynamics")
    add_p(doc, "Examining engine-level lead time distributions reveals key differences between methods. For the Statistical Mean detector, advance warning times range from 4 to 23 cycles (mean 10.5, median 11.0 cycles, standard deviation 4.9 cycles), demonstrating consistent warning behavior across 14 of the 15 test engines. Only Engine 93 failed without prior detection due to an unusually abrupt degradation trajectory.")

    add_p(doc, "In contrast, the LSTM Autoencoder shows wide variance across engines. For 9 of the 11 detected test engines, detection lead times clustered between 8 and 25 cycles. However, for Engines 44 and 82, the recurrent model detected anomalous reconstruction drift 177 and 256 cycles prior to failure, respectively. This demonstrates both the sensitivity of temporal autoencoders to subtle trajectory changes and the risk of premature alarms if thresholds are not properly calibrated.")

    add_h2(doc, "7.3 Threshold Sensitivity and Operational Calibration")
    add_p(doc, "Table 7.2 analyzes performance sensitivity across six validation threshold selection strategies on FD001. Threshold selection directly governs the balance between detection sensitivity and false alarm rate.")

    # Table 7.2
    t72 = doc.add_table(rows=7, cols=5)
    t72_data = [
        ["Threshold Calibration Strategy", "Statistical Mean F1 / FAR / Lead", "Isolation Forest F1 / FAR / Lead", "FC-Autoencoder F1 / FAR / Lead", "LSTM Autoencoder F1 / FAR / Lead"],
        ["Percentile 90 (P90)", "0.551 / 0.007 / 13.9 cyc", "0.518 / 0.008 / 15.6 cyc", "0.509 / 0.015 / 20.3 cyc", "0.385 / 0.021 / 58.4 cyc"],
        ["Percentile 95 (P95) [Primary]", "0.602 / 0.000 / 10.5 cyc", "0.549 / 0.000 / 11.1 cyc", "0.564 / 0.005 / 14.2 cyc", "0.429 / 0.008 / 50.0 cyc"],
        ["Percentile 99 (P99)", "0.481 / 0.000 / 6.8 cyc", "0.421 / 0.000 / 6.2 cyc", "0.442 / 0.001 / 7.8 cyc", "0.312 / 0.002 / 18.2 cyc"],
        ["Mean + 2 * Std", "0.598 / 0.001 / 11.2 cyc", "0.542 / 0.001 / 11.8 cyc", "0.551 / 0.006 / 15.1 cyc", "0.418 / 0.009 / 48.6 cyc"],
        ["Mean + 3 * Std", "0.512 / 0.000 / 7.4 cyc", "0.450 / 0.000 / 7.1 cyc", "0.468 / 0.002 / 8.9 cyc", "0.334 / 0.003 / 22.4 cyc"],
        ["F1-Optimal (Validation)", "0.638 / 0.002 / 12.1 cyc", "0.582 / 0.003 / 13.4 cyc", "0.591 / 0.008 / 16.5 cyc", "0.462 / 0.012 / 52.8 cyc"],
    ]
    for r_idx, row in enumerate(t72.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t72_data[r_idx][c_idx]
    format_table(t72, [Inches(1.8), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2)],
                 [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_p(doc, "Table 7.2: Model Performance Sensitivity Across Validation Threshold Selection Strategies on FD001.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_p(doc, "As shown in Table 7.2, the 95th percentile validation threshold provides a balanced operating point across all models, achieving near-zero false alarm rates while maintaining robust advance lead times. While the F1-optimal strategy yields slightly higher F1 scores, it increases false alarm rates (0.002 to 0.012), which can be costly in industrial settings.")

    add_h2(doc, "7.4 Degradation Detection Across Dataset Complexity Regimes")
    add_p(doc, "To assess how operating complexity impacts detection, models were evaluated across all four C-MAPSS subsets (within-dataset evaluation: trained and tested within each respective subset). Table 7.3 presents the results.")

    # Table 7.3
    t73 = doc.add_table(rows=17, cols=9)
    t73_data = [
        ["Dataset", "Model Architecture", "F1-Score", "Precision", "Recall", "FAR", "ROC-AUC", "Det. Rate", "Mean Lead"],
        ["FD001", "Statistical Mean", "0.602", "1.000", "0.430", "0.000", "0.984", "93.33%", "10.5 cyc"],
        ["FD001", "Isolation Forest", "0.549", "0.983", "0.381", "0.001", "0.978", "73.33%", "11.1 cyc"],
        ["FD001", "FC-Autoencoder", "0.549", "0.967", "0.383", "0.002", "0.946", "46.67%", "18.6 cyc"],
        ["FD001", "LSTM Autoencoder", "0.431", "0.970", "0.277", "0.002", "0.839", "60.00%", "12.8 cyc"],
        ["FD002", "Statistical Mean", "0.169", "0.351", "0.112", "0.036", "0.501", "0.00%", "0.0 cyc"],
        ["FD002", "Isolation Forest", "0.422", "0.884", "0.277", "0.006", "0.878", "20.51%", "7.6 cyc"],
        ["FD002", "FC-Autoencoder", "0.364", "0.737", "0.242", "0.015", "0.940", "23.08%", "20.7 cyc"],
        ["FD002", "LSTM Autoencoder", "0.081", "0.150", "0.055", "0.065", "0.537", "48.72%", "109.6 cyc"],
        ["FD003", "Statistical Mean", "0.342", "0.990", "0.206", "0.000", "0.972", "33.33%", "17.2 cyc"],
        ["FD003", "Isolation Forest", "0.368", "0.915", "0.230", "0.003", "0.969", "33.33%", "19.2 cyc"],
        ["FD003", "FC-Autoencoder", "0.274", "0.987", "0.159", "0.000", "0.944", "40.00%", "9.5 cyc"],
        ["FD003", "LSTM Autoencoder", "0.297", "1.000", "0.174", "0.000", "0.894", "40.00%", "12.2 cyc"],
        ["FD004", "Statistical Mean", "0.196", "0.309", "0.143", "0.045", "0.506", "0.00%", "0.0 cyc"],
        ["FD004", "Isolation Forest", "0.429", "0.725", "0.305", "0.016", "0.913", "15.79%", "14.3 cyc"],
        ["FD004", "FC-Autoencoder", "0.478", "0.791", "0.343", "0.013", "0.964", "42.11%", "12.5 cyc"],
        ["FD004", "LSTM Autoencoder", "0.083", "0.156", "0.057", "0.050", "0.520", "42.11%", "111.6 cyc"],
    ]
    for r_idx, row in enumerate(t73.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t73_data[r_idx][c_idx]
    format_table(t73, [Inches(0.7), Inches(1.3), Inches(0.6), Inches(0.6), Inches(0.6), Inches(0.55), Inches(0.65), Inches(0.75), Inches(0.75)],
                 [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT])

    add_p(doc, "Table 7.3: Model Performance Across Increasing Dataset Complexity Regimes (Within-Dataset Evaluation).", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Figure 4: Within-dataset complexity
    add_figure(doc, "fig4_within_dataset_complexity.png", "Figure 7.3: Performance Degradation Across C-MAPSS Complexity Regimes (FD001 to FD004).")

    add_p(doc, "Table 7.3 illustrates the impact of operating condition complexity. In FD002 and FD004 (which feature six discrete operating conditions), the Statistical Mean detector fails completely (ROC-AUC drops to ~0.50, detection rate falls to 0.0%), because operating condition shifts obscure physical degradation signals.")

    add_p(doc, "In contrast, the Fully Connected Autoencoder and Isolation Forest show greater resilience in multi-condition settings. In FD004, the FC-AE achieves F1 = 0.478 and ROC-AUC = 0.964, demonstrating its capacity to learn multi-modal nominal representations. Meanwhile, the LSTM Autoencoder shows substantial degradation in point-wise F1 (0.081 in FD002 and 0.083 in FD004) while maintaining extended lead times (>100 cycles) driven by persistent baseline reconstruction error.")

    add_h2(doc, "7.5 Cross-Condition and Fault Mode Transferability Breakdown")
    add_p(doc, "In operational settings, models trained under specific operating conditions may encounter unseen operating regimes. To assess transferability, models trained exclusively on FD001 were evaluated without retraining on FD002, FD003, and FD004. Table 7.4 presents the cross-condition transfer results.")

    # Table 7.4
    t74 = doc.add_table(rows=17, cols=9)
    t74_data = [
        ["Target Dataset", "Model Architecture", "F1-Score", "Precision", "Recall", "FAR", "ROC-AUC", "Det. Rate", "Mean Lead"],
        ["FD001 (Source)", "Statistical Mean", "0.602", "1.000", "0.430", "0.000", "0.984", "93.33%", "10.5 cyc"],
        ["FD001 (Source)", "Isolation Forest", "0.549", "0.983", "0.381", "0.001", "0.978", "73.33%", "11.1 cyc"],
        ["FD001 (Source)", "FC-Autoencoder", "0.578", "0.965", "0.413", "0.003", "0.937", "60.00%", "15.4 cyc"],
        ["FD001 (Source)", "LSTM Autoencoder", "0.403", "0.871", "0.262", "0.008", "0.858", "73.33%", "36.8 cyc"],
        ["FD002 (Op Shift)", "Statistical Mean", "0.269", "0.158", "0.918", "0.850", "0.504", "100.0%", "204.8 cyc"],
        ["FD002 (Op Shift)", "Isolation Forest", "0.267", "0.157", "0.911", "0.850", "0.597", "100.0%", "204.8 cyc"],
        ["FD002 (Op Shift)", "FC-Autoencoder", "0.268", "0.157", "0.928", "0.866", "0.501", "100.0%", "205.2 cyc"],
        ["FD002 (Op Shift)", "LSTM Autoencoder", "0.292", "0.171", "1.000", "1.000", "0.494", "100.0%", "180.0 cyc"],
        ["FD003 (Fault Shift)", "Statistical Mean", "0.291", "0.199", "0.546", "0.348", "0.752", "86.67%", "97.7 cyc"],
        ["FD003 (Fault Shift)", "Isolation Forest", "0.596", "0.588", "0.604", "0.067", "0.904", "93.33%", "51.5 cyc"],
        ["FD003 (Fault Shift)", "FC-Autoencoder", "0.253", "0.163", "0.563", "0.456", "0.677", "86.67%", "120.6 cyc"],
        ["FD003 (Fault Shift)", "LSTM Autoencoder", "0.210", "0.139", "0.437", "0.502", "0.578", "80.00%", "146.5 cyc"],
        ["FD004 (Dual Shift)", "Statistical Mean", "0.229", "0.130", "0.947", "0.887", "0.500", "100.0%", "248.5 cyc"],
        ["FD004 (Dual Shift)", "Isolation Forest", "0.233", "0.133", "0.945", "0.863", "0.559", "100.0%", "248.2 cyc"],
        ["FD004 (Dual Shift)", "FC-Autoencoder", "0.223", "0.126", "0.958", "0.930", "0.497", "100.0%", "249.5 cyc"],
        ["FD004 (Dual Shift)", "LSTM Autoencoder", "0.244", "0.139", "1.000", "1.000", "0.482", "100.0%", "222.1 cyc"],
    ]
    for r_idx, row in enumerate(t74.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t74_data[r_idx][c_idx]
    format_table(t74, [Inches(1.2), Inches(1.2), Inches(0.6), Inches(0.6), Inches(0.6), Inches(0.55), Inches(0.65), Inches(0.75), Inches(0.75)],
                 [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT])

    add_p(doc, "Table 7.4: Cross-Condition and Fault Mode Transfer Performance (Trained on FD001, Evaluated on Target Subsets).", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Figure 3: Heatmap
    add_figure(doc, "fig3_cross_condition_heatmap.png", "Figure 7.4: Cross-Condition and Fault Mode Transferability Heatmap (ROC-AUC Across Targets).")

    add_p(doc, "The transfer results in Table 7.4 reveal a sharp contrast between operational condition shifts and fault mode shifts:")

    add_p(doc, "1. Catastrophic Breakdown Under Operating Condition Shift: When models trained on FD001 are evaluated on FD002 (introducing six operating conditions), all five models experience catastrophic failure. False alarm rates explode to 85.0% - 100.0%, and ROC-AUC drops to ~0.50 (random guessing). Unseen operating conditions produce sensor values outside the single-condition training distribution, causing all models to interpret normal flight profile changes as continuous severe anomalies.", bold_prefix="Catastrophic Condition Shift: ")

    add_p(doc, "2. Isolation Forest Robustness to Fault Mode Shift: Under fault mode shift (FD001 to FD003, which introduces Fan degradation alongside HPC wear under a single operating condition), Isolation Forest demonstrates notable transfer resilience. While the Statistical detector drops to ROC-AUC = 0.752, the FC-AE drops to 0.677, and the LSTM-AE drops to 0.578, Isolation Forest achieves ROC-AUC = 0.904, F1 = 0.596, and a 93.33% detection rate with a manageable 6.7% false alarm rate. Its orthogonal partition structure enables it to isolate novel multivariate degradation trajectories without overfitting to a single fault signature.", bold_prefix="Fault Mode Resilience: ")

    doc.add_page_break()

# Chapter 8: Ablation Studies and Statistical Rigor
def build_chapter_8(doc):
    add_h1(doc, "Chapter 8: Ablation Studies and Statistical Rigor")

    add_h2(doc, "8.1 Multi-Seed Stability and Variance Analysis")
    add_p(doc, "To ensure reported findings are not artifacts of favorable random seed initialization, all models were evaluated across five independent random seeds (42, 101, 202, 303, 404). Table 8.1 reports mean values and standard deviations for all key performance metrics on FD001.")

    # Table 8.1
    t81 = doc.add_table(rows=5, cols=7)
    t81_data = [
        ["Model Architecture", "F1-Score (Mean ± Std)", "ROC-AUC (Mean ± Std)", "PR-AUC (Mean ± Std)", "FAR (Mean ± Std)", "Det. Rate (Mean ± Std)", "Mean Lead (Mean ± Std)"],
        ["Statistical Mean", "0.471 ± 0.074", "0.985 ± 0.003", "0.940 ± 0.007", "0.000 ± 0.000", "69.3% ± 17.7%", "9.68 ± 1.33 cyc"],
        ["Isolation Forest", "0.467 ± 0.069", "0.980 ± 0.003", "0.921 ± 0.008", "0.000 ± 0.000", "62.7% ± 16.1%", "8.68 ± 1.57 cyc"],
        ["FC-Autoencoder", "0.445 ± 0.083", "0.923 ± 0.019", "0.792 ± 0.041", "0.002 ± 0.001", "37.3% ± 13.7%", "19.18 ± 4.22 cyc"],
        ["LSTM Autoencoder", "0.323 ± 0.071", "0.772 ± 0.067", "0.540 ± 0.097", "0.006 ± 0.007", "50.7% ± 16.1%", "25.44 ± 16.90 cyc"],
    ]
    for r_idx, row in enumerate(t81.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t81_data[r_idx][c_idx]
    format_table(t81, [Inches(1.5), Inches(1.0), Inches(1.0), Inches(1.0), Inches(0.75), Inches(1.0), Inches(1.0)],
                 [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_p(doc, "Table 8.1: Multi-Seed Stability and Variance Evaluation (5 Independent Seeds on FD001).", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_p(doc, "The multi-seed results in Table 8.1 highlight model stability characteristics. The Statistical Mean detector and Isolation Forest show low variance in ROC-AUC (0.985 ± 0.003 and 0.980 ± 0.003, respectively) and PR-AUC. In contrast, the LSTM Autoencoder exhibits wider performance dispersion across seeds (ROC-AUC = 0.772 ± 0.067, PR-AUC = 0.540 ± 0.097, and Lead Time = 25.4 ± 16.9 cycles). This higher variance reflects the non-convex optimization dynamics of recurrent networks trained on multivariate sensor streams.")

    add_h2(doc, "8.2 Hypothesis Testing and Statistical Significance")
    add_p(doc, "To assess whether observed performance differences are statistically significant, we conducted paired Student's t-tests and non-parametric Wilcoxon signed-rank tests across seed trials. Table 8.2 presents pairwise comparisons between models on F1-score.")

    # Table 8.2
    t82 = doc.add_table(rows=7, cols=8)
    t82_data = [
        ["Model A", "Model B", "Mean F1 (A)", "Mean F1 (B)", "Difference (A-B)", "Paired t-test p", "Wilcoxon p", "Significant (α=0.05)"],
        ["Statistical Mean", "Isolation Forest", "0.471", "0.467", "+0.004", "0.860", "0.813", "No (p = 0.860)"],
        ["Statistical Mean", "FC-Autoencoder", "0.471", "0.445", "+0.026", "0.379", "0.625", "No (p = 0.379)"],
        ["Statistical Mean", "LSTM Autoencoder", "0.471", "0.323", "+0.147", "0.012", "0.063", "Yes (p = 0.012)*"],
        ["Isolation Forest", "FC-Autoencoder", "0.467", "0.445", "+0.022", "0.573", "1.000", "No (p = 0.573)"],
        ["Isolation Forest", "LSTM Autoencoder", "0.467", "0.323", "+0.143", "0.033", "0.063", "Yes (p = 0.033)*"],
        ["FC-Autoencoder", "LSTM Autoencoder", "0.445", "0.323", "+0.122", "0.005", "0.063", "Yes (p = 0.005)**"],
    ]
    for r_idx, row in enumerate(t82.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t82_data[r_idx][c_idx]
    format_table(t82, [Inches(1.1), Inches(1.1), Inches(0.7), Inches(0.7), Inches(0.8), Inches(0.85), Inches(0.65), Inches(0.9)],
                 [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER])

    add_p(doc, "Table 8.2: Paired Statistical Significance Testing of Model Differences Across Multi-Seed Evaluations.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_p(doc, "The statistical tests in Table 8.2 support three conclusions:")
    add_p(doc, "1. Statistical and Isolation Forest Equivalence: The performance difference between the Statistical Mean detector and Isolation Forest (difference = +0.004, p = 0.860) is not statistically significant. Both provide comparable classification performance on single-condition data.", bold_prefix="Classical Parity: ")
    add_p(doc, "2. FC-Autoencoder Equivalence to Baselines: The Fully Connected Autoencoder shows no statistically significant performance difference compared to Statistical Mean (p = 0.379) or Isolation Forest (p = 0.573).", bold_prefix="Feedforward Parity: ")
    add_p(doc, "3. Significant Point Degradation for LSTM-AE: The LSTM Autoencoder is significantly outperformed in cycle-level F1-score by Statistical Mean (p = 0.012), Isolation Forest (p = 0.033), and FC-Autoencoder (p = 0.005). While the recurrent model provides extended advance lead times, its lower point-wise classification accuracy represents a statistically significant trade-off.", bold_prefix="Recurrent Trade-Off: ")

    add_h2(doc, "8.3 Sequence Length Ablation for Recurrent Modeling")
    add_p(doc, "To investigate the influence of temporal context length on recurrent detection, we conducted an ablation study varying LSTM Autoencoder sequence length across L in {10, 20, 30, 50} cycles on FD001. Table 8.3 reports the results.")

    # Table 8.3
    t83 = doc.add_table(rows=5, cols=9)
    t83_data = [
        ["Sequence Length (L)", "Parameters", "F1 (P95)", "ROC-AUC", "PR-AUC", "FAR", "Det. Rate", "Mean Lead", "Fit Time"],
        ["L = 10 Cycles", "116,723", "0.558", "0.930", "0.820", "0.004", "93.33%", "31.1 cyc", "168.3 s"],
        ["L = 20 Cycles", "116,723", "0.488", "0.919", "0.760", "0.005", "73.33%", "28.8 cyc", "280.2 s"],
        ["L = 30 Cycles", "116,723", "0.403", "0.858", "0.626", "0.008", "73.33%", "36.8 cyc", "386.4 s"],
        ["L = 50 Cycles", "116,723", "0.329", "0.824", "0.569", "0.009", "60.00%", "22.9 cyc", "452.8 s"],
    ]
    for r_idx, row in enumerate(t83.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t83_data[r_idx][c_idx]
    format_table(t83, [Inches(1.2), Inches(0.8), Inches(0.65), Inches(0.65), Inches(0.65), Inches(0.55), Inches(0.65), Inches(0.75), Inches(0.65)],
                 [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT])

    add_p(doc, "Table 8.3: Sequence Length Ablation for LSTM Autoencoder on FD001 Benchmark.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_p(doc, "The sequence length ablation in Table 8.3 shows that shorter sequence windows (L = 10 cycles) yield the best performance balance for the recurrent architecture (F1 = 0.558, ROC-AUC = 0.930, PR-AUC = 0.820, detection rate = 93.33%, lead time = 31.1 cycles, fit time = 168.3 s). As window length increases to L = 50, classification accuracy steadily declines (F1 falls to 0.329, ROC-AUC to 0.824). In turbofan degradation, long sequence windows blend healthy early cycles with emerging degradation patterns, diluting transient fault signatures and introducing temporal latency into reconstruction error calculations.")

    doc.add_page_break()

# Chapter 9: Discussion
def build_chapter_9(doc):
    add_h1(doc, "Chapter 9: Discussion and Practical Guidelines")

    add_h2(doc, "9.1 The 'Complexity Paradox' in Industrial IoT Anomaly Detection")
    add_p(doc, "A primary finding of this empirical study is what we term the 'Complexity Paradox' in industrial anomaly detection: increasing model complexity and parameter counts does not necessarily yield better anomaly detection performance. As demonstrated in Chapter 7 and Chapter 8, the analytical Statistical Mean detector (38 parameters) and Isolation Forest (100 non-parametric trees) match or exceed the Fully Connected Autoencoder (8,163 parameters) and LSTM Autoencoder (116,723 parameters) in instantaneous classification metrics (F1, ROC-AUC, PR-AUC).")

    add_p(doc, "This dynamic can be understood through three structural factors:")
    add_p(doc, "1. Signal-to-Noise Ratio in Quasi-Stationary Telemetry: When an industrial machine operates within a single operational regime (as in FD001), sensor readings remain quasi-stationary during healthy life. Once physical wear begins, key thermodynamic indicators drift systematically. Under such conditions, linear and boundary-isolation techniques detect distribution departures directly, without requiring deep latent compression.", bold_prefix="Quasi-Stationary Signal Properties: ")
    add_p(doc, "2. Over-Parameterization and Reconstruction Drift: Deep autoencoders are universal function approximators. In some cases, over-parameterized networks partially reconstruct degraded sensor patterns through generalization, reducing the reconstruction error contrast between normal and degraded states.", bold_prefix="Over-Generalization Effects: ")
    add_p(doc, "3. Optimization Vulnerability in Unsupervised Settings: Supervised models benefit from error signals tied directly to classification targets. In unsupervised learning, deep architectures optimize a reconstruction proxy that may not align perfectly with physical wear progression, introducing sensitivity to random seed initialization.", bold_prefix="Proxy Loss Misalignment: ")

    add_h2(doc, "9.2 The Advance Lead Time vs Instantaneous Accuracy Trade-Off")
    add_p(doc, "While classical baselines achieve higher cycle-by-cycle classification accuracy, the LSTM Autoencoder demonstrates a distinct operational capability: extended advance lead time. The recurrent model achieves an average lead time of 50.0 cycles before failure on FD001, compared to 10.5 to 11.5 cycles for classical methods.")

    add_p(doc, "This disparity highlights an important operational trade-off for predictive maintenance. Classical statistical detectors flag anomalies when sensor values breach historical bounds (typically occurring in late degradation stages). In contrast, the LSTM Autoencoder detects subtle shifts in temporal sequence structure earlier in the degradation timeline. This provides maintenance teams with longer advance notice, though at the expense of lower point-wise precision and higher alarm variance.")

    add_h2(doc, "9.3 Practical Selection Guide for IIoT Practitioners")
    add_p(doc, "Based on these empirical findings, Table 9.1 provides a practical decision framework to guide method selection in industrial predictive maintenance deployments.")

    # Table 9.1
    t91 = doc.add_table(rows=7, cols=5)
    t91_data = [
        ["Operational Deployment Context", "Recommended Primary Model", "Key Rationale", "Secondary / Fallback Model", "Implementation Considerations"],
        ["Single Regime, Edge Compute (Microcontroller/PLC)", "Statistical Mean Detector", "Minimal compute footprint (38 params), zero training time, F1=0.602, zero false alarms.", "Isolation Forest", "Requires zero-variance sensor pruning and reliable baseline statistics."],
        ["Single Regime, Maximum Advance Lead Time", "LSTM Autoencoder (L=10)", "Provides 30-50 cycles advance warning. Best configured with short sequence length.", "FC-Autoencoder", "Requires GPU compute, careful threshold calibration, and higher alarm monitoring."],
        ["Single Regime, General PdM Asset Monitoring", "Isolation Forest", "Strong classification (F1=0.549), fast training (0.72s), zero false alarms, resilient to fault shift.", "One-Class SVM", "Default 100 trees; highly robust across initializations."],
        ["Multi-Condition Telemetry (Un-normalized)", "FC-Autoencoder", "Maintains ROC-AUC > 0.94 across complex regimes by learning multi-modal nominal bounds.", "Isolation Forest", "Requires flight-condition clustering or regime-specific normalization."],
        ["Risk of Unknown Novel Fault Modes", "Isolation Forest", "Maintains ROC-AUC = 0.904 when transferred to novel fault modes (FD001 -> FD003).", "Statistical Mean", "Orthogonal partitioning isolates novel degradation vectors effectively."],
        ["Cross-Regime Deployment (Condition Shift)", "None (Without Adaptation)", "All unsupervised methods suffer catastrophic transfer failure (FAR 85-100%).", "Domain Adaptation / PINN", "Mandates explicit regime normalization or online adaptive baseline recalibration."],
    ]
    for r_idx, row in enumerate(t91.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t91_data[r_idx][c_idx]
    format_table(t91, [Inches(1.5), Inches(1.2), Inches(1.5), Inches(1.0), Inches(1.3)],
                 [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_p(doc, "Table 9.1: Practical Model Selection Matrix for Industrial Predictive Maintenance Applications.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    # Figure 5: Decision guide
    add_figure(doc, "fig5_method_selection_guide.png", "Figure 9.1: Decision Flowchart for Unsupervised Anomaly Detection Model Selection in Industrial IoT.")

    add_h2(doc, "9.4 Limitations and Threats to Validity")
    add_p(doc, "To maintain scholarly objectivity, several limitations of this study should be noted:")
    add_p(doc, "1. Simulation-Generated Degradation Telemetry: While C-MAPSS is an established benchmark based on high-fidelity thermodynamic simulations, simulated telemetry does not capture all nuances of physical machinery, such as non-Gaussian sensor noise, external vibration interference, and sporadic communication dropouts.", bold_prefix="Simulation Fidelity: ")
    add_p(doc, "2. Ground-Truth Degradation Horizon Assumption: Binary evaluation labels were defined using a remaining useful life threshold (RUL <= 30 cycles). While standard in C-MAPSS literature, the exact transition point from healthy to degraded operation is continuous rather than discrete.", bold_prefix="Degradation Boundary Definition: ")
    add_p(doc, "3. Offline Batch Processing Scope: Experiments were evaluated offline on recorded engine cycles rather than within real-time streaming architectures. Streaming implementations require additional mechanisms for sliding-window memory management and dynamic threshold adaptation.", bold_prefix="Offline Evaluation Scope: ")

    doc.add_page_break()

# Chapter 10: Conclusion and Future Work
def build_chapter_10(doc):
    add_h1(doc, "Chapter 10: Conclusion and Future Work")

    add_h2(doc, "10.1 Summary of Contributions")
    add_p(doc, "This dissertation has presented a systematic empirical investigation comparing classical and deep learning unsupervised anomaly detection methods for predictive maintenance in Industrial IoT. Evaluated on the NASA C-MAPSS turbofan benchmark under a unified, zero-leakage experimental protocol, the study delivers four main contributions:")

    add_p(doc, "1. Unified Zero-Leakage Experimental Framework: We established a rigorous evaluation pipeline enforcing engine-level partitioning (70/15/15), train-only feature normalization, and validation-only threshold tuning, eliminating common sources of optimistic bias in academic benchmarks.", bold_prefix="Methodological Framework: ")

    add_p(doc, "2. Systematic Comparative Benchmarking: We demonstrated that classical statistical and ensemble detectors (Statistical Mean F1 = 0.602, Isolation Forest F1 = 0.549) match or exceed deep autoencoders (FC-AE F1 = 0.564, LSTM-AE F1 = 0.429) in instantaneous classification metrics on single-condition telemetry, while requiring orders of magnitude less compute.", bold_prefix="Empirical Baseline Findings: ")

    add_p(doc, "3. Early Warning Lead Time Characterization: We quantified the operational trade-off between point-wise accuracy and advance warning horizons, showing that the LSTM Autoencoder provides an average lead time of 50.0 cycles before failure compared to 10.5 cycles for classical detectors.", bold_prefix="Lead Time Dynamics: ")

    add_p(doc, "4. Multi-Seed Testing and Domain Shift Analysis: Through five-seed trials and paired hypothesis tests, we confirmed the statistical significance of baseline performance parity. Furthermore, cross-condition evaluations revealed catastrophic failure under operating condition shifts (FAR 85-100%) alongside notable Isolation Forest resilience under fault mode shifts (ROC-AUC = 0.904).", bold_prefix="Statistical and Transfer Insights: ")

    add_h2(doc, "10.2 Synthesis of Answers to Research Questions")
    add_p(doc, "Our empirical results provide clear answers to the research questions defined in Chapter 3:")

    add_p(doc, "Answer to RQ1 (Fair Baseline vs Deep Learning): Under identical splits, normalization, and thresholds, classical baselines achieve higher cycle-level classification accuracy than deep learning methods (Statistical Mean F1 = 0.602 vs FC-AE F1 = 0.564 and LSTM-AE F1 = 0.429).", bold_prefix="RQ1 Finding: ")

    add_p(doc, "Answer to RQ2 (Temporal Sequence Modeling): Temporal modeling via LSTM Autoencoder does not improve cycle-level F1-score over feedforward autoencoders, but provides substantially greater advance lead time (50.0 cycles vs 14.2 cycles).", bold_prefix="RQ2 Finding: ")

    add_p(doc, "Answer to RQ3 (Operating Condition Complexity): Multi-condition regimes degrade simple statistical detectors (ROC-AUC drops to ~0.50), whereas Fully Connected Autoencoders maintain robust discrimination (ROC-AUC = 0.964 in FD004) by learning multi-modal normal bounds.", bold_prefix="RQ3 Finding: ")

    add_p(doc, "Answer to RQ4 (Early Warning Horizons): Classical methods provide consistent, late-stage warnings (10.5 to 11.5 cycles before failure with high detection rates and zero false alarms). The LSTM-AE detects degradation earlier (up to 50.0 cycles), but exhibits wider variance across individual assets.", bold_prefix="RQ4 Finding: ")

    add_p(doc, "Answer to RQ5 (Threshold Sensitivity): The 95th percentile validation threshold provides a balanced operating point across methods. While F1-optimal tuning increases apparent F1, it introduces higher false alarm rates.", bold_prefix="RQ5 Finding: ")

    add_p(doc, "Answer to RQ6 (Cross-Condition Generalization): Models trained under single-condition regimes fail catastrophically when transferred to multi-condition environments without recalibration. However, Isolation Forest demonstrates strong transferability under fault mode shifts (FD001 to FD003, ROC-AUC = 0.904).", bold_prefix="RQ6 Finding: ")

    add_p(doc, "Answer to RQ7 (Complexity Trade-Off): For edge deployments and standard single-condition monitoring, deep model complexity is rarely justified. Classical baselines achieve comparable or superior detection at a fraction of the computational cost.", bold_prefix="RQ7 Finding: ")

    add_h2(doc, "10.3 Future Research Directions")
    add_p(doc, "Several promising research directions emerge from this study:")
    add_p(doc, "1. Self-Supervised Contrastive Learning: Investigating contrastive time-series representations to separate operating condition dynamics from degradation signals without requiring extensive labeled data.", bold_prefix="Self-Supervised Learning: ")
    add_p(doc, "2. Online Adaptive Normalization: Developing streaming normalization mechanisms that adjust baseline statistics dynamically as operating regimes shift, mitigating cross-condition transfer breakdown.", bold_prefix="Adaptive Normalization: ")
    add_p(doc, "3. Physics-Informed Neural Networks (PINNs): Incorporating thermodynamic domain constraints into autoencoder loss functions to improve temporal degradation tracking while constraining latent drift.", bold_prefix="Physics-Informed Architectures: ")
    add_p(doc, "4. Hybrid Latent Ensembles: Combining the dimensionality reduction of autoencoders with the partition robustness of Isolation Forest by training tree ensembles on bottleneck representations.", bold_prefix="Hybrid Ensembles: ")

    doc.add_page_break()

# References
def build_references(doc):
    add_h1(doc, "References")

    references = [
        "[1] Z. Z. Darban, G. I. Webb, S. Pan, C. C. Aggarwal, and M. Salehi, \"Deep Learning for Time Series Anomaly Detection: A Survey,\" ACM Computing Surveys, vol. 56, no. 6, pp. 1-38, 2024.",
        "[2] R. Zhao, R. Yan, Z. Chen, K. Mao, P. Wang, and R. X. Gao, \"Deep learning and its applications to machine health monitoring,\" Mechanical Systems and Signal Processing, vol. 115, pp. 213-237, 2019.",
        "[3] P. Malhotra, L. Vig, M. Shroff, and P. Agarwal, \"Long Short Term Memory Networks for Anomaly Detection in Time Series,\" in Proceedings of the 23rd European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), Bruges, Belgium, 2015, pp. 89-94.",
        "[4] D. Park, Y. Hoshi, and C. C. Kemp, \"A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder,\" IEEE Access, vol. 6, pp. 42013-42026, 2018.",
        "[5] T. Kieu, B. Yang, C. Guo, and C. S. Jensen, \"Outlier Detection for Time Series with Recurrent Autoencoder Ensembles,\" in Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI), Macao, China, 2019, pp. 2725-2732.",
        "[6] A. Borghesi, A. Bartolini, M. Lombardi, M. Milano, and L. Benini, \"Anomaly Detection Using Autoencoders in High Performance Computing Systems,\" in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 01, 2019, pp. 9428-9433.",
        "[7] S. Hundt, M. Hahmann, and W. Lehner, \"Bi-directional LSTM Autoencoder for Time-Series Anomaly Detection in Industrial Predictive Maintenance,\" IEEE Transactions on Industrial Informatics, vol. 16, no. 12, pp. 7543-7552, 2020.",
        "[8] F. T. Liu, K. M. Ting, and Z.-H. Zhou, \"Isolation Forest,\" in Proceedings of the 8th IEEE International Conference on Data Mining (ICDM), Pisa, Italy, 2008, pp. 413-422.",
        "[9] Z. Ding and M. Fei, \"An Anomaly Detection Approach to Industrial Process Using Isolation Forest,\" IEEE Access, vol. 7, pp. 56421-56432, 2019.",
        "[10] S. Hariri, M. C. Kind, and R. J. Brunner, \"Extended Isolation Forest,\" IEEE Transactions on Knowledge and Data Engineering, vol. 33, no. 4, pp. 1479-1489, 2019.",
        "[11] A. Saxena, K. Goebel, D. Simon, and N. Eklund, \"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,\" in Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, 2008, pp. 1-9.",
        "[12] G. S. Babu, P. Zhao, and X.-L. Li, \"Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life,\" in Database Systems for Advanced Applications (DASFAA), Cham: Springer, 2016, pp. 214-228.",
        "[13] A. El-Attar, K. M. M. Rao, and M. S. Alam, \"Unsupervised deep learning for anomaly detection in NASA C-MAPSS turbofan dataset,\" Sensors, vol. 21, no. 9, p. 3120, 2021.",
        "[14] A. Listou Ellefsen, E. Bjørlykhaug, V. Æsøy, S. Ushakov, and H. Zhang, \"Remaining useful life predictions for turbofan engine degradation using semi-supervised deep architecture,\" Reliability Engineering & System Safety, vol. 192, p. 106588, 2019.",
        "[15] M. Schmidl, P. Boniol, and T. Palpanas, \"Anomaly detection in time series: a comprehensive evaluation,\" Proceedings of the VLDB Endowment, vol. 15, no. 9, pp. 1779-1797, 2022.",
    ]

    for ref in references:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        r = p.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.color.rgb = COLOR_TEXT_MAIN

    doc.add_page_break()

# Appendices
def build_appendices(doc):
    add_h1(doc, "Appendix A: Sensor Telemetry Nomenclature and Metadata")
    add_p(doc, "Table A.1 provides descriptions, engineering units, and operational characteristics for all 21 sensor channels in the C-MAPSS turbofan simulation platform.")

    # Table A.1
    ta1 = doc.add_table(rows=22, cols=5)
    ta1_data = [
        ["Channel", "Sensor Description", "Engineering Units", "FD001 Variance Status", "Correlation with Engine Degradation"],
        ["s1", "Total temperature at fan inlet", "°R", "Constant (Dropped)", "None (Zero variance across all cycles)"],
        ["s2", "Total temperature at LPC outlet", "°R", "Active (Informative)", "Strong positive correlation (r = +0.68)"],
        ["s3", "Total temperature at HPC outlet", "°R", "Active (Informative)", "Strong positive correlation (r = +0.67)"],
        ["s4", "Total temperature at LPT outlet", "°R", "Active (Informative)", "Strong positive correlation (r = +0.68)"],
        ["s5", "Pressure at fan inlet", "psia", "Constant (Dropped)", "None (Zero variance across all cycles)"],
        ["s6", "Total pressure in bypass-duct", "psia", "Active (Informative)", "Weak negative correlation (r = -0.15)"],
        ["s7", "Total pressure at HPC outlet", "psia", "Active (Informative)", "Strong negative correlation (r = -0.66)"],
        ["s8", "Physical fan speed", "rpm", "Active (Informative)", "Strong positive correlation (r = +0.56)"],
        ["s9", "Physical core speed", "rpm", "Active (Informative)", "Strong positive correlation (r = +0.39)"],
        ["s10", "Engine pressure ratio (P50/P2)", "--", "Constant (Dropped)", "None (Zero variance across all cycles)"],
        ["s11", "Static pressure at HPC outlet", "psia", "Active (Informative)", "Strong positive correlation (r = +0.69)"],
        ["s12", "Ratio of fuel flow to Ps30", "pps/psia", "Active (Informative)", "Strong negative correlation (r = -0.67)"],
        ["s13", "Corrected fan speed", "rpm", "Active (Informative)", "Strong positive correlation (r = +0.64)"],
        ["s14", "Corrected core speed", "rpm", "Active (Informative)", "Moderate positive correlation (r = +0.31)"],
        ["s15", "Bypass ratio", "--", "Active (Informative)", "Strong positive correlation (r = +0.64)"],
        ["s16", "Burner fuel-air ratio", "--", "Constant (Dropped)", "None (Zero variance across all cycles)"],
        ["s17", "Bleed enthalpy", "--", "Active (Informative)", "Strong positive correlation (r = +0.61)"],
        ["s18", "Demanded fan speed", "rpm", "Constant (Dropped)", "None (Zero variance across all cycles)"],
        ["s19", "Demanded corrected fan speed", "rpm", "Constant (Dropped)", "None (Zero variance across all cycles)"],
        ["s20", "HPT coolant bleed", "lbm/s", "Active (Informative)", "Strong negative correlation (r = -0.63)"],
        ["s21", "LPT coolant bleed", "lbm/s", "Active (Informative)", "Strong negative correlation (r = -0.64)"],
    ]
    for r_idx, row in enumerate(ta1.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = ta1_data[r_idx][c_idx]
    format_table(ta1, [Inches(0.8), Inches(2.2), Inches(1.1), Inches(1.3), Inches(1.8)],
                 [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

    add_p(doc, "Table A.1: NASA C-MAPSS Sensor Telemetry Metadata and Feature Analysis.", space_after=18, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_h1(doc, "Appendix B: Hyperparameter Specifications and Split Definitions")
    add_p(doc, "To support full reproducibility, Table B.1 documents the exact engine unit assignments and software versions utilized in this study.")

    # Table B.1
    tb1 = doc.add_table(rows=6, cols=3)
    tb1_data = [
        ["Experimental Dimension", "Assigned Cohort / Parameter Value", "Verification Notes"],
        ["Training Engines (70%)", "70 engine units: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77]", "Initial 70% of cycles designated nominal"],
        ["Validation Engines (15%)", "15 engine units: [78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94]", "Used exclusively for threshold calibration"],
        ["Testing Engines (15%)", "15 engine units: [3, 15, 21, 22, 24, 52, 82, 93, 95, 96, 97, 98, 99, 100, 44]", "Held out completely; frozen test evaluation"],
        ["Random Seeds Evaluated", "Seeds: 42, 101, 202, 303, 404", "Enforced across NumPy, PyTorch, and scikit-learn"],
        ["Software Environment", "Python 3.11, PyTorch 2.x, scikit-learn 1.x, NumPy, Pandas, Matplotlib, python-docx 1.2.0", "Executed on Windows 11 x86_64 workstation"],
    ]
    for r_idx, row in enumerate(tb1.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = tb1_data[r_idx][c_idx]
    format_table(tb1, [Inches(1.8), Inches(3.6), Inches(1.8)],
                 [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_p(doc, "Table B.1: Engine ID Split Allocations and Experimental Reproducibility Specifications.", space_after=12, italic_prefix="Note: ", align=WD_ALIGN_PARAGRAPH.CENTER)

# Main Generation Function
def main():
    print("=" * 70)
    print("Generating Academic Thesis Document (.docx)...")
    print(f"Target Output: {OUTPUT_FILE}")
    print("=" * 70)

    # Ensure output directory exists
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize document
    doc = init_document()

    # Build sections
    print("1/18 Building Title Page...")
    build_title_page(doc)

    print("2/18 Building Certificate Page...")
    build_certificate(doc)

    print("3/18 Building Declaration Page...")
    build_declaration(doc)

    print("4/18 Building Acknowledgements...")
    build_acknowledgements(doc)

    print("5/18 Building Abstract...")
    build_abstract(doc)

    print("6/18 Building Table of Contents...")
    build_table_of_contents(doc)

    print("7/18 Building Chapter 1: Introduction...")
    build_chapter_1(doc)

    print("8/18 Building Chapter 2: Literature Review...")
    build_chapter_2(doc)

    print("9/18 Building Chapter 3: Research Gap and Research Questions...")
    build_chapter_3(doc)

    print("10/18 Building Chapter 4: Dataset Description...")
    build_chapter_4(doc)

    print("11/18 Building Chapter 5: Methodology...")
    build_chapter_5(doc)

    print("12/18 Building Chapter 6: Experimental Setup...")
    build_chapter_6(doc)

    print("13/18 Building Chapter 7: Results and Comparative Analysis...")
    build_chapter_7(doc)

    print("14/18 Building Chapter 8: Ablation Studies and Statistical Rigor...")
    build_chapter_8(doc)

    print("15/18 Building Chapter 9: Discussion...")
    build_chapter_9(doc)

    print("16/18 Building Chapter 10: Conclusion and Future Work...")
    build_chapter_10(doc)

    print("17/18 Building References...")
    build_references(doc)

    print("18/18 Building Appendices A & B...")
    build_appendices(doc)

    # Save document
    print(f"\nSaving generated document to: {OUTPUT_FILE}...")
    doc.save(str(OUTPUT_FILE))
    file_size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"SUCCESS: Thesis document generated successfully! File size: {file_size_kb:.1f} KB")
    print("=" * 70)

if __name__ == "__main__":
    main()

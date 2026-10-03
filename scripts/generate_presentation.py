"""
generate_presentation.py
Generate PowerPoint presentation (.pptx) for BTech thesis defense.

Author: Tusher Tarafder
University: KIIT, Bhubaneswar
Supervisor: Prof. (Dr.) Sujata Dash
Title: Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT
Output: docs/thesis_defense.pptx
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml import parse_xml

# ==============================================================================
# COLOR PALETTE - Clean White & Professional Blue Theme
# ==============================================================================
COLOR_NAVY_DARK   = RGBColor(16, 37, 66)     # #102542 - Primary dark header / main titles
COLOR_ROYAL_BLUE  = RGBColor(26, 86, 160)    # #1A56A0 - Primary theme blue / card headers
COLOR_DEEP_BLUE   = RGBColor(12, 53, 106)    # #0C356A - Accent dark blue
COLOR_TEAL_ACCENT = RGBColor(0, 140, 170)    # #008CAA - Accent teal for metrics/tags
COLOR_CARD_BG     = RGBColor(245, 248, 252)  # #F5F8FC - Soft blue-gray card fill
COLOR_CARD_BG_ALT = RGBColor(238, 244, 250)  # #EEF4FA - Slightly deeper card fill
COLOR_CARD_BORDER = RGBColor(210, 222, 236)  # #D2DEEC - Subtle clean card border
COLOR_BORDER_HEX  = "D2DEEC"                 # Hex for XML borders
COLOR_BORDER_DARK = "1A56A0"                 # Hex for primary border
COLOR_TEXT_DARK   = RGBColor(24, 34, 48)     # #182230 - Main body text
COLOR_TEXT_MUTED  = RGBColor(90, 105, 125)   # #5A697D - Secondary / footer / labels
COLOR_WHITE       = RGBColor(255, 255, 255)  # #FFFFFF - Crisp white
COLOR_HIGHLIGHT   = RGBColor(20, 120, 80)    # #147850 - Success / highlight green
COLOR_ALERT       = RGBColor(190, 40, 40)    # #BE2828 - Warning / alert red
COLOR_ROW_ALT     = RGBColor(248, 250, 254)  # #F8FAFE - Table alternating row

FONT_NAME = "Calibri"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def set_cell_border(cell, color_hex=COLOR_BORDER_HEX, width="12700"):
    """Applies thin professional border to a table cell via oxml."""
    tcPr = cell._tc.get_or_add_tcPr()
    for edge in ('lnL', 'lnR', 'lnT', 'lnB'):
        tcPr.append(parse_xml(
            f'<a:{edge} xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            f'w="{width}" cmpd="s"><a:solidFill><a:srgbClr val="{color_hex}"/></a:solidFill></a:{edge}>'
        ))

def add_header(slide, title_text, category_text):
    """Adds a standard structured header with category breadcrumb and title."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.95))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    # Category breadcrumb
    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = FONT_NAME
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_ROYAL_BLUE
    p_cat.space_after = Pt(2)

    # Title
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_NAME
    p_title.font.size = Pt(26)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_NAVY_DARK

    # Subtle horizontal divider line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(1.42), Inches(11.733), Inches(0.02)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_CARD_BORDER
    line.line.color.rgb = COLOR_CARD_BORDER

def add_footer(slide, slide_num, total_slides=25):
    """Adds standard defense footer with author info and slide numbering."""
    footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.92), Inches(11.733), Inches(0.38))
    tf = footer_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.text = "Tusher Tarafder | BTech Thesis Defense (KIIT) | Supervisor: Prof. (Dr.) Sujata Dash"
    p.font.name = FONT_NAME
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_TEXT_MUTED

    # Slide number on right
    p_num = tf.add_paragraph()
    p_num.alignment = PP_ALIGN.RIGHT
    # In python-pptx, paragraphs in same textframe stack, so create dedicated right box for slide num
    slide_num_box = slide.shapes.add_textbox(Inches(10.5), Inches(6.92), Inches(2.033), Inches(0.38))
    tf_num = slide_num_box.text_frame
    tf_num.word_wrap = False
    tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
    p2 = tf_num.paragraphs[0]
    p2.text = f"Slide {slide_num} of {total_slides}"
    p2.alignment = PP_ALIGN.RIGHT
    p2.font.name = FONT_NAME
    p2.font.size = Pt(10)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_ROYAL_BLUE

def create_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
    """Creates a stylized card container with soft fill and thin border."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)
    return card

def add_bullet_list(text_frame, items, font_size=Pt(18), text_color=COLOR_TEXT_DARK, space_after=Pt(8)):
    """
    Renders structured bullet points.
    Items can be strings or tuples: (main_bullet, [sub_bullet_1, sub_bullet_2])
    """
    first = True
    for item in items:
        if isinstance(item, tuple):
            main_text, sub_items = item
            p = text_frame.paragraphs[0] if first else text_frame.add_paragraph()
            first = False
            p.text = "•  " + main_text
            p.font.name = FONT_NAME
            p.font.size = font_size
            p.font.bold = True
            p.font.color.rgb = text_color
            p.space_after = Pt(3)

            for sub in sub_items:
                p_sub = text_frame.add_paragraph()
                p_sub.text = "     -  " + sub
                p_sub.font.name = FONT_NAME
                p_sub.font.size = Pt(font_size.pt - 3)
                p_sub.font.color.rgb = COLOR_TEXT_MUTED if not sub.startswith("**") else text_color
                p_sub.space_after = Pt(3)
        else:
            p = text_frame.paragraphs[0] if first else text_frame.add_paragraph()
            first = False
            p.text = "•  " + item
            p.font.name = FONT_NAME
            p.font.size = font_size
            p.font.color.rgb = text_color
            p.space_after = space_after

def create_styled_table(slide, left, top, width, height, headers, rows_data, col_widths=None, highlight_rows=None, highlight_cols=None):
    """Creates a beautifully styled academic table with headers, borders, and alternating rows."""
    num_rows = len(rows_data) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
    table = table_shape.table

    if col_widths and len(col_widths) == num_cols:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = w

    # Format header row
    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_ROYAL_BLUE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        set_cell_border(cell, color_hex="102542", width="15000")
        tf = cell.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.08)
        tf.margin_right = Inches(0.08)
        tf.margin_top = Inches(0.06)
        tf.margin_bottom = Inches(0.06)
        p = tf.paragraphs[0]
        p.text = header
        p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
        p.font.name = FONT_NAME
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE

    # Format data rows
    for row_idx, row in enumerate(rows_data):
        actual_row = row_idx + 1
        is_highlighted = highlight_rows and (row_idx in highlight_rows)
        bg = COLOR_CARD_BG if is_highlighted else (COLOR_WHITE if row_idx % 2 == 0 else COLOR_ROW_ALT)

        for col_idx, val in enumerate(row):
            cell = table.cell(actual_row, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            set_cell_border(cell, color_hex=COLOR_BORDER_HEX, width="12700")
            tf = cell.text_frame
            tf.word_wrap = True
            tf.margin_left = Inches(0.08)
            tf.margin_right = Inches(0.08)
            tf.margin_top = Inches(0.05)
            tf.margin_bottom = Inches(0.05)
            p = tf.paragraphs[0]
            p.text = str(val)
            p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
            p.font.name = FONT_NAME
            p.font.size = Pt(12)
            p.font.bold = bool(is_highlighted or (highlight_cols and col_idx in highlight_cols))
            if is_highlighted:
                p.font.color.rgb = COLOR_NAVY_DARK
            else:
                p.font.color.rgb = COLOR_TEXT_DARK

    return table_shape

def add_metric_callout(slide, left, top, width, height, stat_value, stat_label, subtext=""):
    """Creates a punchy metric stat card."""
    create_card(slide, left, top, width, height, bg_color=COLOR_WHITE, border_color=COLOR_ROYAL_BLUE)
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.1)
    tf.margin_bottom = Inches(0.1)

    p0 = tf.paragraphs[0]
    p0.text = stat_value
    p0.alignment = PP_ALIGN.CENTER
    p0.font.name = FONT_NAME
    p0.font.size = Pt(28)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_ROYAL_BLUE
    p0.space_after = Pt(2)

    p1 = tf.add_paragraph()
    p1.text = stat_label
    p1.alignment = PP_ALIGN.CENTER
    p1.font.name = FONT_NAME
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY_DARK
    p1.space_after = Pt(2)

    if subtext:
        p2 = tf.add_paragraph()
        p2.text = subtext
        p2.alignment = PP_ALIGN.CENTER
        p2.font.name = FONT_NAME
        p2.font.size = Pt(10)
        p2.font.color.rgb = COLOR_TEXT_MUTED

# ==============================================================================
# MAIN PRESENTATION BUILDER
# ==============================================================================
def build_thesis_presentation(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    print("Building slides for BTech Thesis Defense presentation...")

    # --------------------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # --------------------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)

    # Background banner / card
    bg_rect = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg_rect.fill.solid()
    bg_rect.fill.fore_color.rgb = COLOR_CARD_BG
    bg_rect.line.fill.background()

    # Top accent bar
    top_bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.35))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_NAVY_DARK
    top_bar.line.fill.background()

    # Main Center Card
    main_card = create_card(slide1, Inches(0.9), Inches(0.7), Inches(11.533), Inches(6.1), bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)

    # University Badge Header
    badge_box = slide1.shapes.add_textbox(Inches(1.2), Inches(0.9), Inches(10.933), Inches(0.5))
    tf_b = badge_box.text_frame
    tf_b.word_wrap = True
    p_b = tf_b.paragraphs[0]
    p_b.text = "KALINGA INSTITUTE OF INDUSTRIAL TECHNOLOGY (KIIT) • SCHOOL OF COMPUTER ENGINEERING"
    p_b.font.name = FONT_NAME
    p_b.font.size = Pt(12)
    p_b.font.bold = True
    p_b.font.color.rgb = COLOR_ROYAL_BLUE

    # Title & Subtitle Box
    title_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.4), Inches(10.933), Inches(2.2))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_top = tf_t.margin_left = tf_t.margin_right = 0

    p_t = tf_t.paragraphs[0]
    p_t.text = "Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT"
    p_t.font.name = FONT_NAME
    p_t.font.size = Pt(32)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_NAVY_DARK
    p_t.space_after = Pt(10)

    p_sub = tf_t.add_paragraph()
    p_sub.text = "A Rigorous Empirical Benchmark, Multi-Dimensional Evaluation, and Leakage Prevention Framework"
    p_sub.font.name = FONT_NAME
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = COLOR_TEAL_ACCENT
    p_sub.space_after = Pt(14)

    # Divider
    div = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(3.7), Inches(10.933), Inches(0.02))
    div.fill.solid()
    div.fill.fore_color.rgb = COLOR_CARD_BORDER
    div.line.fill.background()

    # Left Column: Candidate & Degree
    left_meta = slide1.shapes.add_textbox(Inches(1.2), Inches(3.9), Inches(5.2), Inches(2.5))
    tf_lm = left_meta.text_frame
    tf_lm.word_wrap = True
    tf_lm.margin_top = tf_lm.margin_left = tf_lm.margin_right = 0

    p_cand_lbl = tf_lm.paragraphs[0]
    p_cand_lbl.text = "DEGREE CANDIDATE:"
    p_cand_lbl.font.size = Pt(11)
    p_cand_lbl.font.bold = True
    p_cand_lbl.font.color.rgb = COLOR_ROYAL_BLUE
    p_cand_lbl.space_after = Pt(2)

    p_name = tf_lm.add_paragraph()
    p_name.text = "Tusher Tarafder"
    p_name.font.size = Pt(20)
    p_name.font.bold = True
    p_name.font.color.rgb = COLOR_NAVY_DARK
    p_name.space_after = Pt(4)

    p_deg = tf_lm.add_paragraph()
    p_deg.text = "Bachelor of Technology in Computer Science & Engineering\nSchool of Computer Engineering, KIIT Deemed to be University"
    p_deg.font.size = Pt(14)
    p_deg.font.color.rgb = COLOR_TEXT_MUTED

    # Right Column: Supervisor & Date
    right_meta = slide1.shapes.add_textbox(Inches(6.8), Inches(3.9), Inches(5.3), Inches(2.5))
    tf_rm = right_meta.text_frame
    tf_rm.word_wrap = True
    tf_rm.margin_top = tf_rm.margin_left = tf_rm.margin_right = 0

    p_sup_lbl = tf_rm.paragraphs[0]
    p_sup_lbl.text = "THESIS SUPERVISOR:"
    p_sup_lbl.font.size = Pt(11)
    p_sup_lbl.font.bold = True
    p_sup_lbl.font.color.rgb = COLOR_ROYAL_BLUE
    p_sup_lbl.space_after = Pt(2)

    p_sup = tf_rm.add_paragraph()
    p_sup.text = "Prof. (Dr.) Sujata Dash"
    p_sup.font.size = Pt(20)
    p_sup.font.bold = True
    p_sup.font.color.rgb = COLOR_NAVY_DARK
    p_sup.space_after = Pt(4)

    p_sup_aff = tf_rm.add_paragraph()
    p_sup_aff.text = "Senior Professor, School of Computer Engineering\nDate of Defense: October 2026 • Bhubaneswar, Odisha, India"
    p_sup_aff.font.size = Pt(14)
    p_sup_aff.font.color.rgb = COLOR_TEXT_MUTED

    # --------------------------------------------------------------------------
    # SLIDE 2: OUTLINE / DEFENSE AGENDA
    # --------------------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Presentation Outline & Defense Agenda", "STRUCTURE OF THE DISSERTATION")
    add_footer(slide2, 2)

    # 2 Column Cards
    card_l = create_card(slide2, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l = slide2.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    items_l = [
        ("I. Context & Research Foundations", [
            "Industrial IoT & Predictive Maintenance Motivation",
            "Problem Statement: The Unproven Deep Learning Assumption",
            "Literature Review & Refined Research Gap",
            "Core Research Questions (RQ1 through RQ7)"
        ]),
        ("II. Benchmark Dataset & Leakage Controls", [
            "NASA C-MAPSS Turbofan Degradation Benchmark",
            "5-Pillar Data Leakage Prevention Protocol"
        ]),
        ("III. Unsupervised Detection Frameworks", [
            "Statistical Z-Score Baseline (38 parameters)",
            "Isolation Forest & One-Class Support Vector Machine",
            "Fully Connected Autoencoder (FC-AE: 8,163 parameters)",
            "Recurrent LSTM Autoencoder (LSTM-AE: 116,723 parameters)"
        ])
    ]
    add_bullet_list(tf_l, items_l, font_size=Pt(17))

    card_r = create_card(slide2, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r = slide2.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    items_r = [
        ("IV. Empirical Findings & Breakthroughs", [
            "FD001 Comprehensive Benchmark Evaluation",
            "Finding 1: Surprising Power of Simple Baselines",
            "Finding 2: LSTM-AE 5x Earlier Warning (50 vs 10 cycles)",
            "Finding 3: The Classification-Lead Time Paradox"
        ]),
        ("V. Robustness & Multi-Regime Analysis", [
            "Multi-Condition Degradation Performance (FD002/FD004)",
            "Cross-Condition & Cross-Fault Domain Transfer Collapse"
        ]),
        ("VI. Ablation Studies & Synthesis", [
            "Multi-Seed Reproducibility (5 Seeds) & Significance Tests",
            "Sequence Length Sensitivity Analysis (Seq=10 Optimal)",
            "Industrial Method Selection Matrix (5 Scenarios)",
            "Limitations, Future Research Directions & Conclusions"
        ])
    ]
    add_bullet_list(tf_r, items_r, font_size=Pt(17))

    # --------------------------------------------------------------------------
    # SLIDE 3: INTRODUCTION & MOTIVATION
    # --------------------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Predictive Maintenance in Industrial IoT", "INTRODUCTION & MOTIVATION")
    add_footer(slide3, 3)

    # 3 Structured Panels
    col_w = Inches(3.75)
    gap = Inches(0.24)

    # Panel 1: Industrial IoT
    c1 = create_card(slide3, Inches(0.8), Inches(1.65), col_w, Inches(5.0))
    tb1 = slide3.shapes.add_textbox(Inches(0.95), Inches(1.8), col_w - Inches(0.3), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "1. Industrial IoT Telemetry"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_ROYAL_BLUE
    p1.space_after = Pt(8)
    add_bullet_list(tf1, [
        "Modern industrial assets (turbofan engines, wind turbines, centrifugal pumps) operate with dozens of telemetry sensors.",
        "Sensors capture high-frequency physical dynamics: temperatures, pressures, rotor speeds, vibrations.",
        "Critical operational challenge: Real-time degradation tracking under severe operating stresses."
    ], font_size=Pt(15), text_color=COLOR_TEXT_DARK)

    # Panel 2: The Maintenance Imperative
    c2 = create_card(slide3, Inches(0.8) + col_w + gap, Inches(1.65), col_w, Inches(5.0))
    tb2 = slide3.shapes.add_textbox(Inches(0.95) + col_w + gap, Inches(1.8), col_w - Inches(0.3), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "2. Cost of Unplanned Downtime"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_ROYAL_BLUE
    p2.space_after = Pt(8)
    add_bullet_list(tf2, [
        "Reactive Maintenance: Run-to-failure causes catastrophic destruction, safety hazards, and multi-million dollar outages.",
        "Scheduled Maintenance: Calendar-based overhauls replace healthy components prematurely, wasting 30-40% useful life.",
        "Predictive Maintenance (PdM): Condition-based intervention before failure saves $50B+ across global industries."
    ], font_size=Pt(15), text_color=COLOR_TEXT_DARK)

    # Panel 3: The Machine Learning Dilemma
    c3 = create_card(slide3, Inches(0.8) + (col_w + gap)*2, Inches(1.65), col_w, Inches(5.0))
    tb3 = slide3.shapes.add_textbox(Inches(0.95) + (col_w + gap)*2, Inches(1.8), col_w - Inches(0.3), Inches(4.7))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p3 = tf3.paragraphs[0]
    p3.text = "3. The Unsupervised Imperative"
    p3.font.size = Pt(18)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_ROYAL_BLUE
    p3.space_after = Pt(8)
    add_bullet_list(tf3, [
        "Label Scarcity: Real machines run normally >99.9% of their lifespan. Catastrophic failure records are virtually non-existent.",
        "Supervised RUL algorithms fail in real deployment due to zero annotated training failures.",
        "Unsupervised anomaly detection is the only scalable paradigm: Learn nominal health; flag deviations as degradation."
    ], font_size=Pt(15), text_color=COLOR_TEXT_DARK)

    # --------------------------------------------------------------------------
    # SLIDE 4: PROBLEM STATEMENT
    # --------------------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Problem Statement: The Unproven DL Assumption", "THE CORE RESEARCH CHALLENGE")
    add_footer(slide4, 4)

    card_l4 = create_card(slide4, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l4 = slide4.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l4 = tb_l4.text_frame
    tf_l4.word_wrap = True
    items_l4 = [
        ("The Pervasive Deep Learning Assumption", [
            "Current literature heavily promotes complex deep learning architectures (Transformers, deep LSTMs, VAEs) as indispensable.",
            "Default presumption: Multi-sensor industrial telemetry requires highly parameterized non-linear networks.",
            "Baselines are often omitted, untuned, or used as token strawmen."
        ]),
        ("Operational Consequences in Production", [
            "Massive Parameter Bloat: 100,000+ parameters vs. 38 parameters.",
            "Prohibitive Training & Retraining Costs: Hundreds of GPU hours for edge IIoT gateways.",
            "Total Lack of Physical Interpretability: Black-box latent spaces cannot explain which subsystem is degrading."
        ])
    ]
    add_bullet_list(tf_l4, items_l4, font_size=Pt(17))

    card_r4 = create_card(slide4, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r4 = slide4.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r4 = tb_r4.text_frame
    tf_r4.word_wrap = True
    items_r4 = [
        ("The Three Industrial Dilemmas", [
            "1. Labeled Data Scarcity: Operators do not possess run-to-failure trajectories for supervised model training.",
            "2. Extreme Cost of False Alarms: Unscheduled aircraft grounding or turbine inspection costs over $100,000 per spurious alert.",
            "3. Disconnect from Physical Reality: Point-wise evaluation metrics (F1, Accuracy) treat cycle-by-cycle classification identically, ignoring lead time."
        ]),
        ("Core Research Hypothesis", [
            "Under rigorous, leak-free experimental conditions, simpler statistical and classical machine learning models can rival deep neural networks in detection accuracy, while deep recurrent models offer advantages strictly in temporal early warning."
        ])
    ]
    add_bullet_list(tf_r4, items_r4, font_size=Pt(17))

    # --------------------------------------------------------------------------
    # SLIDE 5: RESEARCH GAP
    # --------------------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Research Gap: Lack of Rigorous Multi-Dimensional Evaluation", "LITERATURE REVIEW & SYNTHESIS")
    add_footer(slide5, 5)

    # Top summary callout banner
    banner = create_card(slide5, Inches(0.8), Inches(1.65), Inches(11.733), Inches(1.1), bg_color=COLOR_WHITE, border_color=COLOR_ROYAL_BLUE)
    tb_b = slide5.shapes.add_textbox(Inches(1.0), Inches(1.72), Inches(11.333), Inches(0.95))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    p_b0 = tf_b.paragraphs[0]
    p_b0.text = "THE EVIDENCE-BASED RESEARCH GAP IN C-MAPSS ANOMALY DETECTION:"
    p_b0.font.size = Pt(11)
    p_b0.font.bold = True
    p_b0.font.color.rgb = COLOR_ROYAL_BLUE
    p_b1 = tf_b.add_paragraph()
    p_b1.text = "While individual aspects of unsupervised anomaly detection have been studied in isolation, literature lacks a unified, methodologically rigorous study evaluating statistical, classical, and deep models side-by-side under strict leakage prevention, multi-seed statistical testing, and dual physical-lead-time evaluation."
    p_b1.font.size = Pt(14)
    p_b1.font.color.rgb = COLOR_NAVY_DARK

    # 4 Flaws of Prior Literature
    w4 = Inches(2.76)
    gap4 = Inches(0.23)

    flaws = [
        ("Flaw 1: Weak Baselines", [
            "Papers compare new deep models only against other deep models.",
            "Rarely test a well-tuned statistical z-score or Isolation Forest.",
            "Assumes deep learning superiority without verification."
        ]),
        ("Flaw 2: Pervasive Leakage", [
            "Random timestep splits mix early & degraded cycles of same unit.",
            "Normalization fit on entire dataset including test units.",
            "Thresholds tuned post-hoc on test data to inflate F1/AUC."
        ]),
        ("Flaw 3: Single Condition", [
            ">70% of C-MAPSS papers test only on FD001 (sea-level, 1 condition).",
            "Ignores operational shifts (6 flight regimes in FD002/FD004).",
            "Models fail when deployed across varying regimes."
        ]),
        ("Flaw 4: Aggregate Metrics", [
            "Only report dataset-wide F1 / ROC-AUC point classification.",
            "Ignore detection lead time: How early before failure did it alert?",
            "Conceals false alarm burdens on healthy operational phases."
        ])
    ]

    for idx, (title_flaw, bullets_flaw) in enumerate(flaws):
        left_pos = Inches(0.8) + (w4 + gap4) * idx
        card_f = create_card(slide5, left_pos, Inches(2.95), w4, Inches(3.7))
        tb_f = slide5.shapes.add_textbox(left_pos + Inches(0.12), Inches(3.1), w4 - Inches(0.24), Inches(3.4))
        tf_f = tb_f.text_frame
        tf_f.word_wrap = True
        p_ft = tf_f.paragraphs[0]
        p_ft.text = title_flaw
        p_ft.font.size = Pt(15)
        p_ft.font.bold = True
        p_ft.font.color.rgb = COLOR_ROYAL_BLUE
        p_ft.space_after = Pt(8)
        add_bullet_list(tf_f, bullets_flaw, font_size=Pt(13), text_color=COLOR_TEXT_DARK)

    # --------------------------------------------------------------------------
    # SLIDE 6: RESEARCH QUESTIONS (RQ1 - RQ7)
    # --------------------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Research Questions (RQ1 - RQ7)", "FORMAL INVESTIGATION SCOPE")
    add_footer(slide6, 6)

    # 2 Column layout of RQs
    c_l6 = create_card(slide6, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l6 = slide6.shapes.add_textbox(Inches(0.95), Inches(1.8), Inches(5.4), Inches(4.7))
    tf_l6 = tb_l6.text_frame
    tf_l6.word_wrap = True

    rqs_left = [
        ("RQ1: Baseline vs. Deep Learning Under Fair Conditions", [
            "When Isolation Forest, Statistical Detector, and Autoencoders are evaluated under strictly identical leak-free splits, how does detection performance compare?"
        ]),
        ("RQ2: Does Explicit Temporal Modeling Improve Detection?", [
            "Does an LSTM Autoencoder modeling temporal sequences capture degradation better than a feedforward Autoencoder treating cycles independently?"
        ]),
        ("RQ3: How Does Detection Behave Across Operating Regimes?", [
            "How does performance degrade when moving from single-condition (FD001) to multi-condition (FD002/FD004) and multi-fault (FD003) datasets?"
        ]),
        ("RQ4: How Early Can Each Method Detect Degradation?", [
            "How many cycles before failure does the first sustained warning trigger, and how does lead time distribute across engines?"
        ])
    ]
    add_bullet_list(tf_l6, rqs_left, font_size=Pt(15))

    c_r6 = create_card(slide6, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r6 = slide6.shapes.add_textbox(Inches(6.95), Inches(1.8), Inches(5.433), Inches(4.7))
    tf_r6 = tb_r6.text_frame
    tf_r6.word_wrap = True

    rqs_right = [
        ("RQ5: What is the Sensitivity vs. False Alarm Trade-off?", [
            "How does validation threshold calibration affect operational precision, detection persistence, and false alarm rate?"
        ]),
        ("RQ6: Do Models Generalize to Completely Unseen Engines?", [
            "When models are evaluated on isolated test engines (engine-level split), does performance degrade or demonstrate genuine pattern learning?"
        ]),
        ("RQ7: Is Added Complexity Computationally Justified?", [
            "Does the marginal performance gain from complex models (LSTM-AE > FC-AE > IF > Stat) justify a 14x-3,000x parameter and training cost increase?"
        ]),
        ("Guiding Principle of Dissertation", [
            "We do not advocate for novel complexity. We pursue objective empirical truth: identifying precisely where each method succeeds and fails."
        ])
    ]
    add_bullet_list(tf_r6, rqs_right, font_size=Pt(15))

    # --------------------------------------------------------------------------
    # SLIDE 7: DATASET: C-MAPSS OVERVIEW
    # --------------------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Dataset: NASA C-MAPSS Benchmark", "BENCHMARK SPECIFICATION")
    add_footer(slide7, 7)

    # Top description card
    top_c7 = create_card(slide7, Inches(0.8), Inches(1.65), Inches(11.733), Inches(1.3))
    tb_t7 = slide7.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(11.333), Inches(1.1))
    tf_t7 = tb_t7.text_frame
    tf_t7.word_wrap = True
    add_bullet_list(tf_t7, [
        "Commercial Modular Aero-Propulsion System Simulation (NASA Ames Research Center): Realistic turbofan engine run-to-failure telemetry.",
        "21 sensor measurements (temperatures, pressures, fan speeds) + 3 operational settings per cycle.",
        "Systematic Feature Selection: 7 invariant features (zero variance across cycles: s1, s5, s10, s16, s18, s19, setting3) removed. 14 active sensors retained."
    ], font_size=Pt(14))

    # 4 Subsets Table
    headers_7 = ["Subset", "Train Units", "Test Units", "Flight Conditions", "Fault Modes", "Complexity Level", "Primary Failure Mechanism"]
    data_7 = [
        ["FD001", "100", "100", "1 (Sea Level)", "1 (HPC Degradation)", "Low (Single Regime)", "High-Pressure Compressor Wear"],
        ["FD002", "260", "259", "6 Operating Regimes", "1 (HPC Degradation)", "High (Multi-Regime)", "HPC Wear + Flight Envelope Variation"],
        ["FD003", "100", "100", "1 (Sea Level)", "2 (HPC + Fan)", "Medium (Dual Fault)", "HPC Wear & Fan Blade Degradation"],
        ["FD004", "249", "248", "6 Operating Regimes", "2 (HPC + Fan)", "Extreme (Combined)", "HPC + Fan Degradation + 6 Regimes"]
    ]
    widths_7 = [Inches(1.1), Inches(1.2), Inches(1.2), Inches(1.8), Inches(1.7), Inches(2.1), Inches(2.633)]
    create_styled_table(slide7, Inches(0.8), Inches(3.1), Inches(11.733), Inches(2.2), headers_7, data_7, col_widths=widths_7, highlight_rows=[0])

    # Bottom notes
    bot_c7 = create_card(slide7, Inches(0.8), Inches(5.45), Inches(11.733), Inches(1.25), bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER)
    tb_b7 = slide7.shapes.add_textbox(Inches(1.0), Inches(5.52), Inches(11.333), Inches(1.05))
    tf_b7 = tb_b7.text_frame
    tf_b7.word_wrap = True
    p_b7 = tf_b7.paragraphs[0]
    p_b7.text = "RESEARCH METHODOLOGY HIGHLIGHT ON SUBSET COMPLEXITY:"
    p_b7.font.size = Pt(11)
    p_b7.font.bold = True
    p_b7.font.color.rgb = COLOR_ROYAL_BLUE
    p_b7_sub = tf_b7.add_paragraph()
    p_b7_sub.text = "FD001 serves as the primary controlled benchmark for baseline vs. deep learning comparison. FD002 and FD004 test model adaptability under severe operational regime shifts. FD003 enables cross-fault domain generalization analysis (testing models trained on HPC wear against novel Fan degradation)."
    p_b7_sub.font.size = Pt(13)
    p_b7_sub.font.color.rgb = COLOR_TEXT_DARK

    # --------------------------------------------------------------------------
    # SLIDE 8: METHODOLOGY OVERVIEW
    # --------------------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Methodology Overview: 5-Method Benchmark", "SYSTEM ARCHITECTURE")
    add_footer(slide8, 8)

    # 5 Pipeline Stages as horizontal cards
    w_p = Inches(2.15)
    g_p = Inches(0.24)
    stages = [
        ("1. Data Pipeline", "Engine-level splitting (70/15/15). Healthy 70% normal cutoff. Invariant feature filtering."),
        ("2. Normalization", "Train-only statistics (mean/std). Zero test exposure. Scaler parameters frozen."),
        ("3. Model Training", "5 unsupervised paradigms: Statistical, Isolation Forest, OC-SVM, FC-AE, LSTM-AE."),
        ("4. Thresholding", "Calibrated exclusively on validation set (95th percentile & k-sigma). Zero test peeking."),
        ("5. Dual Metrics", "Point classification (F1, AUC, FAR) + Asset early warning (Lead Time, 5-cycle persistence).")
    ]
    for idx, (st_title, st_desc) in enumerate(stages):
        l_pos = Inches(0.8) + (w_p + g_p) * idx
        card_p = create_card(slide8, l_pos, Inches(1.65), w_p, Inches(2.2), bg_color=COLOR_WHITE, border_color=COLOR_ROYAL_BLUE)
        tb_p = slide8.shapes.add_textbox(l_pos + Inches(0.08), Inches(1.75), w_p - Inches(0.16), Inches(2.0))
        tf_p = tb_p.text_frame
        tf_p.word_wrap = True
        p_pt = tf_p.paragraphs[0]
        p_pt.text = st_title
        p_pt.font.size = Pt(14)
        p_pt.font.bold = True
        p_pt.font.color.rgb = COLOR_ROYAL_BLUE
        p_pt.space_after = Pt(4)
        p_pd = tf_p.add_paragraph()
        p_pd.text = st_desc
        p_pd.font.size = Pt(12)
        p_pd.font.color.rgb = COLOR_TEXT_DARK

    # Bottom comparison summary of the 5 models
    bot_c8 = create_card(slide8, Inches(0.8), Inches(4.1), Inches(11.733), Inches(2.6))
    tb_b8 = slide8.shapes.add_textbox(Inches(1.0), Inches(4.25), Inches(11.333), Inches(2.3))
    tf_b8 = tb_b8.text_frame
    tf_b8.word_wrap = True
    p_b8_title = tf_b8.paragraphs[0]
    p_b8_title.text = "THE 5 EVALUATED UNSUPERVISED PARADIGMS:"
    p_b8_title.font.size = Pt(14)
    p_b8_title.font.bold = True
    p_b8_title.font.color.rgb = COLOR_NAVY_DARK
    p_b8_title.space_after = Pt(6)

    model_overview = [
        "Statistical Z-Score: Zero-training parametric distance detector (38 scalar parameters; instant closed-form computation).",
        "Isolation Forest: Non-parametric ensemble of random decision trees isolating anomalies in sparse regions.",
        "One-Class SVM: Kernel-based maximum-margin boundary construction in high-dimensional reproducing Hilbert space.",
        "Fully Connected Autoencoder (FC-AE): 7-layer symmetric neural bottleneck manifold reconstruction (8,163 parameters).",
        "LSTM Recurrent Autoencoder (LSTM-AE): 2-layer sequence-to-sequence temporal encoder-decoder (116,723 parameters)."
    ]
    add_bullet_list(tf_b8, model_overview, font_size=Pt(14))

    # --------------------------------------------------------------------------
    # SLIDE 9: STATISTICAL THRESHOLD DETECTOR
    # --------------------------------------------------------------------------
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Method 1: Statistical Threshold Detector", "PARAMETRIC BASELINE")
    add_footer(slide9, 9)

    card_l9 = create_card(slide9, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l9 = slide9.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l9 = tb_l9.text_frame
    tf_l9.word_wrap = True
    items_l9 = [
        ("Mathematical Formulation", [
            "Sensor Z-Score: Standardization using healthy baseline training distribution:",
            "     z_{i,t} = |x_{i,t} - mu_{i,train}| / sigma_{i,train}",
            "Mean Composite Score: S_mean(t) = (1/M) * sum(z_{i,t})",
            "Max Composite Score: S_max(t) = max_i(z_{i,t})",
            "Where M is the number of active sensors (M = 19 or 14)."
        ]),
        ("Detection Rule & Calibration", [
            "Threshold tau calibrated on validation healthy set:",
            "Percentile Rule: tau = Percentile_95(S_val)",
            "K-Sigma Rule: tau = mu_S + k * sigma_S"
        ])
    ]
    add_bullet_list(tf_l9, items_l9, font_size=Pt(17))

    card_r9 = create_card(slide9, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r9 = slide9.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r9 = tb_r9.text_frame
    tf_r9.word_wrap = True
    items_r9 = [
        ("Key Architectural Strengths", [
            "Total Parameters: Exactly 38 scalar values (19 feature means + 19 standard deviations).",
            "Training Time: 0.00 seconds (deterministic single-pass aggregation).",
            "Inference Latency: < 0.01 ms per cycle (vectorized arithmetic; trivial microcontroller implementation).",
            "Full Interpretability: Operators immediately identify which sensor deviated."
        ]),
        ("Theoretical Limitations", [
            "Assumes static operational regime: Fails completely when machine operating speed or altitude varies.",
            "Ignores non-linear sensor correlations.",
            "Point-by-point evaluation without temporal context."
        ])
    ]
    add_bullet_list(tf_r9, items_r9, font_size=Pt(17))

    # --------------------------------------------------------------------------
    # SLIDE 10: ISOLATION FOREST & ONE-CLASS SVM
    # --------------------------------------------------------------------------
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Methods 2 & 3: Isolation Forest & One-Class SVM", "CLASSICAL MACHINE LEARNING BASELINES")
    add_footer(slide10, 10)

    # Left: Isolation Forest
    card_l10 = create_card(slide10, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l10 = slide10.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l10 = tb_l10.text_frame
    tf_l10.word_wrap = True
    items_l10 = [
        ("Isolation Forest (Liu et al.)", [
            "Core Principle: Isolates anomalies rather than profiling normal points.",
            "Tree Partitioning: Recursively partitions feature space using random axis-aligned splits.",
            "Anomaly Score Formula:",
            "     s(x, n) = 2^(- E[h(x)] / c(n))",
            "Where h(x) is path length to terminal node and c(n) is average path length of unsuccessful search.",
            "Configuration: 100 trees, 256 sub-sample size.",
            "Fit Time: 0.72s. Non-parametric, highly scalable."
        ])
    ]
    add_bullet_list(tf_l10, items_l10, font_size=Pt(17))

    # Right: One-Class SVM
    card_r10 = create_card(slide10, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r10 = slide10.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r10 = tb_r10.text_frame
    tf_r10.word_wrap = True
    items_r10 = [
        ("One-Class SVM (Schölkopf et al.)", [
            "Core Principle: Maps healthy training data into high-dimensional feature space via kernel function.",
            "Kernel Formulation: Radial Basis Function (RBF):",
            "     K(x, x') = exp(-gamma * ||x - x'||^2)",
            "Optimization: Constructs maximal-margin hyperplane separating normal points from the origin in Hilbert space.",
            "Configuration: RBF kernel, gamma='scale', nu=0.05.",
            "Fit Time: 0.22s. Memory-efficient support vector representation; sharp non-linear envelope."
        ])
    ]
    add_bullet_list(tf_r10, items_r10, font_size=Pt(17))

    # --------------------------------------------------------------------------
    # SLIDE 11: FEEDFORWARD AUTOENCODER (FC-AE)
    # --------------------------------------------------------------------------
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Method 4: Fully Connected Autoencoder (FC-AE)", "DEEP NON-TEMPORAL ARCHITECTURE")
    add_footer(slide11, 11)

    card_l11 = create_card(slide11, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l11 = slide11.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l11 = tb_l11.text_frame
    tf_l11.word_wrap = True
    items_l11 = [
        ("Architecture Specification", [
            "Symmetric 7-Layer Bottleneck Compression:",
            "     Input (19) -> Dense(64, ReLU)",
            "     -> Dense(32, ReLU) -> Bottleneck(16, ReLU)",
            "     -> Dense(32, ReLU) -> Dense(64, ReLU)",
            "     -> Output (19, Linear Reconstruction)",
            "Parameter Count: Exactly 8,163 trainable weights & biases.",
            "Loss Function: Mean Squared Error (MSE):",
            "     L_rec = (1/M) * sum(x_{i,t} - x_hat_{i,t})^2"
        ])
    ]
    add_bullet_list(tf_l11, items_l11, font_size=Pt(17))

    card_r11 = create_card(slide11, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r11 = slide11.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r11 = tb_r11.text_frame
    tf_r11.word_wrap = True
    items_r11 = [
        ("Operational Mechanics & Scoring", [
            "Training Optimization: Adam optimizer (lr=1e-3), batch size 64, early stopping (patience=10) on validation loss.",
            "Fit Time: 42.33s on CPU/GPU.",
            "Anomaly Scoring: Point-wise reconstruction residual:",
            "     Score(t) = ||x_t - x_hat_t||_2^2",
            "Key Strength: Successfully learns non-linear correlations across multi-sensor combinations.",
            "Limitation: Treats each cycle independently; blind to temporal rate of change and sequential drift."
        ])
    ]
    add_bullet_list(tf_r11, items_r11, font_size=Pt(17))

    # --------------------------------------------------------------------------
    # SLIDE 12: LSTM AUTOENCODER (LSTM-AE)
    # --------------------------------------------------------------------------
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "Method 5: LSTM Recurrent Autoencoder (LSTM-AE)", "DEEP TEMPORAL SEQUENCE MODEL")
    add_footer(slide12, 12)

    card_l12 = create_card(slide12, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l12 = slide12.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l12 = tb_l12.text_frame
    tf_l12.word_wrap = True
    items_l12 = [
        ("Architecture Specification", [
            "Sequence-to-Sequence Recurrent Bottleneck:",
            "     Input Sequence: (Batch, SeqLen=30, Features=19)",
            "     Encoder Layer 1: LSTM (hidden=64, return_seq=True)",
            "     Encoder Layer 2: LSTM (hidden=32, return_seq=False)",
            "     Latent Space: 32-dimensional context vector",
            "     Decoder: RepeatVector(T) -> LSTM(32) -> LSTM(64)",
            "     TimeDistributed(Dense(19, Linear))",
            "Parameter Count: Exactly 116,723 trainable parameters (~14.3x larger than FC-AE; 3,071x larger than Statistical)."
        ])
    ]
    add_bullet_list(tf_l12, items_l12, font_size=Pt(17))

    card_r12 = create_card(slide12, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r12 = slide12.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r12 = tb_r12.text_frame
    tf_r12.word_wrap = True
    items_r12 = [
        ("Training & Theoretical Advantage", [
            "Sliding Window Generation: Window size T=30 cycles, stride 1; strict engine boundary containment.",
            "Optimization: Adam (lr=1e-3), 60 epochs, MSE sequence loss.",
            "Fit Time: 491.88s (~8.2 minutes) — 680x slower than IF.",
            "Temporal Hypothesis: By encoding temporal dynamics, the model learns the trajectory of health over time, capturing rate-of-wear rather than instantaneous value out-of-bounds."
        ])
    ]
    add_bullet_list(tf_r12, items_r12, font_size=Pt(17))

    # --------------------------------------------------------------------------
    # SLIDE 13: LEAKAGE PREVENTION - 5 MEASURES
    # --------------------------------------------------------------------------
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "Data Leakage Prevention: 5 Integrity Measures", "METHODOLOGICAL RIGOR")
    add_footer(slide13, 13)

    w_lp = Inches(2.15)
    g_lp = Inches(0.24)
    leakage_pillars = [
        ("1. Engine-Level Split", [
            "70% Train, 15% Val, 15% Test strictly by engine ID.",
            "Zero mixing of observations across splits.",
            "Models never see future or past cycles of test engines."
        ]),
        ("2. Train-Only Scaler", [
            "Normalization (mean/std) fit strictly on training units.",
            "Zero test or validation data in statistics.",
            "Scaler frozen before transforming test sets."
        ]),
        ("3. Normal Life Cutoff", [
            "Training restricted to initial 70% of engine lifespans.",
            "Degraded cycles strictly excluded from training.",
            "Guarantees true unsupervised anomaly baseline."
        ]),
        ("4. Window Boundaries", [
            "Sliding windows clipped at engine transitions.",
            "Zero cross-engine sequence stitching.",
            "Window contains only past and current timesteps."
        ]),
        ("5. Val Thresholding", [
            "All thresholds calibrated exclusively on validation set.",
            "Zero threshold tuning on test data.",
            "Completely unseen test set evaluation."
        ])
    ]

    for idx, (p_title, p_bullets) in enumerate(leakage_pillars):
        l_pos = Inches(0.8) + (w_lp + g_lp) * idx
        card_lp = create_card(slide13, l_pos, Inches(1.65), w_lp, Inches(5.0))
        tb_lp = slide13.shapes.add_textbox(l_pos + Inches(0.1), Inches(1.8), w_lp - Inches(0.2), Inches(4.7))
        tf_lp = tb_lp.text_frame
        tf_lp.word_wrap = True
        p_lpt = tf_lp.paragraphs[0]
        p_lpt.text = p_title
        p_lpt.font.size = Pt(15)
        p_lpt.font.bold = True
        p_lpt.font.color.rgb = COLOR_ROYAL_BLUE
        p_lpt.space_after = Pt(8)
        add_bullet_list(tf_lp, p_bullets, font_size=Pt(13), text_color=COLOR_TEXT_DARK)

    # --------------------------------------------------------------------------
    # SLIDE 14: RESULTS: FD001 COMPREHENSIVE COMPARISON TABLE
    # --------------------------------------------------------------------------
    slide14 = prs.slides.add_slide(blank_layout)
    add_header(slide14, "Results: FD001 Comprehensive Model Benchmark", "BENCHMARK EVALUATION")
    add_footer(slide14, 14)

    headers_14 = ["Model", "Precision", "Recall", "F1 Score", "FAR", "ROC-AUC", "PR-AUC", "Det. Rate", "Lead Time", "Fit Time", "Parameters"]
    data_14 = [
        ["Statistical Mean", "1.000", "0.430", "0.602", "0.000", "0.984", "0.932", "93.3%", "10.5 cyc", "0.00s", "38"],
        ["Statistical Max", "0.691", "0.288", "0.407", "0.022", "0.949", "0.749", "33.3%", "24.4 cyc", "0.00s", "38"],
        ["Isolation Forest", "1.000", "0.378", "0.549", "0.000", "0.976", "0.905", "73.3%", "11.1 cyc", "0.72s", "Non-param"],
        ["One-Class SVM", "0.995", "0.417", "0.588", "0.000", "0.972", "0.913", "86.7%", "11.5 cyc", "0.22s", "Convex QP"],
        ["Autoencoder (FC)", "0.931", "0.404", "0.564", "0.005", "0.912", "0.769", "60.0%", "14.2 cyc", "42.33s", "8,163"],
        ["LSTM Autoencoder", "0.880", "0.284", "0.429", "0.008", "0.901", "0.689", "73.3%", "50.0 cyc", "491.88s", "116,723"]
    ]
    widths_14 = [Inches(1.8), Inches(0.95), Inches(0.95), Inches(1.0), Inches(0.85), Inches(1.05), Inches(1.05), Inches(1.05), Inches(1.1), Inches(0.95), Inches(1.033)]
    create_styled_table(slide14, Inches(0.8), Inches(1.65), Inches(11.733), Inches(3.2), headers_14, data_14, col_widths=widths_14, highlight_rows=[0, 5])

    # 3 Summary callouts at bottom
    add_metric_callout(slide14, Inches(0.8), Inches(5.1), Inches(3.7), Inches(1.55), "0.602 / 0.984", "Top F1 & ROC-AUC", "Statistical Mean Detector (38 params)")
    add_metric_callout(slide14, Inches(4.8), Inches(5.1), Inches(3.7), Inches(1.55), "50.0 Cycles", "Top Advance Warning", "LSTM Autoencoder (5x vs Baselines)")
    add_metric_callout(slide14, Inches(8.8), Inches(5.1), Inches(3.733), Inches(1.55), "0.72s vs 492s", "Compute Efficiency", "Isolation Forest fits 680x faster")

    # --------------------------------------------------------------------------
    # SLIDE 15: KEY FINDING 1: SIMPLE BASELINES ARE STRONGEST
    # --------------------------------------------------------------------------
    slide15 = prs.slides.add_slide(blank_layout)
    add_header(slide15, "Key Finding 1: Simple Baselines Are Surprisingly Strong", "CORE EMPIRICAL DISCOVERY")
    add_footer(slide15, 15)

    card_l15 = create_card(slide15, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l15 = slide15.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l15 = tb_l15.text_frame
    tf_l15.word_wrap = True
    items_l15 = [
        ("Empirical Evidence on FD001", [
            "Statistical Mean detector achieves highest F1 (0.602) and highest ROC-AUC (0.984) among ALL tested models.",
            "Beats 8,163-param FC-AE (F1=0.564, AUC=0.912) and 116,723-param LSTM-AE (F1=0.429, AUC=0.901).",
            "Zero false alarms on normal operating period (FAR = 0.000).",
            "Highest detection rate: Catches 93.3% of failing engines before breakdown."
        ]),
        ("Why Do Simple Baselines Excel?", [
            "Single operating condition produces steady, monotonic physical drift in key sensors (T24, T50, Ps30).",
            "Z-score directly mirrors underlying thermodynamic degradation without neural distortion."
        ])
    ]
    add_bullet_list(tf_l15, items_l15, font_size=Pt(17))

    card_r15 = create_card(slide15, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r15 = slide15.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r15 = tb_r15.text_frame
    tf_r15.word_wrap = True
    items_r15 = [
        ("Why Do Neural Networks Underperform in Point F1?", [
            "Reconstruction Generalization: Neural autoencoders generalize too well, partially reconstructing mild degradation states.",
            "Reconstruction Smoothing: High-capacity decoders smooth out subtle step deviations.",
            "Hyperparameter Sensitivity: Learning rates, latent dimensions, and epoch limits introduce non-trivial variance."
        ]),
        ("Decisive Takeaway for Industry", [
            "In single-operating-condition industrial assets, deploying deep learning without evaluating a statistical baseline is an expensive, unnecessary engineering error.",
            "A 38-parameter statistical model is mathematically optimal for static conditions."
        ])
    ]
    add_bullet_list(tf_r15, items_r15, font_size=Pt(17))

    # --------------------------------------------------------------------------
    # SLIDE 16: KEY FINDING 2: LSTM-AE PROVIDES 5X EARLIER WARNING
    # --------------------------------------------------------------------------
    slide16 = prs.slides.add_slide(blank_layout)
    add_header(slide16, "Key Finding 2: LSTM-AE Provides 5x Earlier Warning", "OPERATIONAL IMPACT")
    add_footer(slide16, 16)

    card_l16 = create_card(slide16, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l16 = slide16.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l16 = tb_l16.text_frame
    tf_l16.word_wrap = True
    items_l16 = [
        ("The 50-Cycle Early Warning Horizon", [
            "LSTM Autoencoder triggers sustained warning 50.0 cycles before failure on average.",
            "Statistical Mean detector triggers warning only 10.5 cycles before failure.",
            "Isolation Forest: 11.1 cycles | One-Class SVM: 11.5 cycles.",
            "FC Autoencoder: 14.2 cycles.",
            "LSTM-AE provides nearly 5x more actionable maintenance notice than all baseline methods!"
        ]),
        ("Physical Operational Value", [
            "10 cycles is an emergency grounding alert (disruptive, costly).",
            "50 cycles enables scheduled depot overhaul, parts procurement, and zero passenger flight cancellations."
        ])
    ]
    add_bullet_list(tf_l16, items_l16, font_size=Pt(17))

    card_r16 = create_card(slide16, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r16 = slide16.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r16 = tb_r16.text_frame
    tf_r16.word_wrap = True
    items_r16 = [
        ("Why Recurrent Modeling Enables Early Detection", [
            "Static Baselines Check Magnitude Outliers: They only fire when sensor values reach extreme outer boundaries (late-stage degradation).",
            "LSTM-AE Checks Temporal Dynamics: It learns normal rate-of-change across time windows.",
            "Subtle trajectory drifts violate sequence reconstruction long before values breach static thresholds."
        ]),
        ("Persistence Filtering Validation", [
            "All warnings validated with 5-consecutive-cycle persistence filter.",
            "Confirms that LSTM-AE lead time reflects genuine persistent degradation, not transient noisy spikes."
        ])
    ]
    add_bullet_list(tf_r16, items_r16, font_size=Pt(17))

    # --------------------------------------------------------------------------
    # SLIDE 17: KEY FINDING 3: CLASSIFICATION-LEAD TIME PARADOX
    # --------------------------------------------------------------------------
    slide17 = prs.slides.add_slide(blank_layout)
    add_header(slide17, "Key Finding 3: The Classification-Lead Time Paradox", "METHODOLOGICAL PARADOX")
    add_footer(slide17, 17)

    # Top paradox formula card
    top_c17 = create_card(slide17, Inches(0.8), Inches(1.65), Inches(11.733), Inches(1.3), bg_color=COLOR_WHITE, border_color=COLOR_ROYAL_BLUE)
    tb_t17 = slide17.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(11.333), Inches(1.1))
    tf_t17 = tb_t17.text_frame
    tf_t17.word_wrap = True
    p_t17 = tf_t17.paragraphs[0]
    p_t17.text = "THE CENTRAL PARADOX OF INDUSTRIAL PREDICTIVE MAINTENANCE:"
    p_t17.font.size = Pt(12)
    p_t17.font.bold = True
    p_t17.font.color.rgb = COLOR_ROYAL_BLUE
    p_t17_sub = tf_t17.add_paragraph()
    p_t17_sub.text = "Standard point-wise classification metrics (F1 score, Precision, ROC-AUC) actively penalize models that provide early failure warnings. A model with early warning is mathematically rated as worse by traditional metrics."
    p_t17_sub.font.size = Pt(15)
    p_t17_sub.font.bold = True
    p_t17_sub.font.color.rgb = COLOR_NAVY_DARK

    card_l17 = create_card(slide17, Inches(0.8), Inches(3.1), Inches(5.7), Inches(3.55))
    tb_l17 = slide17.shapes.add_textbox(Inches(1.0), Inches(3.25), Inches(5.3), Inches(3.25))
    tf_l17 = tb_l17.text_frame
    tf_l17.word_wrap = True
    items_l17 = [
        ("Why Point Metrics Penalize Early Detection", [
            "Ground truth defines 'anomaly' using fixed RUL cutoff (e.g., RUL <= 30 cycles).",
            "When LSTM-AE flags degradation at cycle 50 (RUL=50), cycles 50 through 31 are scored as FALSE POSITIVES!",
            "Precision and F1 drop sharply, despite providing superior operational value.",
            "Baselines achieve high F1 precisely because they delay warning until RUL <= 11 (late failure)."
        ])
    ]
    add_bullet_list(tf_l17, items_l17, font_size=Pt(15))

    card_r17 = create_card(slide17, Inches(6.8), Inches(3.1), Inches(5.733), Inches(3.55))
    tb_r17 = slide17.shapes.add_textbox(Inches(7.0), Inches(3.25), Inches(5.333), Inches(3.25))
    tf_r17 = tb_r17.text_frame
    tf_r17.word_wrap = True
    items_r17 = [
        ("The Necessary Evaluation Paradigm", [
            "Point-wise classification alone is an invalid evaluation criterion for Predictive Maintenance.",
            "Must report a Dual-Objective Evaluation:",
            "     1. Discrimination Quality: ROC-AUC / PR-AUC",
            "     2. Operational Advance Warning: Mean Lead Time (cycles) & False Alarm Rate on normal phase",
            "A model with F1=0.43 and Lead=50 cycles is far more valuable than one with F1=0.60 and Lead=10 cycles."
        ])
    ]
    add_bullet_list(tf_r17, items_r17, font_size=Pt(15))

    # --------------------------------------------------------------------------
    # SLIDE 18: OPERATING CONDITION RESULTS (FD001 - FD004)
    # --------------------------------------------------------------------------
    slide18 = prs.slides.add_slide(blank_layout)
    add_header(slide18, "Operating Condition Results: Multi-Regime Breakdown", "ENVIRONMENTAL COMPLEXITY")
    add_footer(slide18, 18)

    headers_18 = ["Dataset", "Conditions / Faults", "Statistical Mean", "Isolation Forest", "Autoencoder (FC)", "LSTM Autoencoder"]
    data_18 = [
        ["FD001", "1 Cond / 1 Fault", "0.984 (F1=0.602)", "0.978 (F1=0.549)", "0.946 (F1=0.549)", "0.839 (F1=0.431)"],
        ["FD002", "6 Cond / 1 Fault", "0.501 (F1=0.169 - FAILS)", "0.878 (F1=0.422)", "0.940 (F1=0.364)", "0.537 (F1=0.081)"],
        ["FD003", "1 Cond / 2 Faults", "0.972 (F1=0.342)", "0.969 (F1=0.368)", "0.944 (F1=0.274)", "0.894 (F1=0.297)"],
        ["FD004", "6 Cond / 2 Faults", "0.506 (F1=0.196 - FAILS)", "0.913 (F1=0.429)", "0.964 (F1=0.478 - BEST)", "0.520 (F1=0.083)"]
    ]
    widths_18 = [Inches(1.3), Inches(2.3), Inches(2.1), Inches(2.0), Inches(2.0), Inches(2.033)]
    create_styled_table(slide18, Inches(0.8), Inches(1.65), Inches(11.733), Inches(2.4), headers_18, data_18, col_widths=widths_18, highlight_rows=[1, 3])

    # 2 Summary cards below table
    card_l18 = create_card(slide18, Inches(0.8), Inches(4.3), Inches(5.7), Inches(2.35))
    tb_l18 = slide18.shapes.add_textbox(Inches(1.0), Inches(4.45), Inches(5.3), Inches(2.05))
    tf_l18 = tb_l18.text_frame
    tf_l18.word_wrap = True
    items_l18 = [
        ("Complete Failure of Statistical Baselines", [
            "On FD002 and FD004 (6 operating conditions), Statistical Mean crashes to AUC = 0.501-0.506 (random guess!).",
            "Detection rate collapses to 0.0%: Fails to detect any engine failure before breakdown.",
            "Reason: Flight altitude and Mach changes dwarf degradation signals under global static scaling."
        ])
    ]
    add_bullet_list(tf_l18, items_l18, font_size=Pt(14))

    card_r18 = create_card(slide18, Inches(6.8), Inches(4.3), Inches(5.733), Inches(2.35))
    tb_r18 = slide18.shapes.add_textbox(Inches(7.0), Inches(4.45), Inches(5.333), Inches(2.05))
    tf_r18 = tb_r18.text_frame
    tf_r18.word_wrap = True
    items_r18 = [
        ("FC-Autoencoder Dominates Complex Regimes", [
            "FC-AE achieves highest ROC-AUC on FD004 (0.964) and FD002 (0.940).",
            "Successfully maps non-linear correlations between operational settings and sensor responses.",
            "Isolation Forest is second best (FD004 AUC=0.913, F1=0.429).",
            "Architectural hierarchy completely inverts under environmental complexity!"
        ])
    ]
    add_bullet_list(tf_r18, items_r18, font_size=Pt(14))

    # --------------------------------------------------------------------------
    # SLIDE 19: CROSS-CONDITION TRANSFER
    # --------------------------------------------------------------------------
    slide19 = prs.slides.add_slide(blank_layout)
    add_header(slide19, "Cross-Condition Transfer: Domain Collapse & Fault Robustness", "GENERALIZATION ACROSS DOMAINS")
    add_footer(slide19, 19)

    card_l19 = create_card(slide19, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l19 = slide19.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l19 = tb_l19.text_frame
    tf_l19.word_wrap = True
    items_l19 = [
        ("Shift 1: Operating Regime Transfer (FD001 -> FD002/FD004)", [
            "Catastrophic Domain Failure across ALL models:",
            "False Alarm Rates surge to 85.0% - 100.0%!",
            "Models trained on sea-level (FD001) interpret routine high-altitude flight settings as severe structural degradation.",
            "ROC-AUC drops to ~0.50 (random chance).",
            "Proves: Operating regime conditioning is mandatory. Unsupervised models cannot transfer zero-shot across operating envelopes."
        ])
    ]
    add_bullet_list(tf_l19, items_l19, font_size=Pt(16))

    card_r19 = create_card(slide19, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r19 = slide19.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r19 = tb_r19.text_frame
    tf_r19.word_wrap = True
    items_r19 = [
        ("Shift 2: Novel Fault Mode Transfer (FD001 -> FD003)", [
            "Trained on HPC degradation; tested on novel Fan Blade failure:",
            "Isolation Forest demonstrates outstanding cross-fault robustness:",
            "     ROC-AUC = 0.904 | F1 = 0.596 | Det. Rate = 93.3% | Lead = 51.5 cyc",
            "Neural models struggle significantly:",
            "     FC-AE: ROC-AUC drops to 0.677 | F1 = 0.253",
            "     LSTM-AE: ROC-AUC drops to 0.578 | F1 = 0.210",
            "Tree-based axis partitioning is inherently robust to novel failure modes; neural manifolds over-fit to specific degradation geometries."
        ])
    ]
    add_bullet_list(tf_r19, items_r19, font_size=Pt(16))

    # --------------------------------------------------------------------------
    # SLIDE 20: ABLATION: MULTI-SEED STABILITY & SIGNIFICANCE
    # --------------------------------------------------------------------------
    slide20 = prs.slides.add_slide(blank_layout)
    add_header(slide20, "Ablation: Multi-Seed Stability & Significance Tests", "STATISTICAL RIGOR (5 RANDOM SEEDS)")
    add_footer(slide20, 20)

    # Multi-seed stability table
    headers_20 = ["Model", "F1 Score (Mean ± Std)", "ROC-AUC (Mean ± Std)", "PR-AUC (Mean ± Std)", "Mean Lead Time (Cycles)"]
    data_20 = [
        ["Statistical Mean", "0.471 ± 0.074", "0.985 ± 0.003", "0.940 ± 0.007", "9.68 ± 1.33 cycles"],
        ["Isolation Forest", "0.467 ± 0.069", "0.980 ± 0.003", "0.921 ± 0.008", "8.68 ± 1.57 cycles"],
        ["Autoencoder (FC)", "0.445 ± 0.083", "0.923 ± 0.019", "0.792 ± 0.041", "19.18 ± 4.22 cycles"],
        ["LSTM Autoencoder", "0.323 ± 0.071", "0.772 ± 0.067", "0.540 ± 0.097", "25.44 ± 16.90 cycles"]
    ]
    widths_20 = [Inches(2.5), Inches(2.3), Inches(2.3), Inches(2.3), Inches(2.333)]
    create_styled_table(slide20, Inches(0.8), Inches(1.65), Inches(11.733), Inches(2.4), headers_20, data_20, col_widths=widths_20, highlight_rows=[0])

    # Significance test results
    card_b20 = create_card(slide20, Inches(0.8), Inches(4.3), Inches(11.733), Inches(2.35))
    tb_b20 = slide20.shapes.add_textbox(Inches(1.0), Inches(4.45), Inches(11.333), Inches(2.05))
    tf_b20 = tb_b20.text_frame
    tf_b20.word_wrap = True
    p_b20_t = tf_b20.paragraphs[0]
    p_b20_t.text = "PAIRED STATISTICAL SIGNIFICANCE TESTS (WILCOXON SIGNED-RANK & PAIRED T-TEST):"
    p_b20_t.font.size = Pt(14)
    p_b20_t.font.bold = True
    p_b20_t.font.color.rgb = COLOR_ROYAL_BLUE
    p_b20_t.space_after = Pt(4)

    sig_tests = [
        "Statistical Mean vs. Isolation Forest: Diff = +0.004, Paired t p = 0.860 (No significant difference — statistically equivalent).",
        "Statistical Mean vs. FC-Autoencoder: Diff = +0.026, Paired t p = 0.379 (No statistically significant difference in point F1).",
        "Statistical Mean vs. LSTM-Autoencoder: Diff = +0.147, Paired t p = 0.012 (Statistically significant advantage for Statistical detector, p < 0.05).",
        "FC-Autoencoder vs. LSTM-Autoencoder: Diff = +0.122, Paired t p = 0.005 (Statistically significant difference, p < 0.01)."
    ]
    add_bullet_list(tf_b20, sig_tests, font_size=Pt(14))

    # --------------------------------------------------------------------------
    # SLIDE 21: ABLATION: SEQUENCE LENGTH IN LSTM-AE
    # --------------------------------------------------------------------------
    slide21 = prs.slides.add_slide(blank_layout)
    add_header(slide21, "Ablation: Sequence Length Sensitivity in LSTM-AE", "TEMPORAL HYPERPARAMETER ANALYSIS")
    add_footer(slide21, 21)

    headers_21 = ["Sequence Length", "Parameters", "F1 (P95)", "F1 (Best)", "ROC-AUC", "PR-AUC", "Mean Lead Time", "Training Fit Time"]
    data_21 = [
        ["SeqLen = 10", "116,723", "0.558", "0.734", "0.930", "0.866", "31.1 cycles", "168.3s (Fastest)"],
        ["SeqLen = 20", "116,723", "0.488", "0.657", "0.919", "0.844", "28.8 cycles", "280.2s"],
        ["SeqLen = 30", "116,723", "0.403", "0.507", "0.858", "0.738", "36.8 cycles", "386.4s"],
        ["SeqLen = 50", "116,723", "0.329", "0.461", "0.824", "0.678", "22.9 cycles", "452.8s (Slowest)"]
    ]
    widths_21 = [Inches(1.8), Inches(1.3), Inches(1.2), Inches(1.2), Inches(1.3), Inches(1.3), Inches(1.8), Inches(1.833)]
    create_styled_table(slide21, Inches(0.8), Inches(1.65), Inches(11.733), Inches(2.4), headers_21, data_21, col_widths=widths_21, highlight_rows=[0])

    card_l21 = create_card(slide21, Inches(0.8), Inches(4.3), Inches(5.7), Inches(2.35))
    tb_l21 = slide21.shapes.add_textbox(Inches(1.0), Inches(4.45), Inches(5.3), Inches(2.05))
    tf_l21 = tb_l21.text_frame
    tf_l21.word_wrap = True
    items_l21 = [
        ("Short Sequences (T=10) Are Optimal", [
            "SeqLen=10 achieves peak F1 (0.558), peak ROC-AUC (0.930), and peak PR-AUC (0.866).",
            "Reduces training time by 63% (168s vs 453s for T=50).",
            "Maintains strong early lead time of 31.1 cycles."
        ])
    ]
    add_bullet_list(tf_l21, items_l21, font_size=Pt(14))

    card_r21 = create_card(slide21, Inches(6.8), Inches(4.3), Inches(5.733), Inches(2.35))
    tb_r21 = slide21.shapes.add_textbox(Inches(7.0), Inches(4.45), Inches(5.333), Inches(2.05))
    tf_r21 = tb_r21.text_frame
    tf_r21.word_wrap = True
    items_r21 = [
        ("The 'Temporal Dilution' Effect", [
            "Long windows (T=50) dilute localized degradation onset.",
            "Longer sequences force the recurrent cell to average over ancient nominal cycles, delaying sharp reconstruction divergence.",
            "Recommendation: Practitioners should use compact temporal windows (T=10-15) for industrial anomaly detection."
        ])
    ]
    add_bullet_list(tf_r21, items_r21, font_size=Pt(14))

    # --------------------------------------------------------------------------
    # SLIDE 22: METHOD SELECTION GUIDE (5 SCENARIOS)
    # --------------------------------------------------------------------------
    slide22 = prs.slides.add_slide(blank_layout)
    add_header(slide22, "Practitioner's Method Selection Guide", "DEPLOYMENT DECISION MATRIX")
    add_footer(slide22, 22)

    headers_22 = ["Deployment Scenario", "Recommended Model", "Primary Justification", "Critical Trade-off / Caution"]
    data_22 = [
        ["1. Edge Microcontroller / PLC (<100KB RAM)", "Statistical Z-Score", "38 scalar params, 0.00s fit, zero compute overhead; top F1 in static regime", "Catastrophic failure if operational settings / speed vary"],
        ["2. Unknown / Emerging Mechanical Fault Modes", "Isolation Forest", "0.72s fit; outstanding cross-fault transfer robustness (AUC=0.904, Det=93.3%)", "Shorter lead time (~11 cycles); point magnitude detection"],
        ["3. Mission-Critical Aviation (Early Warning)", "LSTM Autoencoder (T=10)", "50 cycles advance warning (5x baselines); captures drift velocity", "High training latency (168-490s); lower point-wise F1"],
        ["4. Multi-Regime Industrial Assets (Multi-Speed)", "Autoencoder (FC)", "Dominates complex regimes (FD004 AUC=0.964); models non-linear sensors", "Requires regime-aware normalization and careful tuning"],
        ["5. Two-Tier Production Pipeline", "Hybrid: FC-AE / IF + LSTM-AE", "Tier-1 fast screening on edge + Tier-2 temporal confirmation in cloud", "Increased pipeline engineering and dual threshold tuning"]
    ]
    widths_22 = [Inches(2.5), Inches(2.3), Inches(3.6), Inches(3.333)]
    create_styled_table(slide22, Inches(0.8), Inches(1.65), Inches(11.733), Inches(5.0), headers_22, data_22, col_widths=widths_22)

    # --------------------------------------------------------------------------
    # SLIDE 23: LIMITATIONS & FUTURE WORK
    # --------------------------------------------------------------------------
    slide23 = prs.slides.add_slide(blank_layout)
    add_header(slide23, "Limitations & Future Work", "CRITICAL REFLECTION & ROADMAP")
    add_footer(slide23, 23)

    card_l23 = create_card(slide23, Inches(0.8), Inches(1.65), Inches(5.7), Inches(5.0))
    tb_l23 = slide23.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.3), Inches(4.6))
    tf_l23 = tb_l23.text_frame
    tf_l23.word_wrap = True
    items_l23 = [
        ("Current Study Limitations", [
            "Simulated Benchmark Data: C-MAPSS is thermo-dynamical simulation data; real industrial telemetry suffers from packet dropouts, noise, and sampling jitter.",
            "Approximate Degradation Boundary: Defining anomaly ground truth at RUL <= 30 cycles is a necessary operational heuristic for continuous wear.",
            "Computational Edge Burden: 116,000+ parameter LSTM models cannot execute on ultra-low-power microcontrollers.",
            "Single Benchmark Evaluation: Findings should be verified on rotating machinery (vibration, acoustics)."
        ])
    ]
    add_bullet_list(tf_l23, items_l23, font_size=Pt(16))

    card_r23 = create_card(slide23, Inches(6.8), Inches(1.65), Inches(5.733), Inches(5.0))
    tb_r23 = slide23.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(5.333), Inches(4.6))
    tf_r23 = tb_r23.text_frame
    tf_r23.word_wrap = True
    items_r23 = [
        ("Future Research Roadmap", [
            "Physics-Informed Machine Learning (PIML): Embedding thermodynamic degradation laws directly into autoencoder latent loss functions.",
            "Modern State-Space Models (Mamba / S4): Replacing recurrent LSTMs with sub-quadratic linear state-space models for ultra-fast, long-context temporal detection.",
            "Self-Supervised Masked Reconstruction: Masking sensor subsets to improve multi-sensor root cause attribution.",
            "Physical Test Rig Validation: Validating findings on bearing vibration test rigs (PRONOSTIA / IMS datasets)."
        ])
    ]
    add_bullet_list(tf_r23, items_r23, font_size=Pt(16))

    # --------------------------------------------------------------------------
    # SLIDE 24: CONCLUSIONS (5 KEY CONCLUSIONS)
    # --------------------------------------------------------------------------
    slide24 = prs.slides.add_slide(blank_layout)
    add_header(slide24, "Conclusions: Five Core Thesis Takeaways", "SUMMARY OF CONTRIBUTIONS")
    add_footer(slide24, 24)

    card_c24 = create_card(slide24, Inches(0.8), Inches(1.65), Inches(11.733), Inches(5.0))
    tb_c24 = slide24.shapes.add_textbox(Inches(1.1), Inches(1.85), Inches(11.133), Inches(4.6))
    tf_c24 = tb_c24.text_frame
    tf_c24.word_wrap = True

    conclusions = [
        ("1. Simplicity is Severely Underestimated", [
            "A 38-parameter statistical detector outperforms deep neural networks in point-wise F1 (0.602 vs 0.429) and ROC-AUC (0.984 vs 0.901) in single-condition regimes. Deep learning is not universally superior."
        ]),
        ("2. The Classification-Lead Time Paradox Revealed", [
            "Conventional point classification metrics (F1/AUC) actively penalize early warning. LSTM-AE provides 5x earlier warning (50 cycles vs 10 cycles) despite lower point-wise scores."
        ]),
        ("3. Environmental Complexity Dictates Architecture", [
            "Operational shifts destroy statistical baselines (AUC drops from 0.984 to 0.501). Fully Connected Autoencoders dominate multi-condition environments (FD004 AUC=0.964)."
        ]),
        ("4. Tree Isolation Resists Novel Fault Shifts", [
            "Isolation Forest provides unmatched cross-fault generalization (AUC=0.904, Det Rate=93.3%), far exceeding neural autoencoders when encountering unseen failure modes."
        ]),
        ("5. Methodological Rigor is Mandatory", [
            "Strict engine-level partitioning, train-only normalization, and validation-only thresholding eliminate inflated claims and provide true production-ready benchmarks."
        ])
    ]
    add_bullet_list(tf_c24, conclusions, font_size=Pt(16))

    # --------------------------------------------------------------------------
    # SLIDE 25: THANK YOU / Q&A SLIDE
    # --------------------------------------------------------------------------
    slide25 = prs.slides.add_slide(blank_layout)

    # Full background
    bg25 = slide25.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg25.fill.solid()
    bg25.fill.fore_color.rgb = COLOR_CARD_BG
    bg25.line.fill.background()

    # Top Navy Banner
    top25 = slide25.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.35))
    top25.fill.solid()
    top25.fill.fore_color.rgb = COLOR_NAVY_DARK
    top25.line.fill.background()

    # Center card
    main_c25 = create_card(slide25, Inches(1.2), Inches(0.8), Inches(10.933), Inches(5.9), bg_color=COLOR_WHITE, border_color=COLOR_ROYAL_BLUE)

    tb25 = slide25.shapes.add_textbox(Inches(1.5), Inches(1.1), Inches(10.333), Inches(5.3))
    tf25 = tb25.text_frame
    tf25.word_wrap = True

    p25_badge = tf25.paragraphs[0]
    p25_badge.text = "B.TECH THESIS DEFENSE CONCLUDED"
    p25_badge.font.name = FONT_NAME
    p25_badge.font.size = Pt(12)
    p25_badge.font.bold = True
    p25_badge.font.color.rgb = COLOR_ROYAL_BLUE
    p25_badge.space_after = Pt(8)

    p25_t = tf25.add_paragraph()
    p25_t.text = "Thank You for Your Attention!"
    p25_t.font.name = FONT_NAME
    p25_t.font.size = Pt(36)
    p25_t.font.bold = True
    p25_t.font.color.rgb = COLOR_NAVY_DARK
    p25_t.space_after = Pt(14)

    p25_proj = tf25.add_paragraph()
    p25_proj.text = "Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT"
    p25_proj.font.name = FONT_NAME
    p25_proj.font.size = Pt(18)
    p25_proj.font.bold = True
    p25_proj.font.color.rgb = COLOR_TEAL_ACCENT
    p25_proj.space_after = Pt(20)

    p25_meta = tf25.add_paragraph()
    p25_meta.text = "Author: Tusher Tarafder (B.Tech Computer Science & Engineering)\nSupervisor: Prof. (Dr.) Sujata Dash (School of Computer Engineering)\nInstitution: KIIT Deemed to be University, Bhubaneswar, Odisha\nDefense Date: October 2026"
    p25_meta.font.name = FONT_NAME
    p25_meta.font.size = Pt(15)
    p25_meta.font.color.rgb = COLOR_TEXT_MUTED
    p25_meta.space_after = Pt(24)

    p25_qa = tf25.add_paragraph()
    p25_qa.text = "Questions & Academic Discussion are Cordially Welcomed."
    p25_qa.font.name = FONT_NAME
    p25_qa.font.size = Pt(22)
    p25_qa.font.bold = True
    p25_qa.font.color.rgb = COLOR_ROYAL_BLUE

    # Save presentation
    prs.save(output_path)
    print(f"Presentation successfully created at: {output_path}")
    print(f"Total slides generated: {len(prs.slides)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(current_dir)
    default_output = os.path.join(workspace_root, "docs", "thesis_defense.pptx")
    target_path = sys.argv[1] if len(sys.argv) > 1 else default_output
    build_thesis_presentation(target_path)

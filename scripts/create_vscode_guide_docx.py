"""Generate a comprehensive beginner-friendly VS Code execution guide in Word (.docx) format.
Saves to word/VS_Code_Beginner_Execution_Guide.docx
"""

import os
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

BASE_DIR = Path(__file__).resolve().parent.parent
WORD_DIR = BASE_DIR / "word"
OUTPUT_PATH = WORD_DIR / "VS_Code_Beginner_Execution_Guide.docx"

# Color Palette
COLOR_PRIMARY = RGBColor(0x1F, 0x4E, 0x79)   # Navy Blue
COLOR_SECONDARY = RGBColor(0x2E, 0x75, 0xB6) # Steel Blue
COLOR_DARK = RGBColor(0x26, 0x26, 0x26)      # Off-black
COLOR_GREEN = RGBColor(0x27, 0xAE, 0x60)     # Emerald Green
COLOR_MUTED = RGBColor(0x59, 0x59, 0x59)     # Gray


def set_cell_background(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)


def add_callout(doc, title, text, border_color="2E75B6", bg_color="EBF3FA"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)
    cell = table.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(f"📌 {title}: ")
    run_t.bold = True
    run_t.font.name = "Times New Roman"
    run_t.font.size = Pt(11)
    run_t.font.color.rgb = COLOR_PRIMARY

    run_b = p.add_run(text)
    run_b.font.name = "Times New Roman"
    run_b.font.size = Pt(10.5)
    run_b.font.color.rgb = COLOR_DARK

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(2)
    sp.paragraph_format.space_after = Pt(4)


def add_code_box(doc, command_text, description=None):
    if description:
        dp = doc.add_paragraph()
        dp.paragraph_format.space_before = Pt(4)
        dp.paragraph_format.space_after = Pt(2)
        r = dp.add_run(description)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
        r.italic = True
        r.font.color.rgb = COLOR_MUTED

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)
    cell = table.cell(0, 0)
    set_cell_background(cell, "212529") # Dark Terminal style
    set_cell_margins(cell, top=100, bottom=100, left=160, right=160)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(f"> {command_text}")
    run.font.name = "Consolas"
    run.font.size = Pt(10.5)
    run.bold = True
    run.font.color.rgb = RGBColor(0x00, 0xFF, 0x66) # Terminal green

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(2)
    sp.paragraph_format.space_after = Pt(4)


def build_guide():
    WORD_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()

    # Set 1-inch margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("HOW TO RUN THIS PROJECT IN VS CODE")
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(22)
    run_title.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run("A Complete Step-by-Step Beginner's Guide with Commands, Visuals & Troubleshooting")
    run_sub.font.name = "Times New Roman"
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = COLOR_SECONDARY

    # Metadata box
    meta_table = doc.add_table(rows=3, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Project Name:", "Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT (IIoT-AD)"),
        ("Author / Student:", "Tusher Tarafder (KIIT Bhubaneswar, CSE)"),
        ("Project Root Directory:", "c:\\Users\\KIIT\\Desktop\\IIoT-AD"),
    ]
    for r_idx, (k, v) in enumerate(meta_data):
        c0, c1 = meta_table.cell(r_idx, 0), meta_table.cell(r_idx, 1)
        c0.width, c1.width = Inches(2.0), Inches(4.5)
        set_cell_background(c0, "1F4E79")
        set_cell_background(c1, "F2F5F8")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)

        p0 = c0.paragraphs[0]
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.name = "Times New Roman"
        r0.font.size = Pt(10)
        r0.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(10)
        r1.font.color.rgb = COLOR_DARK

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(6)
    sp.paragraph_format.space_after = Pt(10)

    # Introduction
    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.line_spacing = 1.15
    p_intro.paragraph_format.space_after = Pt(10)
    r_intro = p_intro.add_run(
        "Welcome! This guide is written specifically for you to run, explore, and present your BTech final-year research project in Visual Studio Code (VS Code) with zero friction. Even if you are completely new to Python environments or command lines, simply follow each numbered step below in order."
    )
    r_intro.font.name = "Times New Roman"
    r_intro.font.size = Pt(11)

    # -------------------------------------------------------------
    # STEP 1
    # -------------------------------------------------------------
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(4)
    h1.paragraph_format.keep_with_next = True
    r = h1.add_run("STEP 1: Open the Project in VS Code")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY

    p1 = doc.add_paragraph()
    p1.paragraph_format.line_spacing = 1.15
    p1.paragraph_format.space_after = Pt(6)
    r = p1.add_run("1. Open ")
    r = p1.add_run("Visual Studio Code").bold = True
    r = p1.add_run(" on your laptop.\n")
    r = p1.add_run("2. Click on the top menu bar: ")
    r = p1.add_run("File → Open Folder...").bold = True
    r = p1.add_run(" (or press keyboard shortcut ")
    r = p1.add_run("Ctrl + K, then Ctrl + O").bold = True
    r = p1.add_run(").\n")
    r = p1.add_run("3. Navigate to your Desktop and select the folder: ")
    r = p1.add_run("c:\\Users\\KIIT\\Desktop\\IIoT-AD").bold = True
    r = p1.add_run(".\n4. Click ")
    r = p1.add_run("Select Folder").bold = True
    r = p1.add_run(". You should now see the project file tree on the left-side Explorer window.")

    add_callout(doc, "Tip for Beginners", "If VS Code asks 'Do you trust the authors of the files in this folder?', click 'Yes, I trust the authors'.", "27AE60", "E8F8F5")

    # -------------------------------------------------------------
    # STEP 2
    # -------------------------------------------------------------
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(4)
    h2.paragraph_format.keep_with_next = True
    r = h2.add_run("STEP 2: Open the Built-in Terminal in VS Code")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY

    p2 = doc.add_paragraph()
    p2.paragraph_format.line_spacing = 1.15
    p2.paragraph_format.space_after = Pt(6)
    r = p2.add_run("The Terminal allows you to type commands to run Python scripts directly inside VS Code:\n")
    r = p2.add_run("1. On the top menu bar, click: ")
    r = p2.add_run("Terminal → New Terminal").bold = True
    r = p2.add_run(".\n2. Or simply press the shortcut key: ")
    r = p2.add_run("Ctrl + `").bold = True
    r = p2.add_run(" (the backtick key, located right below the Esc key).\n")
    r = p2.add_run("3. A terminal panel will open at the bottom of your VS Code screen showing: ")
    r = p2.add_run("PS C:\\Users\\KIIT\\Desktop\\IIoT-AD>").bold = True

    # -------------------------------------------------------------
    # STEP 3
    # -------------------------------------------------------------
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(4)
    h3.paragraph_format.keep_with_next = True
    r = h3.add_run("STEP 3: Verify Your Python Installation")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY

    p3 = doc.add_paragraph()
    p3.paragraph_format.line_spacing = 1.15
    p3.paragraph_format.space_after = Pt(4)
    p3.add_run("Type this command in your VS Code terminal and press Enter to ensure Python is recognized:")
    add_code_box(doc, "python --version", "Check Python version")

    p3_res = doc.add_paragraph()
    p3_res.paragraph_format.space_after = Pt(6)
    p3_res.add_run("You should see: ").font.name = "Times New Roman"
    r = p3_res.add_run("Python 3.10.x (or higher).")
    r.bold = True
    r.font.name = "Consolas"

    # -------------------------------------------------------------
    # STEP 4
    # -------------------------------------------------------------
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(4)
    h4.paragraph_format.keep_with_next = True
    r = h4.add_run("STEP 4: Run the Quick System & Test Verification")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY

    p4 = doc.add_paragraph()
    p4.paragraph_format.line_spacing = 1.15
    p4.paragraph_format.space_after = Pt(4)
    p4.add_run("Before running models, test that all 20 data integrity and leakage-prevention tests pass. Type:")
    add_code_box(doc, "pytest tests/ -v", "Execute full unit test suite")

    p4_exp = doc.add_paragraph()
    p4_exp.paragraph_format.space_after = Pt(6)
    r = p4_exp.add_run("Expected Output: ")
    r.bold = True
    r = p4_exp.add_run("20 passed in ~12 seconds (in green text). This proves that data loading, train/val/test splits, normalization, and sequence generators are functioning flawlessly.")

    # -------------------------------------------------------------
    # STEP 5
    # -------------------------------------------------------------
    h5 = doc.add_paragraph()
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(4)
    h5.paragraph_format.keep_with_next = True
    r = h5.add_run("STEP 5: Launch the Interactive Web Dashboard (Streamlit)")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY

    p5 = doc.add_paragraph()
    p5.paragraph_format.line_spacing = 1.15
    p5.paragraph_format.space_after = Pt(4)
    p5.add_run(
        "This is the most impressive component to show your professors and project examiner! It opens an interactive web user interface where you can explore sensor readings, compare all 5 machine learning models, and view transfer heatmaps."
    )
    add_code_box(doc, "streamlit run app.py", "Launch web dashboard")

    p5_steps = doc.add_paragraph()
    p5_steps.paragraph_format.line_spacing = 1.15
    p5_steps.paragraph_format.space_after = Pt(6)
    r = p5_steps.add_run("What happens when you run this:\n")
    r = p5_steps.add_run("1. The terminal will display: ")
    r = p5_steps.add_run("Local URL: http://localhost:8501\n").bold = True
    r = p5_steps.add_run("2. Your default web browser (Chrome, Edge, etc.) will automatically open up with your dashboard.\n")
    r = p5_steps.add_run("3. Use the left sidebar to navigate across the 6 pages:\n")
    r = p5_steps.add_run("   • Overview: Key research metric cards and takeaways\n")
    r = p5_steps.add_run("   • Dataset Explorer: Interactive telemetry graphs for any engine ID\n")
    r = p5_steps.add_run("   • Model Comparison: Interactive bar charts for F1, ROC-AUC, Lead Time\n")
    r = p5_steps.add_run("   • Cross-Condition Analysis: Transfer heatmap & condition shifts\n")
    r = p5_steps.add_run("   • Ablation Studies: Sequence length charts & 5-seed stability\n")
    r = p5_steps.add_run("   • Method Selection Guide: Interactive wizard recommending the right model\n")
    r = p5_steps.add_run("4. ")
    r = p5_steps.add_run("How to stop the app: ").bold = True
    r = p5_steps.add_run("When you want to stop the dashboard and return to the terminal prompt, click into the VS Code terminal and press ")
    r = p5_steps.add_run("Ctrl + C").bold = True
    r = p5_steps.add_run(".")

    add_callout(doc, "Examiner Presentation Tip", "During your final viva/defense, have this dashboard running! Click on 'Dataset Explorer' and show the degradation trajectory of engine #1, then click 'Model Comparison' to explain why the LSTM Autoencoder detects anomalies 5x earlier than baselines.", "1F4E79", "EBF3FA")

    # -------------------------------------------------------------
    # STEP 6
    # -------------------------------------------------------------
    h6 = doc.add_paragraph()
    h6.paragraph_format.space_before = Pt(14)
    h6.paragraph_format.space_after = Pt(4)
    h6.paragraph_format.keep_with_next = True
    r = h6.add_run("STEP 6: Running the Machine Learning Experiments")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY

    p6 = doc.add_paragraph()
    p6.paragraph_format.line_spacing = 1.15
    p6.paragraph_format.space_after = Pt(4)
    p6.add_run("All models are pre-implemented and can be re-executed anytime. Below is the command guide for each experiment:")

    # Table of commands
    cmd_table = doc.add_table(rows=6, cols=3)
    cmd_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cmd_table.columns[0].width = Inches(2.2)
    cmd_table.columns[1].width = Inches(2.6)
    cmd_table.columns[2].width = Inches(1.7)

    headers = ["Task / Model", "Terminal Command", "Approx. Time"]
    for i, h in enumerate(headers):
        c = cmd_table.cell(0, i)
        set_cell_background(c, "1F4E79")
        set_cell_margins(c, 100, 100, 100, 100)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    rows_data = [
        ("1. Baselines (Statistical, IF, OC-SVM)", "python scripts/run_baselines.py", "< 5 seconds"),
        ("2. Feedforward Autoencoder", "python scripts/run_autoencoder.py", "~45 seconds"),
        ("3. LSTM Autoencoder", "python scripts/run_lstm_autoencoder.py", "~8 minutes"),
        ("4. Cross-Condition Generalization", "python scripts/run_cross_condition_experiments.py", "~35 minutes"),
        ("5. Multi-Seed & Ablations", "python scripts/run_ablation_significance.py", "~45 minutes"),
    ]

    for r_idx, (t_task, t_cmd, t_time) in enumerate(rows_data, start=1):
        c0, c1, c2 = cmd_table.cell(r_idx, 0), cmd_table.cell(r_idx, 1), cmd_table.cell(r_idx, 2)
        bg = "F4F6F7" if (r_idx % 2 == 1) else "FFFFFF"
        for c in (c0, c1, c2):
            set_cell_background(c, bg)
            set_cell_margins(c, 80, 80, 100, 100)

        p0 = c0.paragraphs[0]
        r = p0.add_run(t_task)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.bold = True

        p1 = c1.paragraphs[0]
        r = p1.add_run(t_cmd)
        r.font.name = "Consolas"
        r.font.size = Pt(9)
        r.font.color.rgb = COLOR_PRIMARY

        p2 = c2.paragraphs[0]
        r = p2.add_run(t_time)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(6)
    sp.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # STEP 7
    # -------------------------------------------------------------
    h7 = doc.add_paragraph()
    h7.paragraph_format.space_before = Pt(14)
    h7.paragraph_format.space_after = Pt(4)
    h7.paragraph_format.keep_with_next = True
    r = h7.add_run("STEP 7: Regenerating Figures, Word Files & Slides")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY

    p7 = doc.add_paragraph()
    p7.paragraph_format.line_spacing = 1.15
    p7.paragraph_format.space_after = Pt(4)
    p7.add_run("You can also regenerate all academic deliverables with a single command whenever needed:")

    add_code_box(doc, "python scripts/generate_figures.py", "1. Regenerate 6 high-res publication figures (saved to results/figures/publication/)")
    add_code_box(doc, "python scripts/generate_word_doc.py", "2. Regenerate Word thesis with embedded figures (saved to docs/thesis.docx)")
    add_code_box(doc, "python scripts/generate_presentation.py", "3. Regenerate 25-slide defense deck (saved to docs/thesis_defense.pptx)")
    add_code_box(doc, "python scripts/convert_all_md_to_docx.py", "4. Convert all markdown files to Word (saved to word/ folder)")

    # -------------------------------------------------------------
    # STEP 8
    # -------------------------------------------------------------
    h8 = doc.add_paragraph()
    h8.paragraph_format.space_before = Pt(16)
    h8.paragraph_format.space_after = Pt(4)
    h8.paragraph_format.keep_with_next = True
    r = h8.add_run("TROUBLESHOOTING & COMMON QUESTIONS FOR BEGINNERS")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY

    q1 = doc.add_paragraph()
    q1.paragraph_format.space_before = Pt(6)
    q1.paragraph_format.space_after = Pt(2)
    r = q1.add_run("Q1: PowerShell shows an error: 'running scripts is disabled on this system'?")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(11.5)
    r.font.color.rgb = COLOR_SECONDARY

    p_a1 = doc.add_paragraph()
    p_a1.paragraph_format.line_spacing = 1.15
    p_a1.paragraph_format.space_after = Pt(4)
    p_a1.add_run("This is a standard Windows security setting. Fix it in 5 seconds by typing this command into your VS Code terminal:")
    add_code_box(doc, "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass", "Bypass execution policy for current terminal session")

    q2 = doc.add_paragraph()
    q2.paragraph_format.space_before = Pt(6)
    q2.paragraph_format.space_after = Pt(2)
    r = q2.add_run("Q2: How do I select the Python Interpreter in VS Code?")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(11.5)
    r.font.color.rgb = COLOR_SECONDARY

    p_a2 = doc.add_paragraph()
    p_a2.paragraph_format.line_spacing = 1.15
    p_a2.paragraph_format.space_after = Pt(6)
    r = p_a2.add_run("1. Press ")
    r = p_a2.add_run("Ctrl + Shift + P").bold = True
    r = p_a2.add_run(" to open the VS Code Command Palette.\n")
    r = p_a2.add_run("2. Type ")
    r = p_a2.add_run("Python: Select Interpreter").bold = True
    r = p_a2.add_run(" and press Enter.\n")
    r = p_a2.add_run("3. Select your installed Python version (e.g., Python 3.10.x 64-bit).")

    q3 = doc.add_paragraph()
    q3.paragraph_format.space_before = Pt(6)
    q3.paragraph_format.space_after = Pt(2)
    r = q3.add_run("Q3: How do I stop a script that is currently running in the terminal?")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(11.5)
    r.font.color.rgb = COLOR_SECONDARY

    p_a3 = doc.add_paragraph()
    p_a3.paragraph_format.line_spacing = 1.15
    p_a3.paragraph_format.space_after = Pt(6)
    r = p_a3.add_run("Simply click inside the terminal window and press ")
    r = p_a3.add_run("Ctrl + C").bold = True
    r = p_a3.add_run(" on your keyboard. This sends a cancel signal and immediately returns you to the prompt.")

    # -------------------------------------------------------------
    # CHEAT SHEET
    # -------------------------------------------------------------
    h_cs = doc.add_paragraph()
    h_cs.paragraph_format.space_before = Pt(16)
    h_cs.paragraph_format.space_after = Pt(4)
    h_cs.paragraph_format.keep_with_next = True
    r = h_cs.add_run("VS CODE KEYBOARD SHORTCUTS CHEAT SHEET")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY

    cs_table = doc.add_table(rows=6, cols=2)
    cs_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cs_table.columns[0].width = Inches(2.5)
    cs_table.columns[1].width = Inches(4.0)

    cs_headers = ["Shortcut", "What It Does in VS Code"]
    for i, h in enumerate(cs_headers):
        c = cs_table.cell(0, i)
        set_cell_background(c, "1F4E79")
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    cs_data = [
        ("Ctrl + `", "Toggle Terminal (Open / Close bottom terminal)"),
        ("Ctrl + Shift + P", "Open Command Palette (Search any VS Code setting)"),
        ("Ctrl + C", "Stop running command / script in terminal"),
        ("cls  or  clear", "Clear terminal screen text"),
        ("Up Arrow Key (↑)", "Cycle through previous commands without retyping"),
    ]

    for r_idx, (t_key, t_desc) in enumerate(cs_data, start=1):
        c0, c1 = cs_table.cell(r_idx, 0), cs_table.cell(r_idx, 1)
        bg = "F4F6F7" if (r_idx % 2 == 1) else "FFFFFF"
        for c in (c0, c1):
            set_cell_background(c, bg)
            set_cell_margins(c, 70, 70, 100, 100)

        p0 = c0.paragraphs[0]
        r = p0.add_run(t_key)
        r.font.name = "Consolas"
        r.font.size = Pt(9.5)
        r.bold = True
        r.font.color.rgb = COLOR_PRIMARY

        p1 = c1.paragraphs[0]
        r = p1.add_run(t_desc)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)

    doc.save(str(OUTPUT_PATH))
    print(f"Successfully generated: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH):,} bytes)")


if __name__ == "__main__":
    build_guide()

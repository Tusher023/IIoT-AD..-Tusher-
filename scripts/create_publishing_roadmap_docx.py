"""Generate a comprehensive independent research publishing roadmap Word document (.docx)
specifically tailored for an undergraduate student / independent researcher from Bangladesh.
Saves to word/Independent_Research_Publishing_Roadmap_Bangladesh.docx
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
OUTPUT_PATH = WORD_DIR / "Independent_Research_Publishing_Roadmap_Bangladesh.docx"

# Palette
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
    run_t = p.add_run(f"💡 {title}: ")
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


def add_section_header(doc, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = COLOR_PRIMARY


def add_bullet(doc, bold_prefix, text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(1.5)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.line_spacing = 1.15
    r0 = p.add_run(bold_prefix)
    r0.bold = True
    r0.font.name = "Times New Roman"
    r0.font.size = Pt(11)
    r0.font.color.rgb = COLOR_DARK
    r1 = p.add_run(text)
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(11)
    r1.font.color.rgb = COLOR_DARK


def build_publishing_plan():
    WORD_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()

    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("INDEPENDENT RESEARCH PUBLICATION BLUEPRINT")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(22)
    r_title.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("A Complete Step-by-Step Strategic Guide for Undergraduate Researchers & Independent Authors from Bangladesh")
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(12.5)
    r_sub.font.color.rgb = COLOR_SECONDARY

    # Metadata Box
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Author / Candidate:", "Tusher Tarafder (Undergraduate Researcher, Bangladesh)"),
        ("Research Domain:", "Industrial IoT, Anomaly Detection, Machine Learning & Predictive Maintenance"),
        ("Target Paper:", "Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT"),
        ("Key Advantage:", "Rigorous empirical benchmark (162 files, 5 seeds, 7 RQs, 20/20 verified tests)"),
    ]
    for r_idx, (k, v) in enumerate(meta_data):
        c0, c1 = meta_table.cell(r_idx, 0), meta_table.cell(r_idx, 1)
        c0.width, c1.width = Inches(2.2), Inches(4.3)
        set_cell_background(c0, "1F4E79")
        set_cell_background(c1, "F2F5F8")
        set_cell_margins(c0, 50, 50, 100, 100)
        set_cell_margins(c1, 50, 50, 100, 100)

        p0 = c0.paragraphs[0]
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.name = "Times New Roman"
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = COLOR_DARK

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(4)
    sp.paragraph_format.space_after = Pt(8)

    # Executive Overview
    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.line_spacing = 1.15
    p_intro.paragraph_format.space_after = Pt(8)
    r = p_intro.add_run(
        "Publishing high-impact machine learning research independently from Bangladesh is entirely achievable when approached systematically. Your project has an extraordinary advantage: unlike shallow student projects, you have conducted a methodologically rigorous empirical study on NASA C-MAPSS data with zero data leakage, multi-seed statistical significance testing, sequence ablation, and negative result reporting. This blueprint provides the exact roadmap to take this work from your desktop to an internationally indexed publication (Scopus/Web of Science/IEEE)."
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    # -------------------------------------------------------------
    # SECTION 1: PUBLISHING FOR FREE ($0 BUDGET)
    # -------------------------------------------------------------
    add_section_header(doc, "1. THE $0 PUBLICATION REALITY (Dispelling the Fee Myth)")

    p_fee = doc.add_paragraph()
    p_fee.paragraph_format.line_spacing = 1.15
    p_fee.paragraph_format.space_after = Pt(6)
    r = p_fee.add_run(
        "Many students from Bangladesh mistakenly believe that publishing in top journals costs $1,500 to $3,000 in Article Processing Charges (APC). This is completely false. You have two 100% free paths:"
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    add_bullet(doc, "Path A: The Traditional Subscription Model (100% Free): ", "Almost all premier journals (Elsevier, Springer, Wiley, IEEE Transactions) operate under a 'Hybrid' model. When your paper is accepted, you simply select 'Subscription / Traditional Publishing'. The publisher charges libraries to read, and you pay exactly $0.00.")
    add_bullet(doc, "Path B: Developing Country Fee Waivers (Research4Life): ", "Bangladesh is classified by the United Nations as a developing/least-developed economy. Major Open Access publishers (Springer Nature, PLoS, Elsevier, BioMed Central) provide automatic or requested 50% to 100% APC waivers for corresponding authors based in Bangladesh.")
    add_bullet(doc, "Path C: Reputable IEEE Conferences with Student Registration: ", "Local IEEE Bangladesh Section (IEEE BDS) conferences offer heavily subsidized student registration fees (typically 3,000 to 5,000 BDT) compared to international rates.")

    add_callout(doc, "Golden Rule for Students", "Never pay an unknown online journal that promises publication within 3 days for a $50-$200 fee. These are predatory journals that carry zero academic value. Always choose Scopus/Web of Science indexed venues.", "27AE60", "E8F8F5")

    # -------------------------------------------------------------
    # SECTION 2: ESTABLISHING YOUR SCHOLARLY IDENTITY
    # -------------------------------------------------------------
    add_section_header(doc, "2. ESTABLISHING YOUR ACADEMIC IDENTITY (Pre-Submission Checklist)")

    p_id = doc.add_paragraph()
    p_id.paragraph_format.line_spacing = 1.15
    p_id.paragraph_format.space_after = Pt(6)
    r = p_id.add_run("Before submitting anywhere, create these three permanent academic profiles:")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    add_bullet(doc, "1. ORCID iD (Open Researcher and Contributor ID): ", "Register for free at https://orcid.org. This provides your unique 16-digit identifier (e.g., 0000-0002-XXXX-XXXX). It connects all your future papers and prevents author ambiguity.")
    add_bullet(doc, "2. Google Scholar Profile: ", "Create your profile using an institutional or personal email at https://scholar.google.com. This tracks your future citations and h-index.")
    add_bullet(doc, "3. ResearchGate Account: ", "Register at https://www.researchgate.net to connect with international predictive maintenance researchers and share preprints.")

    # -------------------------------------------------------------
    # SECTION 3: ESTABLISHING PRIORITY (PREPRINTS)
    # -------------------------------------------------------------
    add_section_header(doc, "3. PROTECTING YOUR DISCOVERY VIA PREPRINTS (TechRxiv / arXiv)")

    p_prep = doc.add_paragraph()
    p_prep.paragraph_format.line_spacing = 1.15
    p_prep.paragraph_format.space_after = Pt(6)
    r = p_prep.add_run(
        "As an independent student researcher, your biggest concern might be idea theft or someone publishing a similar paper while your manuscript is under review. The solution is uploading a preprint:"
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    add_bullet(doc, "Upload to TechRxiv (IEEE's Preprint Server): ", "TechRxiv (https://www.techrxiv.org) is free, moderated, and assigns an official citable DOI within 48 hours. IEEE, Elsevier, and Springer explicitly permit submitting papers that have been posted to TechRxiv or arXiv.")
    add_bullet(doc, "Immediate Citation Benefit: ", "Once on TechRxiv, you can cite your paper on your CV, LinkedIn, and MS/PhD scholarship applications immediately—even while formal peer review is ongoing.")

    # -------------------------------------------------------------
    # SECTION 4: TARGET VENUE OPTIONS
    # -------------------------------------------------------------
    add_section_header(doc, "4. TARGET VENUES: WHERE TO SUBMIT THIS PAPER")

    p_venue = doc.add_paragraph()
    p_venue.paragraph_format.line_spacing = 1.15
    p_venue.paragraph_format.space_after = Pt(6)
    r = p_venue.add_run(
        "Based on the empirical depth of your project (5 models, 4 C-MAPSS subsets, leakage-free pipeline, early warning paradox), here are the recommended publication options ranked by trajectory:"
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    # Venue Table
    v_table = doc.add_table(rows=6, cols=4)
    v_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    v_table.columns[0].width = Inches(2.2)
    v_table.columns[1].width = Inches(1.3)
    v_table.columns[2].width = Inches(1.5)
    v_table.columns[3].width = Inches(1.5)

    v_headers = ["Target Venue", "Index / Tier", "Review Speed", "Cost / APC"]
    for i, h in enumerate(v_headers):
        c = v_table.cell(0, i)
        set_cell_background(c, "1F4E79")
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    v_rows = [
        ("Discover Applied Sciences (Springer)", "Scopus / ESCI", "4–6 weeks", "$0 (Subscription) or Waiver"),
        ("Internet of Things (Elsevier)", "Scopus / SCIE Q1", "8–12 weeks", "$0 (Subscription)"),
        ("Journal of Industrial Information Integration (Elsevier)", "Scopus / SCIE Q1", "8–12 weeks", "$0 (Subscription)"),
        ("IEEE Region 10 / Bangladesh Section Conferences (e.g. ICECE, TENSYMP, WIECON-ECE)", "Scopus / IEEE Xplore", "6–8 weeks", "Student Reg (~3,500-5,000 BDT)"),
        ("IEEE Access (IEEE)", "Scopus / SCIE Q1", "4–6 weeks", "$1,950 (or waiver request)"),
    ]

    for r_idx, (v_name, v_tier, v_speed, v_cost) in enumerate(v_rows, start=1):
        c0, c1, c2, c3 = v_table.cell(r_idx, 0), v_table.cell(r_idx, 1), v_table.cell(r_idx, 2), v_table.cell(r_idx, 3)
        bg = "F4F6F7" if (r_idx % 2 == 1) else "FFFFFF"
        for c in (c0, c1, c2, c3):
            set_cell_background(c, bg)
            set_cell_margins(c, 70, 70, 90, 90)

        p0 = c0.paragraphs[0]
        r = p0.add_run(v_name)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)

        p1 = c1.paragraphs[0]
        r = p1.add_run(v_tier)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)

        p2 = c2.paragraphs[0]
        r = p2.add_run(v_speed)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)

        p3 = c3.paragraphs[0]
        r = p3.add_run(v_cost)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)
        if "$0" in v_cost or "BDT" in v_cost:
            r.bold = True
            r.font.color.rgb = COLOR_GREEN

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(6)
    sp.paragraph_format.space_after = Pt(6)

    add_callout(doc, "Top Recommendation for Rapid Success", "For your first publication, submit to either a Scopus-indexed IEEE Bangladesh Section conference (quick turnaround, presentation experience, IEEE Xplore indexing) OR a Springer Discover Applied Sciences / Elsevier journal under the $0 Subscription option.", "1F4E79", "EBF3FA")

    # -------------------------------------------------------------
    # SECTION 5: STEP-BY-STEP PUBLISHING ROADMAP (CHRONOLOGICAL)
    # -------------------------------------------------------------
    add_section_header(doc, "5. CHRONOLOGICAL 6-MONTH ROADMAP")

    stages = [
        ("Month 1: Final Formatting & Pre-Submission Polish", [
            ("Plagiarism Verification: ", "Run docs/research_paper.md through Turnitin or iThenticate. Aim for similarity index below 12% (excluding bibliography)."),
            ("Target Template Formatting: ", "Download the Word or LaTeX template of your target journal/conference. If using LaTeX, our existing docs/thesis.tex has already implemented all equations and BibTeX citations."),
            ("Preprint Release: ", "Submit manuscript to TechRxiv or arXiv to establish discovery date and secure an immediate DOI."),
        ]),
        ("Month 2: Submission & Editorial Manager Process", [
            ("Account Setup: ", "Create an author account on the journal's submission portal (Editorial Manager / ScholarOne / IEEE Author Portal)."),
            ("Affiliation Setting: ", "List your affiliation as 'School of Computer Engineering, Kalinga Institute of Industrial Technology, Bhubaneswar' (or your active Bangladesh institution / Independent Researcher, Dhaka, Bangladesh)."),
            ("Cover Letter Submission: ", "Attach a professional cover letter highlighting the research gap and industrial significance (template provided below)."),
            ("Suggesting Reviewers: ", "Enter 3 recommended peer reviewers (researchers who have published on C-MAPSS or predictive maintenance, e.g., from your literature review)."),
        ]),
        ("Month 3–4: Navigating Peer Review", [
            ("Status Tracking: ", "Monitor statuses: 'Under Review' → 'Reviews Completed' → 'Decision in Process'."),
            ("Understanding Decisions: ", "Accept without revision is rare (<1%). 'Major Revision' or 'Minor Revision' is a huge victory—it means the editor wants to publish if you address comments!"),
        ]),
        ("Month 5: Responding to Reviewers (The Revision Stage)", [
            ("Point-by-Point Rebuttal: ", "Create a response document addressing every single reviewer comment with utmost politeness and evidence."),
            ("Rerunning Code: ", "If reviewers ask for extra analysis, use our scripts to instantly generate new numbers, figures, or ablations."),
        ]),
        ("Month 6: Acceptance, Proofs & Indexing", [
            ("Galley Proof Approval: ", "Check author spelling, affiliations, figures, and equation formats in the publisher proof."),
            ("Publication & Scopus Indexing: ", "Your paper goes live online with a formal DOI and appears in Scopus / Google Scholar within 3–4 weeks."),
        ]),
    ]

    for stage_title, stage_items in stages:
        p_st = doc.add_paragraph()
        p_st.paragraph_format.space_before = Pt(8)
        p_st.paragraph_format.space_after = Pt(2)
        p_st.paragraph_format.keep_with_next = True
        r = p_st.add_run(stage_title)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.font.color.rgb = COLOR_SECONDARY

        for prefix, desc in stage_items:
            add_bullet(doc, prefix, desc)

    # -------------------------------------------------------------
    # SECTION 6: READY-TO-USE COVER LETTER TEMPLATE
    # -------------------------------------------------------------
    add_section_header(doc, "6. READY-TO-USE COVER LETTER TEMPLATE")

    p_cl_intro = doc.add_paragraph()
    p_cl_intro.paragraph_format.line_spacing = 1.15
    p_cl_intro.paragraph_format.space_after = Pt(4)
    p_cl_intro.add_run("Copy and customize this professional cover letter when submitting through the journal portal:")

    cl_text = (
        "Dear Editor-in-Chief,\n\n"
        "I am pleased to submit our original research manuscript titled \"Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT: A Systematic Empirical Comparison of Classical and Deep Learning Methods\" for consideration for publication in [Journal Name].\n\n"
        "In this study, we address a critical challenge in Industrial IoT predictive maintenance: evaluating unsupervised anomaly detection methods under realistic conditions when labeled run-to-failure data is limited. While deep learning models (such as Autoencoders and LSTM Autoencoders) are increasingly favored in literature, their practical superiority over classical statistical baselines (such as Mean Z-Score and Isolation Forest) remains poorly characterized under leak-free evaluation.\n\n"
        "Our manuscript makes three fundamental contributions:\n"
        "1. We establish a leak-free evaluation framework on the NASA C-MAPSS dataset enforcing engine-level splits, train-only scaling, and validation-only thresholding.\n"
        "2. We empirically prove that a 38-parameter statistical z-score detector matches Isolation Forest (p = 0.86) and outperforms feedforward autoencoders on single-condition fleets.\n"
        "3. We uncover the 'Classification-Lead Time Paradox', demonstrating that while LSTM Autoencoders achieve lower point-wise F1, they deliver 5x earlier advance warning (50.0 cycles vs. 10.5 cycles).\n\n"
        "This manuscript has not been published elsewhere and is not currently under consideration by any other journal. The authors declare no conflicts of interest.\n\n"
        "Thank you for your consideration of our work.\n\n"
        "Sincerely,\n"
        "Tusher Tarafder\n"
        "Department of Computer Science and Engineering\n"
        "Email: [Your Email Address] | ORCID: [Your ORCID iD]"
    )

    table_cl = doc.add_table(rows=1, cols=1)
    table_cl.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_cl.autofit = False
    table_cl.columns[0].width = Inches(6.5)
    cell_cl = table_cl.cell(0, 0)
    set_cell_background(cell_cl, "F4F5F7")
    set_cell_margins(cell_cl, top=120, bottom=120, left=160, right=160)
    p = cell_cl.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(cl_text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_DARK

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(4)
    sp.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # SECTION 7: HOW TO RESPOND TO REVIEWERS
    # -------------------------------------------------------------
    add_section_header(doc, "7. HOW TO MASTER PEER REVIEW REVISIONS")

    p_rev = doc.add_paragraph()
    p_rev.paragraph_format.line_spacing = 1.15
    p_rev.paragraph_format.space_after = Pt(6)
    r = p_rev.add_run(
        "When you receive reviewer comments, follow the 3 Golden Rules of Rebuttal:"
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    add_bullet(doc, "Rule 1: Always Be Grateful and Respectful: ", "Begin every reply with: 'We sincerely thank the reviewer for this constructive insight, which has substantially improved our manuscript.' Even if a reviewer misunderstood a point, politely clarify without being defensive.")
    add_bullet(doc, "Rule 2: Point-by-Point Itemization: ", "Copy every single sentence of the reviewer's comment in bold, followed immediately by your detailed response and the exact line number/page where changes were made in the revised text.")
    add_bullet(doc, "Rule 3: Show, Don't Just Tell: ", "Whenever a reviewer asks for clarification (e.g. 'Why did you use sequence length 10 instead of 30?'), quote the exact ablation table from our study showing that L=10 achieved higher F1 (0.558) and 63% faster training.")

    # -------------------------------------------------------------
    # SECTION 8: SCHOLARSHIP & CAREER IMPACT
    # -------------------------------------------------------------
    add_section_header(doc, "8. CAREER & HIGHER STUDY IMPACT (From Bangladesh to Global MSc/PhD)")

    p_car = doc.add_paragraph()
    p_car.paragraph_format.line_spacing = 1.15
    p_car.paragraph_format.space_after = Pt(6)
    r = p_car.add_run(
        "For a computer science student from Bangladesh aiming for international graduate scholarships (USA, Canada, Germany, Australia, UK) or high-tier AI industry roles, having a first-author Scopus/IEEE publication is the single most valuable credential on your profile:"
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    add_bullet(doc, "Professor Outreach (Cold Emailing): ", "When emailing prospective professors in the US or Europe for Graduate Research Assistantships (GRA), leading with: 'I am the first author of a paper on leak-free unsupervised predictive maintenance published in [Venue]' increases response rates by over 400%.")
    add_bullet(doc, "Visa & Scholarship Strength: ", "Published papers provide concrete proof of research capability, making your Statement of Purpose (SOP) stand out in top percentile applicant pools.")

    doc.save(str(OUTPUT_PATH))
    print(f"Successfully generated: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH):,} bytes)")


if __name__ == "__main__":
    build_publishing_plan()

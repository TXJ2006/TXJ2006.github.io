from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "XujiangTang_CV_2026.pdf"

BLACK = colors.HexColor("#161616")
NAVY = colors.HexColor("#26384a")
GRAY = colors.HexColor("#555555")
LINE = colors.HexColor("#9aa4ad")


def register_fonts():
    font_sets = [
        (Path("C:/Windows/Fonts"), "times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
        (
            Path("/System/Library/Fonts/Supplemental"),
            "Times New Roman.ttf",
            "Times New Roman Bold.ttf",
            "Times New Roman Italic.ttf",
            "Times New Roman Bold Italic.ttf",
        ),
        (Path("/Library/Fonts"), "Times New Roman.ttf", "Times New Roman Bold.ttf", "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf"),
    ]
    for font_dir, regular, bold, italic, bold_italic in font_sets:
        if all((font_dir / filename).exists() for filename in (regular, bold, italic, bold_italic)):
            break
    else:
        raise FileNotFoundError("Times New Roman fonts were not found in the supported system font directories")

    pdfmetrics.registerFont(TTFont("TimesNewRoman", str(font_dir / regular)))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(font_dir / bold)))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Italic", str(font_dir / italic)))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-BoldItalic", str(font_dir / bold_italic)))
    pdfmetrics.registerFontFamily(
        "TimesNewRoman",
        normal="TimesNewRoman",
        bold="TimesNewRoman-Bold",
        italic="TimesNewRoman-Italic",
        boldItalic="TimesNewRoman-BoldItalic",
    )


def p(text, style):
    return Paragraph(text, style)


def section(title, styles):
    table = Table([[p(title.upper(), styles["Section"])]], colWidths=[178 * mm])
    table.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.65, NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6 * mm),
            ]
        )
    )
    return [Spacer(1, 3.2 * mm), table, Spacer(1, 1.6 * mm)]


def entry(title, subtitle, date, bullets, styles):
    heading = Table(
        [[p(f"<b>{title}</b>", styles["Body"]), p(date, styles["Date"])],
         [p(subtitle, styles["Meta"]), ""]],
        colWidths=[150 * mm, 28 * mm],
        hAlign="LEFT",
    )
    heading.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    flowables = [heading, Spacer(1, 0.7 * mm)]
    flowables.extend(p(f"- {item}", styles["Bullet"]) for item in bullets)
    flowables.append(Spacer(1, 1.4 * mm))
    return KeepTogether(flowables)


def publication(number, text, styles):
    return p(f"<b>{number}.</b> {text}", styles["Publication"])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.35)
    canvas.line(doc.leftMargin, 11 * mm, A4[0] - doc.rightMargin, 11 * mm)
    canvas.setFont("TimesNewRoman", 7.6)
    canvas.setFillColor(GRAY)
    canvas.drawString(doc.leftMargin, 7.2 * mm, "Xujiang Tang | Curriculum Vitae | July 2026")
    canvas.drawRightString(A4[0] - doc.rightMargin, 7.2 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_styles():
    return {
        "Name": ParagraphStyle(
            "Name",
            fontName="TimesNewRoman-Bold",
            fontSize=23,
            leading=24,
            textColor=BLACK,
            alignment=TA_CENTER,
            spaceAfter=1.5,
        ),
        "Tagline": ParagraphStyle(
            "Tagline",
            fontName="TimesNewRoman-Italic",
            fontSize=10.4,
            leading=12,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=2.2,
        ),
        "Contact": ParagraphStyle(
            "Contact",
            fontName="TimesNewRoman",
            fontSize=8.5,
            leading=10.5,
            textColor=GRAY,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "Section": ParagraphStyle(
            "Section",
            fontName="TimesNewRoman-Bold",
            fontSize=10.5,
            leading=12,
            textColor=NAVY,
        ),
        "Body": ParagraphStyle(
            "Body",
            fontName="TimesNewRoman",
            fontSize=9.4,
            leading=11.7,
            textColor=BLACK,
        ),
        "Meta": ParagraphStyle(
            "Meta",
            fontName="TimesNewRoman-Italic",
            fontSize=8.7,
            leading=10.5,
            textColor=GRAY,
        ),
        "Date": ParagraphStyle(
            "Date",
            fontName="TimesNewRoman",
            fontSize=8.5,
            leading=10.5,
            textColor=GRAY,
            alignment=TA_RIGHT,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            fontName="TimesNewRoman",
            fontSize=9.2,
            leading=11.3,
            leftIndent=4 * mm,
            firstLineIndent=-3 * mm,
            textColor=BLACK,
            spaceAfter=0.8,
        ),
        "Publication": ParagraphStyle(
            "Publication",
            fontName="TimesNewRoman",
            fontSize=9.15,
            leading=11.2,
            leftIndent=4 * mm,
            firstLineIndent=-4 * mm,
            textColor=BLACK,
            spaceAfter=3.2,
        ),
        "Profile": ParagraphStyle(
            "Profile",
            fontName="TimesNewRoman",
            fontSize=9.4,
            leading=12,
            textColor=BLACK,
            spaceAfter=2,
        ),
    }


def build():
    register_fonts()
    styles = build_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=12 * mm,
        bottomMargin=15 * mm,
        title="Xujiang Tang - Curriculum Vitae",
        author="Xujiang Tang",
        subject="Academic curriculum vitae",
    )

    story = [
        p("Xujiang Tang", styles["Name"]),
        p("Mathematics, Mathematical Modeling, and Theoretical Machine Learning", styles["Tagline"]),
        p(
            "txj_262538@163.com  |  txj2006.github.io  |  github.com/TXJ2006  |  ORCID: 0009-0008-1127-6420",
            styles["Contact"],
        ),
    ]

    story += section("Research Profile", styles)
    story.append(
        p(
            "Mathematics and Applied Mathematics undergraduate and Research Assistant at HKUST(GZ). Research spans pure and applied mathematics, mathematical modeling, dynamical systems, optimization, and theoretical machine learning, with particular interests in online learning, bandit algorithms, reinforcement learning, and sequential decision-making.",
            styles["Profile"],
        )
    )

    story += section("Education", styles)
    story.append(
        entry(
            "B.Sc. in Mathematics and Applied Mathematics",
            "Yangtze University College of Arts and Sciences, Jingzhou, China",
            "Sep 2023 - Jun 2027",
            [
                "Core strengths: Mathematical Modeling 96; Ordinary Differential Equations 89; Complex Analysis 90; Real Analysis 95; Functional Analysis 99; Abstract Algebra 98.",
                "Additional preparation in probability and mathematical statistics, optimization, numerical computation, and programming.",
            ],
            styles,
        )
    )
    story.append(
        entry(
            "2026 AI4Math Summer School: Lean Formalized Mathematics",
            "Zhejiang University, Hangzhou, China",
            "Jul 2026",
            [
                "Intensive training in Lean-based formal proof, theorem formalization, and computer-assisted mathematical reasoning.",
            ],
            styles,
        )
    )

    story += section("Research Appointment and Experience", styles)
    story.extend(
        [
            entry(
                "Research Assistant",
                "The Hong Kong University of Science and Technology (Guangzhou) | Prof. Tianyuan Jin",
                "2026 - 2027",
                [
                    "Research on online learning, multi-armed bandits, reinforcement learning, and sample- and resource-efficient sequential decision-making.",
                    "Developing mathematical foundations in concentration, martingales, KL divergence, exponential families, and information-theoretic lower bounds.",
                ],
                styles,
            ),
            entry(
                "Algebraic Topology and Polyhedral Products",
                "Ongoing research discussions and guidance from Prof. Stephen D. Theriault, University of Southampton",
                "2026",
                [
                    "Studying Whitehead products, moment-angle complexes, graph Lie algebras, quasi-Lie structures, and low-prime homotopy theory.",
                    "Developing background in prime-local methods, low-characteristic operations, completion methods, and PBW-type arguments.",
                ],
                styles,
            ),
            entry(
                "Bilevel Optimization and Implicit Differentiation",
                "Independent Research",
                "2025 - Present",
                [
                    "Investigated hypergradient computation beyond invertible-Hessian assumptions using differentiable lower-level solution manifolds.",
                    "Developed a curvature-filter framework with broad applicability across AI model classes; submitted to <i>JMLR</i> (Theory and Methods).",
                ],
                styles,
            ),
        ]
    )

    story += section("Additional Research Experience", styles)
    story.extend(
        [
            entry(
                "Deep Hedging under Rough Volatility",
                "Independent Research",
                "2025 - Present",
                [
                    "Developed a fractional-kernel neural architecture for variance-optimal hedging under rough-volatility models.",
                    "Integrated stochastic analysis, neural approximation, and numerical optimization.",
                ],
                styles,
            ),
            entry(
                "Dynamical Systems and Koopman Operator Learning",
                "Research Collaborator, Prof. Yanbing Jia's Group",
                "Mar 2024 - Present",
                [
                    "Proposed a Koopman Operator Linearization Skeleton for long-horizon nonlinear prediction.",
                    "Implemented and evaluated the framework on multiple chaotic-system benchmarks against deep-learning baselines.",
                ],
                styles,
            ),
            entry(
                "Peking University Open-Source LLM Group",
                "Research and Engineering Intern",
                "Jun - Aug 2024",
                [
                    "Contributed to large-language-model deployment, retrieval-augmented generation, and external knowledge integration.",
                ],
                styles,
            ),
        ]
    )

    story.append(PageBreak())
    story += section("Published and Accepted", styles)
    story.extend(
        [
            publication(1, "Q. Li and <b>X. Tang</b>. Robust Optimal Reinsurance and Investment with Inflation Risk: A Game-Theoretic Approach and Explicit Solutions. <i>AIMS Mathematics</i>, 11(3), 7330-7352, 2026. Published. DOI: 10.3934/math.2026302.", styles),
            publication(2, "<b>X. Tang et al.</b> Structure-preserving Koopman Predictive Control for Memristive Neural Dynamics: Input-exact and Commutator-defect Lifting. <i>Biological Cybernetics</i>. Accepted.", styles),
            publication(3, "<b>X. Tang et al.</b> Spectral Network Determinants of Seizure-like Synchronization and Spread in Coupled Hindmarsh-Rose Brain Models. <i>Cognitive Neurodynamics</i>. Accepted.", styles),
        ]
    )

    story += section("Under Review, Submitted, and Manuscripts", styles)
    story.extend(
        [
            publication(1, "<b>X. Tang et al.</b> The Curvature Filter in Bilevel Optimization: Implicit Differentiation Without Nondegeneracy. <i>Journal of Machine Learning Research</i>, Theory and Methods. Submitted.", styles),
            publication(2, "<b>X. Tang</b>, R. Guan, and Z. Wang. Moment Order in Unsupervised Direction Learning. <i>Journal of Machine Learning Research</i>. Submitted.", styles),
            publication(3, "Y. Fan, <b>X. Tang</b>, and R. Guan. Deep Hedging under Rough Volatility: A Fractional Kernel Embedding Approach with Optimal Convergence Rate. <i>Journal of Computational and Applied Mathematics</i>. Under review.", styles),
            publication(4, "<b>X. Tang et al.</b> Trajectory-Level Out-of-Distribution Success Bounds for Deep Autoregressive Models. Manuscript, 2026.", styles),
        ]
    )

    story += section("Selected Technical Notes", styles)
    story.append(
        p(
            "Eight online Markdown notes on sequential decisions: regret and exploration; adaptive data and filtrations; concentration bounds; martingales and optional stopping; KL divergence; exponential-family bandit models; and change-of-measure lower bounds. Available at txj2006.github.io/notes/.",
            styles["Body"],
        )
    )

    story += section("Honors", styles)
    story.append(
        p(
            "Kaggle Silver Medal, Jigsaw Toxic Comment Classification Challenge | Regional Gold Medal, WorldQuant International Quant Championship | Third Prize, UCAS Graduate AI Forum | National Second Prize, National College Students Statistical Modeling Competition | National Second Prize, Hua Zhong Cup Mathematical Modeling Competition",
            styles["Body"],
        )
    )

    story += section("Skills", styles)
    story.append(
        p(
            "<b>Mathematics:</b> mathematical modeling, real and complex analysis, functional analysis, differential equations, abstract algebra, algebraic topology, probability, and optimization.<br/><b>Programming and tools:</b> Python, PyTorch, NumPy, Pandas, C++, R, MATLAB, Lean, Git/GitHub, Linux, LaTeX.<br/><b>Machine learning:</b> online learning, bandit algorithms, reinforcement learning, stochastic modeling, LLM deployment, and retrieval-augmented generation.<br/><b>Languages:</b> Mandarin Chinese; English.",
            styles["Body"],
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()

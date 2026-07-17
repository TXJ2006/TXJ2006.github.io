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
    font_dir = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("TimesNewRoman", str(font_dir / "times.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(font_dir / "timesbd.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Italic", str(font_dir / "timesi.ttf")))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-BoldItalic", str(font_dir / "timesbi.ttf")))
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
        p("Online Learning, Bandit Algorithms, and Mathematical Machine Learning", styles["Tagline"]),
        p(
            "txj_262538@163.com  |  txj2006.github.io  |  github.com/TXJ2006  |  ORCID: 0009-0008-1127-6420",
            styles["Contact"],
        ),
    ]

    story += section("Research Profile", styles)
    story.append(
        p(
            "Undergraduate researcher preparing for doctoral study in online learning, multi-armed bandits, reinforcement learning, and sequential decision-making. Strong mathematical preparation in probability, statistics, and optimization, with experience translating theoretical ideas into reproducible Python and PyTorch implementations.",
            styles["Profile"],
        )
    )

    story += section("Education", styles)
    story.append(
        entry(
            "Visiting Student",
            "The Hong Kong University of Science and Technology (Guangzhou) | Advisor: Prof. Tianyuan Jin",
            "2025 - Present",
            [
                "Research preparation in theoretical machine learning, online learning, bandit algorithms, reinforcement learning, and stochastic optimization.",
                "Systematic study of concentration, martingales, KL divergence, exponential families, and information-theoretic lower bounds for adaptive decisions.",
            ],
            styles,
        )
    )
    story.append(
        entry(
            "B.Sc. in Mathematics and Applied Mathematics",
            "Yangtze University College of Arts and Sciences, Jingzhou, China | GPA: 3.5/4.0",
            "Sep 2023 - Jun 2027",
            [
                "Optimization Theory: 97/100, ranked first in the major; Numerical Computation Methods: 96/100.",
                "Relevant coursework: probability and mathematical statistics, optimization, numerical computation, mathematical analysis, advanced algebra, differential equations, mathematical modeling, and programming.",
            ],
            styles,
        )
    )

    story += section("Relevant Research Preparation", styles)
    story.extend(
        [
            entry(
                "Online Learning, Bandits, and Sequential Decisions",
                "Research Training, HKUST(GZ) | Advisor: Prof. Tianyuan Jin",
                "2025 - Present",
                [
                    "Studied regret minimization, best-arm identification, exploration-exploitation tradeoffs, and adaptive decision-making.",
                    "Authored eight LaTeX notes connecting adaptive probability to confidence bounds, exponential-family models, and change-of-measure lower bounds.",
                    "Developed a focused research agenda around sample- and resource-efficient sequential learning.",
                ],
                styles,
            ),
            entry(
                "Bilevel Optimization and Implicit Differentiation",
                "Independent Research",
                "2025 - Present",
                [
                    "Investigated hypergradient computation beyond invertible-Hessian assumptions using differentiable lower-level solution manifolds.",
                    "Developed a curvature-filter framework based on projection operators and singular behavior near focal sets; submitted to <i>TMLR</i>.",
                ],
                styles,
            ),
            entry(
                "Peking University Open-Source LLM Group",
                "Research and Engineering Intern",
                "Jun 2024 - Aug 2024",
                [
                    "Contributed to large-language-model deployment and retrieval-augmented generation systems.",
                    "Implemented retrieval and knowledge-integration modules for external knowledge use.",
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
        ]
    )

    story.append(PageBreak())
    story += section("Published and Accepted", styles)
    story.extend(
        [
            publication(1, "Q. Li and <b>X. Tang</b>. HKGEduRec: A Knowledge Graph-Enhanced Dynamic Hybrid Framework for Educational Recommendation with Cold-Start Mitigation. <i>IEEE ICMEIM 2025</i>, pp. 1-5. EI-indexed.", styles),
            publication(2, "<b>X. Tang</b> and Q. Li. Logical Gene Encoding: A Bio-Inspired Approach for Energy-Efficient Automated Reasoning. <i>IEIT 2025</i>. EI-indexed. DOI: 10.2991/978-94-6463-803-5_81.", styles),
            publication(3, "Q. Li and <b>X. Tang</b>. Robust Optimal Reinsurance and Investment with Inflation Risk: A Game-Theoretic Approach and Explicit Solutions. <i>AIMS Mathematics</i>. Accepted.", styles),
        ]
    )

    story += section("Under Review and Submitted", styles)
    story.extend(
        [
            publication(1, "<b>X. Tang et al.</b> The Curvature Filter in Bilevel Optimization: Implicit Differentiation Without Nondegeneracy. <i>Transactions on Machine Learning Research</i>. Submitted.", styles),
            publication(2, "Y. Fan, <b>X. Tang</b>, and R. Guan. Deep Hedging under Rough Volatility: A Fractional Kernel Embedding Approach with Optimal Convergence Rate. <i>AIMS Mathematics</i>. Under review.", styles),
            publication(3, "<b>X. Tang et al.</b> Structure-preserving Koopman Predictive Control for Memristive Neural Dynamics: Input-exact and Commutator-defect Lifting. <i>Biological Cybernetics</i>. Under review.", styles),
            publication(4, "<b>X. Tang et al.</b> Spectral Network Determinants of Seizure-like Synchronization and Spread in Coupled Hindmarsh-Rose Brain Models. <i>Cognitive Neurodynamics</i>. Under review.", styles),
        ]
    )

    story += section("Selected Technical Notes", styles)
    story.append(
        p(
            "Eight reproducible LaTeX notes on sequential decisions: regret and exploration; adaptive data and filtrations; concentration bounds; martingales and optional stopping; KL divergence; exponential-family bandit models; and change-of-measure lower bounds. Available at txj2006.github.io/notes/.",
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
            "<b>Programming and tools:</b> Python, PyTorch, NumPy, Pandas, C++, R, MATLAB, Git/GitHub, Linux, LaTeX.<br/><b>Theory and methods:</b> online learning, bandit algorithms, probability, statistical modeling, optimization, stochastic simulation, LLM deployment, retrieval-augmented generation.<br/><b>Languages:</b> Mandarin Chinese; English.",
            styles["Body"],
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()

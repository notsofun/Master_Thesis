"""
RQ1: Topic Language Composition Visualization
=============================================

Visualize the language distribution across each topic using a stacked bar chart.
This script reads from document_topic_mapping.csv and creates interactive HTML plots.

Usage:
    python unsupervised_classification/RQ1/topic_language_composition.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# ── Setup project paths ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "visualizations"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Input data path
DOC_PATH = (
    PROJECT_ROOT
    / "unsupervised_classification"
    / "topic_modeling_results" / "sixth" / "data"
    / "document_topic_mapping.csv"
)

# ── Topic labels (English, from datamap_plot.py) ──
TOPIC_LABELS = {
    -1: "Inclusion and Reform Challenges",
     0: "Separation of Church & State / Japan",
     1: "Christians and Trump's Politics",
     2: "Women and Ordination",
     3: "LGBTQ Issues and Youth Alienation",
     4: "Religion, Sexuality, and Violence",
     5: "Catholic-Lutheran Communion Disputes",
     6: "Church Abuse Victims",
     7: "Religion and Natural Disasters",
     8: "Pro-Life and Women's Autonomy",
     9: "Divine-Human Nature / Jesus",
    10: "Conflicts and Reflections on the Bible",
    11: "2017 Pakistan Religious Violence",
    12: "Communion for Divorced and Remarried",
    13: "Religion in Public vs. Catholic Schools",
    14: "Public Prayer and Religious Freedom",
    15: "Authenticity and Faith in Catholicism",
    16: "Canadian Immigration and Cultural Disputes",
    17: "Pagan-Christian Conflicts",
    18: "Internal Discussions in Catholicism",
    19: "Doctrines of Religion and Salvation",
    20: "Nuns and Religious Culture",
    21: "Adventist Church",
    22: "Social Observations and Religious Critique",
    23: "Historical Critiques of Christian Violence",
    24: "Reforms and Divisions in Catholicism",
    25: "Social Division and Racial Antagonism",
    26: "Priests and Complexities of Faith",
    27: "Bakery and Religious Freedom Legal Conflicts",
    28: "Burke vs. Pope Francis",
    29: "Christian Principles and Criticisms",
    30: "Christian Development / Denominational Differences",
    31: "Antisemitism and Religious Persecution",
    32: "Church Doctrine and Sacramental Validity",
    33: "Healthcare Rights and Institutions",
    34: "Church-State Relations and Charities",
    35: "The Poor, Wealth, and Relief",
    36: "Internal Divisions in Catholicism",
    37: "Religious Beliefs and Social Phenomena",
    38: "Opposition and Impact of Religious Beliefs",
    39: "Religious Controversies and Social Critiques",
    40: "Critique of Money and Religious Manipulation",
    41: "Catholic Stereotypes and Theology",
    42: "Faith and Heresy Debates",
    43: "Missionaries and Confucianism",
    44: "Pope Francis: Conflicts and Mercy",
    45: "Life and Responsibility",
    46: "Religious Beliefs and Social Controversies",
}

# ── Language labels and colors ──
LANG_LABEL = {"en": "English (EN)", "zh": "Chinese (ZH)", "jp": "Japanese (JP)"}
LANG_COLOR = {"en": "#4C78A8", "zh": "#F58518", "jp": "#54A24B"}

# ── Plotly layout base ──
_LAYOUT_BASE = dict(
    font=dict(
        family="Arial, sans-serif",
        size=12,
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
)


def normalize_lang(lang: str) -> str:
    """Normalize language codes: jp → ja, etc."""
    return "ja" if str(lang).lower() in ("ja", "jp") else str(lang).lower()


def create_topic_language_composition_plot(doc_csv_path, output_path):
    """
    Create a stacked bar chart showing language distribution per topic.
    
    Args:
        doc_csv_path: Path to document_topic_mapping.csv
        output_path: Path to save the HTML output
    """
    print(f"Loading data from: {doc_csv_path}")
    df = pd.read_csv(doc_csv_path)
    df = df.drop(df[df['topic'] == -1].index)
    
    # Normalize language column
    df["lang"] = df["lang"].apply(normalize_lang).replace({"ja": "jp"})
    
    # Filter out noise topic (-1) if needed, or keep all
    # For now, keep all including -1
    
    # Count documents per topic and language
    topic_lang = df.groupby(["topic", "lang"], as_index=False).size().rename(columns={"size": "n_docs"})
    
    # Calculate percentages
    topic_lang["pct"] = (
        topic_lang["n_docs"] / topic_lang.groupby("topic")["n_docs"].transform("sum") * 100
    ).round(1)
    
    # Pivot to get language as columns
    all_topics_sorted = sorted(topic_lang["topic"].unique())
    lang_pct_pivot = (
        topic_lang.pivot(index="topic", columns="lang", values="pct")
        .fillna(0)
        .reindex(all_topics_sorted)
    )
    
    print(f"Topics found: {len(lang_pct_pivot)}")
    print(f"Language distribution:\n{lang_pct_pivot}")
    
    # Create interactive stacked bar chart
    fig = go.Figure()
    
    for lang in ["en", "zh", "jp"]:
        if lang not in lang_pct_pivot.columns:
            print(f"Warning: Language {lang} not found in data")
            continue
        
        # Create topic labels with topic number and name
        x_labels = [
            f"T{int(t)}: {TOPIC_LABELS.get(int(t), f'Topic {t}')[:30]}"
            for t in lang_pct_pivot.index
        ]
        
        fig.add_trace(go.Bar(
            name=LANG_LABEL.get(lang, lang),
            x=x_labels,
            y=lang_pct_pivot[lang].values,
            marker_color=LANG_COLOR[lang],
            hovertemplate=(
                f"{LANG_LABEL.get(lang, lang)}<br>"
                "%{x}<br>"
                "Percentage: %{y:.1f}%<extra></extra>"
            ),
        ))
    
    # Update layout
    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="stack",
        title=dict(
            text=(
                "Topic Language Composition (Stacked %)<br>"
                "<sup>Language distribution across each topic; "
                "shows which topics are shared across languages and which are language-dominated</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Topic (with Label)",
            tickangle=-45,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title="Language Distribution (%)",
        ),
        legend=dict(
            title="Language",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=700,
        width=1600,
        hovermode="x unified",
    )
    
    # Save to HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"✅ Plot saved to: {output_path}")
    
    return fig


def create_topic_count_plot(doc_csv_path, output_path):
    """
    Create a bar chart showing document count per topic by language.
    
    Args:
        doc_csv_path: Path to document_topic_mapping.csv
        output_path: Path to save the HTML output
    """
    print(f"Loading data from: {doc_csv_path}")
    df = pd.read_csv(doc_csv_path)
    df = df.drop(df[df['topic'] == -1].index)
    
    # Normalize language column
    df["lang"] = df["lang"].apply(normalize_lang).replace({"ja": "jp"})
    
    # Count documents per topic and language
    topic_lang = df.groupby(["topic", "lang"], as_index=False).size().rename(columns={"size": "n_docs"})
    
    # Pivot to get language as columns
    all_topics_sorted = sorted(topic_lang["topic"].unique())
    lang_count_pivot = (
        topic_lang.pivot(index="topic", columns="lang", values="n_docs")
        .fillna(0)
        .reindex(all_topics_sorted)
    )
    
    print(f"Topics found: {len(lang_count_pivot)}")
    
    # Create interactive grouped bar chart
    fig = go.Figure()
    
    for lang in ["en", "zh", "jp"]:
        if lang not in lang_count_pivot.columns:
            print(f"Warning: Language {lang} not found in data")
            continue
        
        # Create topic labels
        x_labels = [
            f"T{int(t)}: {TOPIC_LABELS.get(int(t), f'Topic {t}')[:30]}"
            for t in lang_count_pivot.index
        ]
        
        fig.add_trace(go.Bar(
            name=LANG_LABEL.get(lang, lang),
            x=x_labels,
            y=lang_count_pivot[lang].values,
            marker_color=LANG_COLOR[lang],
            hovertemplate=(
                f"{LANG_LABEL.get(lang, lang)}<br>"
                "%{x}<br>"
                "Documents: %{y}<extra></extra>"
            ),
        ))
    
    # Update layout
    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="group",
        title=dict(
            text=(
                "Topic Document Count by Language<br>"
                "<sup>Grouped bar chart showing document count for each topic-language combination</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Topic (with Label)",
            tickangle=-45,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title="Document Count",
        ),
        legend=dict(
            title="Language",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=700,
        width=1600,
        hovermode="x unified",
    )
    
    # Save to HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"✅ Plot saved to: {output_path}")
    
    return fig


def main():
    if not DOC_PATH.exists():
        print(f"❌ Error: Cannot find {DOC_PATH}")
        sys.exit(1)
    
    print("=" * 70)
    print("RQ1: Topic Language Composition Visualization")
    print("=" * 70)
    
    # Create stacked percentage plot
    output_pct = DATA_DIR / "topic_language_composition_stacked.html"
    create_topic_language_composition_plot(DOC_PATH, output_pct)
    
    # Create grouped count plot
    output_count = DATA_DIR / "topic_language_composition_grouped.html"
    create_topic_count_plot(DOC_PATH, output_count)
    
    print("\n" + "=" * 70)
    print("✅ All visualizations complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "gemini_annotations_flat.csv"

df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Unique question_ids:", df["question_id"].nunique())
print("Example rows:")
print(df.head(8).to_string(index=False))

if "question_type" not in df.columns and "qtype" in df.columns:
    df = df.rename(columns={"qtype": "question_type"})

QUESTIONS_PER_DOMAIN = 10

def compute_question_type(qid: int) -> str:
    within = qid % QUESTIONS_PER_DOMAIN
    return "niche" if within < 5 else "broad"

df["question_type"] = df["question_id"].apply(compute_question_type)

df["domain"] = df["domain"].str.strip()
df["prompt_type"] = df["prompt_type"].str.strip().str.lower()

for col in ["response_accuracy", "source_validity"]:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace({"nan": ""})  # in case NA got stringified

df["resp_acc_bool"] = df["response_accuracy"].map({"yes": True, "no": False})
df["src_valid_bool"] = df["source_validity"].map({"yes": True, "no": False})

DOMAINS = ["Medicine", "Law", "Tech", "Sports", "Fashion"]
QTYPES = ["niche", "broad"]
PROMPTS = ["direct", "precise", "verification", "icl"]

def pct(series: pd.Series) -> float:
    if series.dropna().empty:
        return float("nan")
    return series.mean() * 100.0

def add_value_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height):
            continue
        ax.text(
            p.get_x() + p.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

acc_niche = []
acc_broad = []

for d in DOMAINS:
    for qt, out_list in [("niche", acc_niche), ("broad", acc_broad)]:
        subset = df[(df["domain"] == d) & (df["question_type"] == qt)]
        out_list.append(pct(subset["resp_acc_bool"]))

fig, ax = plt.subplots(figsize=(7, 4))
x = list(range(len(DOMAINS)))
width = 0.35

ax.bar([i - width/2 for i in x], acc_niche, width=width, label="niche")
ax.bar([i + width/2 for i in x], acc_broad, width=width, label="broad")

ax.set_xticks(x)
ax.set_xticklabels(DOMAINS, rotation=20)
ax.set_ylim(0, 105)
ax.set_ylabel("Response accuracy (%)")
ax.set_title("Gemini 2.5 Pro response accuracy by domain and question type")
ax.grid(axis="y", alpha=0.3)
add_value_labels(ax)
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig("gemini_acc_domain_qtype.png", dpi=200)

src_niche = []
src_broad = []

for d in DOMAINS:
    for qt, out_list in [("niche", src_niche), ("broad", src_broad)]:
        subset = df[(df["domain"] == d) & (df["question_type"] == qt)]
        out_list.append(pct(subset["src_valid_bool"]))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([i - width/2 for i in x], src_niche, width=width, label="niche")
ax.bar([i + width/2 for i in x], src_broad, width=width, label="broad")

ax.set_xticks(x)
ax.set_xticklabels(DOMAINS, rotation=20)
ax.set_ylim(0, 105)
ax.set_ylabel("Source validity (%)")
ax.set_title("Gemini 2.5 Pro source validity by domain and question type")
ax.grid(axis="y", alpha=0.3)
add_value_labels(ax)
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig("gemini_src_domain_qtype.png", dpi=200)

acc_prompt_niche = []
acc_prompt_broad = []

for p in PROMPTS:
    for qt, out_list in [("niche", acc_prompt_niche), ("broad", acc_prompt_broad)]:
        subset = df[(df["prompt_type"] == p) & (df["question_type"] == qt)]
        out_list.append(pct(subset["resp_acc_bool"]))

fig, ax = plt.subplots(figsize=(7, 4))
x = list(range(len(PROMPTS)))
ax.bar([i - width/2 for i in x], acc_prompt_niche, width=width, label="niche")
ax.bar([i + width/2 for i in x], acc_prompt_broad, width=width, label="broad")

ax.set_xticks(x)
ax.set_xticklabels(PROMPTS, rotation=15)
ax.set_ylim(0, 105)
ax.set_ylabel("Response accuracy (%)")
ax.set_title("Gemini 2.5 Pro response accuracy by prompt type and question type")
ax.grid(axis="y", alpha=0.3)
add_value_labels(ax)
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig("gemini_acc_prompt_qtype.png", dpi=200)

src_prompt_niche = []
src_prompt_broad = []

for p in PROMPTS:
    for qt, out_list in [("niche", src_prompt_niche), ("broad", src_prompt_broad)]:
        subset = df[(df["prompt_type"] == p) & (df["question_type"] == qt)]
        out_list.append(pct(subset["src_valid_bool"]))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([i - width/2 for i in x], src_prompt_niche, width=width, label="niche")
ax.bar([i + width/2 for i in x], src_prompt_broad, width=width, label="broad")

ax.set_xticks(x)
ax.set_xticklabels(PROMPTS, rotation=15)
ax.set_ylim(0, 105)
ax.set_ylabel("Source validity (%)")
ax.set_title("Gemini 2.5 Pro source validity by prompt type and question type")
ax.grid(axis="y", alpha=0.3)
add_value_labels(ax)
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig("gemini_src_prompt_qtype.png", dpi=200)
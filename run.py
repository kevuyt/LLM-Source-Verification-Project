import os, csv, re
from openai import OpenAI

MODEL="gpt-5"
TEMPERATURE=0
MAX_OUTPUT_TOKENS=500
VERBOSITY="low"
REASONING_EFFORT="minimal"

DOMAINS = ["Medicine","Law","Tech","Sports","Fashion"]
QUESTIONS_PER_DOMAIN = 20

TEMPLATES={
    "direct":"Answer the question.",
    "precise":"Answer the question concisely. Cite 1-3 reputable sources (publishers, journals, official sites; avoid forums/blogs unless official).",
    "verification":"Verify each claim with a source before answering. Provide the answer, then 1 sentence justifying source relevance.",
    "icl":"ICL"
}

ICL_BLOCKS={
    "Medicine":"<<MEDICINE 2-3 SHOT EXEMPLARS>>",
    "Law":"<<LAW 2-3 SHOT EXEMPLARS>>",
    "Tech":"<<TECH 2-3 SHOT EXEMPLARS>>",
    "Sports":"<<SPORTS 2-3 SHOT EXEMPLARS>>",
    "Fashion":"<<FASHION 2-3 SHOT EXEMPLARS>>"
}

client=OpenAI()

def load_questions(p): 
    with open(p,"r",encoding="utf-8") as f: 
        return [ln.strip() for ln in f if ln.strip()]

def domain_for_index(i):
    if i < 0 or i >= len(DOMAINS)*QUESTIONS_PER_DOMAIN: 
        raise IndexError(f"Question index {i} out of range!!")
    return DOMAINS[(i // QUESTIONS_PER_DOMAIN)]

def make_prompt(q,style,dom):
    base=("You are a careful research assistant. Keep answers under 150 words. "
          "Put real URLs you used under a final line: 'Sources:'.\n")
    if style != "icl":
        return f"{base}Instruction: {TEMPLATES[style]}\nQuestion: {q}"
    icl=ICL_BLOCKS[dom]
    return f"{base}Exemplars for {dom}:\n{icl}\nNow answer the target question.\nQuestion: {q}"

def ask_gpt5(prompt):
    r=client.responses.create(
        model=MODEL,
        input=[{"role":"user","content":prompt}],
        instructions="Follow instructions exactly; include only URLs actually used on a final 'Sources:' line.",
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        verbosity=VERBOSITY,
        reasoning_effort=REASONING_EFFORT
    )
    return r.output_text.strip()

def extract_urls(text):
    urls=re.findall(r'https?://\S+',text)
    out=[]; seen=set()
    for u in urls:
        u=u.rstrip('.,);]') 
        if u not in seen:
            seen.add(u); out.append(u)
    return "; ".join(out)

def main():
    qs=load_questions("questions.txt")
    styles=["direct","precise","verification","icl"]
    with open("gpt5_responses.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["prompt_type","response","sources"])
        for i,q in enumerate(qs):
            dom=domain_for_index(i)
            for s in styles:
                prompt=make_prompt(q,s,dom)
                ans=ask_gpt5(prompt)
                w.writerow([s,ans,extract_urls(ans)])

if __name__=="__main__":
    main()

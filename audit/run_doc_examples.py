"""Execute every ```python block in README.md and all docs/*.md (each file in its own namespace,
blocks cumulative). Reports pass / fail (NameError for intentionally undefined placeholders noted).
Run: python audit/run_doc_examples.py"""
import re, glob, traceback, warnings, io, contextlib
import matplotlib; matplotlib.use("Agg")
warnings.simplefilter("ignore")
files = ["README.md"] + sorted(glob.glob("docs/**/*.md", recursive=True))
for f in files:
    blocks = re.findall(r"```python\n(.*?)```", open(f, encoding="utf-8").read(), flags=re.DOTALL)
    if not blocks:
        continue
    ns = {}
    for i, b in enumerate(blocks):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(b, f"{f}[{i}]", "exec"), ns)
            print(f"PASS {f} block {i}")
        except Exception as e:
            print(f"FAIL {f} block {i}: {type(e).__name__}: {str(e)[:160]}")

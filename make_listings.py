"""Extract exact code excerpts for the appendix listings."""
import re, lift_env as L, inspect
for name in ("probe", "lift_bound", "shield_grip", "shield_time", "features", "run_episode", "score", "model_grip"):
    open(f"paper/code_{name}.py", "w").write(inspect.getsource(getattr(L, name)))
open("paper/code_mj1.py", "w").write(inspect.getsource(L.LiftRig.mj1))
open("paper/code_xml.xml", "w").write(L.XML.strip())
src = open("train_es.py").read(); i = src.index("def main"); open("paper/code_train.py", "w").write(src[i:src.index("if __name__")].rstrip() + "\n")
print("ok")

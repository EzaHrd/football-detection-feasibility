"""Build a split manifest by SOURCE group (not by random frame). Does not modify or copy the original data."""
import pathlib, re, csv, collections
root=pathlib.Path("football_detection/data/work/soccana/V1")
base=lambda s: re.sub(r"_sahi_\d+","",re.sub(r"_\d+$","",s))
SPLIT={"v3":"train","v7":"train","v2_temp":"train","spt_v2":"train","tbd_v2":"val","v12":"test","v5_temp":"test"}
rows=[]
for sp in ["train","test"]:
    for p in sorted((root/"images"/sp).glob("*.jpg")):
        b=base(p.stem); rows.append((str(p.relative_to(root)),b,sp,SPLIT[b]))
out=pathlib.Path("football_detection/data/work/splits"); out.mkdir(parents=True,exist_ok=True)
with open(out/"split_by_source_v1.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["image","source_group","original_hf_split","new_split"]); w.writerows(rows)
c=collections.Counter(r[3] for r in rows); print(c)
# sanity check: no source group appears in more than one split
g=collections.defaultdict(set)
for r in rows: g[r[1]].add(r[3])
assert all(len(v)==1 for v in g.values()); print("OK: each source group is in exactly one split")

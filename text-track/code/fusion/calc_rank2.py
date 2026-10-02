import math

teams = [
    ("LLMs", 0.984649, 0.928239),
    ("南枝大学", 0.975675, 0.904898),
    ("gogogo", 0.975518, 0.906452),
    ("打假打假", 0.974210, 0.952037),
    ("AIGCD", 0.968151, 0.835918),
    ("火眼+金睛", 0.961710, None),
    ("陪跑队不对", 0.961868, 0.916647),
    ("kawhi00", 0.964789, 0.940992),
    ("初浙沥以萧飒", 0.960350, 0.926005),
    ("你们给我好好检测啊", 0.956524, 0.833900),
    ("zxg", 0.948162, 0.887177),
    ("汪汪队立大功", 0.940197, 0.936446),
    ("好易评-OPC", 0.938636, 0.943713),
    ("小马过河队", 0.933736, 0.934680),
    ("yndky", 0.895206, 0.853137),
    ("务必休息半天", 0.885489, 0.893818),
]
rows = []
for name, t, i in teams:
    st = math.exp(4*t)
    si = math.exp(4*i) if i is not None else 1.0
    rows.append((name, t, i, st, si, st+si))
rows.sort(key=lambda r: r[5], reverse=True)
print(f"{'#':>2} {'team':<12} {'text':>8} {'image':>8} {'e^4t':>8} {'e^4i':>8} {'Score':>9}")
for k, (name, t, i, st, si, s) in enumerate(rows, 1):
    mark = "  <-- 我们" if name == "南枝大学" else ""
    print(f"{k:>2} {name:<12} {t:>8.5f} {(i if i is not None else float('nan')):>8.5f} {st:>8.2f} {si:>8.2f} {s:>9.2f}{mark}")

cur = math.exp(4*0.975675) + math.exp(4*0.904898)
print("\n南枝大学 Score =", round(cur, 3))
# to beat rank5 (gogogo 87.07 / 初浙沥 87.23):
r5 = math.exp(4*0.975518) + math.exp(4*0.906452)
r4 = math.exp(4*0.960350) + math.exp(4*0.926005)
print("第5名 gogogo Score =", round(r5, 3), "| 第4名 初浙沥 Score =", round(r4, 3))
need_t = math.log(r5 - math.exp(4*0.904898))/4
print("\n图片固定0.9049: text >", round(need_t, 5), "(需再+%.4f)" % (need_t - 0.975675))
need_i = math.log(r5 - math.exp(4*0.975675))/4
print("文本固定0.9757: image >", round(need_i, 5), "(需再+%.4f)" % (need_i - 0.904898))

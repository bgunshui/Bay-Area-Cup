import math

teams = [
    # name, text_auc, image_auc (from screenshots; image "--" = no submission)
    ("LLMs", 0.984649, 0.918363),
    ("gogogo", 0.975518, 0.906452),
    ("打假打假", 0.973766, 0.951149),
    ("南枝大学", 0.973328, 0.898039),
    ("AIGCD", 0.968151, 0.835918),
    ("火眼+金睛", 0.961662, None),
    ("kawhi00", 0.960573, 0.939957),
    ("初浙沥以萧飒", 0.960350, 0.926005),
    ("你们给我好好检测啊", 0.956524, 0.833900),
    ("陪跑队不对", 0.952627, 0.916647),
    ("zxg", 0.948162, 0.887177),
    ("汪汪队立大功", 0.940197, 0.936444),
    ("好易评-OPC", 0.937868, 0.940545),
    ("小马过河队", 0.933736, 0.934680),
    ("yndky", 0.895206, 0.853137),
    ("务必休息半天", 0.885489, 0.893818),
]

rows = []
for name, t, i in teams:
    st = math.exp(4 * t)
    si = math.exp(4 * i) if i is not None else 1.0  # no image submission -> e^0 = 1
    rows.append((name, t, i, st, si, st + si))

rows.sort(key=lambda r: r[5], reverse=True)
print(f"{'#':>2} {'team':<12} {'text':>8} {'image':>8} {'e^4t':>8} {'e^4i':>8} {'Score':>9}")
for k, (name, t, i, st, si, s) in enumerate(rows, 1):
    mark = "  <-- 我们" if name == "南枝大学" else ""
    print(f"{k:>2} {name:<12} {t:>8.5f} {(i if i is not None else float('nan')):>8.5f} {st:>8.2f} {si:>8.2f} {s:>9.2f}{mark}")

# detail for our team: how far from neighbors
print("\n南枝大学 Score =", round(math.exp(4*0.973328) + math.exp(4*0.898039), 3))
print("差好易评-OPC:", round((math.exp(4*0.937868)+math.exp(4*0.940545)) - (math.exp(4*0.973328)+math.exp(4*0.898039)), 3))
print("差汪汪队:", round((math.exp(4*0.940197)+math.exp(4*0.936444)) - (math.exp(4*0.973328)+math.exp(4*0.898039)), 3))

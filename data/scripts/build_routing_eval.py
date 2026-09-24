"""生成 100 条路由分类评测集（方案 §5.5）。

输出 benchmarks/eval_sets/routing.jsonl，每行 {query, level}；
覆盖 L1-L4 各 25 条，跨五大联赛与多种队名/句式。

设计：
- L1 事实查询：积分榜/排名/赛程/比分/战绩；
- L2 单场快速预测：谁赢/胜平负/胜负；
- L3 深度分析：详细/深度/伤停/战术/盘口/赔率；
- L4 全局模拟：模拟赛季/前四/夺冠/降级概率。
"""

import itertools
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "benchmarks" / "eval_sets" / "routing.jsonl"

# 五大联赛代表队（football-data.co.uk 官方名）
TEAMS = {
    "E0": ["Man United", "Liverpool", "Arsenal", "Chelsea", "Man City", "Tottenham", "Newcastle"],
    "SP1": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Villarreal", "Ath Bilbao"],
    "D1": ["Bayern Munich", "Dortmund", "RB Leipzig", "Leverkusen", "Wolfsburg"],
    "I1": ["Juventus", "Inter", "Milan", "Napoli", "Roma", "Lazio"],
    "F1": ["Paris SG", "Marseille", "Lyon", "Monaco", "Lille"],
}
LEAGUE_NAMES = {"E0": "英超", "SP1": "西甲", "D1": "德甲", "I1": "意甲", "F1": "法甲"}


def _fill(tpl: str, *args) -> str:
    """按 positional placeholder 数量填充。"""
    n = tpl.count("{}")
    a = list(args)
    while len(a) < n:
        a.append(a[-1] if a else "")
    return tpl.format(*a[:n])


def _pairs_league_dist():
    pairs: list[tuple[str, str]] = []

    # ===== L1 事实查询（25 条）=====
    league_tpls = [
        "现在{lg}的积分榜",
        "{lg}最新积分榜",
        "{lg}联赛排名",
        "{lg}这赛季积分榜是什么样",
        "{lg}第几名",
        "{lg}目前积分榜",
        "{lg}的排名情况",
        "{lg}赛季赛程",
        "{lg}这周末比赛几点开始",
    ]
    team_tpls = [
        "{t}最近战绩怎么样",
        "{t}最近5场比分是多少",
        "{t}上一场踢得怎么样",
        "{t}这赛季积分榜第几",
        "{t}的主场战绩",
    ]
    pair_tpls = [
        "{h}和{a}最近一次交手比分是多少",
        "{h}上次对阵{a}什么比分",
    ]
    used: set[str] = set()
    for lg_name in LEAGUE_NAMES.values():
        for tpl in league_tpls:
            if len(pairs) >= 25:
                break
            q = tpl.format(lg=lg_name)
            if q in used:
                continue
            used.add(q)
            pairs.append((q, "L1"))
        if len(pairs) >= 25:
            break
    for teams in TEAMS.values():
        if len(pairs) >= 25:
            break
        for t in teams:
            if len(pairs) >= 25:
                break
            for tpl in team_tpls:
                if len(pairs) >= 25:
                    break
                q = tpl.format(t=t)
                if q in used:
                    continue
                used.add(q)
                pairs.append((q, "L1"))
    for teams in TEAMS.values():
        if len(pairs) >= 25:
            break
        for h, a in itertools.combinations(teams, 2):
            if len(pairs) >= 25:
                break
            for tpl in pair_tpls:
                if len(pairs) >= 25:
                    break
                q = tpl.format(h=h, a=a)
                if q in used:
                    continue
                used.add(q)
                pairs.append((q, "L1"))

    # ===== L2 单场快速预测（25 条）=====
    l2_tpls = [
        "{h}对{a}谁会赢",
        "{h}对{a}胜负",
        "{h}对{a}胜平负",
        "{h}对{a}谁赢面大",
        "{h}对{a}的赛果预测",
        "{h}对{a}哪队更强",
        "{h}对阵{a}胜负如何",
    ]
    used = set()
    for teams in TEAMS.values():
        if len(pairs) - 25 >= 25:
            break
        for h, a in itertools.combinations(teams, 2):
            if len(pairs) - 25 >= 25:
                break
            tpl = l2_tpls[(len(pairs) - 25) % len(l2_tpls)]
            q = tpl.format(h=h, a=a)
            if q in used:
                continue
            used.add(q)
            pairs.append((q, "L2"))

    # ===== L3 深度分析（25 条）=====
    l3_tpls = [
        "详细分析{h}对{a}，考虑双方伤停",
        "深度分析{h}对{a}的战术对抗",
        "从战术层面深度解析{h}对{a}",
        "{h}对{a}详细分析，结合伤停和赛程密度",
        "全面分析{h}对{a}，包含盘口与赔率",
        "分析{h}对{a}的阵容深度",
        "{h}对{a}深度复盘，要考虑伤停影响",
        "{h}对{a}的盘口赔率怎么看",
    ]
    used = set()
    for teams in TEAMS.values():
        if len(pairs) - 50 >= 25:
            break
        for h, a in itertools.combinations(teams, 2):
            if len(pairs) - 50 >= 25:
                break
            tpl = l3_tpls[(len(pairs) - 50) % len(l3_tpls)]
            q = tpl.format(h=h, a=a)
            if q in used:
                continue
            used.add(q)
            pairs.append((q, "L3"))

    # ===== L4 赛季模拟（25 条）=====
    l4_tpls = [
        "模拟本赛季{lg}前四的归属概率",
        "按剩余赛程模拟，{lg}哪些球队降级概率最大",
        "模拟{lg}赛季夺冠概率",
        "{lg}剩余赛程蒙特卡洛模拟前四",
        "本赛季{lg}谁夺冠概率最高",
        "{lg}前四争夺模拟",
        "模拟{lg}赛季最终排名",
        "{lg}降级区模拟",
        "{lg}夺冠概率分布",
        "蒙特卡洛模拟{lg}赛季结局",
    ]
    used = set()
    while len(pairs) - 75 < 25:
        for lg_name in LEAGUE_NAMES.values():
            if len(pairs) - 75 >= 25:
                break
            tpl = l4_tpls[(len(pairs) - 75) % len(l4_tpls)]
            q = tpl.format(lg=lg_name)
            if q in used:
                continue
            used.add(q)
            pairs.append((q, "L4"))

    return pairs


def main() -> None:
    pairs = _pairs_league_dist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for q, lv in pairs:
            f.write(json.dumps({"query": q, "level": lv}, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for _, lv in pairs:
        counts[lv] = counts.get(lv, 0) + 1
    print(f"wrote {len(pairs)} queries to {OUT}")
    for lv in ("L1", "L2", "L3", "L4"):
        print(f"  {lv}: {counts.get(lv, 0)}")


if __name__ == "__main__":
    main()

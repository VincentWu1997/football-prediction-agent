"""确定性 RAG 语料生成器：基于 f_matches 2425 赛季合成 500 篇短文。

设计原则：
- 可复现：固定 SEED=42、固定赛季、固定 PER_WEEK_DOCS，重跑结果一致；
- 合规：用 Player{POSITION}{NN} 代号，不涉及真实球员姓名；
- 结构化/非结构化边界：只生成伤停/战术/采访/赛前新闻，不复制比分/积分/赔率；
- 每篇 150-300 字中文短文，带 doc_source/doc_type/league/team/doc_date 元数据。

产物：data/rag_corpus/docs.jsonl，每行一个 JSON 对象。
"""

import json
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHES_CSV = REPO_ROOT / "data" / "processed" / "matches.csv"
OUT_DIR = REPO_ROOT / "data" / "rag_corpus"
OUT_PATH = OUT_DIR / "docs.jsonl"

SEED = 42
SEASON = "2425"
TARGET_DOCS = 500

POSITIONS = ("GK", "DF", "MF", "FW")
PLAYER_POOL = {pos: [f"{pos}{i:02d}" for i in range(1, 16)] for pos in POSITIONS}

INJURY_TYPES = (
    "腿筋拉伤", "膝盖软骨损伤", "腹股沟拉伤",
    "脚踝扭伤", "小腿肌肉撕裂", "肩锁关节脱位",
)
RECOVERY_WEEKS = (1, 2, 3, 4, 6, 8)

FORMATIONS = ("4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "3-4-3", "4-1-4-1")
TACTICS_FOCUS = (
    "高位逼抢", "防守反击", "控球渗透", "边路传中",
    "中场绞杀", "快速转换", "区域防守", "人盯人防守",
)

INTERVIEW_THEMES = (
    "备战态度", "对手评价", "战术部署", "球员状态",
    "赛季目标", "主场优势", "伤病应对", "阵容轮换",
)

NEWS_THEMES = (
    "近期战绩回顾", "历史交锋记录", "赛季排名形势",
    "主场客场对比", "关键球员表现", "赛程密度影响",
)


def _sample_match(rng: random.Random, df: pd.DataFrame) -> dict:
    """从给定比赛集随机抽一场，返回联赛/日期/主客队。"""
    row = df.sample(n=1, random_state=rng.randint(0, 2**31)).iloc[0]
    return {
        "league": row["league"],
        "match_date": row["match_date"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
    }


def _pick_players(rng: random.Random, n: int) -> list[str]:
    """随机选 n 名球员（跨位置）。"""
    pool = [p for pos in POSITIONS for p in PLAYER_POOL[pos]]
    return rng.sample(pool, k=n)


def _render_injury(rng: random.Random, m: dict) -> dict:
    """伤停报道：1-2 名球员受伤 + 预计缺阵周数。"""
    players = _pick_players(rng, rng.randint(1, 2))
    injury = rng.choice(INJURY_TYPES)
    weeks = rng.choice(RECOVERY_WEEKS)
    team = rng.choice([m["home_team"], m["away_team"]])
    date_str = (m["match_date"] - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    opp = m["away_team"] if team == m["home_team"] else m["home_team"]
    team_slug = team.replace(" ", "")

    player_list = "、".join(players)
    content = (
        f"{date_str} 训练消息：{team} 的 {player_list} 在近期训练中{injury}，"
        f"经医疗团队评估预计缺阵约 {weeks} 周。"
        f"这对主帅的排兵布阵带来不小的考验，"
        f"尤其是即将对阵{opp}的比赛。"
        f"俱乐部医疗部门表示将密切关注恢复进度，力争尽早回归赛场。"
    )
    doc_id = f"injury-{m['league']}-{SEASON}-{date_str}-{team_slug}"
    source = f"https://sports-agent.local/news/{m['league']}/{SEASON}/{date_str}/{team_slug}.html"
    return {
        "doc_id": doc_id,
        "content": content,
        "doc_source": source,
        "doc_type": "injury",
        "league": m["league"],
        "team": team,
        "doc_date": date_str,
    }


def _render_tactics(rng: random.Random, m: dict) -> dict:
    """战术分析：阵型 + 战术焦点 + 关键球员对位。"""
    formation = rng.choice(FORMATIONS)
    focus = rng.choice(TACTICS_FOCUS)
    team = rng.choice([m["home_team"], m["away_team"]])
    key_players = _pick_players(rng, 2)
    date_str = (m["match_date"] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    opp = m["away_team"] if team == m["home_team"] else m["home_team"]
    team_slug = team.replace(" ", "")

    content = (
        f"{date_str} 战术前瞻：{team} 本场比赛预计采用 {formation} 阵型，"
        f"战术核心围绕{focus}展开。"
        f"{key_players[0]} 和 {key_players[1]} 的发挥将至关重要，"
        f"两人在前场的跑动和配合能力决定了球队进攻的层次感。"
        f"面对{opp}的防守体系，"
        f"{team}需要在{focus}和阵地战之间找到平衡，"
        f"才能在主场争取主动。"
    )
    doc_id = f"tactics-{m['league']}-{SEASON}-{date_str}-{team_slug}"
    source = f"https://sports-agent.local/tactics/{m['league']}/{SEASON}/{date_str}/{team_slug}.html"
    return {
        "doc_id": doc_id,
        "content": content,
        "doc_source": source,
        "doc_type": "tactics",
        "league": m["league"],
        "team": team,
        "doc_date": date_str,
    }


def _render_interview(rng: random.Random, m: dict) -> dict:
    """赛前采访：教练引语 + 主题。"""
    theme = rng.choice(INTERVIEW_THEMES)
    team = rng.choice([m["home_team"], m["away_team"]])
    date_str = (m["match_date"] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    opp = m["away_team"] if team == m["home_team"] else m["home_team"]
    team_slug = team.replace(" ", "")

    content = (
        f"{date_str} 赛前新闻发布会：{team} 主帅就{theme}回答了媒体提问。"
        f"他表示：'我们对这场比赛做了充分准备，球员们状态良好。"
        f"对手实力很强，但我们有信心在主场拿出好的表现。'"
        f"当被问及阵容选择时，主帅强调{theme}是本场比赛的关键因素之一，"
        f"教练组已针对{opp}的特点制定了相应方案。"
    )
    doc_id = f"interview-{m['league']}-{SEASON}-{date_str}-{team_slug}"
    source = f"https://sports-agent.local/interview/{m['league']}/{SEASON}/{date_str}/{team_slug}.html"
    return {
        "doc_id": doc_id,
        "content": content,
        "doc_source": source,
        "doc_type": "interview",
        "league": m["league"],
        "team": team,
        "doc_date": date_str,
    }


def _render_news(rng: random.Random, m: dict) -> dict:
    """综合赛前新闻：赛季节点 + 看点。"""
    theme = rng.choice(NEWS_THEMES)
    date_str = (m["match_date"] - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    home_slug = m["home_team"].replace(" ", "")
    away_slug = m["away_team"].replace(" ", "")

    content = (
        f"{date_str} 赛前综述：{m['home_team']} vs {m['away_team']}即将打响。"
        f"本场焦点在于{theme}。"
        f"{m['home_team']}坐镇主场，近期状态起伏不定；"
        f"{m['away_team']}客场作战但本赛季表现可圈可点。"
        f"双方上次交锋中{m['away_team']}略占上风，"
        f"本场{m['home_team']}势必力争复仇。"
        f"赛季进入关键阶段，每一分都对排名产生重要影响。"
    )
    doc_id = f"news-{m['league']}-{SEASON}-{date_str}-{home_slug}-{away_slug}"
    source = (
        f"https://sports-agent.local/news/"
        f"{m['league']}/{SEASON}/{date_str}/{home_slug}-{away_slug}.html"
    )
    return {
        "doc_id": doc_id,
        "content": content,
        "doc_source": source,
        "doc_type": "news",
        "league": m["league"],
        "team": m["home_team"],
        "doc_date": date_str,
    }


RENDERERS = {
    "injury": _render_injury,
    "tactics": _render_tactics,
    "interview": _render_interview,
    "news": _render_news,
}
TYPE_WEIGHTS = [0.30, 0.30, 0.20, 0.20]


def main() -> None:
    rng = random.Random(SEED)
    df = pd.read_csv(MATCHES_CSV, parse_dates=["match_date"])
    df = df[df["season"].astype(str) == SEASON].copy()
    if df.empty:
        raise ValueError(f"赛季 {SEASON} 无数据")

    # 直接从全部比赛中放回抽样，生成 TARGET_DOCS 篇
    docs: list[dict] = []
    for _ in range(TARGET_DOCS):
        m = _sample_match(rng, df)
        doc_type = rng.choices(list(RENDERERS.keys()), weights=TYPE_WEIGHTS)[0]
        doc = RENDERERS[doc_type](rng, m)
        docs.append(doc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 统计
    type_counts = {t: sum(1 for d in docs if d["doc_type"] == t) for t in RENDERERS}
    league_counts = {lg: sum(1 for d in docs if d["league"] == lg) for lg in df["league"].unique()}
    print(f"生成 {len(docs)} 篇文档 -> {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  类型分布: {type_counts}")
    print(f"  联赛分布: {league_counts}")
    print(f"  示例 doc_id: {docs[0]['doc_id']}")


if __name__ == "__main__":
    main()

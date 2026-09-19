-- 首次启动空数据卷时自动执行。详见方案 v2 §3.2。

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============ 结构化赛事数据 ============

CREATE TABLE IF NOT EXISTS f_matches (
    match_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    league        TEXT NOT NULL,          -- football-data.co.uk 代码：E0/SP1/D1/I1/F1
    season        TEXT NOT NULL,          -- 如 2425
    match_date    DATE NOT NULL,
    home_team     TEXT NOT NULL,
    away_team     TEXT NOT NULL,
    fthg          INTEGER,                -- 全场主队进球
    ftag          INTEGER,                -- 全场客队进球
    full_time_res CHAR(1),                -- H/D/A
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (league, season, match_date, home_team, away_team)
);

CREATE TABLE IF NOT EXISTS f_odds (
    match_id BIGINT PRIMARY KEY REFERENCES f_matches(match_id) ON DELETE CASCADE,
    -- Bet365 临场（closing）1X2
    b365h    NUMERIC,
    b365d    NUMERIC,
    b365a    NUMERIC,
    -- Pinnacle 临场（sharp 盘，作为赔率 baseline 首选）
    psh     NUMERIC,
    psd     NUMERIC,
    psa     NUMERIC,
    -- 市场平均临场
    avgh    NUMERIC,
    avgd    NUMERIC,
    avga    NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_matches_league_date ON f_matches (league, match_date);

-- ============ RAG 向量库（只放非结构化文本）============

CREATE TABLE IF NOT EXISTS v_chunks (
    id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content   TEXT NOT NULL,
    doc_source TEXT,                      -- URL / 文件名
    doc_type  TEXT,                       -- news / tactics / interview
    league    TEXT,
    team      TEXT,
    doc_date  DATE,
    embedding vector(1024),               -- bge-m3: 1024 维
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS v_chunks_hnsw ON v_chunks
    USING hnsw (embedding vector_cosine_ops);

-- ============ 预测账本（赛前写入，赛后回填，准确率追踪依据）============

CREATE TABLE IF NOT EXISTS pred_ledger (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id      BIGINT REFERENCES f_matches(match_id) ON DELETE SET NULL,
    level         TEXT NOT NULL CHECK (level IN ('L1', 'L2', 'L3', 'L4')),
    model_version TEXT NOT NULL,
    prob_home     NUMERIC NOT NULL,
    prob_draw     NUMERIC NOT NULL,
    prob_away     NUMERIC NOT NULL,
    rationale     TEXT,                   -- LLM 解释（引用工具/RAG 来源）
    predicted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    actual_outcome CHAR(1),               -- 赛后回填 H/D/A
    settled_at    TIMESTAMPTZ,
    log_loss      NUMERIC,
    brier         NUMERIC,
    rps           NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_ledger_unsettled ON pred_ledger (settled_at) WHERE settled_at IS NULL;

package com.example.sportsbff.data;

import java.util.List;
import java.util.Map;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

/**
 * pred_ledger 预测账本查询（方案 §8 "/api/predictions"）。
 *
 * <p>BFF 直接读 PostgreSQL，不经过 Python runtime——预测账本是展示层需求。
 */
@Repository
public class PredictionRepository {

    private final JdbcTemplate jdbc;

    public PredictionRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    /** 最近 N 条预测记录。 */
    public List<Map<String, Object>> recent(int limit) {
        String sql = """
                SELECT prediction_id, level, model_version,
                       prob_home, prob_draw, prob_away,
                       predicted_at, actual_outcome, settled_at,
                       log_loss, brier, rps
                FROM pred_ledger
                ORDER BY predicted_at DESC
                LIMIT ?
                """;
        return jdbc.queryForList(sql, Math.min(limit, 100));
    }

    /** 未结算预测数。 */
    public int unsettledCount() {
        Integer cnt = jdbc.queryForObject(
                "SELECT COUNT(*) FROM pred_ledger WHERE settled_at IS NULL",
                Integer.class);
        return cnt != null ? cnt : 0;
    }

    /** 已结算预测准确率（按 actual_outcome 对比 argmax 概率）。 */
    public Map<String, Object> accuracySummary() {
        String sql = """
                SELECT
                    COUNT(*)                                                          AS total,
                    COUNT(*) FILTER (WHERE actual_outcome =
                        CASE WHEN prob_home >= prob_draw AND prob_home >= prob_away THEN 'H'
                              WHEN prob_away >= prob_home AND prob_away >= prob_draw THEN 'A'
                              ELSE 'D' END)                                                           AS correct,
                    AVG(log_loss) AS avg_log_loss,
                    AVG(brier)    AS avg_brier,
                    AVG(rps)      AS avg_rps
                FROM pred_ledger
                WHERE settled_at IS NOT NULL
                """;
        return jdbc.queryForMap(sql);
    }
}

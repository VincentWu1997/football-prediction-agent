package com.example.sportsbff.web;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.sportsbff.client.PythonRuntimeClient;
import com.example.sportsbff.data.PredictionRepository;

/**
 * BFF 对外接口（方案 §8）。
 *
 * <ul>
 *   <li>GET  /api/health       公开健康检查
 *   <li>GET  /api/tools        列出经 MCP 发现的全部工具
 *   <li>POST /api/analyze       透传 Python runtime /analyze
 *   <li>POST /api/predict       透传 Python runtime /predict
 *   <li>GET  /api/predictions   pred_ledger 最近记录 + 准确率汇总
 * </ul>
 */
@RestController
@RequestMapping("/api")
public class AnalysisController {

    private final ObjectProvider<ToolCallbackProvider> mcpTools;
    private final PythonRuntimeClient pythonClient;
    private final PredictionRepository ledger;

    public AnalysisController(ObjectProvider<ToolCallbackProvider> mcpTools,
                              PythonRuntimeClient pythonClient,
                              PredictionRepository ledger) {
        this.mcpTools = mcpTools;
        this.pythonClient = pythonClient;
        this.ledger = ledger;
    }

    /** 公开健康检查，不依赖任何下游服务。 */
    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of(
                "service", "sports-bff",
                "status", "ok",
                "python_runtime", pythonClient.isReachable() ? "reachable" : "unreachable"
        );
    }

    /**
     * 列出经 MCP 发现的全部工具（跨语言编排验证点）。
     * 前置：三个 Python MCP Server 已启动，且 application.yml 中 initialized=true。
     */
    @GetMapping("/tools")
    public List<String> tools() {
        ToolCallbackProvider provider = mcpTools.getIfAvailable();
        if (provider == null) {
            return List.of();
        }
        return java.util.Arrays.stream(provider.getToolCallbacks())
                .map(tc -> tc.getToolDefinition().name())
                .sorted()
                .toList();
    }

    /**
     * 分析请求入口：透传 Python FastAPI /analyze。
     * 请求体：{"query": "详细分析曼联对利物浦", "dry_run": false}
     */
    @PostMapping("/analyze")
    public Map<String, Object> analyze(@RequestBody Map<String, Object> request) {
        return pythonClient.analyze(request);
    }

    /**
     * 单场预测：透传 Python FastAPI /predict。
     * 请求体：{"home": "Manchester United", "away": "Liverpool", ...}
     */
    @PostMapping("/predict")
    public Map<String, Object> predict(@RequestBody Map<String, Object> request) {
        return pythonClient.predict(request);
    }

    /**
     * 预测账本查询（方案 §8 "/api/predictions"）。
     * ?limit=20 返回最近 N 条；不带 limit 时返回汇总。
     */
    @GetMapping("/predictions")
    public Map<String, Object> predictions(
            @RequestParam(defaultValue = "20") int limit,
            @RequestParam(defaultValue = "false") boolean summary) {
        if (summary) {
            return Map.of(
                    "accuracy", ledger.accuracySummary(),
                    "unsettled", ledger.unsettledCount()
            );
        }
        return Map.of("predictions", ledger.recent(limit));
    }
}

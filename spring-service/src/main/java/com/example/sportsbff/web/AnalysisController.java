package com.example.sportsbff.web;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.ToolCallbacks;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** BFF 对外接口。 */
@RestController
@RequestMapping("/api")
public class AnalysisController {

    private final ObjectProvider<ToolCallbackProvider> mcpTools;

    public AnalysisController(ObjectProvider<ToolCallbackProvider> mcpTools) {
        this.mcpTools = mcpTools;
    }

    /** 公开健康检查，不依赖任何下游服务。 */
    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of("service", "sports-bff", "status", "ok");
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
        return List.of(ToolCallbacks.from(provider)).stream()
                .map(ToolCallback::getToolMetadata)
                .map(meta -> meta.name())
                .sorted()
                .toList();
    }

    /**
     * 分析请求入口（W8 完整实现：透传 Python FastAPI /analyze，SSE 回传 trace）。
     */
    @PostMapping("/analyze")
    public Map<String, Object> analyze(@RequestBody Map<String, Object> request) {
        return Map.of(
                "status", "not_implemented",
                "todo", "W8: 透传 http://localhost:9000/analyze 并转发 SSE trace",
                "received", request);
    }
}

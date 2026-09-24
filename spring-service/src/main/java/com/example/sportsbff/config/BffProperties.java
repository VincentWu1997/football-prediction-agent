package com.example.sportsbff.config;

import java.util.List;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * BFF 自定义配置项（application.yml 中 bff.* 前缀）。
 *
 * @param pythonRuntimeUrl Python FastAPI 地址
 * @param apiKeys 允许的 API Key 列表（演示用）
 * @param rateLimit 限流参数
 */
@ConfigurationProperties(prefix = "bff")
public record BffProperties(
        String pythonRuntimeUrl,
        List<String> apiKeys,
        RateLimit rateLimit
) {
    public record RateLimit(int capacity, int refillPerMinute) {}
}

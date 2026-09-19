package com.example.sportsbff;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 体育预测 Agent 的企业服务层（BFF）。
 *
 * <p>最小版职责：对外 REST、基础鉴权、通过 MCP Client 发现并调用 Python 侧工具。
 * 完整版（方案 v2 §8）还包括会话记忆、限流、Java RAG 对照、可观测，当前按裁剪预案不实现。
 */
@SpringBootApplication
public class SportsBffApplication {

    public static void main(String[] args) {
        SpringApplication.run(SportsBffApplication.class, args);
    }
}

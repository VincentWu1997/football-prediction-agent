package com.example.sportsbff;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

import com.example.sportsbff.config.BffProperties;

/**
 * 体育预测 Agent 的企业服务层（BFF）。
 *
 * <p>职责：对外 REST、API Key 鉴权、限流、MCP Client 跨语言调 Python 工具、可观测。
 */
@SpringBootApplication
@EnableConfigurationProperties(BffProperties.class)
public class SportsBffApplication {

    public static void main(String[] args) {
        SpringApplication.run(SportsBffApplication.class, args);
    }
}

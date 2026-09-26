package com.example.sportsbff;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import com.example.sportsbff.data.PredictionRepository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.when;

/**
 * BFF 端点烟测：健康检查、鉴权、限流、预测账本。
 *
 * <p>不依赖 Python runtime（/api/analyze 会连接失败，只验证鉴权/路由层）；
 * PredictionRepository 被 mock，CI 无需 PostgreSQL。
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {
                "bff.api-keys=demo-key-001",
                "bff.python-runtime-url=http://localhost:1",
                "spring.ai.mcp.client.initialized=false",
                // CI 无 PostgreSQL：关闭数据源健康探针，否则 /actuator/health 返回 503
                "management.health.db.enabled=false"
        })
class AnalysisControllerTest {

    /** 用 mock 替换账本查询，避免测试环境直连 PostgreSQL。 */
    @MockitoBean
    PredictionRepository ledger;

    @LocalServerPort
    int port;

    @Autowired
    TestRestTemplate rest;

    @Test
    void healthIsPublic() {
        var resp = rest.getForEntity(url("/api/health"), String.class);
        assertThat(resp.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(resp.getBody()).contains("sports-bff");
    }

    @Test
    void predictionsWithoutKeyReturns401() {
        var resp = rest.getForEntity(url("/api/predictions"), String.class);
        assertThat(resp.getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    @Test
    void predictionsWithKeyReturns200() {
        // stub 账本查询返回空列表，不触发真实 JDBC 连接
        when(ledger.recent(anyInt())).thenReturn(java.util.List.of());
        var headers = new HttpHeaders();
        headers.set("X-API-Key", "demo-key-001");
        var resp = rest.exchange(url("/api/predictions"), HttpMethod.GET,
                new HttpEntity<>(headers), String.class);
        assertThat(resp.getStatusCode()).isEqualTo(HttpStatus.OK);
    }

    @Test
    void actuatorHealthIsPublic() {
        var resp = rest.getForEntity(url("/actuator/health"), String.class);
        assertThat(resp.getStatusCode()).isEqualTo(HttpStatus.OK);
    }

    @Test
    void invalidKeyReturns401() {
        var headers = new HttpHeaders();
        headers.set("X-API-Key", "wrong");
        var resp = rest.exchange(url("/api/predictions"), HttpMethod.GET,
                new HttpEntity<>(headers), String.class);
        assertThat(resp.getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    private String url(String path) {
        return "http://localhost:" + port + path;
    }
}

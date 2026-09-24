package com.example.sportsbff.client;

import java.util.Map;

import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import com.example.sportsbff.config.BffProperties;

/**
 * Python FastAPI runtime 代理客户端。
 *
 * <p>BFF 不重写 Python 逻辑，只做透传（方案 §8 "对外 API/BFF"）。
 */
@Component
public class PythonRuntimeClient {

    private final RestClient client;

    public PythonRuntimeClient(BffProperties props) {
        this.client = RestClient.builder()
                .baseUrl(props.pythonRuntimeUrl())
                .build();
    }

    /** 透传 POST /analyze。 */
    public Map<String, Object> analyze(Map<String, Object> request) {
        return client.post()
                .uri("/analyze")
                .body(request)
                .retrieve()
                .body(Map.class);
    }

    /** 透传 POST /predict。 */
    public Map<String, Object> predict(Map<String, Object> request) {
        return client.post()
                .uri("/predict")
                .body(request)
                .retrieve()
                .body(Map.class);
    }

    /** Python runtime 健康探测。 */
    public boolean isReachable() {
        try {
            client.get().uri("/health").retrieve().toEntity(String.class);
            return true;
        } catch (RestClientResponseException e) {
            return e.getStatusCode().is2xxSuccessful();
        } catch (Exception e) {
            return false;
        }
    }
}

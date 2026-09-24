package com.example.sportsbff.config;

import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.ConcurrentHashMap;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import io.github.bucket4j.Bandwidth;
import io.github.bucket4j.Bucket;

/**
 * 简单令牌桶限流过滤器：按客户端 IP 分桶，每分钟 capacity 个令牌。
 *
 * <p>健康检查和 Actuator 端点不限流。
 */
@Component
public class RateLimitFilter extends OncePerRequestFilter {

    private final ConcurrentHashMap<String, Bucket> buckets = new ConcurrentHashMap<>();
    private final BffProperties props;

    public RateLimitFilter(BffProperties props) {
        this.props = props;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest req, HttpServletResponse resp,
                                    FilterChain chain) throws ServletException, IOException {
        String path = req.getRequestURI();
        // 健康检查和 Actuator 不限流
        if (path.startsWith("/actuator") || path.equals("/api/health")) {
            chain.doFilter(req, resp);
            return;
        }
        String ip = extractClientIp(req);
        Bucket bucket = buckets.computeIfAbsent(ip, k -> newBucket());
        if (bucket.tryConsume(1)) {
            chain.doFilter(req, resp);
        } else {
            resp.setStatus(HttpStatus.TOO_MANY_REQUESTS.value());
            resp.setContentType("application/json");
            resp.getWriter().write("""
                    {"error": "rate limit exceeded"}""");
        }
    }

    private Bucket newBucket() {
        int capacity = props.rateLimit() != null ? props.rateLimit().capacity() : 20;
        int refill = props.rateLimit() != null ? props.rateLimit().refillPerMinute() : 20;
        Bandwidth limit = Bandwidth.builder()
                .capacity(capacity)
                .refillIntervally(refill, Duration.ofMinutes(1))
                .build();
        return Bucket.builder().addLimit(limit).build();
    }

    private static String extractClientIp(HttpServletRequest req) {
        String fwd = req.getHeader("X-Forwarded-For");
        if (fwd != null && !fwd.isBlank()) {
            return fwd.split(",")[0].trim();
        }
        return req.getRemoteAddr();
    }
}

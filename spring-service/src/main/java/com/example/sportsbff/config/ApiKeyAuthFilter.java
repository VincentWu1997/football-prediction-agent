package com.example.sportsbff.config;

import java.io.IOException;
import java.util.List;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.preauth.PreAuthenticatedAuthenticationToken;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * API Key 鉴权过滤器：检查请求头 X-API-Key 是否在白名单中。
 *
 * <p>未配置任何 key 时放行（方便本地开发）；配置了 key 但请求未带或不匹配则返回 401。
 */
public class ApiKeyAuthFilter extends OncePerRequestFilter {

    private final List<String> validKeys;

    public ApiKeyAuthFilter(List<String> validKeys) {
        this.validKeys = validKeys == null ? List.of() : validKeys;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest req, HttpServletResponse resp,
                                    FilterChain chain) throws ServletException, IOException {
        // 未配置 key 时跳过鉴权（本地开发模式）
        if (validKeys.isEmpty()) {
            chain.doFilter(req, resp);
            return;
        }
        // 健康检查和 Actuator 端点公开
        String path = req.getRequestURI();
        if (path.equals("/api/health") || path.startsWith("/actuator")) {
            chain.doFilter(req, resp);
            return;
        }
        String key = req.getHeader("X-API-Key");
        if (key != null && validKeys.contains(key)) {
            var auth = new PreAuthenticatedAuthenticationToken(
                    "api-client", key, java.util.List.of());
            SecurityContextHolder.getContext().setAuthentication(auth);
            chain.doFilter(req, resp);
            return;
        }
        resp.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        resp.setContentType("application/json");
        resp.getWriter().write("""
                {"error": "missing or invalid X-API-Key"}""");
    }
}

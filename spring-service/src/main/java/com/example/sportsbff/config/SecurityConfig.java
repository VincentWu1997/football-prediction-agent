package com.example.sportsbff.config;

import java.util.List;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

/**
 * 安全配置：API Key 鉴权（X-API-Key 头），REST API 关闭 CSRF 与 Session。
 *
 * <p>健康检查和 Actuator 端点公开；其余 /api/** 需带有效 API Key。
 */
@Configuration
public class SecurityConfig {

    private final BffProperties props;

    public SecurityConfig(BffProperties props) {
        this.props = props;
    }

    @Bean
    SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .csrf(csrf -> csrf.ignoringRequestMatchers("/api/**", "/actuator/**"))
                .sessionManagement(sm -> sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers("/api/health", "/actuator/**").permitAll()
                        .anyRequest().authenticated())
                .addFilterBefore(new ApiKeyAuthFilter(props.apiKeys()),
                        UsernamePasswordAuthenticationFilter.class);
        return http.build();
    }
}

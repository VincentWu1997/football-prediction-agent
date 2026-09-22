-- W5：v_chunks 幂等入库约束
-- 已初始化的 volume 不会重跑 01_init.sql，故单独建此文件
-- 首次手动执行：docker exec sports-agent-pg psql -U sports -d sports -f /docker-entrypoint-initdb.d/02_rag_unique.sql

ALTER TABLE v_chunks
    ADD CONSTRAINT IF NOT EXISTS v_chunks_doc_source_content_key
    UNIQUE (doc_source, content);

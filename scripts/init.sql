-- ============================================================
-- A股智能投研与预警 Agent - MySQL 初始化脚本
-- ============================================================

-- 设置字符集
SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

-- 使用数据库
USE agent_db;

-- ============================================================
-- 用户表
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '用户ID',
    feishu_open_id VARCHAR(64) UNIQUE NOT NULL COMMENT '飞书用户Open ID',
    nickname VARCHAR(64) DEFAULT NULL COMMENT '用户昵称',
    avatar_url VARCHAR(256) DEFAULT NULL COMMENT '头像URL',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_feishu_open_id (feishu_open_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- ============================================================
-- 用户画像表
-- ============================================================
CREATE TABLE IF NOT EXISTS user_profiles (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '画像ID',
    user_id BIGINT UNIQUE NOT NULL COMMENT '用户ID',
    investment_style ENUM('conservative', 'balanced', 'aggressive') DEFAULT 'balanced' COMMENT '投资风格: conservative-保守型, balanced-平衡型, aggressive-进攻型',
    focus_sectors JSON DEFAULT NULL COMMENT '关注板块列表，如 ["半导体", "新能源"]',
    risk_tolerance INT DEFAULT 5 COMMENT '风险承受度 1-10',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户画像表';

-- ============================================================
-- 预警规则表
-- ============================================================
CREATE TABLE IF NOT EXISTS alert_rules (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '规则ID',
    user_id BIGINT NOT NULL COMMENT '用户ID',
    stock_code VARCHAR(16) NOT NULL COMMENT '股票代码，如 300567',
    stock_name VARCHAR(32) DEFAULT NULL COMMENT '股票名称',
    rule_type ENUM('price_up', 'price_down', 'volume', 'turnover', 'custom') NOT NULL COMMENT '规则类型: price_up-涨幅, price_down-跌幅, volume-成交量, turnover-换手率, custom-自定义',
    threshold DECIMAL(10,4) NOT NULL COMMENT '阈值',
    unit ENUM('percent', 'absolute', 'times') DEFAULT 'percent' COMMENT '单位: percent-百分比, absolute-绝对值, times-倍数',
    status ENUM('active', 'paused', 'deleted') DEFAULT 'active' COMMENT '状态: active-生效, paused-暂停, deleted-删除',
    cooldown_minutes INT DEFAULT 60 COMMENT '冷却时间(分钟)，同一股票触发后需等待的时间',
    last_triggered_at TIMESTAMP NULL DEFAULT NULL COMMENT '上次触发时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_stock (user_id, stock_code),
    INDEX idx_status (status),
    INDEX idx_user_status (user_id, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='预警规则表';

-- ============================================================
-- 预警历史表
-- ============================================================
CREATE TABLE IF NOT EXISTS alert_history (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '历史ID',
    rule_id BIGINT NOT NULL COMMENT '规则ID',
    user_id BIGINT NOT NULL COMMENT '用户ID',
    stock_code VARCHAR(16) NOT NULL COMMENT '股票代码',
    stock_name VARCHAR(32) DEFAULT NULL COMMENT '股票名称',
    trigger_price DECIMAL(10,4) DEFAULT NULL COMMENT '触发时价格',
    trigger_value DECIMAL(10,4) DEFAULT NULL COMMENT '触发值',
    trigger_reason VARCHAR(256) DEFAULT NULL COMMENT '触发原因描述',
    analysis_result TEXT DEFAULT NULL COMMENT 'AI 分析结果',
    feishu_message_id VARCHAR(64) DEFAULT NULL COMMENT '飞书消息ID',
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '发送时间',
    FOREIGN KEY (rule_id) REFERENCES alert_rules(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_rule_time (rule_id, sent_at),
    INDEX idx_user_time (user_id, sent_at),
    INDEX idx_stock_time (stock_code, sent_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='预警历史表';

-- ============================================================
-- 插入测试数据 (可选)
-- ============================================================

-- 测试用户
INSERT INTO users (feishu_open_id, nickname) VALUES 
    ('ou_test_user_001', '测试用户1'),
    ('ou_test_user_002', '测试用户2')
ON DUPLICATE KEY UPDATE nickname = VALUES(nickname);

-- 用户画像
INSERT INTO user_profiles (user_id, investment_style, focus_sectors, risk_tolerance) VALUES 
    (1, 'balanced', '["半导体", "新能源", "医药"]', 5),
    (2, 'aggressive', '["科技", "军工"]', 8)
ON DUPLICATE KEY UPDATE investment_style = VALUES(investment_style);

-- 测试预警规则
INSERT INTO alert_rules (user_id, stock_code, stock_name, rule_type, threshold, unit, status, cooldown_minutes) VALUES 
    (1, '300567', '精测电子', 'price_up', 5.0, 'percent', 'active', 60),
    (1, '600519', '贵州茅台', 'price_down', 3.0, 'percent', 'active', 30),
    (2, '000858', '五粮液', 'volume', 2.0, 'times', 'active', 60)
ON DUPLICATE KEY UPDATE threshold = VALUES(threshold);

-- ============================================================
-- 创建视图：活跃规则统计
-- ============================================================
CREATE OR REPLACE VIEW v_active_rules_stats AS
SELECT 
    u.id AS user_id,
    u.nickname,
    u.feishu_open_id,
    COUNT(ar.id) AS total_rules,
    SUM(CASE WHEN ar.status = 'active' THEN 1 ELSE 0 END) AS active_rules,
    SUM(CASE WHEN ar.status = 'paused' THEN 1 ELSE 0 END) AS paused_rules
FROM users u
LEFT JOIN alert_rules ar ON u.id = ar.user_id
GROUP BY u.id, u.nickname, u.feishu_open_id;

-- ============================================================
-- 完成
-- ============================================================
SELECT 'MySQL 初始化完成!' AS message;

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Rule:
    rule_id: str
    reason: str
    weight: float
    patterns: tuple[str, ...]


RULES: tuple[Rule, ...] = (
    Rule("instruction_override", "instruction override pattern detected", 0.44, (
        r"\b(ignore|disregard|forget|bypass|override)\b.{0,48}\b(previous|prior|system|developer|safety|instruction|rule)s?\b",
        r"\b(bỏ qua|phớt lờ|quên|ghi đè|vượt qua)\b.{0,48}\b(chỉ dẫn|hướng dẫn|quy tắc|hệ thống|trước đó)\b",
    )),
    Rule("role_manipulation", "attempt to alter assistant role", 0.30, (
        r"\b(you are now|act as|switch roles?|new system message|enter .{0,20} mode)\b",
        r"\b(bây giờ bạn là|hãy đóng vai|đổi vai|thông điệp hệ thống mới)\b",
    )),
    Rule("system_prompt_extraction", "attempt to extract hidden instructions", 0.48, (
        r"\b(reveal|show|print|repeat|leak|expose)\b.{0,48}\b(system prompt|hidden instruction|developer message|internal context)\b",
        r"\b(tiết lộ|hiển thị|in ra|lặp lại)\b.{0,48}\b(system prompt|chỉ dẫn ẩn|thông điệp developer|ngữ cảnh nội bộ)\b",
    )),
    Rule("context_takeover", "context priority takeover pattern detected", 0.40, (
        r"\b(follow|obey|execute)\b.{0,40}\b(this|these|following)\b.{0,30}\b(instruction|command)s?\b.{0,24}\b(instead|higher priority|above all)\b",
        r"\b(chỉ làm theo|tuân theo|thực hiện)\b.{0,50}\b(thay vì|ưu tiên cao hơn|trên hết)\b",
    )),
    Rule("rag_reader_targeting", "retrieved-document instruction targets an AI reader", 0.46, (
        r"\b(when|if)\b.{0,28}\b(ai|assistant|model|llm|chatbot)\b.{0,22}\b(reads?|processes?|retrieves?|sees?)\b",
        r"\b(khi|nếu)\b.{0,28}\b(ai|trợ lý|mô hình|llm|chatbot)\b.{0,28}\b(đọc|xử lý|truy xuất|nhìn thấy)\b",
    )),
    Rule("response_control", "attempt to control the model response from retrieved text", 0.35, (
        r"\b(send|return|answer|respond with|output)\b.{0,70}\b(exactly|only|following|instead)\b",
        r"\b(trả lời|phản hồi|xuất ra|gửi)\b.{0,70}\b(chính xác|chỉ|sau đây|thay thế)\b",
    )),
    Rule("concealment", "instruction concealment pattern detected", 0.28, (
        r"\b(do not|don't|never)\b.{0,30}\b(mention|reveal|tell|disclose)\b.{0,28}\b(this|instruction|request|command)\b",
        r"\b(đừng|không được)\b.{0,30}\b(nhắc|tiết lộ|nói)\b.{0,28}\b(chỉ dẫn|yêu cầu|lệnh|điều này)\b",
    )),
    Rule("data_exfiltration", "possible secret or internal-context exfiltration", 0.44, (
        r"\b(reveal|send|upload|exfiltrate|copy|steal)\b.{0,55}\b(api[ _-]?keys?|secret|password|token|credentials?|internal context|private data)\b",
        r"\b(tiết lộ|gửi|tải lên|sao chép|đánh cắp)\b.{0,55}\b(api key|bí mật|mật khẩu|token|thông tin đăng nhập|dữ liệu riêng)\b",
    )),
)


def detect_prompt_injection(text: str) -> dict:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    matched: list[str] = []
    reasons: list[str] = []
    weights: list[float] = []
    for rule in RULES:
        matches = [
            match
            for pattern in rule.patterns
            if (match := re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        ]
        # Security guidance commonly says "do not reveal API keys". That is a
        # prohibition, not an exfiltration command, so suppress this rule when
        # the action verb is directly negated.
        if rule.rule_id == "data_exfiltration" and matches:
            non_negated = []
            for match in matches:
                prefix = text[max(0, match.start() - 24):match.start()].casefold()
                if not re.search(r"(?:do not|don't|never|should not|không được|đừng)\s*$", prefix):
                    non_negated.append(match)
            matches = non_negated
        if matches:
            matched.append(rule.rule_id)
            reasons.append(rule.reason)
            weights.append(rule.weight)

    # No singleton generic keyword rule can block. Multiple independent rule
    # families amplify confidence without summing unboundedly.
    score = 0.0
    if weights:
        score = max(weights)
        if len(weights) > 1:
            score += min(0.35, 0.13 * (len(weights) - 1))
    return {
        "score": round(min(1.0, score), 6),
        "matched_rules": matched,
        "reasons": reasons,
    }

class QuestionAnalyzer:
    def __init__(self, subject):
        self.prompt = \
r'''
당신은 {subject} 과외 수업에서 선생님의 발화를 분석하여, 각 문장이 질문인지 아닌지 구별해야합니다.
주의사항과 질문 제외 사항을 잘 숙지하고, 각 문장의 주변 문맥을 파악하고, 충분히 생각한 다음, 해당 문장이 질문일 경우 idx를 반환하세요.

## 주의 사항
- 질문의 주변 문맥을 고려하여, 단순한 확언문이 아닌 질문인 경우를 판단해야 합니다.
- 문장의 끝에 '?'가 있을 경우 질문으로 간주합니다.
- "왜 ~인지", "어떻게 ~하는지", "무엇을 ~하는지", "왜 ~죠" 등의 형태는 질문으로 간주합니다.

## 질문 제외 사항
다음과 같은 형식의 질문은 제외합니다.
- "네?", "예?", "그렇죠?", "맞죠?", "그쵸?", "아니죠?", "아닌가요?", "뭐죠?"

## 출력 형식
```json
[
    {{
        "idx": [List]
    }}
]
'''
        self.prompt = self.prompt.replace("{subject}", subject)
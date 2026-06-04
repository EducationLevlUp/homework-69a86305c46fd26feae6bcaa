#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

"""
Human-in-the-Loop агент через HumanInTheLoopMiddleware.

Агент останавливается перед каждым вызовом инструмента, запрашивает
подтверждение у пользователя (approve / reject / edit) и возобновляет
выполнение через Command(resume=...).

Запуск:
    python hitl_agent.py

Требования:
    - переменная окружения OPENAI_API_KEY
    - langchain, langgraph, langchain-openai
"""


# ---------------------------------------------------------------------------
# Инструменты
# ---------------------------------------------------------------------------

@tool
def get_weather(city: str, date: str = "сегодня") -> str:
    """Получить текущую погоду для указанного города."""
    return f"В {city} ({date}): +20 °C, солнечно, без осадков."


# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------

def _get_attr_or_item(obj: Any, key: str, default: Any = None) -> Any:
    """Получить значение как атрибут объекта или элемент словаря."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_action_info(action: Any) -> dict:
    """Извлечь name, args, description из action (dict или объект)."""
    return {
        "name": _get_attr_or_item(action, "name", "unknown"),
        "args": _get_attr_or_item(action, "args", {}),
        "description": _get_attr_or_item(action, "description", ""),
    }


# ---------------------------------------------------------------------------
# Инициализация агента
# ---------------------------------------------------------------------------

def create_hitl_agent() -> tuple:
    """
    Создать агента с HumanInTheLoopMiddleware и MemorySaver как checkpointer.

    Returns
    -------
    tuple[agent, MemorySaver]
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "⚠️  OPENAI_API_KEY не установлен.\n"
            "   Установите переменную окружения для работы с LLM:\n"
            "   export OPENAI_API_KEY='sk-...'\n"
            "   Запуск завершён."
        )
        sys.exit(1)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=[get_weather],
        system_prompt="Ты полезный ассистент.",
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "get_weather": True,  # все решения: approve, edit, reject
                },
                description_prefix="Подтвердите вызов инструмента",
            )
        ],
        checkpointer=memory,
    )
    return agent, memory


# ---------------------------------------------------------------------------
# Сбор решений от пользователя
# ---------------------------------------------------------------------------

def collect_decisions(
    action_requests: list,
    review_configs: list,
) -> list[dict]:
    """
    Для каждого action_request вывести информацию и запросить решение.

    Parameters
    ----------
    action_requests : list
        Список действий, ожидающих решения.
    review_configs : list
        Конфигурации разрешённых решений для каждого действия.

    Returns
    -------
    list[dict]
        Список решений в том же порядке, что и action_requests.
    """
    decisions: list[dict] = []

    for i, action in enumerate(action_requests):
        info = _extract_action_info(action)
        name = info["name"]
        args = info["args"]
        description = info["description"]

        print(f"\n{'─' * 40}")
        print(f"🔧 Инструмент: {name}")
        print(f"📦 Аргументы: {args}")
        if description:
            print(f"📝 Описание: {description}")
        print(f"{'─' * 40}")

        # Определяем доступные решения из review_configs
        allowed = ["approve", "reject"]
        if review_configs and i < len(review_configs):
            rc = review_configs[i]
            if isinstance(rc, dict):
                allowed = rc.get("allowed_decisions", allowed)
            elif hasattr(rc, "allowed_decisions"):
                allowed = rc.allowed_decisions

        has_edit = "edit" in allowed
        prompt = "a = approve, r = reject"
        if has_edit:
            prompt += ", e = edit"
        prompt += ": "

        choice = input(prompt).strip().lower()

        if choice in ("a", "approve"):
            decisions.append({"type": "approve"})

        elif choice in ("r", "reject"):
            message = input(
                "Сообщение для агента (причина отказа): "
            ).strip()
            if not message:
                message = "Пользователь отклонил действие"
            decisions.append({"type": "reject", "message": message})

        elif choice in ("e", "edit"):
            if not has_edit:
                print("⚠️  Edit не разрешён для этого действия, одобряю автоматически.")
                decisions.append({"type": "approve"})
            else:
                print("Введите новые аргументы в формате JSON:")
                new_args_str = input("> ").strip()
                try:
                    new_args = json.loads(new_args_str)
                except json.JSONDecodeError:
                    print("⚠️  Неверный JSON, используются исходные аргументы.")
                    new_args = args
                decisions.append({
                    "type": "edit",
                    "edited_action": {
                        "name": name,
                        "args": new_args,
                    },
                })

        else:
            print(f"⚠️  Непонятный ввод '{choice}', одобряю автоматически.")
            decisions.append({"type": "approve"})

    return decisions


# ---------------------------------------------------------------------------
# Интерактивная сессия
# ---------------------------------------------------------------------------

def run_session(agent, config: dict) -> None:
    """
    Запустить интерактивную сессию с HITL.

    Схема:
    1. Первый вызов: agent.invoke({"messages": [...]}, config)
    2. Пока в result есть '__interrupt__':
       - извлечь action_requests и review_configs
       - вывести информацию в терминал
       - запросить решения (approve/reject/edit)
       - agent.invoke(Command(resume={"decisions": ...}), config)
    3. Когда '__interrupt__' нет — вывести финальный ответ.
    """
    print("\n💬 Введите сообщение (или 'exit' для выхода):")

    while True:
        user_input = input("\nВы: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "выход"):
            print("👋 Сессия завершена.")
            break

        # --- Шаг 1: первый вызов агента ---
        result = agent.invoke(
            {"messages": [{"role": "human", "content": user_input}]},
            config=config,
        )

        # --- Шаг 2: цикл обработки прерываний ---
        while "__interrupt__" in result:
            interrupt_entry = result["__interrupt__"][0]
            interrupt_value = interrupt_entry.value

            # Извлекаем action_requests и review_configs
            if isinstance(interrupt_value, dict):
                action_requests = interrupt_value.get("action_requests", [])
                review_configs = interrupt_value.get("review_configs", [])
            else:
                action_requests = getattr(interrupt_value, "action_requests", [])
                review_configs = getattr(interrupt_value, "review_configs", [])

            if not action_requests:
                print("⚠️  Прерывание без action_requests, пропускаю.")
                break

            # Собираем решения от пользователя
            decisions = collect_decisions(action_requests, review_configs)

            # --- Шаг 3: возобновление через Command ---
            result = agent.invoke(
                Command(resume={"decisions": decisions}),
                config=config,
            )

        # --- Шаг 4: вывод финального ответа ---
        if "__interrupt__" not in result and "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                print(f"\n🤖 Агент: {last_message.content}")
            elif isinstance(last_message, dict):
                print(f"\n🤖 Агент: {last_message.get('content', '')}")
            else:
                print(f"\n🤖 Агент: {last_message}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    """Создать агента и запустить интерактивную сессию."""
    agent, memory = create_hitl_agent()
    config = {"configurable": {"thread_id": "сессия-1"}}
    run_session(agent, config)


if __name__ == "__main__":
    main()

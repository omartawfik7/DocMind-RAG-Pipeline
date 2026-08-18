"""agents — Supervisor/Planner, Retrieval, and Analysis agents.

The Supervisor coordinates the other two rather than any single LLM
call doing retrieval, generation, and judgment all at once. See
supervisor.py for the orchestration logic and docs/INTERVIEW_GUIDE.md
for "why multiple agents".
"""
